import argparse
import os
import time
from typing import List, Tuple

import sys
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from path_planner import astar, inflate_obstacles

# Add DrivableAreaController to path so we can import TwinLiteNetPlus
sys.path.insert(0, "/home/maynafiesta/myWorkspace/DrivableAreaController/python")
from model.TwinLiteNetPlus.model import TwinLiteNetPlus

OBSTACLE_CLASSES = {0, 1, 2, 3, 5, 7}

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    if auto:
        dw = np.mod(dw, stride)
        dh = np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, (left, top, right, bottom)


def detection_contacts(result) -> List[Tuple[int, int, int]]:
    if result.boxes is None or result.boxes.cls is None:
        return []

    contacts: List[Tuple[int, int, int]] = []
    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.int().cpu().tolist()
    for box, cls_id in zip(boxes, class_ids):
        if cls_id not in OBSTACLE_CLASSES:
            continue
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) * 0.5)
        bottom_y = int(y2)
        width = max(4, int(x2 - x1))
        contacts.append((cx, bottom_y, width))
    return contacts


def build_cost_grid(occupied: np.ndarray, risk: np.ndarray) -> np.ndarray:
    cost = np.ones(occupied.shape, dtype=np.float32)
    cost += risk.astype(np.float32) * 12.0
    cost[occupied] = 1e6

    # Prefer the center corridor
    _, grid_w = occupied.shape
    x = np.arange(grid_w, dtype=np.float32)
    center_penalty = np.abs(x - (grid_w - 1) * 0.5) / max(1.0, grid_w * 0.5)
    cost += (center_penalty[None, :] ** 2) * 8.0
    return cost


def draw_grid(occupied: np.ndarray, risk: np.ndarray, path: List[Tuple[int, int]], out_size: Tuple[int, int]) -> np.ndarray:
    grid_h, grid_w = occupied.shape
    vis = np.full((grid_h, grid_w, 3), 235, dtype=np.uint8)
    vis[risk] = (0, 210, 255)
    vis[occupied] = (0, 0, 255)

    for y, x in path:
        if 0 <= y < grid_h and 0 <= x < grid_w:
            vis[y, x] = (0, 200, 0)

    vis = cv2.resize(vis, out_size, interpolation=cv2.INTER_NEAREST)
    cell_h = out_size[1] / grid_h
    cell_w = out_size[0] / grid_w
    for y, x in path:
        px = int((x + 0.5) * cell_w)
        py = int((y + 0.5) * cell_h)
        cv2.circle(vis, (px, py), 3, (0, 120, 0), -1)

    cv2.putText(vis, "BEV Occupancy / Risk Grid", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
    cv2.putText(vis, "red: occupied  yellow: risk  green: path", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
    return vis

def smooth_points(points: List[Tuple[int, int]], window: int) -> List[Tuple[int, int]]:
    if len(points) < 3 or window <= 1:
        return points
    window = max(3, window | 1)
    half = window // 2
    arr = np.array(points, dtype=np.float32)
    out = []
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        out.append(tuple(np.mean(arr[lo:hi], axis=0).astype(int)))
    return out

def main() -> None:
    parser = argparse.ArgumentParser(description="Single-camera local occupancy and path planning demo.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="YOLO segmentation model.")
    parser.add_argument("--device", default="cuda:0", help="Inference device: cuda:0 or cpu.")
    parser.add_argument("--grid_h", type=int, default=100, help="Local grid height.")
    parser.add_argument("--grid_w", type=int, default=60, help="Local grid width.")
    parser.add_argument("--risk_radius", type=int, default=3, help="Obstacle inflation radius in grid cells.")
    parser.add_argument("--output", default="outputs/planning_demo.mp4", help="Output video path.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Cannot read first frame.")
    frame_h, frame_w = frame.shape[:2]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    panel_w = 500
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, 20.0, (frame_w + panel_w, frame_h))

    model = YOLO(args.model)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Initialize Drivable Area Model (PyTorch)
    pth_path = "/home/maynafiesta/myWorkspace/DrivableAreaController/python/Weights/TwinLiteNetPlus/large.pth"
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"PyTorch weights not found: {pth_path}")
    
    device = torch.device(args.device)
    da_model = TwinLiteNetPlus(config="large").to(device).eval()
    
    ckpt = torch.load(pth_path, map_location=device, weights_only=True)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    da_model.load_state_dict(cleaned, strict=True)
    if device.type == "cuda":
        da_model = da_model.half()

    grid_shape = (args.grid_h, args.grid_w)
    start = (args.grid_h - 2, args.grid_w // 2)
    goal = (2, args.grid_w // 2)

    # --- FIXED PERSPECTIVE TRANSFORM (IPM) ---
    # Define a fixed trapezoid in the image that represents the ground in front of the car
    horizon_y = int(frame_h * 0.55)
    src_pts = np.array([
        [frame_w * 0.35, horizon_y],       # Top Left
        [frame_w * 0.65, horizon_y],       # Top Right
        [frame_w * 0.95, frame_h - 1],     # Bottom Right
        [frame_w * 0.05, frame_h - 1]      # Bottom Left
    ], dtype=np.float32)
    
    dst_pts = np.array([
        [0, 0],
        [args.grid_w - 1, 0],
        [args.grid_w - 1, args.grid_h - 1],
        [0, args.grid_h - 1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_inv = np.linalg.inv(M)

    last_t = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # --- YOLO Inference ---
        result = model(frame, verbose=False, conf=0.35, device=args.device)[0]
        contacts = detection_contacts(result)

        # --- Drivable Area Inference (PyTorch) ---
        lb, (left, top, right, bottom) = letterbox(frame, new_shape=(640, 640), auto=False)
        x = lb[:, :, ::-1].transpose(2, 0, 1)
        x = np.ascontiguousarray(x)
        
        x_tensor = torch.from_numpy(x).to(device)
        if device.type == "cuda":
            x_tensor = x_tensor.half()
        else:
            x_tensor = x_tensor.float()
        x_tensor = x_tensor.unsqueeze(0) / 255.0
        
        with torch.no_grad():
            da_out, _ = da_model(x_tensor)
            
        da_pred = torch.argmax(da_out, dim=1)[0]
        da_mask_raw = da_pred.cpu().numpy().astype(np.uint8)
        
        y1, y2 = top, 640 - bottom
        x1, x2 = left, 640 - right
        da_mask_cropped = da_mask_raw[y1:y2, x1:x2]
        da_mask = cv2.resize(da_mask_cropped, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

        # Overlay Drivable Area on frame
        overlay = frame.copy()
        color_mask = np.zeros_like(frame, dtype=np.uint8)
        color_mask[da_mask == 1] = (0, 255, 0)
        overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.35, 0)

        # --- Draw the fixed IPM Trapezoid (Helps visualize the projection area) ---
        cv2.polylines(overlay, [src_pts.astype(np.int32)], True, (0, 165, 255), 2)

        # --- Sensor Fusion in BEV ---
        da_bev = cv2.warpPerspective(da_mask, M, (args.grid_w, args.grid_h), flags=cv2.INTER_NEAREST)
        occupied = (da_bev == 0) # Non-drivable is occupied

        # Project YOLO boxes to BEV Grid
        if result.boxes is not None:
            for box, cls_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.int().cpu().tolist()):
                if cls_id not in OBSTACLE_CLASSES:
                    continue
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(overlay, ((x1 + x2) // 2, y2), 4, (0, 255, 255), -1)

        # Add obstacles to grid
        if len(contacts) > 0:
            # Transform all contact points at once
            pts_img = np.array([[[c[0], c[1]]] for c in contacts], dtype=np.float32)
            pts_grid = cv2.perspectiveTransform(pts_img, M)
            
            for i, (gx, gy) in enumerate(pts_grid[:, 0, :]):
                gy_int, gx_int = int(gy), int(gx)
                # Filter points outside the grid or behind the camera
                if 0 <= gy_int < args.grid_h and 0 <= gx_int < args.grid_w:
                    width_px = contacts[i][2]
                    # Estimate radius based on bounding box width and depth (gy)
                    # Objects closer (higher gy) should be larger
                    radius = int(max(1, (width_px / frame_w) * args.grid_w * 0.4))
                    
                    yy, xx = np.ogrid[:args.grid_h, :args.grid_w]
                    occupied |= (yy - gy_int) ** 2 + (xx - gx_int) ** 2 <= radius ** 2

        risk = inflate_obstacles(occupied, args.risk_radius) & ~occupied
        cost = build_cost_grid(occupied, risk)
        
        # Ensure start is always clear
        start = (args.grid_h - 2, args.grid_w // 2)
        cost[start[0], start[1]] = 1.0
        
        # Dynamic goal selection: Find the highest drivable cell near the top center
        goal = (2, args.grid_w // 2)
        found = False
        for y in range(2, args.grid_h // 2):
            for dx in range(args.grid_w // 2):
                if cost[y, args.grid_w // 2 + dx] < 1e6:
                    goal = (y, args.grid_w // 2 + dx)
                    found = True
                    break
                if cost[y, args.grid_w // 2 - dx] < 1e6:
                    goal = (y, args.grid_w // 2 - dx)
                    found = True
                    break
            if found:
                break
                
        path = astar(cost, start, goal)

        # --- Project Path Back to Image ---
        if path and len(path) > 2:
            path_grid_pts = np.array([[[x, y]] for y, x in path], dtype=np.float32)
            path_img_pts = cv2.perspectiveTransform(path_grid_pts, M_inv)
            
            pts = []
            for px, py in path_img_pts[:, 0, :]:
                if py > horizon_y: # Only draw path below horizon
                    pts.append((int(px), int(py)))
            
            pts = smooth_points(pts, window=11)
            if len(pts) >= 2:
                cv2.polylines(overlay, [np.array(pts, dtype=np.int32)], False, (0, 255, 0), 4)

        now = time.time()
        fps = 1.0 / max(now - last_t, 1e-3)
        last_t = now
        cv2.putText(overlay, f"FPS: {fps:.1f}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)

        grid_panel = draw_grid(occupied, risk, path, (panel_w, frame_h))
        canvas = np.zeros((frame_h, frame_w + panel_w, 3), dtype=np.uint8)
        canvas[:, :frame_w] = overlay
        canvas[:, frame_w:] = grid_panel

        out.write(canvas)
        cv2.imshow("BEV Planning Demo", canvas)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
