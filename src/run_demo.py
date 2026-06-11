import argparse
import os
import time
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO

from homography import build_homography, warp_binary_mask
from occupancy_grid import OccupancyGrid


# COCO class ids treated as dynamic/static obstacles for occupancy.
OBSTACLE_CLASSES = {0, 1, 2, 3, 5, 7}


def parse_points(raw: str) -> np.ndarray:
    values = [float(v.strip()) for v in raw.split(",")]
    if len(values) != 8:
        raise ValueError("src_points must contain 8 comma-separated numbers.")
    pts = np.array(values, dtype=np.float32).reshape(4, 2)
    ordered = order_points_tl_tr_br_bl(pts)
    area = cv2.contourArea(ordered.astype(np.float32))
    if area < 1000:
        raise ValueError(
            "Selected polygon area is too small/degenerate. Re-pick wider ground-plane points."
        )
    return ordered


def order_points_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Normalize arbitrary 4-point order to TL, TR, BR, BL for homography.

    This variant is more stable than sum/diff heuristics for narrow perspective trapezoids.
    """
    pts = pts.astype(np.float32)
    y_sorted = pts[np.argsort(pts[:, 1])]
    top = y_sorted[:2]
    bottom = y_sorted[2:]

    top = top[np.argsort(top[:, 0])]          # left -> right
    bottom = bottom[np.argsort(bottom[:, 0])] # left -> right

    tl = top[0]
    tr = top[1]
    bl = bottom[0]
    br = bottom[1]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def masks_to_obstacle_mask(result, shape_hw) -> np.ndarray:
    h, w = shape_hw
    obs = np.zeros((h, w), dtype=np.uint8)
    if result.masks is None:
        return obs
    boxes = result.boxes
    if boxes is None or boxes.cls is None:
        return obs
    class_ids: List[int] = boxes.cls.int().cpu().tolist()
    masks = result.masks.data.cpu().numpy()
    for cls_id, mask in zip(class_ids, masks):
        if cls_id in OBSTACLE_CLASSES:
            m = (mask > 0.5).astype(np.uint8)
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            obs = np.maximum(obs, m)
    return obs


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-camera occupancy grid demo.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="YOLO segmentation model.")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Inference device (e.g. cuda:0, cpu). Use cpu if CUDA/cuDNN mismatch exists.",
    )
    parser.add_argument("--bev_h", type=int, default=500, help="BEV height in pixels.")
    parser.add_argument("--bev_w", type=int, default=500, help="BEV width in pixels.")
    parser.add_argument("--cell_px", type=int, default=5, help="Grid cell size in pixels.")
    parser.add_argument(
        "--src_points",
        required=True,
        help=(
            "4 source points (clockwise) as x1,y1,x2,y2,x3,y3,x4,y4. "
            "These points should bound the drivable ground plane in the image."
        ),
    )
    parser.add_argument("--output", default="outputs/occupancy_demo.mp4", help="Output video.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame.")

    h, w = frame.shape[:2]
    src_pts = parse_points(args.src_points)
    H = build_homography(src_pts, (args.bev_h, args.bev_w))

    if args.bev_h % args.cell_px != 0 or args.bev_w % args.cell_px != 0:
        raise ValueError("bev_h and bev_w must be divisible by cell_px.")

    grid_h = args.bev_h // args.cell_px
    grid_w = args.bev_w // args.cell_px
    occ_grid = OccupancyGrid(grid_h, grid_w)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, 20.0, (w + args.bev_w, h))
    if not out.isOpened():
        raise RuntimeError(f"Cannot open writer: {args.output}")

    model = YOLO(args.model)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    last_t = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_h, frame_w = frame.shape[:2]

        results = model(frame, verbose=False, conf=0.35, device=args.device)
        res = results[0]

        obs_mask = masks_to_obstacle_mask(res, (frame_h, frame_w))

        visible_ground = np.zeros((frame_h, frame_w), dtype=np.uint8)
        cv2.fillConvexPoly(visible_ground, src_pts.astype(np.int32), 1)

        # Restrict obstacle evidence to the selected ground-plane polygon.
        obs_mask_roi = cv2.bitwise_and(obs_mask, visible_ground)
        bev_obs = warp_binary_mask(obs_mask_roi, H, (args.bev_h, args.bev_w))
        bev_seen = warp_binary_mask(visible_ground, H, (args.bev_h, args.bev_w))

        occ_cells = bev_obs.reshape(grid_h, args.cell_px, grid_w, args.cell_px).max(axis=(1, 3)).astype(bool)
        obs_cells = bev_seen.reshape(grid_h, args.cell_px, grid_w, args.cell_px).max(axis=(1, 3)).astype(bool)
        occ_cells &= obs_cells

        occ_grid.update(occ_cells=occ_cells, observed_cells=obs_cells)
        grid_vis_color = occ_grid.visualization_color()
        # Display BEV with the same height as the input frame to avoid black bands.
        grid_vis_color = cv2.resize(grid_vis_color, (args.bev_w, frame_h), interpolation=cv2.INTER_NEAREST)

        overlay = frame.copy()
        overlay[obs_mask > 0] = (0, 0, 255)
        cv2.polylines(overlay, [src_pts.astype(np.int32)], True, (0, 255, 0), 2)
        for i, (x, y) in enumerate(src_pts.astype(np.int32)):
            cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(
                overlay,
                f"P{i}",
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        now = time.time()
        fps = 1.0 / max(now - last_t, 1e-3)
        last_t = now
        cv2.putText(overlay, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)

        canvas = np.zeros((frame_h, frame_w + args.bev_w, 3), dtype=np.uint8)
        canvas[:, :frame_w] = overlay
        canvas[:, frame_w : frame_w + args.bev_w] = grid_vis_color
        out.write(canvas)

        cv2.imshow("Occupancy Grid Demo", canvas)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
