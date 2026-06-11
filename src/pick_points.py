import argparse

import cv2
import numpy as np

clicked = []


def on_mouse(event, x, y, flags, param):
    del flags, param
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
        clicked.append((x, y))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick 4 BEV source points from first video frame.")
    parser.add_argument("--video", required=True, help="Input video path.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Cannot read first frame from video.")

    window = "Pick points (clockwise) - press q to finish"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        vis = frame.copy()
        for i, (x, y) in enumerate(clicked):
            cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(vis, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if len(clicked) == 4:
            poly = np.array(clicked, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [poly], True, (0, 255, 255), 2)
        cv2.imshow(window, vis)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or len(clicked) == 4:
            break

    cv2.destroyAllWindows()

    if len(clicked) != 4:
        print("Please select exactly 4 points.")
        return

    out = ",".join([f"{x},{y}" for x, y in clicked])
    print(out)


if __name__ == "__main__":
    main()
