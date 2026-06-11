from typing import Tuple

import cv2
import numpy as np


def build_homography(src_points: np.ndarray, dst_size: Tuple[int, int]) -> np.ndarray:
    dst_h, dst_w = dst_size
    dst_points = np.array(
        [
            [0, 0],
            [dst_w - 1, 0],
            [dst_w - 1, dst_h - 1],
            [0, dst_h - 1],
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points)


def warp_binary_mask(mask: np.ndarray, H: np.ndarray, dst_size: Tuple[int, int]) -> np.ndarray:
    dst_h, dst_w = dst_size
    warped = cv2.warpPerspective(mask, H, (dst_w, dst_h), flags=cv2.INTER_NEAREST)
    return (warped > 0).astype(np.uint8)
