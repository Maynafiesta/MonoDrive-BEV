import heapq
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

GridPoint = Tuple[int, int]


def inflate_obstacles(occupied: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return occupied.copy()
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    inflated = cv2.dilate(occupied.astype(np.uint8), kernel, iterations=1)
    return inflated.astype(bool)


def astar(cost: np.ndarray, start: GridPoint, goal: GridPoint) -> List[GridPoint]:
    h, w = cost.shape
    blocked = cost >= 1e6

    def in_bounds(p: GridPoint) -> bool:
        y, x = p
        return 0 <= y < h and 0 <= x < w and not blocked[y, x]

    if not in_bounds(start) or not in_bounds(goal):
        return []

    def heuristic(a: GridPoint, b: GridPoint) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    moves = [(-1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, 0), (1, -1), (1, 1)]
    open_heap: List[Tuple[float, GridPoint]] = [(0.0, start)]
    came_from: Dict[GridPoint, Optional[GridPoint]] = {start: None}
    g_score: Dict[GridPoint, float] = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            break

        for dy, dx in moves:
            nxt = (current[0] + dy, current[1] + dx)
            if not in_bounds(nxt):
                continue
            step = 1.4 if dy != 0 and dx != 0 else 1.0
            tentative = g_score[current] + step + float(cost[nxt])
            if tentative < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative
                priority = tentative + heuristic(nxt, goal)
                heapq.heappush(open_heap, (priority, nxt))

    if goal not in came_from:
        return []

    path: List[GridPoint] = []
    current: Optional[GridPoint] = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

