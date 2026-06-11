import numpy as np


class OccupancyGrid:
    def __init__(
        self,
        height: int,
        width: int,
        lo_occ: float = 0.85,
        lo_free: float = 0.4,
        lo_min: float = -4.0,
        lo_max: float = 4.0,
    ) -> None:
        self.height = height
        self.width = width
        self.lo_occ = lo_occ
        self.lo_free = lo_free
        self.lo_min = lo_min
        self.lo_max = lo_max
        self.grid = np.zeros((height, width), dtype=np.float32)
        self.seen = np.zeros((height, width), dtype=bool)

    def update(self, occ_cells: np.ndarray, observed_cells: np.ndarray) -> None:
        self.grid[observed_cells] -= self.lo_free
        self.grid[occ_cells] += self.lo_occ
        np.clip(self.grid, self.lo_min, self.lo_max, out=self.grid)
        self.seen |= observed_cells

    def probability(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.grid))

    def visualization_color(self) -> np.ndarray:
        p = self.probability()
        vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        unseen = ~self.seen
        occ = p > 0.65
        free = p < 0.35
        uncertain = self.seen & ~(occ | free)

        # BGR colors for OpenCV
        vis[unseen] = (80, 80, 80)       # gray: not observed yet
        vis[free] = (245, 245, 245)      # near-white: free
        vis[uncertain] = (0, 215, 255)   # amber: uncertain
        vis[occ] = (0, 0, 255)           # red: occupied
        return vis
