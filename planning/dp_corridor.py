# planning/dp_corridor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import math
import carla

@dataclass
class Corridor:
    s: np.ndarray
    lower: np.ndarray  # 右侧（ey 更小）
    upper: np.ndarray  # 左侧（ey 更大）

class DPCorridor:
    """
    极简走廊：
      - set_window：定义 s 网格（跟车滑窗由外部调用）
      - map_lane_walls_with_ref：按参考线在每个 s 的车道半宽生成左右“地图墙”
      - hard_walls_from_points：把障碍点聚成左右最近“硬墙”
      - fuse_corridor：地图墙 ∩ 硬墙，最小宽度保护 + 轻量平滑
    """

    def __init__(self, s_max=40.0, ds=2.0, ey_span=3.0, dey=0.2):
        self.ds = float(ds)
        self.dey = float(dey)
        self.s_grid = np.arange(0.0, float(s_max) + 1e-6, self.ds)

    # -------- 滑动窗口 --------
    def set_window(self, s0: float, length_m: float):
        K = max(2, int(math.floor(float(length_m) / max(1e-6, self.ds))) + 1)
        self.s_grid = float(s0) + np.arange(K) * self.ds

    def window_range(self) -> Tuple[float, float]:
        if self.s_grid.size == 0:
            return (0.0, 0.0)
        return (float(self.s_grid[0]), float(self.s_grid[-1]))

    # -------- 地图墙（永远有线）--------
    def map_lane_walls_with_ref(self, world, ref, lane_margin: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
        """
        对每个 s，取 ref.se2xy(s,0) 投到道路 waypoint：
        用 waypoint 的 lane_width/2 构造左右边界（左为 +half，右为 -half）。
        lane_margin 为向内收缩 margin。
        """
        amap = world.get_map()
        K = len(self.s_grid)
        left = np.zeros(K, dtype=float)
        right = np.zeros(K, dtype=float)

        for i, s in enumerate(self.s_grid):
            # 取参考线中心点（s,0） → world
            x, y = ref.se2xy(float(s), 0.0)
            # 投到道路
            wp = amap.get_waypoint(
                carla.Location(x=x, y=y, z=0.0),  # 需要在你的工程里 from carla import carla 或提前 import carla
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            if wp is None:
                # 兜底：用上一个的值
                left[i] = left[i-1] if i > 0 else +2.0
                right[i] = right[i-1] if i > 0 else -2.0
                continue

            half = max(0.5, 0.5 * float(wp.lane_width) - float(lane_margin))
            # Frenet 约定：ey>0 左侧，ey<0 右侧
            left[i] = +half
            right[i] = -half

        # 轻量平滑，避免锯齿
        if K >= 3:
            def smooth(u):
                v = u.copy()
                v[1:-1] = (u[:-2] + 2*u[1:-1] + u[2:]) * 0.25
                return v
            left = smooth(left)
            right = smooth(right)

        return left, right

    # -------- 障碍点 → 硬墙 --------
    @staticmethod
    def hard_walls_from_points(
        pts_se: List[Tuple[float, float]],
        s_grid: np.ndarray,
        ds: float,
        safety: float = 0.25
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        左墙（ey>0）取“最小正值-安全裕度”；右墙（ey<0）取“最大负值+安全裕度”。
        """
        K = len(s_grid)
        left = np.full(K, +np.inf, dtype=float)
        right = np.full(K, -np.inf, dtype=float)
        s0 = float(s_grid[0])

        for (s, ey) in pts_se:
            k = int(np.clip(round((s - s0) / max(1e-6, ds)), 0, K - 1))
            if ey >= 0.0:
                left[k] = min(left[k], float(ey))
            else:
                right[k] = max(right[k], float(ey))

        # 插值填洞
        def interp_wall(a: np.ndarray) -> np.ndarray:
            idx = np.where(np.isfinite(a))[0]
            if len(idx) == 0:
                return a
            if len(idx) == 1:
                a[:] = a[idx[0]]
                return a
            return np.interp(np.arange(K), idx, a[idx])

        left = interp_wall(left) - float(safety)
        right = interp_wall(right) + float(safety)

        # 平滑一下
        if K >= 3:
            left[1:-1] = (left[:-2] + 2*left[1:-1] + left[2:]) * 0.25
            right[1:-1] = (right[:-2] + 2*right[1:-1] + right[2:]) * 0.25

        return left, right

    # -------- 融合：地图墙 ∩ 硬墙 --------
    def fuse_corridor(
        self,
        map_left: np.ndarray, map_right: np.ndarray,
        hard_left: Optional[np.ndarray], hard_right: Optional[np.ndarray],
        min_width: float = 1.8,
    ) -> Corridor:
        """
        lower = max(map_right, hard_right) ; upper = min(map_left, hard_left)
        支持 hard_* 为 None（无障碍时只用地图墙）。
        """
        left = map_left.copy()
        right = map_right.copy()

        if hard_left is not None:
            left = np.minimum(left, hard_left)
        if hard_right is not None:
            right = np.maximum(right, hard_right)

        # 最小宽度保护
        width = left - right
        bad = width < float(min_width)
        if np.any(bad):
            mid = 0.5 * (left + right)
            left[bad] = mid[bad] + 0.5 * float(min_width)
            right[bad] = mid[bad] - 0.5 * float(min_width)

        # 轻量时间平滑（如需）：此处不做时序，仅做一次空间平滑
        if len(left) >= 3:
            left[1:-1] = (left[:-2] + 2*left[1:-1] + left[2:]) * 0.25
            right[1:-1] = (right[:-2] + 2*right[1:-1] + right[2:]) * 0.25

        return Corridor(self.s_grid.copy(), right, left)
