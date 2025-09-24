# planning/dp_corridor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from planning.obstacles import collect_obstacles_api


@dataclass
class Corridor:
    s: np.ndarray
    lower: np.ndarray   # ≤0
    upper: np.ndarray   # ≥0


class DPCorridor:
    def __init__(self, s_max=120.0, ds=2.0, ey_span=3.0, dey=0.2,
                 obs_sigma=0.6, smooth_w=0.05, max_step=2):
        self.s_grid = np.arange(0.0, s_max + 1e-6, ds)
        self.ey_grid = np.arange(-ey_span, ey_span + 1e-6, dey)
        self.ds = ds; self.dey = dey
        self.obs_sigma = obs_sigma
        self.smooth_w = smooth_w
        self.max_step = max_step
        self._prev_lower = None
        self._prev_upper = None

    # ---------- window ----------
    def set_window(self, s0: float, length_m: float):
        K = max(2, int(np.floor(float(length_m) / max(1e-6, self.ds))) + 1)
        self.s_grid = float(s0) + np.arange(0, K) * self.ds

    # ---------- lane walls ----------
    def map_lane_walls_with_ref(self, world, ref, safety=0.25):
        import carla
        K = len(self.s_grid)
        amap = world.get_map()

        left = np.full(K, np.nan); right = np.full(K, np.nan)

        # 方案 A：ref.wps 更稳定
        if hasattr(ref, "wps") and getattr(ref, "wps") is not None and len(ref.wps) >= 2:
            s_ref = np.asarray(getattr(ref, "s"))
            lane_w = np.array([float(getattr(wp, "lane_width", 3.5)) for wp in ref.wps], dtype=float)
            for k, s in enumerate(self.s_grid):
                i = int(np.clip(np.searchsorted(s_ref, s) - 1, 0, len(s_ref) - 1))
                w = float(lane_w[i])
                left[k]  = +0.5 * w - safety
                right[k] = -0.5 * w + safety

        # 方案 B：fallback + lane guard
        if np.isnan(left).all() or np.isnan(right).all():
            guard = None
            for k, s in enumerate(self.s_grid):
                x, y = ref.se2xy(float(s), 0.0)
                wp = amap.get_waypoint(carla.Location(x=x, y=y, z=0.0),
                                       project_to_road=True,
                                       lane_type=carla.LaneType.Driving)
                if not wp: continue
                ids = (wp.road_id, wp.section_id, wp.lane_id)
                if guard is None: guard = ids
                if ids != guard:  # 严格不跨道
                    continue
                w = float(getattr(wp, "lane_width", 3.5))
                left[k]  = +0.5 * w - safety
                right[k] = -0.5 * w + safety

        left  = self._interp_smooth(left, prefer_pos=True)
        right = self._interp_smooth(right, prefer_pos=False)
        # 保证右≤0≤左
        left  = np.maximum(left,  0.0)
        right = np.minimum(right, 0.0)
        bad = right > left
        if np.any(bad):
            mid = 0.5 * (left + right)
            half = np.maximum(0.5, 0.5 * (left - right))
            left[bad]  = np.abs(half[bad])
            right[bad] = -np.abs(half[bad])
        return left, right

    @staticmethod
    def _interp_smooth(arr: np.ndarray, prefer_pos: bool):
        K = len(arr)
        out = arr.copy()
        idx = np.where(np.isfinite(out))[0]
        if len(idx) == 0:
            return np.full(K, 0.0 if prefer_pos else 0.0)
        if len(idx) == 1:
            out[:] = out[idx[0]]
        else:
            out[:] = np.interp(np.arange(K), idx, out[idx])
        if K >= 3:
            sm = out.copy()
            sm[1:-1] = (out[:-2] + 2*out[1:-1] + out[2:]) * 0.25
            out = 0.5*out + 0.5*sm
        return out

    # ---------- 硬墙：由障碍点生成左右最近边界，并做 s 向“膨胀” ----------
    @staticmethod
    def hard_walls_from_points(pts_se: List[Tuple[float, float]],
                               s_grid: np.ndarray, ds: float,
                               s_dilate: float = 1.0,  # 纵向膨胀(米)
                               safety: float = 0.20):
        K = len(s_grid)
        left  = np.full(K, +np.inf)
        right = np.full(K, -np.inf)
        s0 = float(s_grid[0])

        for (s, ey) in pts_se:
            k = int(np.clip(round((s - s0) / max(1e-6, ds)), 0, K - 1))
            if ey >= 0.0:
                left[k]  = min(left[k],  ey)
            else:
                right[k] = max(right[k], ey)

        # 插值填洞
        def _interp(a):
            idx = np.where(np.isfinite(a))[0]
            if len(idx) == 0:  return a
            if len(idx) == 1:  a[:] = a[idx[0]]; return a
            return np.interp(np.arange(K), idx, a[idx])

        left  = _interp(left);  right = _interp(right)

        # s 方向形态学“膨胀/闭运算”——让散点连成连续墙
        w = max(1, int(round(s_dilate / max(1e-6, ds))))
        if w > 0:
            # 左墙取“更靠近中线”的最小值；右墙取“更靠近中线”的最大值
            for i in range(K):
                lo = max(0, i - w); hi = min(K, i + w + 1)
                left[i]  = np.min(left[lo:hi])   # min 更靠近 0
                right[i] = np.max(right[lo:hi])  # max 更靠近 0

        left  = left  - safety
        right = right + safety
        return left, right

    # ---------- 软代价（可选） ----------
    def build_cost_map_general(self, world, ego_actor, ref_xy2se,
                               s_center: float, s_back: float = 10.0, s_fwd: float = 20.0,
                               sigma_s: float = 2.5, sigma_y: float = 0.6,
                               horizon_T: float = 2.0, dt: float = 0.2,
                               map_left: Optional[np.ndarray] = None,
                               map_right: Optional[np.ndarray] = None) -> np.ndarray:
        K, M = len(self.s_grid), len(self.ey_grid)
        cost = np.zeros((K, M), dtype=float)
        if ego_actor is None:
            return cost

        pts = collect_obstacles_api(world, ego_actor, ref_xy2se,
                                    s_center=s_center, s_back=s_back, s_fwd=s_fwd,
                                    r_xy=35.0, horizon_T=horizon_T, dt=dt,
                                    static_density=0.20)
        if not pts:
            return cost

        s0 = float(self.s_grid[0]); y0 = float(self.ey_grid[0])
        half_ws = max(1, int(round(sigma_s / max(1e-6, self.ds)) * 3))
        half_wy = max(1, int(round(sigma_y / max(1e-6, self.dey)) * 3))

        for (s_ob, ey_ob) in pts:
            si = int(np.clip(round((s_ob - s0) / self.ds), 0, K - 1))
            if map_left is not None and map_right is not None:
                if (ey_ob < map_right[si]) or (ey_ob > map_left[si]):
                    continue
            yi = int(np.clip(round((ey_ob - y0) / self.dey), 0, M - 1))

            s_lo = max(0, si - half_ws); s_hi = min(K - 1, si + half_ws)
            y_lo = max(0, yi - half_wy); y_hi = min(M - 1, yi + half_wy)
            if s_lo > s_hi or y_lo > y_hi: continue

            S = self.s_grid[s_lo:s_hi+1][:, None]
            Y = self.ey_grid[y_lo:y_hi+1][None, :]
            gs = np.exp(-0.5 * ((S - s_ob) / max(1e-6, sigma_s)) ** 2)
            gy = np.exp(-0.5 * ((Y - ey_ob) / max(1e-6, sigma_y)) ** 2)
            cost[s_lo:s_hi+1, y_lo:y_hi+1] += gs * gy

        if M >= 3 and self.smooth_w > 0.0:
            c2 = cost.copy()
            c2[:, 1:-1] = (cost[:, :-2] + 2.0*cost[:, 1:-1] + cost[:, 2:]) * 0.25
            cost = (1.0 - self.smooth_w)*cost + self.smooth_w*c2
        return cost

    # ---------- 融合：地图墙 + 硬墙（障碍） + （可选）DP ----------
    def fuse_corridor(self,
                      map_left: np.ndarray, map_right: np.ndarray,
                      hard_left: Optional[np.ndarray], hard_right: Optional[np.ndarray],
                      min_width: float = 1.8, temporal_alpha: float = 0.6) -> Corridor:
        """
        最稳的做法：直接取交集——
          upper = min(map_left, hard_left)；lower = max(map_right, hard_right)
        没有 hard_* 时就用地图墙。
        """
        K = len(self.s_grid)
        upper = map_left.copy()
        lower = map_right.copy()
        if hard_left is not None:
            upper = np.minimum(upper, hard_left)
        if hard_right is not None:
            lower = np.maximum(lower, hard_right)

        # 最小宽度 + 合法性
        bad = (upper - lower) < min_width
        if np.any(bad):
            mid = 0.5 * (upper + lower)
            upper[bad] = mid[bad] + 0.5 * min_width
            lower[bad] = mid[bad] - 0.5 * min_width
            upper = np.minimum(upper, map_left)
            lower = np.maximum(lower, map_right)

        # 轻平滑
        upper = self._interp_smooth(upper, prefer_pos=True)
        lower = self._interp_smooth(lower, prefer_pos=False)

        # 时间平滑
        if (self._prev_lower is not None) and (len(self._prev_lower) == K):
            beta = np.clip(float(temporal_alpha), 0.0, 1.0)
            lower = beta * lower + (1.0 - beta) * self._prev_lower
            upper = beta * upper + (1.0 - beta) * self._prev_upper if self._prev_upper is not None else upper

        self._prev_lower = lower.copy()
        self._prev_upper = upper.copy()
        return Corridor(self.s_grid.copy(), lower, upper)
