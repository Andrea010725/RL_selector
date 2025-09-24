# planning/dp_corridor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
from planning.obstacles import collect_obstacles_api

import numpy as np


# 0.9.15 版本里 cone 的命名不统一，这里统一做“包含匹配”关键字集合（命名已修正为 CONE_KEYS_0915）
CONE_KEYS_0915 = (
    "trafficcone",            # 通用关键字，优先匹配
    "traffic_cone",
    "prop.trafficcone",       # 一些包带前缀
    "static.prop.trafficcone",
    "static.prop.trafficcone01",
    "static.prop.trafficcone02",
    "static.prop.cone",       # 个别地图包
)


@dataclass
class Corridor:
    s: np.ndarray      # [K] 纵向网格
    lower: np.ndarray  # [K] ey 下边界（米）
    upper: np.ndarray  # [K] ey 上边界（米）


class DPCorridor:
    """
    动态规划风格（逐行选择）的安全走廊构造器（Frenet: s-纵向, ey-横向）。
    """

    def __init__(
        self,
        s_max: float = 120.0,
        ds: float = 2.0,
        ey_span: float = 3.0,
        dey: float = 0.2,
        obs_sigma: float = 0.6,      # 保留以兼容旧接口
        smooth_w: float = 0.05,      # 列向平滑系数（对 cost 做轻量 1D 平滑）
        max_step: int = 2,
    ):
        self.s_grid = np.arange(0.0, s_max + 1e-6, ds)            # [K]
        self.ey_grid = np.arange(-ey_span, ey_span + 1e-6, dey)   # [M]
        self.ds = ds
        self.dey = dey
        self.obs_sigma = obs_sigma
        self.smooth_w = smooth_w
        self.max_step = max_step

        # 用于跨帧时域平滑
        self._prev_lower: Optional[np.ndarray] = None
        self._prev_upper: Optional[np.ndarray] = None

    def set_window(self, s0: float, length_m: float):
        s0 = float(s0)
        length_m = float(length_m)
        K = max(2, int(np.floor(length_m / max(1e-6, self.ds))) + 1)
        self.s_grid = s0 + np.arange(0, K) * self.ds

    def window_range(self):
        if self.s_grid is None or len(self.s_grid) == 0:
            return (0.0, 0.0)
        return (float(self.s_grid[0]), float(self.s_grid[-1]))

    # =================== 代价图（含采样点输出版） ===================
    def build_cost_map(
        self,
        world,
        ref_xy2se,
        horizon_T: float = 2.0,
        dt: float = 0.2,
        sigma_s: float = 2.5,
        sigma_y: float = 0.6,
        ego_actor=None,
        cones: Optional[List] = None,
        window_s_forward: float = 20.0,
        window_s_back: float = 10.0,
        prefilter_radius: Optional[float] = 60.0,
        return_points: bool = False,
    ):
        """
        仅在 Frenet 本地窗口 [s_ego - window_s_back, s_ego + window_s_forward] 内采样障碍并构建代价图。
        返回 cost_map 或 (cost_map, obs_xy)。
        """
        K, M = len(self.s_grid), len(self.ey_grid)
        cost = np.zeros((K, M), dtype=float)
        obs_xy: List[Tuple[float, float]] = []

        if ego_actor is None:
            return cost if not return_points else (cost, obs_xy)

        ego_tf = ego_actor.get_transform()
        s_ego, _ = ref_xy2se(ego_tf.location.x, ego_tf.location.y)
        s_min = s_ego - float(window_s_back)
        s_max = s_ego + float(window_s_forward)

        steps = max(1, int(horizon_T / dt))
        ego_loc = ego_tf.location

        # --- 动态车辆 ---
        try:
            vehicles = world.get_actors().filter("vehicle.*")
        except Exception:
            vehicles = world.get_actors()

        for v in vehicles:
            try:
                if v.id == ego_actor.id or v.attributes.get("role_name", "") == "hero":
                    continue
                loc = v.get_transform().location
                if prefilter_radius is not None:
                    dx = loc.x - ego_loc.x
                    dy = loc.y - ego_loc.y
                    if dx * dx + dy * dy > prefilter_radius * prefilter_radius:
                        continue
                vel = v.get_velocity()
                x0, y0 = loc.x, loc.y
                vx, vy = vel.x, vel.y
                for k in range(steps + 1):
                    t = k * dt
                    x = x0 + vx * t
                    y = y0 + vy * t
                    s_ob, _ = ref_xy2se(x, y)
                    if s_min <= s_ob <= s_max:
                        obs_xy.append((x, y))
            except Exception:
                continue

        # --- 静态锥桶 ---
        if cones is not None:
            for c in cones:
                try:
                    loc = c.get_transform().location
                    if prefilter_radius is not None:
                        dx = loc.x - ego_loc.x
                        dy = loc.y - ego_loc.y
                        if dx * dx + dy * dy > prefilter_radius * prefilter_radius:
                            continue
                    s_ob, _ = ref_xy2se(loc.x, loc.y)
                    if s_min <= s_ob <= s_max:
                        obs_xy.append((loc.x, loc.y))
                except Exception:
                    continue
        else:
            try:
                static_props = world.get_actors().filter("static.prop.*")
            except Exception:
                static_props = world.get_actors()
            for a in static_props:
                try:
                    tid = getattr(a, "type_id", "").lower()
                    if not any(key in tid for key in CONE_KEYS_0915):
                        continue
                    loc = a.get_transform().location
                    if prefilter_radius is not None:
                        dx = loc.x - ego_loc.x
                        dy = loc.y - ego_loc.y
                        if dx * dx + dy * dy > prefilter_radius * prefilter_radius:
                            continue
                    s_ob, _ = ref_xy2se(loc.x, loc.y)
                    if s_min <= s_ob <= s_max:
                        obs_xy.append((loc.x, loc.y))
                except Exception:
                    continue

        if len(obs_xy) == 0:
            return cost if not return_points else (cost, obs_xy)

        # --- 叠加 2D 高斯代价（注意 s0 基准） ---
        s_grid, ey_grid = self.s_grid, self.ey_grid
        s0 = float(self.s_grid[0])                  # ✅ 修正：索引必须相对 s0
        half_ws = max(1, int(round(sigma_s / max(1e-6, self.ds)) * 3))
        half_wy = max(1, int(round(sigma_y / max(1e-6, self.dey)) * 3))

        for (x, y) in obs_xy:
            try:
                s_ob, ey_ob = ref_xy2se(x, y)
                if not (s_min <= s_ob <= s_max):
                    continue
                si = int(np.clip(round((s_ob - s0) / self.ds), 0, K - 1))     # ✅
                yi = int(np.clip(round((ey_ob - ey_grid[0]) / self.dey), 0, M - 1))

                s_lo = max(0, si - half_ws); s_hi = min(K - 1, si + half_ws)
                y_lo = max(0, yi - half_wy); y_hi = min(M - 1, yi + half_wy)

                S = s_grid[s_lo: s_hi + 1][:, None]
                Y = ey_grid[y_lo: y_hi + 1][None, :]

                gs = np.exp(-0.5 * ((S - s_ob) / max(1e-6, sigma_s)) ** 2)
                gy = np.exp(-0.5 * ((Y - ey_ob) / max(1e-6, sigma_y)) ** 2)
                cost[s_lo: s_hi + 1, y_lo: y_hi + 1] += gs * gy
            except Exception:
                continue

        # --- 列向轻量平滑 ---
        if M >= 3 and self.smooth_w > 0.0:
            cost_smooth = cost.copy()
            cost_smooth[:, 1:-1] = (cost[:, :-2] + 2.0 * cost[:, 1:-1] + cost[:, 2:]) * 0.25
            cost = (1.0 - self.smooth_w) * cost + self.smooth_w * cost_smooth

        cost = np.asarray(cost, dtype=float)
        if cost.ndim != 2:
            raise ValueError(f"[build_cost_map] cost.ndim={cost.ndim}, expect 2D")

        return (cost, obs_xy) if return_points else cost

    # =================== 代价图（通用：已是 Frenet 点） ===================
    def build_cost_map_general(
        self,
        world,
        ego_actor,
        ref_xy2se,
        s_center: float,
        s_back: float = 10.0,
        s_fwd: float = 20.0,
        sigma_s: float = 2.5,
        sigma_y: float = 0.6,
        horizon_T: float = 2.0,
        dt: float = 0.2,
    ) -> np.ndarray:

        K, M = len(self.s_grid), len(self.ey_grid)
        cost = np.zeros((K, M), dtype=float)
        if ego_actor is None:
            return cost

        pts_se = collect_obstacles_api(
            world=world,
            ego=ego_actor,
            ref_xy2se=ref_xy2se,
            s_center=s_center,
            s_back=s_back,
            s_fwd=s_fwd,
            r_xy=35.0,
            horizon_T=horizon_T,
            dt=dt,
            static_density=0.3,
        )

        if not pts_se:
            return cost

        half_ws = max(1, int(round(sigma_s / max(1e-6, self.ds)) * 3))
        half_wy = max(1, int(round(sigma_y / max(1e-6, self.dey)) * 3))
        s0 = float(self.s_grid[0])
        y0 = float(self.ey_grid[0])

        for (s_ob, ey_ob) in pts_se:
            si = int(np.clip(round((s_ob - s0) / self.ds), 0, K - 1))
            yi = int(np.clip(round((ey_ob - y0) / self.dey), 0, M - 1))

            s_lo = max(0, si - half_ws); s_hi = min(K - 1, si + half_ws)
            y_lo = max(0, yi - half_wy); y_hi = min(M - 1, yi + half_wy)
            if s_lo > s_hi or y_lo > y_hi:
                continue

            S = self.s_grid[s_lo:s_hi + 1][:, None]
            Y = self.ey_grid[y_lo:y_hi + 1][None, :]

            gs = np.exp(-0.5 * ((S - s_ob) / max(1e-6, sigma_s)) ** 2)
            gy = np.exp(-0.5 * ((Y - ey_ob) / max(1e-6, sigma_y)) ** 2)
            cost[s_lo:s_hi + 1, y_lo:y_hi + 1] += gs * gy

        return cost

    # =================== 由障碍点生成硬边界（几何墙） ===================
    @staticmethod
    def walls_from_points(
        pts_se: List[Tuple[float, float]],
        s_grid: np.ndarray,
        ds: float,
        safety: float = 0.25
    ):
        K = len(s_grid)
        left = np.full(K, +np.inf, dtype=float)   # 左侧墙（ey>0）：越小越靠中
        right = np.full(K, -np.inf, dtype=float)  # 右侧墙（ey<0）：越大越靠中
        s0 = float(s_grid[0])

        for (s, ey) in pts_se:
            k = int(np.clip(round((s - s0) / max(1e-6, ds)), 0, K - 1))
            if ey >= 0.0:
                left[k] = min(left[k], ey)
            else:
                right[k] = max(right[k], ey)

        def _interp_nan(a: np.ndarray) -> np.ndarray:
            """仅对有限值插值；全缺失时返回 NaN 数组。"""
            idx = np.where(np.isfinite(a))[0]
            out = a.astype(float).copy()
            if len(idx) == 0:
                return np.full_like(a, np.nan)
            if len(idx) == 1:
                out[:] = a[idx[0]]
                return out
            vals = a[idx]
            out[:] = np.interp(np.arange(K), idx, vals)
            return out

        def _smooth_nan(u: np.ndarray) -> np.ndarray:
            v = u.copy()
            for i in range(K):
                lo = max(0, i - 1); hi = min(K, i + 2)
                seg = u[lo:hi]
                m = np.nanmean(seg)
                v[i] = float(m) if np.isfinite(m) else np.nan
            return v

        left  = _smooth_nan(_interp_nan(left))
        right = _smooth_nan(_interp_nan(right))

        # 加上安全边距，并把 NaN 解释为“无墙”（左=+inf，右=-inf）
        left  = np.where(np.isfinite(left),  left  - safety, +np.inf)
        right = np.where(np.isfinite(right), right + safety, -np.inf)

        return left, right

    # =================== 从代价图提取走廊 ===================
    def run_dp(
        self,
        cost_map: np.ndarray,
        row_percentile: float = 60.0,
        min_width: float = 1.8,
        safety_margin: float = 0.20,
        temporal_alpha: float = 0.6,   # 跨帧平滑系数（0~1），None 关闭
    ) -> Corridor:
        """
        每行（固定 s）按分位数取“低代价带”，再加安全余量。
        """
        if cost_map.ndim != 2:
            raise ValueError(f"[run_dp] cost_map.ndim={cost_map.ndim}, expect 2D")

        K, M = cost_map.shape
        s = self.s_grid
        ey = self.ey_grid

        lower = np.full(K, ey[0], dtype=float)
        upper = np.full(K, ey[-1], dtype=float)

        # 行分位数，附加简易行向平滑（抑制毛刺）
        tau_row = np.percentile(cost_map, row_percentile, axis=1)
        if K >= 3:
            tau_row = 0.25*np.r_[tau_row[:1], tau_row[:-1]] + 0.5*tau_row + 0.25*np.r_[tau_row[1:], tau_row[-1:]]

        zero_idx = int(np.clip(round((0.0 - ey[0]) / self.dey), 0, M - 1))

        for k in range(K):
            row = cost_map[k, :]
            tau = tau_row[k]
            free = row < tau

            # 扫描连续 True 段
            segments: List[Tuple[int, int]] = []
            i = 0
            while i < M:
                if free[i]:
                    j = i
                    while j + 1 < M and free[j + 1]:
                        j += 1
                    segments.append((i, j))
                    i = j + 1
                else:
                    i += 1

            if not segments:
                # —— 改进回退：向邻近有段的行插值 —— #
                lo, up = ey[0], ey[-1]  # 默认全宽
                # 找上/下最近有段的行
                k_up = None
                for kk in range(k - 1, -1, -1):
                    if not np.allclose([lower[kk], upper[kk]], [ey[0], ey[-1]]):
                        k_up = kk; break
                k_dn = None
                for kk in range(k + 1, K):
                    if not np.allclose([lower[kk], upper[kk]], [ey[0], ey[-1]]):
                        k_dn = kk; break
                if k_up is not None and k_dn is not None:
                    t = (k - k_up) / max(1, (k_dn - k_up))
                    lo = (1 - t) * lower[k_up] + t * lower[k_dn]
                    up = (1 - t) * upper[k_up] + t * upper[k_dn]
                elif k_up is not None:
                    lo, up = lower[k_up], upper[k_up]
                elif k_dn is not None:
                    lo, up = lower[k_dn], upper[k_dn]

                if up - lo < min_width:
                    mid = 0.5 * (lo + up)
                    lo, up = mid - 0.5 * min_width, mid + 0.5 * min_width
                lower[k] = max(lo, ey[0])
                upper[k] = min(up, ey[-1])
                continue

            # 选择“包含 ey=0 的段”，否则选“最宽段”
            chosen: Optional[Tuple[int, int]] = None
            for (i0, i1) in segments:
                if i0 <= zero_idx <= i1:
                    chosen = (i0, i1)
                    break
            if chosen is None:
                chosen = max(segments, key=lambda seg: seg[1] - seg[0])

            i0, i1 = chosen
            lo = ey[i0] - safety_margin
            up = ey[i1] + safety_margin

            if up - lo < min_width:
                mid = 0.5 * (lo + up)
                lo = mid - 0.5 * min_width
                up = mid + 0.5 * min_width
            lo = max(lo, ey[0]); up = min(up, ey[-1])

            lower[k] = lo
            upper[k] = up

        # —— 跨帧时域平滑（可关闭） —— #
        if temporal_alpha is not None and 0.0 < temporal_alpha < 1.0:
            if self._prev_lower is not None and len(self._prev_lower) == K:
                lower = temporal_alpha * lower + (1.0 - temporal_alpha) * self._prev_lower
                upper = temporal_alpha * upper + (1.0 - temporal_alpha) * self._prev_upper
            self._prev_lower = lower.copy()
            self._prev_upper = upper.copy()

        return Corridor(s.copy(), lower, upper)
