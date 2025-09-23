# planning/dp_corridor.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class Corridor:
    s: np.ndarray      # [K] 纵向网格
    lower: np.ndarray  # [K] ey 下边界（米）
    upper: np.ndarray  # [K] ey 上边界（米])

class DPCorridor:
    """
    动态规划生成安全走廊（Frenet: s-纵向, ey-横向）。
    - 代价由：动态障碍物排斥 + 平滑/转向变化 组成
    - DP 输出不是单条轨迹，而是每个 s 的可行 ey 带宽 -> (lower, upper)
    参考：附件“基于DP的动态障碍物轨迹规划 与 NMPC 控制”的 DP 部分。:contentReference[oaicite:1]{index=1}
    """
    def __init__(self, s_max: float=120.0, ds: float=2.0, ey_span: float=3.0, dey: float=0.2,
                 obs_sigma: float=0.6, smooth_w: float=0.05, max_step: int=2):
        self.s_grid = np.arange(0.0, s_max+1e-6, ds)            # [K]
        self.ey_grid = np.arange(-ey_span, ey_span+1e-6, dey)   # [M]
        self.ds = ds
        self.dey = dey
        self.obs_sigma = obs_sigma
        self.smooth_w = smooth_w
        self.max_step = max_step

    # === 你可以把这部分替换成更好的障碍物预测 ===
    def build_cost_map(self, world, ref_xy2se, horizon_T=3.0, dt=0.2) -> np.ndarray:
        """
        读取 CARLA 动态车辆，做短时常速预测，投影到 Frenet，叠加高斯排斥代价。
        ref_xy2se: callable(x, y) -> (s, ey)
        返回: cost[K, M]
        """
        actors = world.get_actors().filter("vehicle.*")
        K, M = len(self.s_grid), len(self.ey_grid)
        cost = np.zeros((K, M), dtype=float)

        steps = max(1, int(horizon_T/dt))
        for _ in range(steps):
            for veh in actors:
                try:
                    tr = veh.get_transform()
                    vel = veh.get_velocity()
                    x = tr.location.x + vel.x*dt
                    y = tr.location.y + vel.y*dt
                    s_ob, ey_ob = ref_xy2se(x, y)
                    si = int(np.clip(round(s_ob/self.ds), 0, K-1))
                    # 高斯横向扩散
                    dlat = (self.ey_grid - ey_ob)
                    cost[si, :] += np.exp(-0.5*(dlat/self.obs_sigma)**2)
                except Exception:
                    continue

        # 平滑/转向变化惩罚（简化版）
        if M >= 3:
            lap = np.abs(self.ey_grid[2:] - 2*self.ey_grid[1:-1] + self.ey_grid[:-2])
            cost[:,1:-1] += self.smooth_w * lap
        return cost

    def run_dp(self, cost_map: np.ndarray) -> Corridor:
        """
        基于 cost_map 做单源 DP，限制横向步进幅度（车辆机动约束），
        再按分位数阈值提取上下边界（带安全 buffer）。
        """
        K, M = cost_map.shape
        J = np.full((K, M), 1e12, dtype=float)
        ptr = np.zeros((K, M), dtype=int)
        J[0,:] = cost_map[0,:]

        for k in range(1, K):
            for i in range(M):
                i_min = max(0, i - self.max_step)
                i_max = min(M-1, i + self.max_step)
                lateral = np.abs(np.arange(i_min, i_max+1) - i)*0.5  # 横向突变惩罚（可调）
                cand = J[k-1, i_min:i_max+1] + cost_map[k, i] + lateral
                j = np.argmin(cand)
                J[k,i] = cand[j]
                ptr[k,i] = i_min + j

        # 走廊提取：用分位数阈值挑“低代价带”形成上下界
        thr = np.percentile(J, 40)   # 越低越窄越保守；你可改 30~60 做风格调节
        lower = np.full(K, self.ey_grid[0], dtype=float)
        upper = np.full(K, self.ey_grid[-1], dtype=float)
        for k in range(K):
            valid = np.where(J[k,:] < thr)[0]
            if len(valid) > 0:
                lower[k] = self.ey_grid[valid.min()] - 0.20  # 安全 buffer
                upper[k] = self.ey_grid[valid.max()] + 0.20
        return Corridor(self.s_grid.copy(), lower, upper)

