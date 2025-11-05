import sys
sys.path.append("/home/ajifang/czw/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
import carla
import numpy as np
import ipdb


class LaneRef:
    def __init__(self, amap: carla.Map, seed_wp: carla.Waypoint, step: float = 1.0, max_len= 500.0):
        pts, wps = [], []
        wp = seed_wp
        guard_ids = (wp.road_id, wp.section_id, wp.lane_id)
        dist = 0.0
        pts.append((wp.transform.location.x, wp.transform.location.y))
        wps.append(wp)
        while dist < max_len:
            nxts = wp.next(step)
            if not nxts:
                break
            wp = nxts[0]
            if (wp.road_id, wp.section_id, wp.lane_id) != guard_ids:
                break
            pts.append((wp.transform.location.x, wp.transform.location.y))
            wps.append(wp)
            dist += step
        self.P = np.asarray(pts, dtype=float)  # [N,2]
        d = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(d)])  # [N]
        tang = np.diff(self.P, axis=0)
        tang = np.vstack([tang, tang[-1]])
        self.tang = tang / (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
        self.wps = wps  # 保存Waypoints，便于车道宽获取
        self.step = float(step)

    def _segment_index_and_t(self, x, y):
        P = self.P
        xy = np.array([x, y], dtype=float)
        v = xy - P[:-1]
        seg = P[1:] - P[:-1]
        seg_len2 = (seg[:, 0] ** 2 + seg[:, 1] ** 2) + 1e-9
        t = np.clip((v[:, 0] * seg[:, 0] + v[:, 1] * seg[:, 1]) / seg_len2, 0.0, 1.0)
        proj = P[:-1] + seg * t[:, None]
        dist2 = np.sum((proj - xy[None, :]) ** 2, axis=1)
        i = int(np.argmin(dist2))
        return i, float(t[i]), proj[i]

    def xy2se(self, x: float, y: float):
        i, t, proj = self._segment_index_and_t(x, y)
        s_val = self.s[i] + t * (self.s[i + 1] - self.s[i])
        tx, ty = self.tang[i]
        nx, ny = -ty, tx
        ey = (x - proj[0]) * nx + (y - proj[1]) * ny
        return float(s_val), float(ey)

    def se2xy(self, s: float, ey: float):
        s = float(np.clip(s, self.s[0], self.s[-1]))
        i = int(np.searchsorted(self.s, s) - 1)
        i = max(0, min(i, len(self.s) - 2))
        r = (s - self.s[i]) / max(1e-9, self.s[i + 1] - self.s[i])
        base = self.P[i] * (1 - r) + self.P[i + 1] * r
        tx, ty = self.tang[i]
        nx, ny = -ty, tx
        x = base[0] + ey * nx
        y = base[1] + ey * ny
        return float(x), float(y)
