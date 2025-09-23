# utils/control_mapping.py
def ax_to_throttle_brake(ax: float, v: float, a_max: float = 2.0, a_min: float = -4.0):
    """
    线性映射：ax∈[a_min, a_max] → throttle/brake∈[0,1]
    """
    ax = max(a_min, min(a_max, ax))
    if ax >= 0:
        return float(ax / a_max), 0.0
    else:
        return 0.0, float(-ax / (-a_min))

def delta_to_steer(delta_rad: float, max_steer_rad: float = 0.5) -> float:
    """
    前轮转角→CARLA steer ∈ [-1,1]
    """
    return float(max(-1.0, min(1.0, delta_rad / max_steer_rad)))
