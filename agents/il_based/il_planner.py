# agents/il_based/il_planner.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import math

# 统一低层动作三元组
LowLevelAction = Tuple[float, float, float]  # (throttle, brake, steer)

try:
    import sys
    from planners.base import PlannerBase  # 你之前定义的抽象基类
except Exception:
    # 兜底：如果你的项目里 base 在别处，请把导入路径改成你自己的
    from planners.base import PlannerBase

# CARLA 导入（确保你的 sys.path 已在主程序里注入 egg 与 PythonAPI 路径）
import carla
from agents.navigation.basic_agent import BasicAgent


class ILPlanner(PlannerBase):
    """
    Imitation Learning Planner（占位实现）：
    - 这里使用 CARLA 自带的 BasicAgent 来产生一个“模仿式”的控制，
      仅作为你的 IL 模型尚未接入前的替身。
    - 关键点：本 planner 只“计算控制并返回”，不会 set_autopilot() 或直接写车，
      避免和环境/选择器冲突。
    """

    def __init__(self, v_ref_kmh: float = 25.0, lookahead_m: float = 80.0):
        self.v_ref_kmh = float(v_ref_kmh)
        self.lookahead_m = float(lookahead_m)
        self._world: Optional[carla.World] = None
        self._ego: Optional[carla.Actor] = None
        self._map: Optional[carla.Map] = None
        self._agent: Optional[BasicAgent] = None
        self._ready: bool = False

    # Router 会在 reset()/env.reset() 后调用它
    def attach_context(self,
                       world: carla.World,
                       ego: carla.Actor,
                       lane_ref=None) -> None:
        """将运行上下文注入到 IL Planner。"""
        self._world = world
        self._ego = ego
        self._map = world.get_map()
        # 初始化 BasicAgent（不接管车辆）
        self._agent = BasicAgent(self._ego, target_speed=self.v_ref_kmh)
        # 给一个“前方 lookahead_m”处的目标，避免需要全局路由
        try:
            ego_tf = self._ego.get_transform()
            forward = ego_tf.get_forward_vector()
            goal_loc = ego_tf.location + carla.Location(x=forward.x * self.lookahead_m,
                                                        y=forward.y * self.lookahead_m,
                                                        z=0.0)
            # project_to_road=True，拿到一个合理的目标点
            wp = self._map.get_waypoint(goal_loc, project_to_road=True,
                                        lane_type=carla.LaneType.Driving)
            if wp is not None:
                dest = wp.transform.location
            else:
                dest = goal_loc
            self._agent.set_destination((dest.x, dest.y, dest.z))
            self._ready = True
        except Exception:
            # 即使目标设置失败，也允许继续（BasicAgent 会退化为局部避障）
            self._ready = True

    def reset(self):
        # 可选：若你想在每个 episode 开头重置内部状态，在这里做
        pass

    def _to_low_level(self, control: carla.VehicleControl) -> LowLevelAction:
        """
        将 CARLA 的 VehicleControl 映射成统一三元组：
        throttle ∈ [0,1], brake ∈ [0,1], steer ∈ [-1,1]
        """
        th = float(max(0.0, min(1.0, control.throttle)))
        br = float(max(0.0, min(1.0, control.brake)))
        st = float(max(-1.0, min(1.0, control.steer)))
        return th, br, st

    def plan(self, obs, info: Dict[str, Any] = None) -> LowLevelAction:
        """
        主接口：返回 (throttle, brake, steer)
        注意：不调用 ego.apply_control；由上层环境统一执行。
        """
        if not self._ready or self._agent is None:
            # 上下文未注入，保守制动
            return 0.0, 1.0, 0.0

        # 使用 BasicAgent 的局部策略来生成控制（不写车，仅取值）
        try:
            ctrl: carla.VehicleControl = self._agent.run_step(debug=False)
            return self._to_low_level(ctrl)
        except Exception:
            # 异常时紧急制动
            return 0.0, 1.0, 0.0
