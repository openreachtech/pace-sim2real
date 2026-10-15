# Licensed under the Apache License 2.0

"""PACE fitting task for Anaguma.

The robot is ``ANAGUMA_UNITREE_CFG`` from tsubame_isaac (the asset the locomotion and jump
policies are trained with), with its actuators swapped for :class:`PaceDCMotorCfg` at the
training gains kp=120 / kd=3.

Timing matches training: 200 Hz physics (sim.dt = 5 ms) and decimation 4, so one env.step
is one 20 ms sample of ``scripts/pace/anaguma/chirp_collection.py`` (the 50 Hz CAN loop on
the robot). The delay parameter is counted in physics steps (5 ms).
"""

import torch

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from tsubame_isaac.assets import ANAGUMA_UNITREE_CFG

from pace_sim2real import PaceCfg, PaceSim2realEnvCfg, PaceSim2realSceneCfg
from pace_sim2real.utils import PaceDCMotorCfg


def _pace_actuator_cfg(joint_names_expr: list[str], peak_torque: float, rated_speed: float, max_speed: float):
    """ENCOS torque-speed envelope as a DC motor.

    Full torque up to `rated_speed`, then tapering linearly to 0 at `max_speed`: the same
    curve as tsubame_isaac's UnitreeActuator (X1, X2, Y1) in the driving direction.
    """
    return PaceDCMotorCfg(
        joint_names_expr=joint_names_expr,
        saturation_effort=peak_torque * max_speed / (max_speed - rated_speed),
        effort_limit=peak_torque,
        velocity_limit=max_speed,
        stiffness={".*": 120.0},  # P gain [Nm/rad], training / deploy value
        damping={".*": 3.0},  # D gain [Nm s/rad], training / deploy value
        encoder_bias={".*": 0.0},
        friction={".*": 0.0},
        dynamic_friction={".*": 0.0},
        viscous_friction={".*": 0.0},
        max_delay=10,  # [physics steps] = 50 ms
    )


ANAGUMA_PACE_ACTUATORS = {
    # ENCOS EC-A6408-P2-30.25H
    "hip_thigh": _pace_actuator_cfg([".*_hip_joint", ".*_thigh_joint"], peak_torque=70.0, rated_speed=12.86, max_speed=13.40),
    # ENCOS EC-A6408-P2-25
    "knee": _pace_actuator_cfg([".*_knee_joint"], peak_torque=60.0, rated_speed=13.93, max_speed=15.60),
}


@configclass
class AnagumaPaceCfg(PaceCfg):
    """Pace configuration for Anaguma."""

    robot_name: str = "anaguma"
    data_dir: str = "anaguma/chirp_data_real.pt"  # data/anaguma/chirp_data_real.pt
    bounds_params: torch.Tensor = torch.zeros((49, 2))  # 12 + 12 + 12 + 12 + 1 = 49 parameters to optimize
    # Columns of chirp_data.pt: the robot's physical / CAN order FL, FR, RL, RR x hip, thigh, calf.
    # Isaac names the rear legs BL/BR and the calf "knee"; axes and limits are the same.
    joint_order: list[str] = [
        "FL_hip_joint", "FL_thigh_joint", "FL_knee_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_knee_joint",
        "BL_hip_joint", "BL_thigh_joint", "BL_knee_joint",
        "BR_hip_joint", "BR_thigh_joint", "BR_knee_joint",
    ]

    def __post_init__(self):
        self.bounds_params[:12, 0] = 1e-4
        self.bounds_params[:12, 1] = 0.2  # armature [kg m^2] (nominal 0.058 hip/thigh, 0.039 knee)
        self.bounds_params[12:24, 1] = 2.0  # viscous friction [Nm s/rad]
        self.bounds_params[24:36, 1] = 2.0  # coulomb friction [Nm] (the real fit reached 1.0)
        # encoder bias [rad], effectively fixed at 0: it nearly cancels in the score and on the real
        # data it drifted to the +-0.1 bounds, dragging armature / friction with it
        self.bounds_params[36:48, 0] = -1e-4
        self.bounds_params[36:48, 1] = 1e-4
        self.bounds_params[48, 1] = 10.0  # delay [physics steps of 5 ms]


@configclass
class AnagumaPaceSceneCfg(PaceSim2realSceneCfg):
    """Anaguma hanging in the air (the base is fixed by PaceSim2realEnvCfg)."""

    robot: ArticulationCfg = ANAGUMA_UNITREE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        actuators=ANAGUMA_PACE_ACTUATORS,
    )


@configclass
class AnagumaPaceEnvCfg(PaceSim2realEnvCfg):

    scene: AnagumaPaceSceneCfg = AnagumaPaceSceneCfg()
    sim2real: PaceCfg = AnagumaPaceCfg()

    def __post_init__(self):
        super().__post_init__()

        # same physics rate as training; one env.step = one 20 ms sample of the chirp data
        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.render_interval = self.decimation


@configclass
class AnagumaMujocoPaceCfg(AnagumaPaceCfg):
    """Fits the MuJoCo stand-in data (scripts/pace/anaguma/chirp_collection.py --backend mujoco)."""

    robot_name: str = "anaguma_mujoco"
    data_dir: str = "anaguma_mujoco/chirp_data.pt"


@configclass
class AnagumaMujocoPaceEnvCfg(AnagumaPaceEnvCfg):
    sim2real: PaceCfg = AnagumaMujocoPaceCfg()
