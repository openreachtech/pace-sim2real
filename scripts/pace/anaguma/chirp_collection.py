# Licensed under the Apache License 2.0

"""Chirp data collection for Anaguma, on the real robot or on a MuJoCo stand-in.

Hardware counterpart of ``scripts/pace/data_collection.py``: it plays a linear
chirp as absolute joint-angle targets and records the encoder angles, then
writes ``time``, ``dof_pos`` and ``des_dof_pos`` to
``data/<robot_name>/chirp_data.pt`` for ``scripts/pace/fit.py``.

Procedure (real robot):

1. Suspend the body so that all four feet are off the ground.
2. Run this script with the PD gains the policy was trained with (default
   kp=120, kd=3, the deploy values of tsubame_control's AnagumaControl).
3. The script ramps slowly from the measured pose to the first chirp target,
   holds there, then logs the chirp. Every sample sends exactly one target per
   motor and records the feedback angle that arrives in reply, i.e. the angle
   measured *before* that target takes effect -- the same order as
   ``data_collection.py`` (read joint_pos, then env.step(target)).
4. After the chirp it ramps back to the start pose and disables the motors.

Backends:

* ``--backend real``    ENCOS motors over CAN through tsubame_control's
  ``encos_lib`` (run on the robot with the ROS 2 workspace sourced).
* ``--backend mujoco``  Anaguma's MJCF with the body welded in the air. The
  motor firmware's MIT-mode PD is emulated at the physics rate with the target
  held for one sample period, so the same collection loop and the same output
  file can be checked without hardware.

Columns of ``dof_pos``/``des_dof_pos`` follow ``JOINT_ORDER`` (the physical /
CAN order FL, FR, RL, RR x hip, thigh, calf). The fitting task's
``PaceCfg.joint_order`` must list the Isaac joint names in this same order.

Examples::

    # dry run in MuJoCo (no hardware)
    python scripts/pace/anaguma/chirp_collection.py --backend mujoco \\
        --mjcf ~/tsubame/ros2_ws/src/tsubame_robot/tsubame_description/mjcf/anaguma.xml --plot

    # real robot (suspended!)
    python scripts/pace/anaguma/chirp_collection.py --backend real --plot
"""

from __future__ import annotations

import argparse
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Physical / CAN / MuJoCo actuator order (tsubame_control anaguma_parameters.JOINT_NAME).
JOINT_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

# URDF joint limits [rad] (anaguma.urdf / anaguma.xml), per joint type.
JOINT_LIMITS = {
    "hip": (-0.785398, 0.785398),
    "thigh": (-1.745329, 2.356194),
    "calf": (-0.523599, 1.396263),
}

# Chirp shape per joint type: target = center + direction * amplitude * sin(phase).
# The legs hang freely, so these stay well inside JOINT_LIMITS.
CHIRP_CENTER = {"hip": 0.0, "thigh": 0.3, "calf": 0.6}
CHIRP_AMPLITUDE = {"hip": 0.15, "thigh": 0.3, "calf": 0.3}

# Sign of the oscillation per joint, chosen so that the reaction moments on the
# hanging body cancel:
#   hip   -- left/right mirrored: all four feet swing outward together. The
#            rear hip axes are -x in the URDF (front +x), hence FL+, FR-, RL-, RR+
#            (same pattern as the standby pose 0.276/-0.276/-0.276/0.276).
#   thigh/calf -- front and rear legs in anti-phase, cancelling the pitch moment.
# Only the oscillation is signed, not the center, so a flipped joint stays in range.
CHIRP_DIRECTION = [
    +1.0, +1.0, +1.0,  # FL
    -1.0, +1.0, +1.0,  # FR
    -1.0, -1.0, -1.0,  # RL
    +1.0, -1.0, -1.0,  # RR
]

# Velocity-dependent torque limit of the ENCOS motors (AnagumaControl.torque_speed_curve):
# full torque Y1 up to |vel| = X1, tapering linearly to 0 at |vel| = X2.
TORQUE_SPEED_CURVE = {
    "hip": (12.86, 13.40, 70.0),
    "thigh": (12.86, 13.40, 70.0),
    "calf": (13.93, 15.60, 60.0),
}

LIMIT_MARGIN = 0.05  # [rad] abort when a measured angle leaves the URDF range by more than this


def joint_type(name: str) -> str:
    return name.split("_")[1]


def joint_limits() -> np.ndarray:
    return np.array([JOINT_LIMITS[joint_type(name)] for name in JOINT_ORDER])


def build_chirp(duration: float, dt: float, f0: float, f1: float, amplitude_scale: float):
    """Linear chirp f0 -> f1 over `duration`, as absolute joint targets [num_steps, 12]."""
    num_steps = int(round(duration / dt))
    t = np.arange(num_steps) * dt
    # phase = 2*pi*(f0*t + (f1-f0)/(2*duration)*t^2)
    phase = 2 * math.pi * (f0 * t + (f1 - f0) / (2 * duration) * t ** 2)
    center = np.array([CHIRP_CENTER[joint_type(name)] for name in JOINT_ORDER])
    amplitude = amplitude_scale * np.array([CHIRP_AMPLITUDE[joint_type(name)] for name in JOINT_ORDER])
    direction = np.array(CHIRP_DIRECTION)
    targets = center + direction * amplitude * np.sin(phase)[:, None]
    return t, targets


def check_within_limits(targets: np.ndarray):
    limits = joint_limits()
    low, high = targets.min(axis=0), targets.max(axis=0)
    for i, name in enumerate(JOINT_ORDER):
        if low[i] < limits[i, 0] or high[i] > limits[i, 1]:
            raise ValueError(f"Chirp for {name} spans [{low[i]:.3f}, {high[i]:.3f}] rad, outside the joint "
                             f"limits [{limits[i, 0]:.3f}, {limits[i, 1]:.3f}]. Lower --amplitude_scale.")


def cosine_ramp(start: np.ndarray, goal: np.ndarray, num_steps: int) -> np.ndarray:
    """Targets easing from start to goal with zero velocity at both ends (TrajectoryPlayer's ease)."""
    s = np.arange(1, num_steps + 1) / max(num_steps, 1)
    w = (1.0 - np.cos(np.pi * s)) / 2.0
    return start + w[:, None] * (goal - start)


@dataclass
class Feedback:
    pos: np.ndarray
    temperature: np.ndarray
    error_code: np.ndarray
    ok: np.ndarray  # per motor: a reply arrived for this command


class SafetyAbort(RuntimeError):
    pass


# --------------------------------------------------------------------------------------
# Backends: exchange() sends one target per joint and returns the feedback measured
# before that target takes effect.
# --------------------------------------------------------------------------------------


class RealBackend:
    """ENCOS motors (MIT mode) over CAN, via tsubame_control's encos_lib."""

    realtime = True

    def __init__(self, kp: float, kd: float):
        # Imported here so that the MuJoCo backend runs without the ROS 2 workspace.
        from tsubame_control.encos_lib.encos_lib import CanController, CanMotor
        from tsubame_control.tsubame_utils.parameters import anaguma_parameters as P

        assert list(P.JOINT_NAME) == JOINT_ORDER, "JOINT_ORDER drifted from tsubame_control's JOINT_NAME"
        self.kp, self.kd = kp, kd
        configs = P.select_motor_configs("anaguma")
        self.controllers = {}
        self.motors = []
        for name in JOINT_ORDER:
            cfg = configs[name]
            if cfg.can_device not in self.controllers:
                self.controllers[cfg.can_device] = CanController(bus_name=cfg.can_device)
            self.motors.append(CanMotor(self.controllers[cfg.can_device], motor_id=cfg.can_id,
                                        motor_dir=cfg.motor_dir, motor_type=cfg.motor_type))
        self.last_pos = np.zeros(len(JOINT_ORDER))

    def start(self) -> np.ndarray:
        limits = joint_limits()
        for name, motor in zip(JOINT_ORDER, self.motors):
            if not motor.enable_motor():
                raise RuntimeError(f"Motor {name} failed to enable.")
        time.sleep(1.0)
        # Zero-gain probe: reads the absolute encoders without applying torque.
        fb = self._exchange(np.zeros(len(JOINT_ORDER)), kp=0.0, kd=0.0)
        if not fb.ok.all():
            missing = [n for n, ok in zip(JOINT_ORDER, fb.ok) if not ok]
            raise RuntimeError(f"No feedback from {missing}.")
        for i, name in enumerate(JOINT_ORDER):
            print(f"[INIT] {name:16s} pos {fb.pos[i]:+.3f} rad  temp {fb.temperature[i]:.0f} C")
            if not (limits[i, 0] - 0.15 <= fb.pos[i] <= limits[i, 1] + 0.15):
                raise RuntimeError(f"{name} reads {fb.pos[i]:.3f} rad, outside its limits: the encoder zero "
                                   f"has drifted, re-zero it before running.")
            # encos_lib clips outgoing targets to this range.
            self.motors[i].set_angle_range(limits[i, 0], limits[i, 1], deg=False)
        return fb.pos

    def exchange(self, target: np.ndarray) -> Feedback:
        return self._exchange(target, self.kp, self.kd)

    def _exchange(self, target, kp, kd) -> Feedback:
        n = len(self.motors)
        ok = np.zeros(n, dtype=bool)
        # One motor at a time, as anaguma_main's CAN loop does.
        for i, motor in enumerate(self.motors):
            ok[i] = motor.send_rad_command(float(target[i]), 0.0, kp, kd, 0.0)
        pos = np.array([m.current_position for m in self.motors])
        pos[~ok] = self.last_pos[~ok]
        self.last_pos = pos
        return Feedback(
            pos=pos,
            temperature=np.array([m.current_temp for m in self.motors]),
            error_code=np.array([m.current_error for m in self.motors]),
            ok=ok,
        )

    def damp(self, duration: float, dt: float):
        """Zero stiffness, damping only: the legs settle under gravity without a jolt."""
        for _ in range(int(duration / dt)):
            self._exchange(self.last_pos, kp=0.0, kd=self.kd)
            time.sleep(dt)

    def close(self):
        for name, motor in zip(JOINT_ORDER, self.motors):
            try:
                motor.disable_motor()
            except Exception as e:  # keep disabling the others
                print(f"[WARN] disabling {name} failed: {e}")
        for controller in self.controllers.values():
            controller.shutdown()


class MujocoBackend:
    """Suspended Anaguma in MuJoCo with the motor's MIT-mode PD emulated per physics step."""

    def __init__(self, mjcf: Path, kp: float, kd: float, dt: float, delay_steps: int, viewer: bool):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(self._fixed_base_xml(mjcf))
        self.data = mujoco.MjData(self.model)
        physics_dt = self.model.opt.timestep
        self.substeps = int(round(dt / physics_dt))
        if not math.isclose(self.substeps * physics_dt, dt, rel_tol=1e-6):
            raise ValueError(f"--dt {dt} is not a multiple of the MuJoCo timestep {physics_dt}.")
        self.kp, self.kd = kp, kd
        self.delay_steps = delay_steps
        self.qpos_adr = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in JOINT_ORDER])
        self.dof_adr = np.array([self.model.jnt_dofadr[self.model.joint(n).id] for n in JOINT_ORDER])
        self.act_ids = np.array([self.model.actuator(n).id for n in JOINT_ORDER])
        self.curves = np.array([TORQUE_SPEED_CURVE[joint_type(n)] for n in JOINT_ORDER])
        self.target_history = []
        # With the viewer, pace the samples in real time so the motion can be watched.
        self.realtime = viewer
        self.viewer = None
        if viewer:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    @staticmethod
    def _fixed_base_xml(mjcf: Path) -> str:
        """The robot MJCF with the body lifted and welded in place, mesh paths made absolute.

        The free joint is kept and the base pinned with a weld constraint: removing the
        joint instead would merge base_link into the world body, which turns off MuJoCo's
        parent/child contact filter and locks the hips against the base mesh.
        """
        root = ET.parse(mjcf).getroot()
        compiler = root.find("compiler")
        if compiler is not None:
            compiler.set("meshdir", str((mjcf.parent / compiler.get("meshdir", "")).resolve()))
        base = root.find("worldbody/body")
        base.set("pos", "0 0 1.0")
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        ET.SubElement(equality, "weld", body1=base.get("name"))
        return ET.tostring(root, encoding="unicode")

    def start(self) -> np.ndarray:
        mujoco = self.mujoco
        mujoco.mj_resetData(self.model, self.data)
        # Let the unpowered legs settle under gravity, like a robot just hung up.
        for _ in range(int(1.0 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        return self.data.qpos[self.qpos_adr].copy()

    def exchange(self, target: np.ndarray) -> Feedback:
        mujoco = self.mujoco
        fb = Feedback(
            pos=self.data.qpos[self.qpos_adr].copy(),
            temperature=np.full(len(JOINT_ORDER), 25.0),
            error_code=np.zeros(len(JOINT_ORDER), dtype=int),
            ok=np.ones(len(JOINT_ORDER), dtype=bool),
        )
        # The firmware holds the latest target until the next one arrives.
        for _ in range(self.substeps):
            self.target_history.append(np.asarray(target, dtype=float))
            del self.target_history[:-(self.delay_steps + 1)]
            applied = self.target_history[0]
            pos = self.data.qpos[self.qpos_adr]
            vel = self.data.qvel[self.dof_adr]
            torque = self.kp * (applied - pos) - self.kd * vel
            x1, x2, y1 = self.curves.T
            max_torque = np.where(np.abs(vel) < x1, y1, np.maximum(y1 - y1 / (x2 - x1) * (np.abs(vel) - x1), 0.0))
            self.data.ctrl[self.act_ids] = np.clip(torque, -max_torque, max_torque)
            mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()
        return fb

    def damp(self, duration: float, dt: float):
        pos = self.data.qpos[self.qpos_adr].copy()
        kp, self.kp = self.kp, 0.0
        for _ in range(int(duration / dt)):
            self.exchange(pos)
        self.kp = kp

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


# --------------------------------------------------------------------------------------
# Collection loop
# --------------------------------------------------------------------------------------


class Clock:
    """Fixed-rate schedule. On hardware it sleeps until each sample's deadline."""

    def __init__(self, dt: float, realtime: bool):
        self.dt, self.realtime = dt, realtime
        self.t0 = time.perf_counter()
        self.k = 0

    def tick(self) -> float:
        """Wait for the next sample slot; returns the time since the clock started."""
        if self.realtime:
            deadline = self.t0 + self.k * self.dt
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            elapsed = time.perf_counter() - self.t0
        else:
            elapsed = self.k * self.dt
        self.k += 1
        return elapsed


def play(backend, clock: Clock, targets: np.ndarray, max_temp: float, max_missed: int, record: bool):
    """Send `targets` one per sample. Returns measured angles and send times when `record`."""
    limits = joint_limits()
    dof_pos = np.zeros_like(targets)
    stamps = np.zeros(len(targets))
    missed = np.zeros(targets.shape[1], dtype=int)
    for k, target in enumerate(targets):
        stamps[k] = clock.tick()
        fb = backend.exchange(target)
        dof_pos[k] = fb.pos

        missed = np.where(fb.ok, 0, missed + 1)
        bad = [
            (f"no feedback from {JOINT_ORDER[i]} for {missed[i]} samples", missed[i] >= max_missed)
            for i in range(len(JOINT_ORDER))
        ] + [
            (f"{JOINT_ORDER[i]} at {fb.pos[i]:.3f} rad, outside its limits",
             fb.ok[i] and not (limits[i, 0] - LIMIT_MARGIN <= fb.pos[i] <= limits[i, 1] + LIMIT_MARGIN))
            for i in range(len(JOINT_ORDER))
        ] + [
            (f"{JOINT_ORDER[i]} at {fb.temperature[i]:.0f} C", fb.temperature[i] > max_temp)
            for i in range(len(JOINT_ORDER))
        ] + [
            (f"{JOINT_ORDER[i]} reports error code {fb.error_code[i]}", fb.error_code[i] != 0)
            for i in range(len(JOINT_ORDER))
        ]
        reasons = [msg for msg, hit in bad if hit]
        if reasons:
            raise SafetyAbort("; ".join(reasons))
        if record and (k + 1) % int(round(1.0 / clock.dt)) == 0:
            print(f"[INFO]: {(k + 1) * clock.dt:5.1f} / {len(targets) * clock.dt:.1f} s")
    return dof_pos, stamps


def plot(path: Path, t, dof_pos, des_dof_pos):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(t, des_dof_pos[:, i], "--", label="target")
        ax.plot(t, dof_pos[:, i], label="measured")
        ax.set_title(JOINT_ORDER[i])
        ax.grid(True)
    axes[0, 0].legend()
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Joint position [rad]")
    fig.tight_layout()
    fig.savefig(path)
    print(f"[INFO]: Plot saved to {path}")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".project-root").exists():
            return parent
    return Path.cwd()


def main():
    parser = argparse.ArgumentParser(description="Anaguma chirp data collection (real robot or MuJoCo).")
    parser.add_argument("--backend", choices=["real", "mujoco"], required=True)
    parser.add_argument("--robot_name", type=str, default=None,
                        help="Output goes to data/<robot_name>/chirp_data.pt "
                             "(default: anaguma for real, anaguma_mujoco for mujoco).")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="Sample interval [s]. Must match the sample interval of the fitting task "
                             "(anaguma_main's CAN loop runs at 50 Hz = 0.02 s).")
    parser.add_argument("--min_frequency", type=float, default=0.1, help="Chirp start frequency [Hz].")
    parser.add_argument("--max_frequency", type=float, default=10.0, help="Chirp end frequency [Hz].")
    parser.add_argument("--duration", type=float, default=20.0, help="Chirp duration [s].")
    parser.add_argument("--amplitude_scale", type=float, default=1.0,
                        help="Multiplies the per-joint chirp amplitudes (CHIRP_AMPLITUDE).")
    parser.add_argument("--kp", type=float, default=120.0, help="P gain [Nm/rad] (training value).")
    parser.add_argument("--kd", type=float, default=3.0, help="D gain [Nm s/rad] (training value, max 5.0).")
    parser.add_argument("--ramp_time", type=float, default=3.0, help="Time to move to the first target [s].")
    parser.add_argument("--hold_time", type=float, default=1.0, help="Settling time at the first target [s].")
    parser.add_argument("--max_temp", type=float, default=70.0, help="Abort above this motor temperature [C].")
    parser.add_argument("--max_missed", type=int, default=5, help="Abort after this many missed replies in a row.")
    parser.add_argument("--plot", action="store_true", help="Save a target-vs-measured plot next to the data.")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt (real backend).")
    sim = parser.add_argument_group("mujoco backend")
    sim.add_argument("--mjcf", type=Path,
                     default=Path.home() / "tsubame/ros2_ws/src/tsubame_robot/tsubame_description/mjcf/anaguma.xml",
                     help="Anaguma MJCF (the robot file, not the scene).")
    sim.add_argument("--sim_delay_steps", type=int, default=0,
                     help="Physics steps the targets lag behind the command, to emulate actuator delay "
                          "(anaguma_deploy.yaml sim.actuator.delay_steps).")
    sim.add_argument("--viewer", action="store_true", help="Show the MuJoCo passive viewer.")
    args = parser.parse_args()

    if args.kd > 5.0:
        parser.error("--kd above 5.0 is clipped by the ENCOS firmware (encos_lib KD_MAX).")
    robot_name = args.robot_name or ("anaguma" if args.backend == "real" else "anaguma_mujoco")

    t, chirp = build_chirp(args.duration, args.dt, args.min_frequency, args.max_frequency, args.amplitude_scale)
    check_within_limits(chirp)
    print(f"[INFO]: {len(t)} samples at {1 / args.dt:.0f} Hz, chirp {args.min_frequency}-{args.max_frequency} Hz")
    for i, name in enumerate(JOINT_ORDER):
        print(f"        {name:16s} [{chirp[:, i].min():+.3f}, {chirp[:, i].max():+.3f}] rad")

    if args.backend == "real":
        if not args.yes and input("Is the body suspended with all feet off the ground? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            return
        backend = RealBackend(args.kp, args.kd)
    else:
        backend = MujocoBackend(args.mjcf.expanduser(), args.kp, args.kd, args.dt, args.sim_delay_steps,
                                args.viewer)

    out_dir = project_root() / "data" / robot_name
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        start_pos = backend.start()
        approach = np.concatenate([
            cosine_ramp(start_pos, chirp[0], int(args.ramp_time / args.dt)),
            np.repeat(chirp[:1], int(args.hold_time / args.dt), axis=0),
        ])
        clock = Clock(args.dt, backend.realtime)
        print("[INFO]: Moving to the first target...")
        play(backend, clock, approach, args.max_temp, args.max_missed, record=False)

        print("[INFO]: Recording chirp...")
        clock = Clock(args.dt, backend.realtime)
        dof_pos, stamps = play(backend, clock, chirp, args.max_temp, args.max_missed, record=True)

        print("[INFO]: Returning to the start pose...")
        play(backend, Clock(args.dt, backend.realtime), cosine_ramp(chirp[-1], start_pos, int(args.ramp_time / args.dt)),
             args.max_temp, args.max_missed, record=False)
    except BaseException as e:
        # Any failure (safety check, Ctrl-C, CAN error): let the legs go limp under damping.
        print(f"[ERROR]: Stopping: {e!r}. Damping the joints and disabling the motors.")
        try:
            backend.damp(1.0, args.dt)
        finally:
            backend.close()
        raise SystemExit(1) if isinstance(e, (SafetyAbort, KeyboardInterrupt)) else e
    backend.damp(0.5, args.dt)
    backend.close()

    jitter = np.diff(stamps) - args.dt
    print(f"[INFO]: Sample interval error: mean {jitter.mean() * 1e3:+.2f} ms, max {np.abs(jitter).max() * 1e3:.2f} ms")
    rmse = np.sqrt(np.mean((dof_pos - chirp) ** 2, axis=0))
    for name, e in zip(JOINT_ORDER, rmse):
        print(f"        {name:16s} tracking RMSE {e:.4f} rad")

    import torch

    data_file = out_dir / "chirp_data.pt"
    torch.save({
        "time": torch.tensor(t, dtype=torch.float32),
        "dof_pos": torch.tensor(dof_pos, dtype=torch.float32),
        "des_dof_pos": torch.tensor(chirp, dtype=torch.float32),
        # Not read by fit.py; kept to check the recording afterwards.
        "time_measured": torch.tensor(stamps, dtype=torch.float64),
        "joint_order": JOINT_ORDER,
        "kp": args.kp,
        "kd": args.kd,
    }, data_file)
    print(f"[INFO]: Saved {data_file}")

    if args.plot:
        plot(out_dir / "chirp_data.png", t, dof_pos, chirp)


if __name__ == "__main__":
    main()
