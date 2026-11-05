"""armkin: forward/inverse kinematics, Jacobians and trajectories for serial manipulators."""

from .models import planar_2r, planar_2r_ik, planar_3r, ur5
from .robot import IKResult, Link, SerialRobot
from .trajectory import Trajectory, quintic
from .transforms import dh_transform, make_pose, pose_error, rot_x, rot_y, rot_z, rotation_log

__version__ = "0.1.0"

__all__ = [
    "IKResult",
    "Link",
    "SerialRobot",
    "Trajectory",
    "dh_transform",
    "make_pose",
    "planar_2r",
    "planar_2r_ik",
    "planar_3r",
    "pose_error",
    "quintic",
    "rot_x",
    "rot_y",
    "rot_z",
    "rotation_log",
    "ur5",
]
