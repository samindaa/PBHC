# Unified Robot Control Interface for HumanoidVerse
# Weiji Xie @ 2025.03.04

import mujoco
import os
import time
import signal
from typing import Union
import ml_collections

import torch
from humanoidverse.deploy import URCIRobot
from scipy.spatial.transform import Rotation as R
import logging
from utils.config_utils import *  # noqa: E402, F403
# add argparse arguments

from typing import Dict, Optional, Tuple
from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from loguru import logger
from humanoidverse.utils.helpers import np2torch, torch2np
from humanoidverse.utils.real.rotation_helper import *
from humanoidverse.utils.noise_tool import noise_process_dict, RadialPerturbation
from description.robots.dtype import RobotExitException

import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from humanoidverse.common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, MotorMode
# from common.rotation_helper import get_gravity_orientation, transform_imu_data
from humanoidverse.common.remote_controller import RemoteController, KeyMap
# from config import Config


class Controller:

    def __init__(self) -> None:
        # self.config = config
        self.remote_controller = RemoteController()

        # Initializing process variables
        # self.qj = np.zeros(config.num_actions, dtype=np.float32)
        # self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        # self.action = np.zeros(config.num_actions, dtype=np.float32)
        # self.target_dof_pos = config.default_angles.copy()
        # self.obs = np.zeros(config.num_obs, dtype=np.float32)
        # self.cmd = np.array([0.0, 0, 0])
        # self.counter = 0

        self.config = ml_collections.config_dict.create(
            dof_idx=[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28
            ],
            kps=[
                100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40, 400,
                400, 400, 100, 100, 50, 50, 20, 20, 20, 100, 100, 50, 50, 20,
                20, 20
            ],
            kds=[
                2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5, 2, 2, 2, 2, 1, 1,
                1, 2, 2, 2, 2, 1, 1, 1
            ],
            use_dof_idx=[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 22, 23, 24, 25
            ])

        # g1 and h1_2 use the hg msg type
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmdHG)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateHG)
        self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        # while self.low_state.tick == 0:
        #     time.sleep(0.02)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        # TODO: this needs be enabled
        while True:
            start_str = input("Enter s: ")
            if start_str in ["s", "S"]:
                break

        # while self.remote_controller.button[KeyMap.start] != 1:
        #     create_zero_cmd(self.low_cmd)
        #     self.send_cmd(self.low_cmd)
        #     time.sleep(0.02)

    def move_to_default_pos(self, default_pos):
        print(f"Moving to default pos: {len(default_pos)=} {default_pos}")
        # move time 2s
        total_time = 2
        num_step = int(total_time / 0.02)

        dof_size = len(self.config.dof_idx)
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[
                self.config.dof_idx[i]].q

        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = self.config.dof_idx[j]
                # PBHC 23DOF
                if motor_idx >= len(default_pos):
                    continue
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (
                    1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(0.02)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        # while self.remote_controller.button[KeyMap.A] != 1:
        #     for i in range(len(self.config.leg_joint2motor_idx)):
        #         motor_idx = self.config.leg_joint2motor_idx[i]
        #         self.low_cmd.motor_cmd[
        #             motor_idx].q = self.config.default_angles[i]
        #         self.low_cmd.motor_cmd[motor_idx].qd = 0
        #         self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
        #         self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
        #         self.low_cmd.motor_cmd[motor_idx].tau = 0
        #     for i in range(len(self.config.arm_waist_joint2motor_idx)):
        #         motor_idx = self.config.arm_waist_joint2motor_idx[i]
        #         self.low_cmd.motor_cmd[
        #             motor_idx].q = self.config.arm_waist_target[i]
        #         self.low_cmd.motor_cmd[motor_idx].qd = 0
        #         self.low_cmd.motor_cmd[
        #             motor_idx].kp = self.config.arm_waist_kps[i]
        #         self.low_cmd.motor_cmd[
        #             motor_idx].kd = self.config.arm_waist_kds[i]
        #         self.low_cmd.motor_cmd[motor_idx].tau = 0
        #     self.send_cmd(self.low_cmd)
        #     time.sleep(self.config.control_dt)

    def get_state(self, data):
        """Get mujoco data proxy."""
        # TODO: Get the current joint position and velocity
        data.qpos[:3] = np.array([0, 0, 1])
        data.qpos[3:7] = np.array([1, 0, 0, 0])
        for i, j in enumerate(self.config.use_dof_idx):
            data.qpos[7 + i] = self.low_state.motor_state[j].q
            data.qpos[6 + i] = self.low_state.motor_state[j].dq

        # print("XXX", data.qpos)
        # print("YYY", data.qvel)

        return data

    def send_target(self, target_dof_pos):
        """Send target_dof_pos to hardware."""
        for motor_idx in self.config.dof_idx:
            if motor_idx in self.config.use_dof_idx:
                tgt = target_dof_pos[self.config.use_dof_idx.index(motor_idx)]
            else:
                tgt = 0.0
            self.low_cmd.motor_cmd[motor_idx].q = tgt
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[motor_idx]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[motor_idx]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        self.send_cmd(self.low_cmd)


class RealRobot(URCIRobot):
    REAL = True
    HANG = True

    print_torque = lambda tau: print(
        f"tau (norm, max) = {np.linalg.norm(tau):.2f}, \t{np.max(tau):.2f}",
        end='\r')

    def __init__(self, cfg):
        super().__init__(cfg)

        # Initialize DDS communication
        # TODO: change this to 0 and comm interface.
        ChannelFactoryInitialize(1, "lo")
        self.controller = Controller()

        # Enter the zero torque state, press the start key to continue executing
        self.controller.zero_torque_state()

        # Move to the default position
        self.controller.move_to_default_pos(self.dof_init_pose)

        # Enter the default position state, press the A key to continue executing
        self.controller.default_pos_state()

        def signal_handler(sig, frame):
            logger.info("Ctrl+C  Exiting safely...")
            raise RobotExitException("Mujoco Robot Exiting")

        signal.signal(signal.SIGINT, signal_handler)

        self.decimation = cfg.simulator.config.sim.control_decimation
        self.sim_dt = 1 / cfg.simulator.config.sim.fps
        assert self.dt == self.decimation * self.sim_dt
        # self._subtimer = 0

        self.model = mujoco.MjModel.from_xml_path(
            os.path.join(cfg.robot.asset.asset_root,
                         cfg.robot.asset.xml_file))  # type: ignore
        print("XML", cfg.robot.asset.xml_file)
        self.data = mujoco.MjData(self.model)  # type: ignore
        self.model.opt.timestep = self.sim_dt
        print("timestep", self.model.opt.timestep)

        self.num_ctrl = self.data.ctrl.shape[0]
        assert self.num_ctrl == self.num_actions, f"Number of control DOFs {self.num_ctrl} does not match number of actions {self.num_actions}"

        logger.info("Initializing Mujoco Robot")
        logger.info("Task Name: {}".format(cfg.log_task_name))
        logger.info("Robot Type: {}".format(cfg.robot.asset.robot_type))

        logger.info(
            f"decimation: {self.decimation}, sim_dt: {self.sim_dt}, dt: {self.dt}"
        )
        logger.info(f"xml_file: {cfg.robot.asset.xml_file}")
        # print(self.decimation, self.sim_dt, self.dt)
        self.Reset()

        ###
        # mujoco.mj_step(self.model, self.data)  # type: ignore
        ###

    def _reset(self):
        self.data.qpos[:3] = np.array(self.cfg.robot.init_state.pos)
        self.data.qpos[3:7] = np.array(
            self.cfg.robot.init_state.rot)[[3, 0, 1, 2]]  # XYZW to WXYZ
        # self.data.qpos[3:7] = np.array([0,0,0.7184,-0.6956])[[3,0,1,2]]  #DEBUG: init quat of JingjiTaiji
        # self.data.qpos[3:7] = np.array([0,0,0.7455,-0.6665])[[3,0,1,2]]  #DEBUG: init quat of NewTaiji
        # self.data.qpos[3:7] = np.array([0,0,0.6894,0.7244])[[3,0,1,2]]  #DEBUG: init quat of Shaolinquan
        self.data.qpos[7:] = self.dof_init_pose
        self.data.qvel[:] = 0
        self.cmd = np.array(self.cfg.deploy.defcmd)
        self.controller.move_to_default_pos(self.cfg.robot.init_state.pos)
        logging.info("Rest done")

    @staticmethod
    def pd_control(target_q, q, kp, target_dq, dq, kd):
        '''Calculates torques from position commands
        '''
        return (target_q - q) * kp + (target_dq - dq) * kd

    def _get_state(self):
        '''Extracts physical states from the mujoco data structure
        '''
        self.data = self.controller.get_state(self.data)
        data = self.data
        self.q_raw = data.qpos.astype(np.double)[7:]  # 19 dim
        self.q = self.q_raw.copy()
        # WXYZ
        # 3 dim base pos + 4 dim quat + 12 dim actuation angles
        self.dq_raw = data.qvel.astype(np.double)[6:]  # 18 dim ?????
        self.dq = self.dq_raw.copy()
        # 3 dim base vel + 3 dim omega + 12 dim actuation vel

        self.pos = data.qpos.astype(np.double)[:3]
        self.quat_raw = data.qpos.astype(np.double)[3:7][[1, 2, 3,
                                                          0]]  # WXYZ to XYZW
        self.quat = self.quat_raw.copy()
        self.vel = data.qvel.astype(np.double)[:3]
        self.omega_raw = data.qvel.astype(np.double)[3:6]
        self.omega = self.omega_raw.copy()

        r = R.from_quat(self.quat)  # R.from_quat: need xyzw
        self.rpy = quaternion_to_euler_array(self.quat)  # need xyzw
        self.rpy[self.rpy > math.pi] -= 2 * math.pi
        self.gvec = r.apply(np.array([0., 0., -1.]),
                            inverse=True).astype(np.double)

    def _sanity_check(self, target_q):
        unsafe_dof = np.where((np.abs(target_q - self.q) > 2.2)
                              | (np.abs(self.dq) > 20))[0]
        if len(unsafe_dof) > 0:
            for motor_idx in unsafe_dof:
                logger.error(f"Action of joint {motor_idx} is too large.\n"
                             f"target q\t: {target_q[motor_idx]} \n"
                             f"target dq\t: {0} \n"
                             f"q\t\t: {self.q[motor_idx]} \n"
                             f"dq\t\t: {self.dq[motor_idx]}\n")
                # breakpoint()

    _motor_offset = np.array([
        3, 0.5, 2, -0.5, -1, 1, -2, 1, -.3, 1, 0.3, 0.1, 0, 0, 0, 0, 1, 0, -1,
        -2, 0, 0, 0
    ]) * (np.pi / 180)  # [23]

    def _apply_action(self, target_q):
        """Apply hardware action."""

        # TODO: 50Hz is handle by urcirobot.
        # TODO: if the last action is not presisted
        self._sanity_check(target_q)

        ## 50Hz and sleep
        self.GetState()

        tau = self.pd_control(target_q, self.q, self.kp, 0, self.dq,
                              self.kd)  # Calc torques

        tau = np.clip(tau, -self.tau_limit, self.tau_limit)  # Clamp torques

        # MujocoRobot.print_torque(tau)
        # tau*=0
        # print(np.linalg.norm(target_q-self.q), np.linalg.norm(self.dq), np.linalg.norm(tau))
        # self.data.qpos[:3] = np.array([0,0,1])

        self.data.ctrl[:] = tau

        if self.HANG:
            # self.data.ctrl[14] = 0.5
            self.data.qpos[:3] = np.array([0, 0, 1])
            self.data.qpos[3:7] = np.array([1, 0, 0, 0])

        ## TODO
        # mujoco.mj_step(self.model, self.data)  # type: ignore
        ##
        # self.controller.send_target(self.data.ctrl)

        # self.tracking()
        # self.render_step()
        # self._subtimer += 1

    # Noise Version ApplyAction & Obs
    def ApplyAction(self, action):
        URCIRobot.ApplyAction(self, action)

        # breakpoint()

    def Obs(self):

        # return {k: torch2np(v) for k, v in self.obs_buf_dict.items()}

        actor_obs = torch2np(self.obs_buf_dict['actor_obs']).reshape(1, -1)

        return {'actor_obs': actor_obs}

    def _get_motion_to_save_torch(
            self) -> Tuple[float, Dict[str, torch.Tensor]]:

        from scipy.spatial.transform import Rotation as sRot

        motion_time = (self.timer) * self.dt

        root_trans = self.pos
        root_rot = self.quat  # XYZW
        root_rot_vec = torch.from_numpy(
            sRot.from_quat(root_rot).as_rotvec()).float()  # type: ignore
        dof = self.q
        # T, num_env, J, 3
        # print(self._motion_lib.mesh_parsers.dof_axis)
        pose_aa = torch.cat([
            root_rot_vec[..., None, :],
            torch.from_numpy(self._dof_axis * dof[..., None]),
            torch.zeros((self.num_augment_joint, 3))
        ],
                            axis=0)  # type: ignore

        return motion_time, {
            'root_trans_offset': torch.from_numpy(root_trans),
            'root_rot': torch.from_numpy(root_rot),  # 统一save xyzw
            'dof': torch.from_numpy(dof),
            'pose_aa': pose_aa,
            'action': torch.from_numpy(self.act),
            'actor_obs': np2torch(self.obs_buf_dict['actor_obs']),
            'terminate': torch.zeros((1, )),
            'dof_vel': torch.from_numpy(self.dq),
            'root_lin_vel': torch.from_numpy(self.vel),
            'root_ang_vel': torch.from_numpy(self.omega),
            'clock_time': torch.from_numpy(np.array([time.time()])),
            'tau': torch.from_numpy(self.data.ctrl).clone(),
            'cmd': torch.from_numpy(self.cmd),
            'root_rot_raw': torch.from_numpy(self.quat_raw),
            'root_ang_vel_raw': torch.from_numpy(self.omega_raw),
        }

    def _get_motion_to_save_np(self) -> Tuple[float, Dict[str, np.ndarray]]:

        from scipy.spatial.transform import Rotation as sRot

        motion_time = (self.timer) * self.dt

        root_trans = self.pos
        root_rot = self.quat  # XYZW
        root_rot_vec = np.array(sRot.from_quat(root_rot).as_rotvec(),
                                dtype=np.float32)  # type: ignore
        dof = self.q
        # T, num_env, J, 3
        # print(self._motion_lib.mesh_parsers.dof_axis)
        pose_aa = np.concatenate([
            root_rot_vec[..., None, :],
            np.array(self._dof_axis * dof[..., None]),
            np.zeros((self.num_augment_joint, 3))
        ],
                                 axis=0)  # type: ignore

        return motion_time, {
            'root_trans_offset': (root_trans).copy(),
            'root_rot': (root_rot).copy(),  # 统一save xyzw
            'dof': (dof).copy(),
            'pose_aa': pose_aa,
            'action': (self.act).copy(),
            'actor_obs': (self.obs_buf_dict['actor_obs']),
            'terminate': np.zeros((1, )),
            'dof_vel': (self.dq).copy(),
            'root_lin_vel': (self.vel).copy(),
            'root_ang_vel': (self.omega).copy(),
            'clock_time': (np.array([time.time()])),
            'tau': (self.data.ctrl).copy(),
            'cmd': (self.cmd).copy()
        }

    _get_motion_to_save = _get_motion_to_save_np

    def tracking(self):
        # breakpoint()
        # print(np.linalg.norm(self.data.xpos[6]-self.data.xpos[12]))
        if np.any(self.data.contact.pos[:, 2] > 0.01):
            names_list = self.model.names.decode('utf-8').split('\x00')[:40]
            res = np.zeros((6, 1), dtype=np.float64)
            geom_name = lambda x: (names_list[self.model.geom_bodyid[x] + 1])
            geom_force = lambda x: mujoco.mj_contactForce(
                self.model, self.data, x, res)  #type:ignore

            for contact in self.data.contact:
                if contact.pos[
                        2] > 0.01 and contact.geom1 != 0 and contact.geom2 != 0:
                    geom1_name = geom_name(contact.geom1)
                    geom2_name = geom_name(contact.geom2)
                    logger.warning(
                        f"Warning!!! Collision between '{geom1_name,contact.geom1}' and '{geom2_name,contact.geom2}' at position {contact.pos}."
                    )
                    # breakpoint()
