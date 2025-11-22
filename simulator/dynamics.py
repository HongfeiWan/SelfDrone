#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动力学模块：实现基础动力学和四旋翼动力学
基于C++版本的dynamics.hpp和quadrotor_dynamics.hpp
支持批量处理和GPU加速
"""

import math
import numpy as np
import torch
from typing import Dict, Optional

class Dynamics:
    """
    基础动力学类（批量处理版本）
    处理刚体的位置、速度、加速度、姿态（四元数）和角速度
    支持jerk控制模式，通过控制力jerk和力矩jerk来平滑控制
    
    状态格式: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz] (13维)
    动作格式（jerk控制）: [force_jerk_x, force_jerk_y, force_jerk_z, torque_jerk_x, torque_jerk_y, torque_jerk_z] (6维)
        - force_jerk: 力jerk (N/s³)，控制世界坐标系中力的变化率
        - torque_jerk: 力矩jerk (N·m/s³)，控制体坐标系中力矩的变化率
    """
    def __init__(self, config: Dict, device: torch.device):
        """
        初始化动力学对象
        Args:
            config: 配置字典，支持从 simulator.dynamics 读取参数
            device: 计算设备
        """
        self.device = device
        # 获取dynamics配置
        if 'simulator' in config and isinstance(config.get('simulator'), dict):
            simulator_config = config['simulator']
            dynamics_config = simulator_config.get('dynamics', {})
            if not dynamics_config:
                dynamics_config = simulator_config
        else:
            dynamics_config = config.get('dynamics', config)
        
        # Jerk控制参数（固定使用jerk控制）
        self.max_force_jerk = dynamics_config.get('max_force_jerk', 100.0)  # 最大力jerk (N/s³)
        self.max_torque_jerk = dynamics_config.get('max_torque_jerk', 50.0)  # 最大力矩jerk (N·m/s³)
        self.max_force_accel = dynamics_config.get('max_force_accel', 50.0)  # 最大力加速度 (N/s²)
        self.max_torque_accel = dynamics_config.get('max_torque_accel', 25.0)  # 最大力矩加速度 (N·m/s²)
        
        # 当前控制状态（jerk控制）
        self.current_force_accel = None  # (N, 3) 当前力加速度 (N/s²)
        self.current_torque_accel = None  # (N, 3) 当前力矩加速度 (N·m/s²)
        
        # 延迟初始化：不在 __init__ 中注入参数，等外部 reset(vehicle_params) 再设置
        self.vehicle_params = None
        self.num_agents = 0
        self.mass = None  # (N,)
        self.inertia_mat = None  # (N, 3, 3)
    
    def step(self, states: torch.Tensor, actions: torch.Tensor, dt: float) -> torch.Tensor:
        """
        对一批刚体状态进行一步更新（使用jerk控制）
        Args:
            states: 当前状态，形状为 (N, 13) [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
            actions: jerk控制动作，形状为 (N, 6) 
                [force_jerk_x, force_jerk_y, force_jerk_z, torque_jerk_x, torque_jerk_y, torque_jerk_z]
                - force_jerk: 力jerk (N/s³)，控制世界坐标系中力的变化率
                - torque_jerk: 力矩jerk (N·m/s³)，控制体坐标系中力矩的变化率
            dt: 时间步长 (s)
        Returns:
            下一时刻状态，形状为 (N, 13)
        """
        assert states.ndim == 2 and states.shape[1] == 13, f"States shape must be (N, 13), but got {states.shape}"
        assert actions.ndim == 2 and actions.shape[1] == 6, f"Actions shape must be (N, 6) [force_jerk_x, force_jerk_y, force_jerk_z, torque_jerk_x, torque_jerk_y, torque_jerk_z], but got {actions.shape}"
        
        batch_size = states.shape[0]
        assert batch_size == self.num_agents, f"batch_size ({batch_size}) must match num_agents ({self.num_agents})"
        
        # Jerk控制模式：通过jerk积分得到力和力矩
        forces, torques = self._jerk_control_to_forces_torques(states, actions, dt)
        
        # 解包状态
        pos = states[:, :3]  # (N, 3)
        q = states[:, 3:7]  # (N, 4) [w, x, y, z]
        vel = states[:, 7:10]  # (N, 3)
        w_body = states[:, 10:13]  # (N, 3)
        
        # 归一化四元数
        q = q / (torch.norm(q, dim=1, keepdim=True) + 1e-9)
        
        # 计算加速度并更新位置和速度
        acceleration = forces / self.mass.unsqueeze(1)  # (N, 3)
        new_pos = pos + vel * dt
        new_vel = vel + acceleration * dt
        
        # 欧拉方程：更新角速度
        # w_dot = I^-1 * (Tau - w × (I * w))
        I_w = torch.bmm(self.inertia_mat, w_body.unsqueeze(2)).squeeze(2)  # (N, 3)
        w_cross_Iw = torch.cross(w_body, I_w, dim=1)  # (N, 3)
        I_inv = torch.inverse(self.inertia_mat)  # (N, 3, 3)
        w_dot = torch.bmm(I_inv, (torques - w_cross_Iw).unsqueeze(2)).squeeze(2)  # (N, 3)
        new_w_body = w_body + dt * w_dot
        
        # 从角速度更新四元数
        # q_dot = 0.5 * Q(q) * [0; w_body]
        # 构建四元数乘法矩阵（批量）
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        q_transmat = torch.stack([
            torch.stack([w, -x, -y, -z], dim=1),
            torch.stack([x,  w, -z,  y], dim=1),
            torch.stack([y,  z,  w, -x], dim=1),
            torch.stack([z, -y,  x,  w], dim=1)
        ], dim=1)  # (N, 4, 4)
        
        w_body_withzero = torch.cat([
            torch.zeros(batch_size, 1, device=self.device),
            new_w_body
        ], dim=1)  # (N, 4)
        
        d_q = 0.5 * torch.bmm(q_transmat, w_body_withzero.unsqueeze(2)).squeeze(2)  # (N, 4)
        new_q = q + d_q * dt
        
        # 归一化四元数
        new_q = new_q / (torch.norm(new_q, dim=1, keepdim=True) + 1e-9)
        
        # 组合新状态
        new_states = torch.cat([new_pos, new_q, new_vel, new_w_body], dim=1)  # (N, 13)
        return new_states
    
    def reset(self, vehicle_params: Dict[str, torch.Tensor]):
        """
        重置动力学参数
        Args:
            vehicle_params: 批量车辆参数，包含：
                - 'mass': (N,) 质量
                - 'inertia': (N, 3, 3) 惯性矩阵，或 (N, 3) 对角惯性矩阵
        """
        if vehicle_params is None:
            raise ValueError("reset(vehicle_params) 需要提供车辆参数")
        
        self.mass = vehicle_params['mass'].to(self.device)  # (N,)
        
        inertia = vehicle_params['inertia'].to(self.device)
        if inertia.ndim == 2 and inertia.shape[1] == 3:
            # 对角惯性矩阵，转换为 (N, 3, 3)
            self.inertia_mat = torch.diag_embed(inertia)  # (N, 3, 3)
        elif inertia.ndim == 3 and inertia.shape[1] == 3 and inertia.shape[2] == 3:
            # 完整惯性矩阵
            self.inertia_mat = inertia
        else:
            raise ValueError(f"Invalid inertia shape: {inertia.shape}, expected (N, 3) or (N, 3, 3)")
        
        self.num_agents = self.mass.shape[0]
        self.vehicle_params = vehicle_params
        
        # 重置jerk控制状态
        self.current_force_accel = torch.zeros(self.num_agents, 3, device=self.device)
        self.current_torque_accel = torch.zeros(self.num_agents, 3, device=self.device)
    
    def _jerk_control_to_forces_torques(self, states: torch.Tensor, jerk_actions: torch.Tensor, dt: float):
        """
        将jerk控制动作转换为力和力矩
        Jerk控制流程：
        1. 接收jerk动作 [force_jerk_x, force_jerk_y, force_jerk_z, torque_jerk_x, torque_jerk_y, torque_jerk_z]
        2. 积分得到加速度：a_new = a_old + jerk * dt
        3. 限制加速度范围
        4. 将加速度转换为力和力矩
        
        Args:
            states: 当前状态 (N, 13)
            jerk_actions: jerk控制动作 (N, 6) [force_jerk_x, force_jerk_y, force_jerk_z, torque_jerk_x, torque_jerk_y, torque_jerk_z]
            dt: 时间步长 (s)
        Returns:
            tuple: (forces, torques)
                - forces: 世界坐标系中的力 (N, 3)
                - torques: 体坐标系中的力矩 (N, 3)
        """
        batch_size = states.shape[0]
        
        # 初始化控制状态（如果尚未初始化或batch_size变化）
        if (self.current_force_accel is None or 
            self.current_force_accel.shape[0] != batch_size):
            self.current_force_accel = torch.zeros(batch_size, 3, device=self.device)
            self.current_torque_accel = torch.zeros(batch_size, 3, device=self.device)
        
        # ========== 步骤1: 解包jerk动作 ==========
        force_jerk = jerk_actions[:, :3]  # (N, 3) 力jerk (N/s³)
        torque_jerk = jerk_actions[:, 3:6]  # (N, 3) 力矩jerk (N·m/s³)
        
        # ========== 步骤2: 限制jerk范围 ==========
        force_jerk = torch.clamp(force_jerk, -self.max_force_jerk, self.max_force_jerk)
        torque_jerk = torch.clamp(torque_jerk, -self.max_torque_jerk, self.max_torque_jerk)
        
        # ========== 步骤3: 积分jerk得到加速度 ==========
        # a_new = a_old + jerk * dt
        new_force_accel = self.current_force_accel + force_jerk * dt  # (N, 3) 力加速度 (N/s²)
        new_torque_accel = self.current_torque_accel + torque_jerk * dt  # (N, 3) 力矩加速度 (N·m/s²)
        
        # ========== 步骤4: 限制加速度范围 ==========
        new_force_accel = torch.clamp(new_force_accel, -self.max_force_accel, self.max_force_accel)
        new_torque_accel = torch.clamp(new_torque_accel, -self.max_torque_accel, self.max_torque_accel)
        
        # 更新控制状态
        self.current_force_accel = new_force_accel
        self.current_torque_accel = new_torque_accel
        
        # ========== 步骤5: 将加速度转换为力和力矩 ==========
        # 当前力 = 当前力加速度 * dt（简化模型，假设从0开始）
        # 更准确的方式是：force = force_old + force_accel * dt
        # 这里简化处理，直接使用加速度积分
        forces = new_force_accel * dt  # (N, 3) 世界坐标系中的力
        torques = new_torque_accel * dt  # (N, 3) 体坐标系中的力矩
        
        return forces, torques

class QuadrotorDynamics:
    """
    四旋翼动力学类（批量处理版本）
    添加电机模型和四旋翼特定的力和力矩计算
    使用jerk控制模式，通过控制推力jerk和角加速度jerk来平滑控制四旋翼
    
    状态格式: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz] (13维)
    动作格式: [thrust_jerk, roll_jerk, pitch_jerk, yaw_jerk] (4维)
        - thrust_jerk: 推力jerk (N/s³)，控制垂直方向加速度的变化率
        - roll_jerk: 滚转角加速度jerk (rad/s³)
        - pitch_jerk: 俯仰角加速度jerk (rad/s³)
        - yaw_jerk: 偏航角加速度jerk (rad/s³)
    """
    
    def __init__(self, config: Dict, device: torch.device):
        """
        初始化四旋翼动力学对象
        Args:
            config: 配置字典，支持从 simulator.dynamics 读取参数
            device: 计算设备
        """
        self.device = device
        
        # 获取dynamics配置
        if 'simulator' in config and isinstance(config.get('simulator'), dict):
            simulator_config = config['simulator']
            dynamics_config = simulator_config.get('dynamics', {})
            if not dynamics_config:
                dynamics_config = simulator_config
        else:
            dynamics_config = config.get('dynamics', config)
        
        # 四旋翼参数（从配置读取，带默认值）
        self.arm_length = dynamics_config.get('arm_length', 0.22)  # 臂长 (m)
        self.min_rpm = dynamics_config.get('min_rpm', 0.0)
        self.max_rpm = dynamics_config.get('max_rpm', 35000.0)
        self.g = dynamics_config.get('gravity', 9.81)  # 重力加速度 (m/s^2)
        self.k_F = dynamics_config.get('k_F', 3 * 8.98132e-9)  # 升力系数
        self.k_T = dynamics_config.get('k_T', 0.07 * (3 * 0.062) * self.k_F)  # 扭矩系数
        
        # 电机位置（X型四旋翼布局）
        # 坐标系：forward x, left y, up z
        sqrt2_half = self.arm_length * np.sqrt(2) / 2.0
        self.motor_pos = torch.tensor([
            [ sqrt2_half, -sqrt2_half, 0.0],  # 电机0
            [-sqrt2_half,  sqrt2_half, 0.0],  # 电机1
            [ sqrt2_half,  sqrt2_half, 0.0],  # 电机2
            [-sqrt2_half, -sqrt2_half, 0.0]   # 电机3
        ], dtype=torch.float32, device=device)  # (4, 3)
        
        # 延迟初始化
        self.vehicle_params = None
        self.num_agents = 0
        self.mass = None  # (N,)
        self.inertia_mat = None  # (N, 3, 3)
        
        # Jerk控制参数（默认启用jerk控制）
        self.use_jerk_control = dynamics_config.get('use_jerk_control', True)  # 是否使用jerk控制（默认True）
        self.max_thrust_jerk = dynamics_config.get('max_thrust_jerk', 50.0)  # 最大推力jerk (N/s³)
        self.max_angular_jerk = dynamics_config.get('max_angular_jerk', 20.0)  # 最大角加速度jerk (rad/s³)
        self.max_thrust_accel = dynamics_config.get('max_thrust_accel', 20.0)  # 最大推力加速度 (N/s²)
        self.max_angular_accel = dynamics_config.get('max_angular_accel', 10.0)  # 最大角加速度 (rad/s²)
        
        # 当前控制状态（jerk控制模式）
        self.current_thrust_accel = None  # (N,) 当前推力加速度 (N/s²)
        self.current_angular_accel = None  # (N, 3) 当前角加速度 [roll, pitch, yaw] (rad/s²)
        self.current_motorRPM = None  # (N, 4) 当前电机RPM，用于状态跟踪
    
    def quaternion_to_rotation_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """
        将四元数转换为旋转矩阵（批量处理）
        Args:
            q: 四元数 [w, x, y, z]，形状为 (N, 4) 或 (4,)
        Returns:
            旋转矩阵，形状为 (N, 3, 3) 或 (3, 3)
        """
        if q.ndim == 1:
            q = q.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # 归一化
        norm = torch.sqrt(w**2 + x**2 + y**2 + z**2)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        
        # 构建旋转矩阵
        R = torch.stack([
            torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)], dim=1),
            torch.stack([2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)], dim=1),
            torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)], dim=1)
        ], dim=1)
        
        if squeeze_output:
            R = R.squeeze(0)
        return R
    
    def step(self, states: torch.Tensor, actions: torch.Tensor, dt: float) -> torch.Tensor:
        """
        对一批四旋翼状态进行一步更新（使用jerk控制）
        Args:
            states: 当前状态，形状为 (N, 13) [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
            actions: jerk控制动作，形状为 (N, 4) [thrust_jerk, roll_jerk, pitch_jerk, yaw_jerk]
                - thrust_jerk: 推力jerk (N/s³)，控制垂直方向加速度的变化率
                - roll_jerk: 滚转角加速度jerk (rad/s³)
                - pitch_jerk: 俯仰角加速度jerk (rad/s³)
                - yaw_jerk: 偏航角加速度jerk (rad/s³)
                注意：如果 use_jerk_control=False，则actions为电机RPM [rpm0, rpm1, rpm2, rpm3]
            dt: 时间步长 (s)
        Returns:
            下一时刻状态，形状为 (N, 13)
        """
        assert states.ndim == 2 and states.shape[1] == 13, f"States shape must be (N, 13), but got {states.shape}"
        assert actions.ndim == 2 and actions.shape[1] == 4, f"Actions shape must be (N, 4), but got {actions.shape}"
        
        batch_size = states.shape[0]
        assert batch_size == self.num_agents, f"batch_size ({batch_size}) must match num_agents ({self.num_agents})"
        
        # 根据控制模式处理动作
        if self.use_jerk_control:
            # Jerk控制模式：通过jerk积分得到加速度，再转换为RPM
            motorRPM = self._jerk_control_to_rpm(states, actions, dt)
        else:
            # 直接RPM控制模式（向后兼容）
            motorRPM = torch.clamp(actions, self.min_rpm, self.max_rpm)  # (N, 4)
        
        # 解包状态
        pos = states[:, :3]  # (N, 3)
        q = states[:, 3:7]  # (N, 4) [w, x, y, z]
        vel = states[:, 7:10]  # (N, 3)
        w_body = states[:, 10:13]  # (N, 3)
        
        # 归一化四元数
        q = q / (torch.norm(q, dim=1, keepdim=True) + 1e-9)
        
        # 计算旋转矩阵（批量）
        R_body2world = self.quaternion_to_rotation_matrix(q)  # (N, 3, 3)
        
        # 计算每个电机的升力
        F_motor = self.k_F * motorRPM ** 2  # (N, 4)
        
        # 体坐标系z轴方向
        z_body = torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0).expand(batch_size, -1)  # (N, 3)
        
        # 计算重力在体坐标系中的表示
        g_world = torch.tensor([0.0, 0.0, -self.g], device=self.device).unsqueeze(0).expand(batch_size, -1)  # (N, 3)
        R_body2world_T = R_body2world.transpose(1, 2)  # (N, 3, 3)
        g_body = self.mass.unsqueeze(1) * torch.bmm(R_body2world_T, g_world.unsqueeze(2)).squeeze(2)  # (N, 3)
        
        # 计算体坐标系中的总力
        total_thrust = torch.sum(F_motor, dim=1, keepdim=True)  # (N, 1)
        Force_body = total_thrust * z_body + g_body  # (N, 3)
        
        # 转换到世界坐标系
        Force_world = torch.bmm(R_body2world, Force_body.unsqueeze(2)).squeeze(2)  # (N, 3)
        
        # 计算力矩
        # 1. 由电机升力产生的力矩（力臂 × 力）
        torque_temp = torch.zeros(batch_size, 3, device=self.device)  # (N, 3)
        motor_pos_expanded = self.motor_pos.unsqueeze(0).expand(batch_size, -1, -1)  # (N, 4, 3)
        for i in range(4):
            motor_force = F_motor[:, i:i+1].unsqueeze(2) * z_body.unsqueeze(1)  # (N, 1, 3)
            torque_temp += torch.cross(motor_pos_expanded[:, i], motor_force.squeeze(1), dim=1)  # (N, 3)
        
        # 2. 由电机反扭矩产生的力矩
        # 电机0和1顺时针（负），电机2和3逆时针（正）
        motor_torque = (
            -self.k_T * motorRPM[:, 0:1]**2 * z_body
            -self.k_T * motorRPM[:, 1:2]**2 * z_body
            +self.k_T * motorRPM[:, 2:3]**2 * z_body
            +self.k_T * motorRPM[:, 3:4]**2 * z_body
        )  # (N, 3)
        
        Torque_body = torque_temp + motor_torque  # (N, 3)
        
        # 使用基础动力学更新状态
        # 计算加速度并更新位置和速度
        acceleration = Force_world / self.mass.unsqueeze(1)  # (N, 3)
        new_pos = pos + vel * dt
        new_vel = vel + acceleration * dt
        
        # 欧拉方程：更新角速度
        I_w = torch.bmm(self.inertia_mat, w_body.unsqueeze(2)).squeeze(2)  # (N, 3)
        w_cross_Iw = torch.cross(w_body, I_w, dim=1)  # (N, 3)
        I_inv = torch.inverse(self.inertia_mat)  # (N, 3, 3)
        w_dot = torch.bmm(I_inv, (Torque_body - w_cross_Iw).unsqueeze(2)).squeeze(2)  # (N, 3)
        new_w_body = w_body + dt * w_dot
        
        # 从角速度更新四元数
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        q_transmat = torch.stack([
            torch.stack([w, -x, -y, -z], dim=1),
            torch.stack([x,  w, -z,  y], dim=1),
            torch.stack([y,  z,  w, -x], dim=1),
            torch.stack([z, -y,  x,  w], dim=1)
        ], dim=1)  # (N, 4, 4)
        
        w_body_withzero = torch.cat([
            torch.zeros(batch_size, 1, device=self.device),
            new_w_body
        ], dim=1)  # (N, 4)
        
        d_q = 0.5 * torch.bmm(q_transmat, w_body_withzero.unsqueeze(2)).squeeze(2)  # (N, 4)
        new_q = q + d_q * dt
        
        # 归一化四元数
        new_q = new_q / (torch.norm(new_q, dim=1, keepdim=True) + 1e-9)
        
        # 组合新状态
        new_states = torch.cat([new_pos, new_q, new_vel, new_w_body], dim=1)  # (N, 13)
        return new_states
    
    def reset(self, vehicle_params: Dict[str, torch.Tensor]):
        """
        重置四旋翼参数
        Args:
            vehicle_params: 批量车辆参数，包含：
                - 'mass': (N,) 质量
                - 'inertia': (N, 3, 3) 惯性矩阵，或 (N, 3) 对角惯性矩阵
        """
        if vehicle_params is None:
            raise ValueError("reset(vehicle_params) 需要提供车辆参数")
        
        self.mass = vehicle_params['mass'].to(self.device)  # (N,)
        
        inertia = vehicle_params['inertia'].to(self.device)
        if inertia.ndim == 2 and inertia.shape[1] == 3:
            # 对角惯性矩阵，转换为 (N, 3, 3)
            self.inertia_mat = torch.diag_embed(inertia)  # (N, 3, 3)
        elif inertia.ndim == 3 and inertia.shape[1] == 3 and inertia.shape[2] == 3:
            # 完整惯性矩阵
            self.inertia_mat = inertia
        else:
            raise ValueError(f"Invalid inertia shape: {inertia.shape}, expected (N, 3) or (N, 3, 3)")
        
        self.num_agents = self.mass.shape[0]
        self.vehicle_params = vehicle_params
        
        # 重置jerk控制状态
        if self.use_jerk_control:
            self.current_thrust_accel = torch.zeros(self.num_agents, device=self.device)
            self.current_angular_accel = torch.zeros(self.num_agents, 3, device=self.device)
            self.current_motorRPM = torch.zeros(self.num_agents, 4, device=self.device)
    
    def _jerk_control_to_rpm(self, states: torch.Tensor, jerk_actions: torch.Tensor, dt: float) -> torch.Tensor:
        """
        将jerk控制动作转换为电机RPM
        
        Jerk控制流程：
        1. 接收jerk动作 [thrust_jerk, roll_jerk, pitch_jerk, yaw_jerk]
        2. 积分得到加速度：a_new = a_old + jerk * dt
        3. 限制加速度范围
        4. 将期望推力和力矩转换为电机RPM（控制分配）
        
        Args:
            states: 当前状态 (N, 13)
            jerk_actions: jerk控制动作 (N, 4) [thrust_jerk, roll_jerk, pitch_jerk, yaw_jerk]
                - thrust_jerk: 推力jerk (N/s³)
                - roll_jerk: 滚转角加速度jerk (rad/s³)
                - pitch_jerk: 俯仰角加速度jerk (rad/s³)
                - yaw_jerk: 偏航角加速度jerk (rad/s³)
            dt: 时间步长 (s)
        Returns:
            电机RPM (N, 4) [rpm0, rpm1, rpm2, rpm3]
        """
        batch_size = states.shape[0]
        
        # 初始化控制状态（如果尚未初始化或batch_size变化）
        if (self.current_thrust_accel is None or 
            self.current_thrust_accel.shape[0] != batch_size):
            self.current_thrust_accel = torch.zeros(batch_size, device=self.device)
            self.current_angular_accel = torch.zeros(batch_size, 3, device=self.device)
            self.current_motorRPM = torch.zeros(batch_size, 4, device=self.device)
        
        # ========== 步骤1: 解包jerk动作 ==========
        thrust_jerk = jerk_actions[:, 0]  # (N,) 推力jerk (N/s³)
        roll_jerk = jerk_actions[:, 1]    # (N,) 滚转角加速度jerk (rad/s³)
        pitch_jerk = jerk_actions[:, 2]   # (N,) 俯仰角加速度jerk (rad/s³)
        yaw_jerk = jerk_actions[:, 3]     # (N,) 偏航角加速度jerk (rad/s³)
        
        # ========== 步骤2: 限制jerk范围 ==========
        thrust_jerk = torch.clamp(thrust_jerk, -self.max_thrust_jerk, self.max_thrust_jerk)
        angular_jerk = torch.stack([roll_jerk, pitch_jerk, yaw_jerk], dim=1)  # (N, 3)
        angular_jerk = torch.clamp(angular_jerk, -self.max_angular_jerk, self.max_angular_jerk)
        
        # ========== 步骤3: 积分jerk得到加速度 ==========
        # a_new = a_old + jerk * dt
        new_thrust_accel = self.current_thrust_accel + thrust_jerk * dt  # (N,) 推力加速度 (N/s²)
        new_angular_accel = self.current_angular_accel + angular_jerk * dt  # (N, 3) 角加速度 (rad/s²)
        
        # ========== 步骤4: 限制加速度范围 ==========
        new_thrust_accel = torch.clamp(new_thrust_accel, -self.max_thrust_accel, self.max_thrust_accel)
        new_angular_accel = torch.clamp(new_angular_accel, -self.max_angular_accel, self.max_angular_accel)
        
        # 更新控制状态
        self.current_thrust_accel = new_thrust_accel
        self.current_angular_accel = new_angular_accel
        
        # ========== 步骤5: 将加速度转换为期望的力和力矩 ==========
        # 期望推力 = 悬停推力 + 推力加速度 * dt
        hover_thrust = self.mass * self.g  # (N,) 悬停所需推力
        desired_thrust = hover_thrust + new_thrust_accel * dt  # (N,) 期望推力
        max_thrust = self.mass * self.g * 2.0  # (N,) 最大推力（2倍重力）
        desired_thrust = torch.clamp(desired_thrust, min=torch.zeros_like(desired_thrust), max=max_thrust)  # 限制在0到2倍重力之间
        
        # 期望角加速度转换为期望力矩: Tau = I * alpha_desired
        desired_torque = torch.bmm(self.inertia_mat, new_angular_accel.unsqueeze(2)).squeeze(2)  # (N, 3)
        
        # ========== 步骤6: 控制分配 - 将期望推力和力矩转换为RPM ==========
        # 对于X型四旋翼布局，控制分配关系：
        # T = k_F * (rpm0² + rpm1² + rpm2² + rpm3²)  # 总推力
        # Mx = k_F * l * (rpm2² - rpm0²) / sqrt(2)   # 滚转力矩
        # My = k_F * l * (rpm3² - rpm1²) / sqrt(2)   # 俯仰力矩
        # Mz = k_T * (rpm2² + rpm3² - rpm0² - rpm1²) # 偏航力矩
        
        # 计算基础悬停RPM²
        hover_rpm_sq = hover_thrust / (4.0 * self.k_F)  # (N,) 悬停时每个电机的RPM²
        
        # 计算推力增量对应的RPM²增量
        delta_T = desired_thrust - hover_thrust  # (N,) 推力增量
        delta_rpm_sq = delta_T / (4.0 * self.k_F)  # (N,) 平均分配到4个电机
        
        # 计算力矩对应的RPM²增量
        l = self.arm_length  # 臂长
        sqrt2 = np.sqrt(2.0)
        delta_Mx = desired_torque[:, 0]  # (N,) 滚转力矩增量
        delta_My = desired_torque[:, 1]  # (N,) 俯仰力矩增量
        delta_Mz = desired_torque[:, 2]  # (N,) 偏航力矩增量
        
        # 控制分配：从悬停RPM²开始，加上推力和力矩增量
        rpm_sq_base = hover_rpm_sq.unsqueeze(1).expand(-1, 4)  # (N, 4) 基础RPM²
        
        # 推力分配：平均分配到4个电机
        rpm_sq = rpm_sq_base + delta_rpm_sq.unsqueeze(1) / 4.0  # (N, 4)
        
        # 力矩分配（X型布局）：
        # 电机0: 前右，顺时针，负滚转和负偏航
        rpm_sq[:, 0] += (-delta_Mx / (self.k_F * l * sqrt2) - delta_Mz / (4.0 * self.k_T)) / 2.0
        # 电机1: 后左，顺时针，负俯仰和负偏航
        rpm_sq[:, 1] += (-delta_My / (self.k_F * l * sqrt2) - delta_Mz / (4.0 * self.k_T)) / 2.0
        # 电机2: 后右，逆时针，正滚转和正偏航
        rpm_sq[:, 2] += (delta_Mx / (self.k_F * l * sqrt2) + delta_Mz / (4.0 * self.k_T)) / 2.0
        # 电机3: 前左，逆时针，正俯仰和正偏航
        rpm_sq[:, 3] += (delta_My / (self.k_F * l * sqrt2) + delta_Mz / (4.0 * self.k_T)) / 2.0
        
        # ========== 步骤7: 转换为RPM并限制范围 ==========
        rpm_sq = torch.clamp(rpm_sq, 0.0, self.max_rpm ** 2)
        motorRPM = torch.sqrt(rpm_sq)  # (N, 4)
        motorRPM = torch.clamp(motorRPM, self.min_rpm, self.max_rpm)
        
        # 更新当前RPM状态（用于跟踪）
        self.current_motorRPM = motorRPM
        
        return motorRPM

class DiscreteActionSpace:
    """
    离散动作空间定义，支持三维jerk控制
    对于地面车辆（自行车模型），z轴jerk始终为0
    """
    def __init__(self, config: Dict, device: torch.device):
        """
        初始化离散动作空间
        Args:
            config: 配置字典，包含jerk范围参数
            device: 计算设备
        """
        if config is None:
            config = {}
        # 定义离散动作：三维jerk [jerk_x, jerk_y, jerk_z]
        # jerk_x (纵向): [-15, -4, 0, 4] m/s³
        # jerk_y (横向): [-4, 0, 4] m/s³
        # jerk_z (垂直): 始终为0（地面车辆）
        min_long_jerk = config.get('min_longitudinal_jerk', -15.0)
        max_long_jerk = config.get('max_longitudinal_jerk', 4.0)
        min_lat_jerk = config.get('min_lateral_jerk', -4.0)
        max_lat_jerk = config.get('max_lateral_jerk', 4.0)
        
        self.jerk_x_values = [min_long_jerk, -max_long_jerk, 0, max_long_jerk]
        self.jerk_y_values = [min_lat_jerk, 0, max_lat_jerk]
        
        # 创建所有可能的动作组合（三维jerk，z始终为0）
        self.actions = []
        for jerk_x in self.jerk_x_values:
            for jerk_y in self.jerk_y_values:
                self.actions.append([jerk_x, jerk_y, 0.0])  # [jerk_x, jerk_y, jerk_z=0]
        self.num_actions = len(self.actions)  # 12个动作
        self.actions_tensor = torch.tensor(self.actions, dtype=torch.float32, device=device)

    def get_action(self, action_idx: torch.Tensor) -> torch.Tensor:
        """
        根据动作索引获取实际动作值（三维jerk）
        Args:
            action_idx: 动作索引张量，形状为 (N,) 或标量
            
        Returns:
            torch.Tensor: 实际动作值，形状为 (N, 3) 或 (3,)
                格式: [jerk_x, jerk_y, jerk_z=0]
        """
        if action_idx.ndim == 0:  # 标量
            return self.actions_tensor[action_idx]
        else:  # 批量
            return self.actions_tensor[action_idx]
        
    def get_all_actions(self) -> torch.Tensor:
        """获取所有动作（三维jerk）"""
        return self.actions_tensor

class KinematicBicycleModel:
    """
    实现了一个精确且批量化的运动学自行车模型（三维jerk控制）。
    该模型使用运动学方程的解析解，以确保在离散时间步长内的物理准确性，
    特别是在转弯场景中。所有操作都已完全向量化，以实现最高性能。
    使用三维jerk控制：
    - 动作格式: [jerk_x, jerk_y, jerk_z] (3维)
    - jerk_x: 纵向jerk (x方向，前进/后退)
    - jerk_y: 横向jerk (y方向，左右)
    - jerk_z: 垂直jerk (z方向，对于地面车辆始终为0)
    
    支持离散动作空间，通过jerk控制实现平滑的加速度变化。
    """
    def __init__(self, config: Dict, device: torch.device):
        """
        初始化动力学模型。
        Args:
            config (Dict): 包含车辆物理参数的配置字典。
                支持从simulator.dynamics子配置中读取参数，也支持直接从根级别读取（向后兼容）。
            device (torch.device): 计算设备。
            vehicle_params (Dict[str, torch.Tensor]): 批量车辆参数，必需参数，包含：
                - 'length': (N,) 车辆长度
                - 'width': (N,) 车辆宽度
                - 'wheelbase': (N,) 车辆轴距
                批量参数应与world_init采样顺序一致。
        """
        self.device = device
        # 获取dynamics配置，支持嵌套配置结构
        # 首先尝试从 simulator.dynamics 获取，如果没有则从根级别的 dynamics 获取，最后回退到整个config
        if 'simulator' in config and isinstance(config.get('simulator'), dict):
            simulator_config = config['simulator']
            dynamics_config = simulator_config.get('dynamics', {})
            if not dynamics_config:  # 如果simulator.dynamics不存在，尝试直接使用simulator配置
                dynamics_config = simulator_config
        else:
            dynamics_config = config.get('dynamics', config)
        
        max_steer_deg = dynamics_config.get('vehicle_max_steer_angle')
        if max_steer_deg is None:
            max_steer_deg = 35.0  # 默认值
        self.max_steer_rad = math.radians(max_steer_deg) # 将角度转换为弧度
        
        # 延迟初始化：不在 __init__ 中注入车辆参数，等外部 reset(vehicle_params) 再设置
        self.vehicle_params = None
        self.num_vehicles = 0
        self.Cthrottle = None
        self.Csteer = None
        self.Cacc = None
        self.Cvel = None
        
        # 转向角限制参数
        self.max_steering_rate = dynamics_config.get('max_steering_rate', 0.6)  # 最大转向角变化率 (rad/s)
        
        # 加速度约束参数
        self.max_longitudinal_accel = dynamics_config.get('max_longitudinal_accel', 2.5)  # 最大纵向加速度 (m/s²)
        self.min_longitudinal_accel = dynamics_config.get('min_longitudinal_accel', -5.0)  # 最小纵向加速度 (m/s²)
        self.max_lateral_accel = dynamics_config.get('max_lateral_accel', 4.0)       # 最大横向加速度 (m/s²)
        self.min_lateral_accel = dynamics_config.get('min_lateral_accel', -4.0)      # 最小横向加速度 (m/s²)
        
        # 速度约束参数
        self.max_velocity = dynamics_config.get('max_velocity', 20.0)  # 最大速度 (m/s)
        self.min_velocity = dynamics_config.get('min_velocity', -2.0)  # 最小速度 (m/s)
        
        # 数值稳定性参数
        self.curvature_epsilon = float(dynamics_config.get('curvature_epsilon', 1e-5))      # 曲率计算的数值稳定性参数
        #self.steering_epsilon = float(dynamics_config.get('steering_epsilon', 1e-5))       # 转向角计算的数值稳定性参数
        #self.straight_motion_threshold = float(dynamics_config.get('straight_motion_threshold', 1e-5))  # 直线运动判断阈值 (rad)

        # 使用离散动作空间
        self.discrete_action_space = DiscreteActionSpace(dynamics_config, device)
        
        # 当前加速度状态（用于jerk控制）
        # 这些将在step方法中根据批量大小动态调整
        self.current_along = None
        self.current_alat = None
        self.current_steering_angle = None  # 当前有效转向角

    def step(self, states: torch.Tensor, actions: torch.Tensor, dt: float) -> torch.Tensor:
        """
        对一批车辆状态进行一步精确更新（使用三维jerk控制）。
        Args:
            states (torch.Tensor): 形状为 (N, 4) 的当前状态张量 [x, y, yaw, speed]。
            actions (torch.Tensor): 动作索引张量，形状为 (N,)，对应离散动作空间中的索引。
                离散动作空间返回三维jerk: [jerk_x, jerk_y, jerk_z=0]
                - jerk_x: 纵向jerk (x方向)
                - jerk_y: 横向jerk (y方向)
                - jerk_z: 垂直jerk (始终为0，地面车辆)
            dt (float): 模拟时间步长 (s)。
        Returns:
            torch.Tensor: 形状为 (N, 4) 的下一时刻状态张量。
        """
        # 检查输入维度
        assert states.ndim == 2 and states.shape[1] == 4, f"States shape must be (N, 4), but got {states.shape}"
        # 获取批次大小
        batch_size = states.shape[0]
        # 离散动作空间：actions是动作索引
        assert actions.ndim == 1, f"Discrete actions shape must be (N,), but got {actions.shape}"
        # 使用批量wheelbase参数，要求batch_size与车辆参数数量匹配
        assert batch_size == self.num_vehicles, f"batch_size ({batch_size}) must match num_vehicles ({self.num_vehicles})"
        wheelbases = self.vehicle_params['wheelbase']  # (batch_size,)
        
        # 检查并初始化控制状态（确保batch_size正确）
        if (self.current_along is None or self.current_along.shape[0] != batch_size or
            self.current_alat is None or self.current_alat.shape[0] != batch_size):
            print(f"Initializing dynamics state for batch_size: {batch_size}")
            self.current_along = torch.zeros(batch_size, device=self.device)
            self.current_alat = torch.zeros(batch_size, device=self.device)
            # 同时重置prev_along以确保一致性
            if hasattr(self, 'prev_along'):
                self.prev_along = torch.zeros(batch_size, device=self.device)
        
        # 获取实际的三维jerk动作
        jerk_actions = self.discrete_action_space.get_action(actions)  # (N, 3) [jerk_x, jerk_y, jerk_z]
        # 更新当前加速度和转向角（jerk控制）
        # 对于地面车辆，只使用x和y方向的jerk，z方向jerk始终为0
        along_jerk = jerk_actions[:, 0]  # 纵向jerk (x方向)
        alat_jerk = jerk_actions[:, 1]   # 横向jerk (y方向)
        # jerk_actions[:, 2] 始终为0，忽略
        
        # 更新加速度和转向角（应用控制系数）
        new_along = self.current_along + along_jerk * dt * self.Cthrottle
        new_alat = self.current_alat + alat_jerk * dt * self.Csteer
        
        # 检测纵向加速度符号变化：a(t-1)_long * a(t)_long < 0
        accel_sign_change = (self.current_along * new_along) < 0
        
        # 如果纵向加速度改变符号，将横向加速度设置为0,纵向加速度设置为0
        if torch.any(accel_sign_change):
            new_alat = torch.where(accel_sign_change, torch.zeros_like(new_alat), new_alat)
            new_along = torch.where(accel_sign_change, torch.zeros_like(new_along), new_along)
            # 如果纵向加速度改变符号，将纵向加速度设置为0，并保持当前的横向加速度
            # 会使得agent更容易停在原地，或者驾驶时速度变化更平缓
        
        # 应用约束：a(t)_long ← clip(a(t)_long, min_long_accel, max_long_accel*Cacc), a(t)_lat ← clip(a(t)_lat, min_lat_accel, max_lat_accel)
        max_along = torch.tensor(self.max_longitudinal_accel, device=self.device) * self.Cacc
        min_along = torch.tensor(self.min_longitudinal_accel, device=self.device)
        along = torch.clamp(new_along, min_along, max_along)
        min_alat = torch.tensor(self.min_lateral_accel, device=self.device)
        max_alat = torch.tensor(self.max_lateral_accel, device=self.device)
        alat = torch.clamp(new_alat, min_alat, max_alat) # 横向加速度约束（从jerk计算的，用于转向角计算）
        
        # 更新当前纵向加速度状态（横向加速度稍后会根据转向角重新计算）
        self.current_along = along
            
        # 从状态张量中解包
        x, y, yaw, speed = states.T
        # 使用梯形法则更新速度：v(t) = v(t-1) + 0.5 * (a(t)_long + a(t-1)_long) * dt
        # 需要保存前一步的纵向加速度用于梯形积分
        if not hasattr(self, 'prev_along'):
            self.prev_along = torch.zeros_like(self.current_along)
        
        # 梯形法则：v(t) = v(t-1) + 0.5 * (a(t)_long + a(t-1)_long) * dt
        new_speed = speed + 0.5 * (along + self.prev_along) * dt
        
        # 检测速度符号变化：v(t-1) * v(t) < 0
        speed_sign_change = (speed * new_speed) < 0
        
        # 如果速度改变符号，将速度设置为0
        if torch.any(speed_sign_change):
            new_speed = torch.where(speed_sign_change, torch.zeros_like(new_speed), new_speed)
        
        # 应用速度约束：v(t) ← clip(v(t), min_velocity, max_velocity*Cvel)
        max_vel = torch.tensor(self.max_velocity, device=self.device) * self.Cvel
        min_vel = torch.tensor(self.min_velocity, device=self.device)
        new_speed = torch.clamp(new_speed, min_vel, max_vel)
        
        # 更新前一步的纵向加速度
        self.prev_along = along.clone()

        # 根据横向加速度计算目标转向角（使用当前速度 v^(t)，即 new_speed）
        # 原文：从横向加速度反推转向角，使用当前时刻的速度
        target_steering_angle = self.calculate_steering_angle(alat, new_speed, wheelbases)
        
        # 初始化当前转向角（如果还没有初始化）
        if self.current_steering_angle is None or self.current_steering_angle.shape[0] != batch_size:
            self.current_steering_angle = torch.zeros(batch_size, device=self.device)
        
        # 计算转向角变化：δφ = φ_target - φ(t-1)
        steering_change = target_steering_angle - self.current_steering_angle
        
        # 限制转向角变化率：δφ = clip(δφ, -δmax*dt, δmax*dt)
        max_change = self.max_steering_rate * dt
        limited_steering_change = torch.clamp(steering_change, -max_change, max_change)
        
        # 更新有效转向角：φ(t) = clip(φ(t-1) + δφ, -φmax, φmax)
        new_steering_angle = self.current_steering_angle + limited_steering_change
        steering_angle = torch.clamp(new_steering_angle, -self.max_steer_rad, self.max_steer_rad)
        
        # 更新当前转向角状态
        self.current_steering_angle = steering_angle
        
        # 根据有效转向角更新曲率和横向加速度（物理一致性修正）
        # 原文：ρ^(-1) ← tan(φ^(t)) / l_wb
        effective_curvature = torch.tan(steering_angle) / wheelbases
        # 原文：a_lat(t) ← (v^(t))^2 * ρ^(-1)，使用当前时刻的速度 v^(t)，即 new_speed
        # 注意：这里根据实际转向角重新计算横向加速度，确保物理一致性
        # （之前从jerk计算的alat只是用于反推目标转向角）
        effective_alat = new_speed ** 2 * effective_curvature
        
        # 在计算 effective_alat 后，再次应用约束
        min_alat = torch.tensor(self.min_lateral_accel, device=self.device)
        max_alat = torch.tensor(self.max_lateral_accel, device=self.device)
        effective_alat = torch.clamp(effective_alat, min_alat, max_alat)
        # 更新横向加速度状态（用于下一次step的jerk计算）
        self.current_alat = effective_alat
        
        # 使用时间步内的平均速度进行位移计算，提高精度
        avg_speed = (speed + new_speed) / 2.0

        # 使用自行车动力学模型更新车辆位置
        # 计算位移：d = 0.5(v(t) + v(t-1)) * Δt
        displacement = avg_speed * dt
        
        # 计算角位移：θ = d * ρ^(-1)
        angular_displacement = displacement * effective_curvature
        
        # 原文：Δx = ρ sin(θ), Δy = ρ cos(θ)
        # 其中 ρ = 1 / ρ^(-1)，θ = d * ρ^(-1)
        # 需要处理曲率为0的情况（直线运动）
        # curvature_threshold = self.curvature_epsilon
        # is_straight = torch.abs(effective_curvature) < curvature_threshold
        
        # 计算半径：ρ = 1 / ρ^(-1)
        # radius = torch.where(is_straight,
        #                     torch.ones_like(effective_curvature),  # 直线时占位，实际不会使用
        #                     1.0 / effective_curvature)
        
        # 位置变化：Δx = ρ sin(θ), Δy = ρ cos(θ)
        # dx_curved = radius * torch.sin(angular_displacement)
        # dy_curved = radius * torch.cos(angular_displacement)
             
        # 根据是否直线选择相应的计算方式
        dx = displacement * torch.cos(yaw)
        dy = displacement * torch.sin(yaw)
        
        # 偏航角变化
        d_yaw = angular_displacement 
        
        # --- 计算新状态 ---
        new_x = x + dx
        new_y = y + dy
        new_yaw = yaw + d_yaw

        # 归一化偏航角到 [-pi, pi]
        new_yaw = torch.atan2(torch.sin(new_yaw), torch.cos(new_yaw))
        
        # 将新状态组合成一个张量返回
        new_states = torch.stack([new_x, new_y, new_yaw, new_speed], dim=1)
        return new_states
    
    def reset(self, vehicle_params: Dict[str, torch.Tensor]):
        """
        回合重置：
        - 接收新一批车辆参数（长度、宽度、轴距），更新批大小
        - 重新采样驾驶风格参数（Cthrottle/Csteer/Cacc/Cvel）
        - 清空内部控制状态（沿用 reset_control_state）
        """
        # 1) 更新车辆参数与批大小
        if vehicle_params is None:
            raise ValueError("reset(vehicle_params) 需要提供新一批车辆参数")
        self.vehicle_params = {
            'length': vehicle_params['length'].to(self.device),
            'width': vehicle_params['width'].to(self.device),
            'wheelbase': vehicle_params['wheelbase'].to(self.device)
        }
        self.num_vehicles = self.vehicle_params['wheelbase'].shape[0]

        # 2) 重新采样驾驶风格参数（批量）
        Cthrottle, Csteer, Cacc, Cvel = self._sample_driving_style(size=self.num_vehicles)
        self.Cthrottle = Cthrottle
        self.Csteer = Csteer
        self.Cacc = Cacc
        self.Cvel = Cvel

        # 3) 清空控制积分状态
        """重置控制状态（加速度和转向角）"""
        # 清除所有控制状态变量，让step方法在需要时重新初始化正确的batch_size
        self.current_along = None
        self.current_alat = None
        self.current_steering_angle = None
        # 清除前一步的纵向加速度，让step方法重新创建
        if hasattr(self, 'prev_along'):
            delattr(self, 'prev_along')
        print("Dynamics control state reset - variables cleared for fresh initialization")

    def calculate_steering_angle(self, alat: torch.Tensor, speed: torch.Tensor, wheelbases: torch.Tensor, epsilon: float = None) -> torch.Tensor:
        """
        根据横向加速度和速度计算转向角
        Args:
            alat (torch.Tensor): 横向加速度
            speed (torch.Tensor): 速度
            wheelbases (torch.Tensor): 批量轴距参数，形状为 (N,)，必需参数
            epsilon (float): 数值稳定性参数，如果为None则使用配置中的默认值
        Returns:
            torch.Tensor: 转向角 (弧度)
        """
        # 使用配置中的默认值或传入的参数
        if epsilon is None:
            epsilon = self.curvature_epsilon
        
        # 使用传入的wheelbase值
        L_used = wheelbases
            
        # 计算曲率：ρ^(-1) = alat / max(v^2, ε)
        speed_squared = torch.clamp(speed ** 2, min=epsilon)
        curvature = alat / speed_squared
        
        # 应用数值稳定性：ρ^(-1) ← sign(ρ^(-1)) * max(|ρ^(-1)|, ε)
        curvature_sign = torch.sign(curvature)
        curvature_magnitude = torch.clamp(torch.abs(curvature), min=epsilon)
        curvature = curvature_sign * curvature_magnitude

        # 计算转向角：φ = arctan(ρ^(-1) * lwb)
        steering_angle = torch.atan(curvature * L_used)
        return steering_angle
    
    def get_discrete_action_space(self) -> DiscreteActionSpace:
        """获取离散动作空间"""
        return self.discrete_action_space

    def _sample_driving_style(self, size: int):
        """
        采样驾驶风格参数（内部方法）
        Args:
            size: 批量大小
        Returns:
            tuple: (Cthrottle, Csteer, Cacc, Cvel)，每个都是形状为 (size,) 的tensor
                - Cthrottle: 油门控制系数 [0.8, 1.2]
                - Csteer: 转向控制系数 [0.8, 1.2]
                - Cacc: 加速度限制系数 [0.8, 1.0]
                - Cvel: 速度限制系数 [0.8, 1.0]
        """
        # 从均匀分布采样驾驶风格参数
        Cthrottle = torch.rand(size, device=self.device) * 0.4 + 0.8  # [0.8, 1.2]
        Csteer = torch.rand(size, device=self.device) * 0.4 + 0.8     # [0.8, 1.2]
        Cacc = torch.rand(size, device=self.device) * 0.2 + 0.8        # [0.8, 1.0]
        Cvel = torch.rand(size, device=self.device) * 0.2 + 0.8         # [0.8, 1.0]
        return Cthrottle, Csteer, Cacc, Cvel

# 示例用法
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 60)
    # ========== 测试1: Dynamics类（基础动力学，6维jerk控制）==========
    print("\n[测试1] Dynamics类 - 6维jerk控制")
    print("-" * 60)
    config_dynamics = {
        'simulator': {
            'dynamics': {
                'max_force_jerk': 100.0,
                'max_torque_jerk': 50.0,
                'max_force_accel': 50.0,
                'max_torque_accel': 25.0
            }
        }
    }
    dynamics = Dynamics(config_dynamics, device)
    batch_size = 2
    vehicle_params = {
        'mass': torch.tensor([1.0, 1.5], device=device),  # (2,)
        'inertia': torch.tensor([
            [0.01, 0.01, 0.02],  # 刚体1
            [0.015, 0.015, 0.03]  # 刚体2
        ], device=device)  # (2, 3) 对角惯性矩阵
    }
    dynamics.reset(vehicle_params)
    
    # 初始状态：位置在原点，姿态为水平
    initial_states = torch.zeros(batch_size, 13, device=device)
    initial_states[:, 3] = 1.0  # qw = 1.0 (单位四元数)
    
    # Jerk控制动作: [force_jerk_x, force_jerk_y, force_jerk_z, torque_jerk_x, torque_jerk_y, torque_jerk_z]
    actions_dynamics = torch.tensor([
        [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 刚体1: x方向力jerk
        [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]   # 刚体2: y方向力jerk
    ], device=device)
    
    dt = 0.01
    states = initial_states
    for i in range(10):
        states = dynamics.step(states, actions_dynamics, dt)
        if i % 1 == 0:
            pos = states[:, :3]
            vel = states[:, 7:10]
            print(f"Step {i}:")
            print(f"  Positions:\n{pos}")
            print(f"  Velocities:\n{vel}")
    
    # ========== 测试2: QuadrotorDynamics类（四旋翼，4维jerk控制）==========
    print("\n[测试2] QuadrotorDynamics类 - 4维jerk控制")
    print("-" * 60)
    config_quad = {
        'simulator': {
            'dynamics': {
                'arm_length': 0.22,
                'min_rpm': 0.0,
                'max_rpm': 35000.0,
                'gravity': 9.81,
                'k_F': 3 * 8.98132e-9,
                'k_T': 0.07 * (3 * 0.062) * (3 * 8.98132e-9),
                'max_thrust_jerk': 50.0,
                'max_angular_jerk': 20.0,
                'max_thrust_accel': 20.0,
                'max_angular_accel': 10.0
            }
        }
    }
    
    quad = QuadrotorDynamics(config_quad, device)
    batch_size = 3
    vehicle_params = {
        'mass': torch.tensor([1.0, 1.2, 0.8], device=device),  # (3,)
        'inertia': torch.tensor([
            [0.01, 0.01, 0.02],  # 四旋翼1
            [0.012, 0.012, 0.024],  # 四旋翼2
            [0.008, 0.008, 0.016]   # 四旋翼3
        ], device=device)  # (3, 3) 对角惯性矩阵
    }
    quad.reset(vehicle_params)
    
    # 初始状态：位置在原点上方1米，姿态为水平
    initial_states = torch.zeros(batch_size, 13, device=device)
    initial_states[:, 2] = 1.0  # z = 1.0m
    initial_states[:, 3] = 1.0  # qw = 1.0 (单位四元数)
    
    # Jerk控制动作: [thrust_jerk, roll_jerk, pitch_jerk, yaw_jerk]
    # 悬停状态：thrust_jerk = 0（保持悬停推力）
    actions_quad = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],      # 四旋翼1: 悬停
        [5.0, 0.0, 0.0, 0.0],      # 四旋翼2: 增加推力
        [0.0, 2.0, 0.0, 0.0]       # 四旋翼3: 滚转jerk
    ], device=device)
    
    states = initial_states
    for i in range(10):
        states = quad.step(states, actions_quad, dt)
        if i % 5 == 0:
            pos = states[:, :3]
            vel = states[:, 7:10]
            print(f"Step {i}:")
            print(f"  Positions:\n{pos}")
            print(f"  Velocities:\n{vel}")
    
    # ========== 测试3: KinematicBicycleModel类（自行车模型，3维jerk控制）==========
    print("\n[测试3] KinematicBicycleModel类 - 3维jerk控制（离散动作空间）")
    print("-" * 60)
    config_bike = {
        'simulator': {
            'dynamics': {
                'vehicle_max_steer_angle': 35.0,
                'max_steering_rate': 0.6,
                'max_longitudinal_accel': 2.5,
                'min_longitudinal_accel': -5.0,
                'max_lateral_accel': 4.0,
                'min_lateral_accel': -4.0,
                'max_velocity': 20.0,
                'min_velocity': -2.0
            }
        }
    }
    
    bike = KinematicBicycleModel(config_bike, device)
    batch_size = 2
    vehicle_params = {
        'length': torch.tensor([4.5, 5.0], device=device),  # (2,)
        'width': torch.tensor([1.8, 2.0], device=device),    # (2,)
        'wheelbase': torch.tensor([2.7, 3.0], device=device) # (2,)
    }
    bike.reset(vehicle_params)
    
    # 初始状态: [x, y, yaw, speed]
    initial_states = torch.tensor([
        [0.0, 0.0, 0.0, 5.0],  # 车辆1: 在原点，朝x方向，速度5m/s
        [0.0, 0.0, 1.57, 3.0]  # 车辆2: 在原点，朝y方向，速度3m/s
    ], device=device)
    
    # 离散动作索引（对应3维jerk: [jerk_x, jerk_y, jerk_z=0]）
    # 动作空间有12个离散动作，这里使用动作索引
    actions_bike = torch.tensor([0, 5], device=device)  # 选择不同的离散动作
    
    states = initial_states
    for i in range(10):
        states = bike.step(states, actions_bike, dt)
        if i % 5 == 0:
            print(f"Step {i}:")
            print(f"  States:\n{states}")
    
    print("\n" + "=" * 60)
    print("所有测试完成！")

