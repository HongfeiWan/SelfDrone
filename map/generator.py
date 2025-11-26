#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于区块（Chunk）的地图生成系统 - 完全GPU加速
- 每个区块：16×16×256 体素
- 每个体素：0.5m×0.5m×0.5m
- 视野范围：21×21 区块（以agent为中心）
- 支持动态加载新区块
- 所有体素数据存储在GPU上作为tensor
- 暴露体素计算完全在GPU上完成
- 使用vispy的GPU高速渲染
"""
import numpy as np
import torch
import json
import os
import time
from typing import Dict, Tuple, Optional, List
from concurrent.futures import ThreadPoolExecutor
from vispy import app, scene, gloo
from vispy.util.transforms import perspective, translate, rotate
from vispy.geometry import MeshData

# 加载配置文件
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config/defalt_config.json')
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)
    CHUNK_CONFIG = CONFIG.get('chunk')
    HORIZON = CONFIG.get('horizon')

class Section:
    def __init__(self, section_y: int, chunk, device):
        self.section_y = section_y  # 从chunk底部起第几层section
        self.chunk = chunk          # 所属chunk，可用于反查
        self.voxels = torch.zeros(
            (chunk.CHUNK_SIZE_X, chunk.CHUNK_SECTION_HEIGHT, chunk.CHUNK_SIZE_Z),
            dtype=torch.uint8, device=device)

class Chunk:
    """
    区块类：存储 16×16×256 的体素数据，每个chunk用chunk_x、chunk_z索引
    """
    CHUNK_SIZE_X = CHUNK_CONFIG.get('CHUNK_SIZE_X')
    CHUNK_SIZE_Z = CHUNK_CONFIG.get('CHUNK_SIZE_Z')
    CHUNK_SECTION_HEIGHT = CHUNK_CONFIG.get('SECTION_HEIGHT')  # 每 section 高度
    CHUNK_SIZE_Y = CHUNK_CONFIG.get('CHUNK_SIZE_Y')             # 总高度
    VOXEL_SIZE = CHUNK_CONFIG.get('VOXEL_SIZE')

    def __init__(self, chunk_x: int, chunk_z: int, device: torch.device):
        self.chunk_x = chunk_x
        self.chunk_z = chunk_z
        self.device = device
        self.num_sections = int(np.ceil(self.CHUNK_SIZE_Y / self.CHUNK_SECTION_HEIGHT))
        # 直接分配全chunk四维大张量：[section, x, y_in_section, z]
        self.voxels_all = torch.zeros(
            (self.num_sections, self.CHUNK_SIZE_X, self.CHUNK_SECTION_HEIGHT, self.CHUNK_SIZE_Z),
            dtype=torch.uint8, device=device)
        self._generate_ground()

    @staticmethod
    def world_to_chunk_idx(x: float, z: float) -> Tuple[int, int]:
        idx_x = int(np.floor(x / (Chunk.CHUNK_SIZE_X * Chunk.VOXEL_SIZE)))
        idx_z = int(np.floor(z / (Chunk.CHUNK_SIZE_Z * Chunk.VOXEL_SIZE)))
        return idx_x, idx_z

    def get_world_bounds(self) -> Tuple[float, float, float, float]:
        min_x = self.chunk_x * self.CHUNK_SIZE_X * self.VOXEL_SIZE
        max_x = (self.chunk_x + 1) * self.CHUNK_SIZE_X * self.VOXEL_SIZE
        min_z = self.chunk_z * self.CHUNK_SIZE_Z * self.VOXEL_SIZE
        max_z = (self.chunk_z + 1) * self.CHUNK_SIZE_Z * self.VOXEL_SIZE
        return min_x, max_x, min_z, max_z

    def get_center(self) -> Tuple[float, float]:
        min_x, max_x, min_z, max_z = self.get_world_bounds()
        center_x = (min_x + max_x) / 2
        center_z = (min_z + max_z) / 2
        return center_x, center_z

    def _generate_ground(self):
        # 地面层在全chunk的y=0，位于第0个section的第0行
        self.voxels_all[0, :, 0, :] = 1

    def get_section_idx_for_y(self, world_y: float) -> int:
        # 全局y -> chunk内section索引
        idx = int(np.floor(world_y / (self.CHUNK_SECTION_HEIGHT * self.VOXEL_SIZE)))
        if 0 <= idx < self.num_sections:
            return idx
        raise IndexError('y坐标不在此chunk覆盖的区间内')

    def get_voxel(self, world_x: float, world_y: float, world_z: float) -> int:
        # 转为chunk内local坐标
        lx = int(np.floor((world_x - self.get_world_bounds()[0]) / self.VOXEL_SIZE))
        lz = int(np.floor((world_z - self.get_world_bounds()[2]) / self.VOXEL_SIZE))
        section_idx = self.get_section_idx_for_y(world_y)
        y_local = int(np.floor(world_y / self.VOXEL_SIZE)) % self.CHUNK_SECTION_HEIGHT
        return int(self.voxels_all[section_idx, lx, y_local, lz].item())

    def set_voxel(self, world_x: float, world_y: float, world_z: float, value: int):
        lx = int(np.floor((world_x - self.get_world_bounds()[0]) / self.VOXEL_SIZE))
        lz = int(np.floor((world_z - self.get_world_bounds()[2]) / self.VOXEL_SIZE))
        section_idx = self.get_section_idx_for_y(world_y)
        y_local = int(np.floor(world_y / self.VOXEL_SIZE)) % self.CHUNK_SECTION_HEIGHT
        self.voxels_all[section_idx, lx, y_local, lz] = value

    def contains_point(self, world_x: float, world_y: float, world_z: float) -> bool:
        min_x, max_x, min_z, max_z = self.get_world_bounds()
        min_y = 0.0
        max_y = self.num_sections * self.CHUNK_SECTION_HEIGHT * self.VOXEL_SIZE
        return (
            min_x <= world_x < max_x and
            min_y <= world_y < max_y and
            min_z <= world_z < max_z
        )

    def to(self, device: torch.device):
        """将Chunk数据移动到指定设备"""
        if self.device != device:
            self.device = device
            self.voxels_all = self.voxels_all.to(device)
        return self

class Agent:
    """
    Agent类：表示在环境中移动的智能体
    具有长、宽、高三个维度属性，以及当前位置信息
    """
    def __init__(self, length: float, width: float, height: float, 
                 initial_x: float = 0.0, initial_y: float = 0.0, initial_z: float = 0.0,
                 initial_qw: float = 1.0, initial_qx: float = 0.0, 
                 initial_qy: float = 0.0, initial_qz: float = 0.0,
                 device: Optional[torch.device] = None):
        """
        初始化Agent
        Args:
            length: Agent的长度（x方向，米）
            width: Agent的宽度（z方向，米）
            height: Agent的高度（y方向，米）
            initial_x: 初始x位置（世界坐标，米）
            initial_y: 初始y位置（世界坐标，米）
            initial_z: 初始z位置（世界坐标，米）
            initial_qw: 初始四元数w分量（姿态）
            initial_qx: 初始四元数x分量（姿态）
            initial_qy: 初始四元数y分量（姿态）
            initial_qz: 初始四元数z分量（姿态）
            device: 计算设备（GPU或CPU）
        """
        self.length = length      # x方向长度
        self.width = width        # z方向宽度
        self.height = height      # y方向高度
        
        # 当前位置（世界坐标）
        self.x = initial_x
        self.y = initial_y
        self.z = initial_z
        
        # 当前姿态（四元数：w, x, y, z）
        self.qw = initial_qw
        self.qx = initial_qx
        self.qy = initial_qy
        self.qz = initial_qz

        self.device = device if device is not None else torch.device("cpu")
    
    def update_position(self, new_x: float, new_y: float, new_z: float):
        """
        更新Agent的位置
        Args:
            new_x: 新的x位置
            new_y: 新的y位置
            new_z: 新的z位置
        """
        self.x = new_x
        self.y = new_y
        self.z = new_z
    
    def update_orientation(self, new_qw: float, new_qx: float, new_qy: float, new_qz: float):
        """
        更新Agent的姿态（四元数）
        Args:
            new_qw: 新的四元数w分量
            new_qx: 新的四元数x分量
            new_qy: 新的四元数y分量
            new_qz: 新的四元数z分量
        """
        self.qw = new_qw
        self.qx = new_qx
        self.qy = new_qy
        self.qz = new_qz
    
    def get_bounding_box(self) -> Dict[str, float]:
        """
        获取Agent的包围盒（考虑长宽高）
        Returns:
            {'x_min': x_min, 'x_max': x_max, 'z_min': z_min, 'z_max': z_max, 
             'y_min': y_min, 'y_max': y_max}
        """
        half_length = self.length / 2
        half_width = self.width / 2
        return {
            'x_min': self.x - half_length,
            'x_max': self.x + half_length,
            'z_min': self.z - half_width,
            'z_max': self.z + half_width,
            'y_min': self.y,
            'y_max': self.y + self.height
        }

if __name__=="__main__":
    # 示例：在世界坐标(0, 2, 0) 生成一个agent
    device = torch.device("cuda")
    agent = Agent(length=1.0, width=1.0, height=0.5,
                  initial_x=4.0, initial_y=2.0, initial_z=4.0,
                  device=device)
    print(f"创建Agent: 位置=({agent.x}, {agent.y}, {agent.z})")

    # ========== Vispy 可视化 ==========
    # 引入通用可视化工具
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.visualize import MapVisualizer

    # 传入 device 以便 Visualizer 和 Cache 正确处理 GPU 数据
    visualizer = MapVisualizer(title="Chunk可视化", device=device)
    
    # 根据Agent位置加载可见范围内所有chunk并可视化
    chunks_to_render = visualizer.update_chunks(agent.x, agent.z, Chunk, HORIZON, device)
    
    # 可视化agent
    visualizer.visualize_agent(agent)
    
    # 设置相机跟随Agent
    world_max_y = Chunk.CHUNK_SIZE_Y * Chunk.VOXEL_SIZE
    horizon_chunks = HORIZON.get("chunk_radius", 1) if isinstance(HORIZON, dict) else (HORIZON or 1)
    camera_dist = max(world_max_y, horizon_chunks * Chunk.CHUNK_SIZE_X * Chunk.VOXEL_SIZE * 2)
    visualizer.set_camera_target(np.array([agent.x, agent.y, agent.z]), distance=camera_dist)
    
    # ========== 键盘控制集成 ==========
    from utils.control import KeyboardController
    controller = KeyboardController(move_speed=Chunk.VOXEL_SIZE) # 每次移动一个体素大小
    controller.bind_agent(agent)
    controller.bind_visualizer(visualizer, camera_follow=True)
    # 绑定Chunk更新参数，以便移动时自动加载地图
    controller.bind_chunk_updater(Chunk, HORIZON, device)
    
    # 绑定事件到Canvas
    visualizer.canvas.events.key_press.connect(controller.on_key_press)
    visualizer.canvas.events.key_release.connect(controller.on_key_release)
    
    visualizer.run()
