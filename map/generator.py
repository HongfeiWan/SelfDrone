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
        # 稀疏存储：仅存储非空体素的坐标和值
        # 格式：indices=(N, 3), values=(N,)
        # 但为了保持与现有接口兼容，我们可能需要一个包装器或者修改访问方式
        # 这里先用 dense tensor，后续优化为 sparse
        # 根据用户请求：大部分是空气(0)，不需要存储
        # PyTorch 稀疏张量支持有限，对于 uint8 可能更有限
        # 简单方案：使用 Dense Tensor 但只在内存中，序列化时转为 Sparse 或 Coordinate List
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
        
        # 初始化为 None，表示全空（空气）
        # 这是一个简单的稀疏优化：如果 voxels_all 为 None，则表示整个 Chunk 为空
        # 或者，我们可以使用 PyTorch 的 sparse_coo_tensor
        # 为了兼容性，先保留 dense tensor，但提供 compress/decompress 方法
        # 用户意图是“存储时优化”，所以我们可以保持运行时 dense，存储时 sparse
        
        # 运行时：Dense Tensor (GPU)
        self.voxels_all = torch.zeros(
            (self.num_sections, self.CHUNK_SIZE_X, self.CHUNK_SECTION_HEIGHT, self.CHUNK_SIZE_Z),
            dtype=torch.uint8, device=device)
        self._generate_ground()

    def compress(self):
        """将体素数据压缩为稀疏格式（坐标列表），用于存储"""
        # 只有非零值需要存储
        indices = torch.nonzero(self.voxels_all, as_tuple=False) # (N, 4) -> [section, x, y, z]
        if indices.shape[0] == 0:
            return None
        
        values = self.voxels_all[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]
        # 返回稀疏表示：(indices, values, shape)
        # 注意：indices 和 values 都在 GPU 上，可以直接返回
        return {
            'indices': indices,
            'values': values,
            'shape': self.voxels_all.shape
        }

    def decompress(self, sparse_data):
        """从稀疏格式解压回 Dense Tensor"""
        # 确保 voxels_all 已分配
        if self.voxels_all is None:
            self.voxels_all = torch.zeros(
                (self.num_sections, self.CHUNK_SIZE_X, self.CHUNK_SECTION_HEIGHT, self.CHUNK_SIZE_Z),
                dtype=torch.uint8, device=self.device)

        if sparse_data is None:
            # 全空
            self.voxels_all.zero_()
            return

        indices = sparse_data['indices']
        values = sparse_data['values']
        shape = sparse_data['shape']
        
        # 确保 indices 和 values 在正确的设备上
        if indices.device != self.device:
            indices = indices.to(self.device)
        if values.device != self.device:
            values = values.to(self.device)
            
        self.voxels_all.zero_()
        self.voxels_all[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]] = values

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
            if self.voxels_all is not None:
                self.voxels_all = self.voxels_all.to(device)
                 
            # 同时移动压缩数据（如果存在）
            if hasattr(self, 'compressed_data') and self.compressed_data is not None:
                # compressed_data 是一个字典 {'indices': ..., 'values': ..., 'shape': ...}
                for k, v in self.compressed_data.items():
                    if hasattr(v, 'to'):
                        self.compressed_data[k] = v.to(device)
        return self

    def clone(self):
        """创建Chunk的副本（深拷贝）"""
        new_chunk = Chunk(self.chunk_x, self.chunk_z, self.device)
        
        # 复制稠密数据
        if self.voxels_all is not None:
            new_chunk.voxels_all = self.voxels_all.clone()
        else:
            new_chunk.voxels_all = None
            
        # 复制压缩数据
        if hasattr(self, 'compressed_data') and self.compressed_data is not None:
            new_chunk.compressed_data = {}
            for k, v in self.compressed_data.items():
                if hasattr(v, 'clone'):
                    new_chunk.compressed_data[k] = v.clone()
                else:
                    new_chunk.compressed_data[k] = v # shape等元数据
                    
        return new_chunk

class Agent:
    """
    Agent类：表示在环境中移动的智能体
    具有长、宽、高三个维度属性，以及当前位置信息
    """
    def __init__(self, length: float, width: float, height: float, 
                 initial_x: float = 0.0, initial_y: float = 0.0, initial_z: float = 0.0,
                 initial_qw: float = 1.0, initial_qx: float = 0.0, 
                 initial_qy: float = 0.0, initial_qz: float = 0.0,
                 fov_lat_range: Tuple[float, float] = (-7.0, 52.0),
                 fov_lon_range: Tuple[float, float] = (0.0, 360.0),
                 sensor_mount_rotation: Tuple[float, float, float] = (0.0, 0.0, -30.0),
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
            fov_lat_range: 视角的纬度范围 (min_pitch, max_pitch)，单位度。默认(-7, 52)。
            fov_lon_range: 视角的经度范围 (min_yaw, max_yaw)，单位度。默认(0, 360)。
            sensor_mount_rotation: 传感器相对于Agent本体坐标系的安装角度偏移 (pitch_offset, yaw_offset, roll_offset)，单位度。
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

        # 传感器/视角参数
        self.fov_lat_range = fov_lat_range
        self.fov_lon_range = fov_lon_range
        self.sensor_mount_rotation = sensor_mount_rotation # (pitch, yaw, roll) offset in degrees

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
    # 返回值为 (N, 2) 的 IntTensor，每行 [chunk_x, chunk_z]
    visible_chunk_indices = visualizer.update_chunks(agent.x, agent.z, Chunk, HORIZON, device)
    
    # 初始时同时可视化 Agent 和其观测到的 Sections
    visualizer.update_agent_visuals(agent, visible_chunk_indices)
    
    # 设置相机跟随Agent
    world_max_y = Chunk.CHUNK_SIZE_Y * Chunk.VOXEL_SIZE
    horizon_chunks = HORIZON.get("chunk_radius", 1) if isinstance(HORIZON, dict) else (HORIZON or 1)
    camera_dist = max(world_max_y, horizon_chunks * Chunk.CHUNK_SIZE_X * Chunk.VOXEL_SIZE * 2)

    
    # ========== 键盘控制集成 ==========
    from utils.control import KeyboardController
    controller = KeyboardController(move_speed=Chunk.VOXEL_SIZE) # 每次移动一个体素大小
    controller.bind_agent(agent)
    controller.bind_visualizer(visualizer, camera_follow=True)
    # 绑定Chunk更新参数，以便移动时自动加载地图
    controller.bind_chunk_updater(Chunk, HORIZON, device)
    # 将初始可见的 Chunk 索引传递给控制器，用于后续 Section 可视化
    controller.visible_chunk_indices = visible_chunk_indices
    
    # 绑定事件到Canvas
    visualizer.canvas.events.key_press.connect(controller.on_key_press)
    visualizer.canvas.events.key_release.connect(controller.on_key_release)
    
    try:
        visualizer.run()
    finally:
        # 程序退出时保存地图
        if hasattr(visualizer, 'chunk_cache'):
            visualizer.chunk_cache.save_to_disk()
            #visualizer.chunk_cache.clear_disk_cache() # 可选：是否保留历史
