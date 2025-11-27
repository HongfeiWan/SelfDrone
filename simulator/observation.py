import torch
import numpy as np
from typing import Any, Tuple, Optional
from map.generator import Chunk

class ObservationSystem:
    """
    观察系统：负责计算Agent视野范围内的有效数据，基于GPU加速。
    """
    def __init__(self, device: torch.device):
        self.device = device

    def get_visible_sections(self, agent: Any, view_radius: float) -> torch.Tensor:
        """
        高效获取Agent视野范围内所有涉及的Section索引。
        利用AABB（轴对齐包围盒）进行快速粗筛，通过GPU批量生成索引。
        Args:
            agent: Agent对象，需包含 x, y, z, height 属性
            view_radius: 视野半径 (米)，即 AABB 的半边长
        Returns:
            sections: (N, 3) IntTensor, 每一行是 [chunk_x, section_y, chunk_z]
                      即唯一的 Section ID 坐标
        """
        # 1. 获取 Section 的物理尺寸 (缓存这些值以提高性能)
        # 假设 Chunk 在 X/Z 平面是正方形，Section 高度可能不同
        # 注意：这里假设所有 Chunk 配置相同，直接从 Chunk 类读取
        voxel_size = Chunk.VOXEL_SIZE
        section_size_x = Chunk.CHUNK_SIZE_X * voxel_size
        section_size_z = Chunk.CHUNK_SIZE_Z * voxel_size
        section_size_y = Chunk.CHUNK_SECTION_HEIGHT * voxel_size
        max_sections_y = Chunk.CHUNK_SIZE_Y // Chunk.CHUNK_SECTION_HEIGHT
        
        # 2. 计算 Agent 的视野 AABB (世界坐标)
        center_x = agent.x
        center_y = agent.y + agent.height * 0.5
        center_z = agent.z
        
        min_x = center_x - view_radius
        max_x = center_x + view_radius
        min_y = center_y - view_radius
        max_y = center_y + view_radius
        min_z = center_z - view_radius
        max_z = center_z + view_radius
        
        # 3. 转换为 Grid/Section 索引范围
        # 使用 floor 确保覆盖边缘，并在 CPU 上计算边界（通常只有 2 个数，很快）
        # 或者全部转为 Tensor 计算以保持流水线，但这里标量计算更直接
        min_cx = int(np.floor(min_x / section_size_x))
        max_cx = int(np.floor(max_x / section_size_x))
        
        min_sy = int(np.floor(min_y / section_size_y))
        max_sy = int(np.floor(max_y / section_size_y))
        
        min_cz = int(np.floor(min_z / section_size_z))
        max_cz = int(np.floor(max_z / section_size_z))
        
        # 限制高度范围 (物理世界的硬约束)
        min_sy = max(0, min_sy)
        max_sy = min(max_sy, max_sections_y - 1)
        
        # 4. GPU 批量生成索引
        # 如果范围无效，返回空
        if min_cx > max_cx or min_sy > max_sy or min_cz > max_cz:
             return torch.empty((0, 3), dtype=torch.int, device=self.device)
             
        # 使用 torch.arange 生成范围
        # 注意：arange 是 [start, end)，所以要 +1
        rx = torch.arange(min_cx, max_cx + 1, device=self.device, dtype=torch.int)
        ry = torch.arange(min_sy, max_sy + 1, device=self.device, dtype=torch.int)
        rz = torch.arange(min_cz, max_cz + 1, device=self.device, dtype=torch.int)
        
        # 生成网格
        # meshgrid ('ij') -> (nx, ny, nz)
        # 我们希望最终顺序是 (chunk_x, section_y, chunk_z)
        grid_x, grid_y, grid_z = torch.meshgrid(rx, ry, rz, indexing='ij')
        
        # 堆叠并重塑为 (N, 3)
        sections = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)
        
        return sections

    def get_frustum_culled_sections(self, agent: Any, view_radius: float, 
                                    chunk_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        (高级) 获取经视锥剔除后的 Section 索引。
        不仅检查 AABB，还检查是否在视锥内。
        """
        # 1. 先获取 AABB 内的所有 sections
        candidate_sections = self.get_visible_sections(agent, view_radius)
        if candidate_sections.shape[0] == 0:
            return candidate_sections

        # 2. 如果提供了 chunk_indices（(M, 2) -> [cx, cz]），
        #    先在 GPU 上根据 (chunk_x, chunk_z) 进行过滤，保证仅保留 horizon 范围内的 sections。
        if chunk_indices is not None and chunk_indices.numel() > 0:
            # candidate_sections: (N, 3) -> 取 (cx, cz)
            sec_pairs = candidate_sections[:, (0, 2)]              # (N, 2)
            chunk_indices = chunk_indices.to(self.device)          # (M, 2)
            # 利用广播做集合相交：
            # sec_pairs[:, None, :]  vs  chunk_indices[None, :, :]
            # 得到 (N, M) 的布尔矩阵，表示每个 section 是否在给定 chunks 内
            sp = sec_pairs.unsqueeze(1)        # (N, 1, 2)
            cp = chunk_indices.unsqueeze(0)    # (1, M, 2)
            eq = (sp == cp).all(dim=2)         # (N, M)
            mask = eq.any(dim=1)               # (N,)
            candidate_sections = candidate_sections[mask]
            if candidate_sections.shape[0] == 0:
                return candidate_sections

        # 3. 精确的视锥剔除：将 Section 中心点转换到传感器坐标系，用 FOV 角度范围筛选
        if not (hasattr(agent, 'fov_lat_range') and hasattr(agent, 'fov_lon_range')):
            # 如果没有 FOV 配置，直接返回 AABB + horizon 过滤结果
            return candidate_sections

        # 3.1 计算每个 Section 的中心点（世界坐标）
        voxel_size = Chunk.VOXEL_SIZE
        section_size_x = Chunk.CHUNK_SIZE_X * voxel_size
        section_size_z = Chunk.CHUNK_SIZE_Z * voxel_size
        section_size_y = Chunk.CHUNK_SECTION_HEIGHT * voxel_size

        # candidate_sections: (N, 3) -> [chunk_x, section_y, chunk_z]
        # 计算每个 section 的中心点（世界坐标）
        section_centers_world = torch.zeros(
            (candidate_sections.shape[0], 3), 
            dtype=torch.float32, 
            device=self.device
        )
        section_centers_world[:, 0] = candidate_sections[:, 0].float() * section_size_x + section_size_x * 0.5
        section_centers_world[:, 1] = candidate_sections[:, 1].float() * section_size_y + section_size_y * 0.5
        section_centers_world[:, 2] = candidate_sections[:, 2].float() * section_size_z + section_size_z * 0.5

        # 3.2 转换到 Agent body 坐标系
        # Agent 位置（世界坐标）
        agent_pos = torch.tensor(
            [agent.x, agent.y + agent.height * 0.5, agent.z],
            dtype=torch.float32,
            device=self.device
        )
        # 相对于 Agent 的向量
        vec_to_section = section_centers_world - agent_pos.unsqueeze(0)  # (N, 3)

        # 四元数转旋转矩阵（Body -> World）
        qw, qx, qy, qz = agent.qw, agent.qx, agent.qy, agent.qz
        # 归一化
        q_norm = torch.sqrt(torch.tensor(qw*qw + qx*qx + qy*qy + qz*qz, dtype=torch.float32, device=self.device)) + 1e-9
        qw, qx, qy, qz = qw/q_norm, qx/q_norm, qy/q_norm, qz/q_norm

        # 旋转矩阵 R_body_world（列向量是 body 坐标轴在世界坐标系的方向）
        # R = [1-2y^2-2z^2, 2xy-2wz,     2xz+2wy]
        #     [2xy+2wz,     1-2x^2-2z^2, 2yz-2wx]
        #     [2xz-2wy,     2yz+2wx,     1-2x^2-2y^2]
        R_bw = torch.tensor([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy + qw*qz),     2*(qx*qz - qw*qy)],
            [2*(qx*qy - qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz + qw*qx)],
            [2*(qx*qz + qw*qy),     2*(qy*qz - qw*qx),     1 - 2*(qx*qx + qy*qy)]
        ], dtype=torch.float32, device=self.device)

        # World -> Body: R_world_body = R_body_world^T
        R_wb = R_bw.T
        vec_body = vec_to_section @ R_wb.T  # (N, 3)

        # 3.3 转换到传感器坐标系（应用 sensor_mount_rotation）
        mount_rot = getattr(agent, 'sensor_mount_rotation', (0.0, 0.0, 0.0))
        pitch, yaw, roll = torch.tensor(mount_rot, dtype=torch.float32, device=self.device)
        p, y, r = torch.deg2rad(pitch), torch.deg2rad(yaw), torch.deg2rad(roll)

        # 欧拉角转旋转矩阵：R_sensor_body（Roll -> Pitch -> Yaw）
        cos_p, sin_p = torch.cos(p), torch.sin(p)
        cos_y, sin_y = torch.cos(y), torch.sin(y)
        cos_r, sin_r = torch.cos(r), torch.sin(r)

        Rx = torch.tensor([
            [1, 0, 0],
            [0, cos_p, -sin_p],
            [0, sin_p, cos_p]
        ], dtype=torch.float32, device=self.device)

        Ry = torch.tensor([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y]
        ], dtype=torch.float32, device=self.device)

        Rz = torch.tensor([
            [cos_r, -sin_r, 0],
            [sin_r, cos_r, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)

        R_sb = Ry @ Rx @ Rz  # (3, 3)
        vec_sensor = vec_body @ R_sb.T  # (N, 3)

        # 3.4 转换为球坐标（lat, lon）
        # 传感器坐标系：Y-up, Z-forward
        # x = r * cos(lat) * sin(lon)
        # y = r * sin(lat)
        # z = r * cos(lat) * cos(lon)
        r = torch.norm(vec_sensor, dim=1, keepdim=True)  # (N, 1)
        r = torch.clamp(r, min=1e-6)  # 避免除零

        # lat (pitch): lat = arcsin(y / r)
        lat_rad = torch.asin(vec_sensor[:, 1] / r.squeeze(1))  # (N,)
        lat_deg = torch.rad2deg(lat_rad)  # (N,)

        # lon (yaw): lon = atan2(x, z)
        lon_rad = torch.atan2(vec_sensor[:, 0], vec_sensor[:, 2])  # (N,)
        lon_deg = torch.rad2deg(lon_rad)  # (N,)
        # 归一化到 [0, 360)
        lon_deg = torch.where(lon_deg < 0, lon_deg + 360.0, lon_deg)

        # 3.5 判断是否在 FOV 盲区范围内
        # FOV 范围定义的是"盲区"（不可见区域），在盲区内的点看不到，盲区外的点可以看到
        lat_min, lat_max = agent.fov_lat_range
        lon_min, lon_max = agent.fov_lon_range

        # 纬度范围检查：是否在盲区锥形内
        lat_in_blind = (lat_deg >= lat_min) & (lat_deg <= lat_max)

        # 经度范围检查：是否在盲区锥形内（处理跨 0 度的情况，例如 [350, 10]）
        if lon_max >= lon_min:
            lon_in_blind = (lon_deg >= lon_min) & (lon_deg <= lon_max)
        else:
            # 跨 0 度：例如 [350, 10] -> lon >= 350 或 lon <= 10 表示在盲区内
            lon_in_blind = (lon_deg >= lon_min) | (lon_deg <= lon_max)

        # 组合掩码：在盲区锥形内的点（lat 和 lon 都在盲区范围内）
        blind_zone_mask = lat_in_blind & lon_in_blind

        # 可见区域 = 不在盲区内的点
        visible_mask = ~blind_zone_mask

        # 3.6 应用视锥剔除：只保留可见区域内的 sections
        visible_sections = candidate_sections[visible_mask]

        return visible_sections

    def get_observed_chunk_indices(self, agent_x: float, agent_z: float, ChunkClass: Any, horizon: Any) -> torch.Tensor:
        """
        根据Agent位置和Horizon设置计算观测范围内的Chunk索引（GPU 张量版本）
        Args:
            agent_x: Agent的x坐标
            agent_z: Agent的z坐标
            ChunkClass: Chunk类引用，必须包含 world_to_chunk_idx 静态方法
            horizon: 视野配置或半径
        Returns:
            indices: (N, 2) IntTensor，device 与 ObservationSystem 一致；每行是 [cx, cz]
        """
        # 世界坐标 -> 中心 chunk 索引（通常是纯数学/整数运算，开销极小）
        chunk_x, chunk_z = ChunkClass.world_to_chunk_idx(agent_x, agent_z)
        h = horizon.get("chunk_radius", 1) if isinstance(horizon, dict) else (horizon or 1)

        # 在 GPU 上一次性生成所有 (cx, cz) 组合
        rx = torch.arange(chunk_x - h, chunk_x + h + 1, device=self.device, dtype=torch.int)
        rz = torch.arange(chunk_z - h, chunk_z + h + 1, device=self.device, dtype=torch.int)

        grid_x, grid_z = torch.meshgrid(rx, rz, indexing='ij')  # (nx, nz)
        indices = torch.stack([grid_x.reshape(-1), grid_z.reshape(-1)], dim=1)  # (N, 2)

        return indices

