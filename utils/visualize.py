import numpy as np
import torch
from typing import List, Tuple, Optional, Any, Union, Dict
from vispy import app, scene
from vispy.scene.visuals import Line, Mesh, Markers
from vispy.geometry import MeshData
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.cache import LFUCache
from simulator.observation import ObservationSystem
from map.generator import Chunk # 需要获取常量

class MapVisualizer:
    """
    通用地图/Agent可视化工具
    """
    def __init__(self, title: str = "Map Visualization", size: Tuple[int, int] = (1200, 900), bg_color: str = 'white', device: Union[str, torch.device] = 'cpu'):
        self.canvas = scene.SceneCanvas(keys='interactive', show=True, size=size, bgcolor=bg_color, title=title)
        self.view = self.canvas.central_widget.add_view()
        # 使用 TurntableCamera，后续通过 set_camera_target 实现相机跟随
        self.view.camera = scene.TurntableCamera(up='y', distance=60)
        
        # 存储可视化对象引用
        self.chunk_visuals = []
        self.section_visuals = []
        self.agent_visuals = []
        self.observation_visuals = [] # 存储观测到的Section
        self.show_sections = True
        
        # Chunk缓存 (传入device以便加载时自动移回GPU)
        # map_path 对应 cache.py 中 LFUCache.__init__ 的参数
        self.chunk_cache = LFUCache(capacity=100, map_path="map/world_map.pt", device=device)
        
        # 初始化观测系统
        self.obs_system = ObservationSystem(device=device)

    def set_camera_target(self, target: Union[np.ndarray, List[float], Tuple[float, float, float]]):
        """
        设置相机关注点（用于键盘控制时的相机跟随）。
        target: (x, y, z) 世界坐标
        """
        # TurntableCamera 使用 center 属性作为观察中心
        if self.view is not None and getattr(self.view, "camera", None) is not None:
            # 确保转为 numpy 数组，长度为 3
            center = np.asarray(target, dtype=np.float32).reshape(3,)
            self.view.camera.center = center

    def _compute_chunk_vis_data(self, chunk: Any) -> Dict[str, Any]:
        """计算单个Chunk的可视化数据（顶点等）"""
        min_x, max_x, min_z, max_z = chunk.get_world_bounds()
        voxel_size = chunk.VOXEL_SIZE
        section_height_world = chunk.CHUNK_SECTION_HEIGHT * voxel_size
        chunk_size_y_world = chunk.CHUNK_SIZE_Y * voxel_size

        # 1. Chunk 边界框
        box_vertices = np.array([
            [min_x, 0, min_z], [max_x, 0, min_z],
            [max_x, 0, max_z], [min_x, 0, max_z],
            [min_x, chunk_size_y_world, min_z], [max_x, chunk_size_y_world, min_z],
            [max_x, chunk_size_y_world, max_z], [min_x, chunk_size_y_world, max_z],
        ], dtype=np.float32)
        
        box_edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ], dtype=np.int32)
        
        chunk_edge_vertices = box_vertices[box_edges.flatten()]

        # 2. Section 分隔线
        section_edge_vertices_list = []
        for s in range(1, chunk.num_sections):
            y = s * section_height_world
            section_vertices = np.array([
                [min_x, y, min_z], [max_x, y, min_z],
                [max_x, y, max_z], [min_x, y, max_z],
            ], dtype=np.float32)
            edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
            section_edge_vertices_list.append(section_vertices[edges.flatten()])
        
        if section_edge_vertices_list:
            section_edge_vertices = np.concatenate(section_edge_vertices_list, axis=0)
        else:
            section_edge_vertices = np.empty((0, 3), dtype=np.float32)

        # 3. 地面体素
        ground_layer = chunk.voxels_all[0, :, 0, :]
        if hasattr(ground_layer, 'cpu'):
            ground_layer = ground_layer.cpu()
        if hasattr(ground_layer, 'numpy'):
            ground_layer = ground_layer.numpy()
        
        lx, lz = np.nonzero(ground_layer)
        ground_centers = np.empty((0, 3), dtype=np.float32)
        if len(lx) > 0:
            ground_world_x = min_x + (lx + 0.5) * voxel_size
            ground_world_z = min_z + (lz + 0.5) * voxel_size
            ground_world_y = np.zeros_like(ground_world_x) + voxel_size * 0.5
            ground_centers = np.stack([ground_world_x, ground_world_y, ground_world_z], axis=1)

        return {
            'chunk_edge_vertices': chunk_edge_vertices,
            'section_edge_vertices': section_edge_vertices,
            'ground_centers': ground_centers,
            'voxel_size': voxel_size # 保存以便后续使用
        }

    def visualize_chunks(self, chunks_data: List[Dict[str, Any]]):
        """
        可视化一组Chunk数据
        Args:
            chunks_data: 包含可视化数据的字典列表
        """
        if not chunks_data:
            return

        chunk_edge_vertices_list = []
        section_edge_vertices_list = []
        all_ground_centers = []
        
        voxel_size = chunks_data[0]['voxel_size'] # 假设一致

        for data in chunks_data:
            chunk_edge_vertices_list.append(data['chunk_edge_vertices'])
            if data['section_edge_vertices'].size > 0:
                section_edge_vertices_list.append(data['section_edge_vertices'])
            if data['ground_centers'].size > 0:
                all_ground_centers.append(data['ground_centers'])

        # 创建 Vispy Visuals
        if chunk_edge_vertices_list:
            chunk_edge_vertices = np.concatenate(chunk_edge_vertices_list, axis=0)
            chunk_line = Line(pos=chunk_edge_vertices, connect='segments', 
                              color=[0, 0, 0, 0.6], width=1.5, method='gl')
            self.view.add(chunk_line)
            self.chunk_visuals.append(chunk_line)

        if section_edge_vertices_list:
            section_edge_vertices = np.concatenate(section_edge_vertices_list, axis=0)
            section_line = Line(pos=section_edge_vertices, connect='segments', 
                                color=[0, 0, 0, 0.3], width=1, method='gl')
            self.view.add(section_line)
            self.chunk_visuals.append(section_line)

        if all_ground_centers:
            ground_centers_global = np.concatenate(all_ground_centers, axis=0)
            
            # Mesh 渲染 (立方体)
            half_voxel = voxel_size * 0.5
            vertex_offsets = np.array([
                [-half_voxel, -half_voxel, -half_voxel],
                [half_voxel, -half_voxel, -half_voxel],
                [half_voxel, half_voxel, -half_voxel],
                [-half_voxel, half_voxel, -half_voxel],
                [-half_voxel, -half_voxel, half_voxel],
                [half_voxel, -half_voxel, half_voxel],
                [half_voxel, half_voxel, half_voxel],
                [-half_voxel, half_voxel, half_voxel],
            ], dtype=np.float32)
            
            # 向量化生成所有体素的顶点
            ground_vertices = (ground_centers_global[:, np.newaxis, :] + vertex_offsets[np.newaxis, :, :]).reshape(-1, 3)
            
            base_faces = np.array([
                [0, 1, 2], [0, 2, 3],
                [4, 6, 5], [4, 7, 6],
                [0, 3, 7], [0, 7, 4],
                [1, 5, 6], [1, 6, 2],
                [0, 4, 5], [0, 5, 1],
                [3, 2, 6], [3, 6, 7],
            ], dtype=np.uint32)
            
            num_cubes = len(ground_centers_global)
            # 向量化生成所有面索引
            all_faces = (base_faces[np.newaxis, :, :] + (np.arange(num_cubes, dtype=np.uint32) * 8)[:, np.newaxis, np.newaxis]).reshape(-1, 3)

            ground_colors = np.zeros((ground_vertices.shape[0], 4), dtype=np.float32)
            ground_colors[:] = [0.6, 0.4, 0.2, 1.0] # 棕色
            
            ground_mesh_data = MeshData(vertices=ground_vertices, faces=all_faces, vertex_colors=ground_colors)
            ground_mesh = Mesh(meshdata=ground_mesh_data, shading='flat')
            self.view.add(ground_mesh)
            self.chunk_visuals.append(ground_mesh)

            # 快速可视化 (Markers)
            markers = Markers()
            markers.set_data(ground_centers_global, face_color=[0.4, 0.7, 0.9, 0.8], size=4)
            self.view.add(markers)
            self.chunk_visuals.append(markers)
            
            # print(f"✓ 已可视化地面网格: {len(ground_centers_global)}个体素")

    def update_chunks(self, agent_x: float, agent_z: float, ChunkClass: Any, horizon: Any, device: Any) -> torch.Tensor:
        """
        根据Agent位置和Horizon设置动态加载并更新可视化的chunks，
        并返回当前视野内的Chunk索引张量。
        Args:
            agent_x: Agent的x坐标
            agent_z: Agent的z坐标
            ChunkClass: Chunk类引用，必须包含 world_to_chunk_idx 静态方法和构造函数
            horizon: 视野配置或半径
            device: torch device
        Returns:
            torch.Tensor: (N, 2) IntTensor，每行 [cx, cz]，位于 ObservationSystem 的 device 上
        """
        # 清除旧的Chunks可视化
        for visual in self.chunk_visuals:
            visual.parent = None
        self.chunk_visuals = []
        
        # 计算需要的chunks（GPU张量形式的索引）
        chunk_indices = self.obs_system.get_observed_chunk_indices(agent_x, agent_z, ChunkClass, horizon)
        horizon_chunks = horizon.get("chunk_radius", 1) if isinstance(horizon, dict) else (horizon or 1)

        chunks_to_render: List[Any] = []
        vis_data_list: List[Dict[str, Any]] = []

        # 遍历张量的每一行 [cx, cz]，按需取出标量索引用于 Python 端对象/缓存
        for i in range(chunk_indices.shape[0]):
            cx = int(chunk_indices[i, 0].item())
            cz = int(chunk_indices[i, 1].item())
            # 尝试从缓存获取
            cached_item = self.chunk_cache.get((cx, cz))
            chunk = None
            vis_data = None
            if cached_item is not None:
                # 情况1: L1缓存，包含完整对象（兼容旧格式）
                if isinstance(cached_item, dict) and 'chunk' in cached_item:
                    chunk = cached_item['chunk']
                    vis_data = cached_item.get('vis_data')
                
                # 情况2: 仅包含压缩数据 (compressed_data)，可选带 vis_data
                elif isinstance(cached_item, dict) and 'compressed_data' in cached_item:
                    # 重建 Chunk
                    chunk = ChunkClass(cx, cz, device)
                    chunk.decompress(cached_item['compressed_data'])
                    vis_data = cached_item.get('vis_data')

            if chunk is None:
                # 创建新Chunk (全新生成)
                chunk = ChunkClass(cx, cz, device)

            # 无论 chunk 是否来自缓存，只要没有 vis_data 就现算一份
            if vis_data is None:
                vis_data = self._compute_chunk_vis_data(chunk)

            # 无论 vis_data 是否来自缓存，始终将最新的压缩体素数据写入缓存
            # 缓存中只保留 tensor 友好的压缩数据 + 可选的可视化派生数据
            if hasattr(chunk, 'compress'):
                compressed = chunk.compress()
                cache_payload = {'compressed_data': compressed, 'vis_data': vis_data}
            else:
                # 后备：如果 Chunk 不支持 compress，则直接放入，但这不再推荐
                cache_payload = {'raw_chunk': chunk, 'vis_data': vis_data}

            self.chunk_cache.put((cx, cz), cache_payload)
            chunks_to_render.append(chunk)
            vis_data_list.append(vis_data)
        print(f"加载/渲染 {len(chunks_to_render)} 个 chunk，范围半径={horizon_chunks}")

        # 重新可视化chunks
        self.visualize_chunks(vis_data_list)
        
        # 返回当前视野内的 Chunk 索引（GPU 张量）
        return chunk_indices

    def visualize_agent(self, agent: Any, color: list = [1, 0, 0, 1]):
        """
        可视化Agent (包围盒)
        Args:
            agent: 必须包含 x, y, z, length, width, height 属性
            color: RGBA颜色
        """
        agent_half = np.array([agent.length, agent.height, agent.width]) * 0.5
        agent_offsets = np.array([
            [-agent_half[0], -agent_half[1], -agent_half[2]],
            [ agent_half[0], -agent_half[1], -agent_half[2]],
            [ agent_half[0],  agent_half[1], -agent_half[2]],
            [-agent_half[0],  agent_half[1], -agent_half[2]],
            [-agent_half[0], -agent_half[1],  agent_half[2]],
            [ agent_half[0], -agent_half[1],  agent_half[2]],
            [ agent_half[0],  agent_half[1],  agent_half[2]],
            [-agent_half[0],  agent_half[1],  agent_half[2]],
        ], dtype=np.float32)
        center_pos = np.array([agent.x, agent.y + agent.height * 0.5, agent.z])
        agent_vertices = agent_offsets + center_pos
        
        agent_edges = np.array([
            [0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7]
        ], dtype=np.int32)
        
        agent_edge_vertices = agent_vertices[agent_edges.flatten()].reshape(-1,3)
        agent_line = Line(pos=agent_edge_vertices, connect='segments', color=color, width=3, method='gl')
        self.view.add(agent_line)
        self.agent_visuals.append(agent_line)

        # 绘制Agent的局部坐标轴 (X=红, Y=绿, Z=蓝)
        # 长度为Agent最长边的一半
        axis_len = max(agent.length, agent.height, agent.width) * 0.8
        # 局部坐标轴的起点（Agent中心）
        center_pos = np.array([agent.x, agent.y + agent.height * 0.5, agent.z])
        # 如果Agent有旋转矩阵或四元数，应在此处应用旋转
        # 这里假设Agent有姿态属性，尝试获取旋转矩阵
        # 简单起见，这里先画未旋转的轴。若需要跟随旋转，需要Agent提供旋转矩阵或四元数计算
        # 构建坐标轴顶点 [Start, End]
        # X轴 (红)
        x_axis = np.array([center_pos, center_pos + np.array([axis_len, 0, 0])])
        # Y轴 (绿)
        y_axis = np.array([center_pos, center_pos + np.array([0, axis_len, 0])])
        # Z轴 (蓝)
        z_axis = np.array([center_pos, center_pos + np.array([0, 0, axis_len])])
        
        # 如果Agent有四元数属性，进行旋转变换
        if hasattr(agent, 'qw') and hasattr(agent, 'qx'):
            # 简单的四元数转旋转矩阵逻辑
            w, x, y, z = agent.qw, agent.qx, agent.qy, agent.qz
            # 归一化
            norm = np.sqrt(w*w + x*x + y*y + z*z) + 1e-9
            w, x, y, z = w/norm, x/norm, y/norm, z/norm
            
            # 旋转矩阵列向量 (对应局部x, y, z轴在世界坐标系的方向)
            # R = [1-2y^2-2z^2, 2xy-2wz,     2xz+2wy]
            #     [2xy+2wz,     1-2x^2-2z^2, 2yz-2wx]
            #     [2xz-2wy,     2yz+2wx,     1-2x^2-2y^2]
            
            rx = np.array([1 - 2*(y**2 + z**2), 2*(x*y + w*z),     2*(x*z - w*y)])
            ry = np.array([2*(x*y - w*z),     1 - 2*(x**2 + z**2), 2*(y*z + w*x)])
            rz = np.array([2*(x*z + w*y),     2*(y*z - w*x),     1 - 2*(x**2 + y**2)])
            
            # 更新轴的终点
            x_axis[1] = center_pos + rx * axis_len
            y_axis[1] = center_pos + ry * axis_len
            z_axis[1] = center_pos + rz * axis_len
            
        line_x = Line(pos=x_axis, color='red', width=3, method='gl')
        self.view.add(line_x)
        self.agent_visuals.append(line_x)
        
        line_y = Line(pos=y_axis, color='green', width=3, method='gl')
        self.view.add(line_y)
        self.agent_visuals.append(line_y)
        
        line_z = Line(pos=z_axis, color='blue', width=3, method='gl')
        self.view.add(line_z)
        self.agent_visuals.append(line_z)

        # 绘制FOV视锥
        if hasattr(agent, 'fov_lat_range') and hasattr(agent, 'fov_lon_range'):
            fov_dist = axis_len * 10.0 # 视锥长度，设大一点以便观察
            lat_min, lat_max = agent.fov_lat_range
            lon_min, lon_max = agent.fov_lon_range
            mount_rot = getattr(agent, 'sensor_mount_rotation', (0, 0, 0))
            
            # 辅助函数：欧拉角转旋转矩阵
            def euler_to_rot_mat(pitch, yaw, roll): # degrees
                p, y, r = np.radians(pitch), np.radians(yaw), np.radians(roll)
                # Rx (Pitch)
                Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
                # Ry (Yaw)
                Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
                # Rz (Roll)
                Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
                return Ry @ Rx @ Rz # 顺序: Roll -> Pitch -> Yaw
            
            # 传感器安装旋转矩阵 R_sensor_body
            R_sb = euler_to_rot_mat(mount_rot[0], mount_rot[1], mount_rot[2])
            
            # Body到World的旋转矩阵 R_body_world (复用之前的 rx, ry, rz)
            if hasattr(agent, 'qw'):
                 R_bw = np.column_stack((rx, ry, rz))
            else:
                 R_bw = np.eye(3)
                 
            # 总旋转 R_total = R_body_world * R_sensor_body
            R_total = R_bw @ R_sb
            
            fov_lines = []
            num_segments = 36
            
            # 1. 绘制 lat_min 和 lat_max 的圆环 (或弧线)
            for lat in [lat_min, lat_max]:
                circle_points = []
                # 如果 lon 范围是 360，画整圆；否则画弧
                lon_steps = np.linspace(lon_min, lon_max, num_segments)
                for lon in lon_steps:
                    # 转为弧度
                    lat_rad = np.radians(lat)
                    lon_rad = np.radians(lon) # 注意：lon通常0度对应Z轴(前)
                    # 简单的球坐标映射 (Y-up, Z-forward)
                    # y = sin(lat)
                    # z = cos(lat) * cos(lon)
                    # x = cos(lat) * sin(lon)
                    x = fov_dist * np.cos(lat_rad) * np.sin(lon_rad)
                    y = fov_dist * np.sin(lat_rad)
                    z = fov_dist * np.cos(lat_rad) * np.cos(lon_rad)
                    
                    p_sensor = np.array([x, y, z])
                    p_world = center_pos + R_total @ p_sensor
                    circle_points.append(p_world)
                
                # 闭合圆环 (如果是360度)
                if abs(lon_max - lon_min) >= 359.9:
                    circle_points.append(circle_points[0])
                
                # 转换为 Line 数据
                circle_points = np.array(circle_points)
                fov_lines.append(circle_points)
            
            # 2. 绘制连接线 (经度边界或骨架)
            corners_lon = [lon_min, lon_max]
            if abs(lon_max - lon_min) >= 359.9:
                corners_lon = np.linspace(0, 360, 8, endpoint=False) # 画8条骨架线
            
            for lon in corners_lon:
                lon_rad = np.radians(lon)
                pts = [] # 用于存储该经度上的上下两个端点
                
                for lat in [lat_min, lat_max]:
                    lat_rad = np.radians(lat)
                    x = fov_dist * np.cos(lat_rad) * np.sin(lon_rad)
                    y = fov_dist * np.sin(lat_rad)
                    z = fov_dist * np.cos(lat_rad) * np.cos(lon_rad)
                    p_end = center_pos + R_total @ np.array([x, y, z])
                    pts.append(p_end)
                    
                    # 画中心射线 (可选，保留以显示锥体感)
                    fov_lines.append(np.array([center_pos, p_end]))
                
                # 将上下两点连起来
                fov_lines.append(np.array(pts))

            # 渲染
            for line_points in fov_lines:
                vis_line = Line(pos=line_points, color='yellow', width=1, connect='strip', method='gl')
                self.view.add(vis_line)
                self.agent_visuals.append(vis_line)

    def clear_agent_visuals(self):
        """清除所有Agent相关的可视化元素"""
        for visual in self.agent_visuals:
            visual.parent = None
        self.agent_visuals = []

    def visualize_observations(self, agent: Any, chunk_indices: torch.Tensor):
        """
        可视化Agent观测到的Sections
        """
        # 清除旧的观测可视化
        for visual in self.observation_visuals:
            visual.parent = None
        self.observation_visuals = []
        
        # 获取观测到的 Section 索引
        # 视距应与 FOV 设置相关，这里取一个合理值（上界），
        # 实际 horizon 范围由 chunk_indices 进一步约束
        # 雷达半径30m
        view_radius = 30.0

        visible_sections = self.obs_system.get_frustum_culled_sections(
            agent, view_radius=view_radius, chunk_indices=chunk_indices
        )
        if visible_sections.shape[0] == 0:
            return
            
        # 转换为 numpy
        sections_np = visible_sections.cpu().numpy()
        
        # 获取 Chunk 常量
        chunk_sx = Chunk.CHUNK_SIZE_X * Chunk.VOXEL_SIZE
        chunk_sz = Chunk.CHUNK_SIZE_Z * Chunk.VOXEL_SIZE
        section_sy = Chunk.CHUNK_SECTION_HEIGHT * Chunk.VOXEL_SIZE
        
        # 基础 AABB 顶点 (相对于Section原点)
        base_box = np.array([
            [0, 0, 0], [chunk_sx, 0, 0], [chunk_sx, 0, chunk_sz], [0, 0, chunk_sz],
            [0, section_sy, 0], [chunk_sx, section_sy, 0], [chunk_sx, section_sy, chunk_sz], [0, section_sy, chunk_sz]
        ], dtype=np.float32)
        
        base_edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ], dtype=np.int32)
        
        base_line_vertices = base_box[base_edges.flatten()] # (24, 3)
        
        # 计算偏移量: cx*sx, sy*sy, cz*sz
        offsets = np.zeros((len(sections_np), 3), dtype=np.float32)
        offsets[:, 0] = sections_np[:, 0] * chunk_sx
        offsets[:, 1] = sections_np[:, 1] * section_sy
        offsets[:, 2] = sections_np[:, 2] * chunk_sz
        
        # 批量生成线条: (N, 1, 3) + (1, 24, 3) -> (N, 24, 3) -> (-1, 3)
        all_lines = (offsets[:, np.newaxis, :] + base_line_vertices[np.newaxis, :, :]).reshape(-1, 3)
        
        # 绘制高亮框 (橙色)
        vis = Line(pos=all_lines, connect='segments', color=[1, 0.6, 0, 0.8], width=3, method='gl')
        self.view.add(vis)
        self.observation_visuals.append(vis)

    def update_agent_visuals(self, agent: Any, chunk_indices: Optional[torch.Tensor] = None):
        """更新Agent的可视化（先清除再重绘），可选叠加观测到的Sections"""
        self.clear_agent_visuals()
        self.visualize_agent(agent)
        if chunk_indices is not None:
            self.visualize_observations(agent, chunk_indices)

    def run(self):
        app.run()
