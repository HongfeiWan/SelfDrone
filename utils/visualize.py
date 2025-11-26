import numpy as np
import torch
from typing import List, Tuple, Optional, Any, Union, Dict
from vispy import app, scene
from vispy.scene.visuals import Line, Mesh, Markers
from vispy.geometry import MeshData
from utils.cache import LFUCache

class MapVisualizer:
    """
    通用地图/Agent可视化工具
    """
    def __init__(self, title: str = "Map Visualization", size: Tuple[int, int] = (1200, 900), bg_color: str = 'white', device: Union[str, torch.device] = 'cpu'):
        self.canvas = scene.SceneCanvas(keys='interactive', show=True, size=size, bgcolor=bg_color, title=title)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(up='y', distance=60)
        
        # 存储可视化对象引用
        self.chunk_visuals = []
        self.agent_visuals = []
        
        # Chunk缓存 (传入device以便加载时自动移回GPU)
        self.chunk_cache = LFUCache(capacity=100, save_dir="map/chunks_cache", device=device)

    def set_camera_target(self, target_pos: np.ndarray, distance: Optional[float] = None):
        """设置相机跟随的目标中心"""
        self.view.camera.center = target_pos
        if distance is not None:
            self.view.camera.distance = distance
            
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

    def update_chunks(self, agent_x: float, agent_z: float, ChunkClass: Any, horizon: Any, device: Any) -> List[Any]:
        """
        根据Agent位置和Horizon设置动态加载并更新可视化的chunks
        Args:
            agent_x: Agent的x坐标
            agent_z: Agent的z坐标
            ChunkClass: Chunk类引用，必须包含 world_to_chunk_idx 静态方法和构造函数
            horizon: 视野配置或半径
            device: torch device
        Returns:
            List[Chunk]: 当前渲染的chunks列表
        """
        # 清除旧的Chunks可视化
        for visual in self.chunk_visuals:
            visual.parent = None
        self.chunk_visuals = []
        
        # 计算需要的chunks
        chunk_x, chunk_z = ChunkClass.world_to_chunk_idx(agent_x, agent_z)
        horizon_chunks = horizon.get("chunk_radius", 1) if isinstance(horizon, dict) else (horizon or 1)
        chunks_to_render = []
        vis_data_list = []
        
        for dx in range(-horizon_chunks, horizon_chunks + 1):
            for dz in range(-horizon_chunks, horizon_chunks + 1):
                cx = chunk_x + dx
                cz = chunk_z + dz
                # 尝试从缓存获取
                cached_item = self.chunk_cache.get((cx, cz))
                if cached_item:
                    chunk = cached_item['chunk']
                    vis_data = cached_item['vis_data']
                else:
                    # 创建新Chunk
                    chunk = ChunkClass(cx, cz, device)
                    # 计算可视化数据
                    vis_data = self._compute_chunk_vis_data(chunk)
                    # 存入缓存
                    self.chunk_cache.put((cx, cz), {'chunk': chunk, 'vis_data': vis_data})
                chunks_to_render.append(chunk)
                vis_data_list.append(vis_data)
        print(f"加载/渲染 {len(chunks_to_render)} 个 chunk，范围半径={horizon_chunks}")
        # 重新可视化
        self.visualize_chunks(vis_data_list)
        return chunks_to_render

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

    def clear_agent_visuals(self):
        """清除所有Agent相关的可视化元素"""
        for visual in self.agent_visuals:
            visual.parent = None
        self.agent_visuals = []

    def update_agent_visuals(self, agent: Any):
        """更新Agent的可视化（先清除再重绘）"""
        self.clear_agent_visuals()
        self.visualize_agent(agent)

    def run(self):
        app.run()
