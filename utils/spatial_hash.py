import torch
from typing import Tuple, Optional, Dict, Any

class SpatialHash3D:
    """
    一个统一的、基于GPU加速的批量化3D空间哈希（无限网格版本），用于高效查询几何图元。
    
    特点：
    1. 无边界限制：使用哈希映射支持无限世界。
    2. 自动配置：直接从Chunk配置推导网格尺寸。
    3. 功能：支持静态几何体索引和动态物体临近对查询。
    """
    def __init__(self, chunk_class: Any, table_size: int = 1000000, device: torch.device = torch.device('cpu')):
        """
        初始化 SpatialHash3D。
        
        Args:
            chunk_class: 包含 CHUNK_SIZE_X 和 VOXEL_SIZE 等配置的 Chunk 类或配置对象。
                         用于自动计算 cell_size (即 Section 的物理尺寸)。
            table_size (int): 哈希表的大小（桶的数量）。默认100万，适用于中大规模场景。
                              越大哈希冲突越少，但显存占用稍高。
            device (torch.device): 用于存储张量的设备。
        """
        # 自动推导 cell_size (Section 物理尺寸 = Chunk宽 * Voxel大小)
        self.cell_size = float(chunk_class.CHUNK_SIZE_X * chunk_class.VOXEL_SIZE)
        self.table_size = table_size
        self.device = device
        # 用于哈希计算的大质数
        self.primes = torch.tensor([73856093, 19349663, 83492791], dtype=torch.long, device=device)
        # 用于静态几何体索引的属性 (CSR格式)
        self.static_sorted_items = torch.empty((0,), dtype=torch.long, device=self.device)
        self.static_cell_starts = torch.empty((0,), dtype=torch.long, device=self.device)

    def get_grid_coords(self, points: torch.Tensor) -> torch.Tensor:
        """将世界坐标转换为3D网格整数坐标 (ix, iy, iz)。"""
        # points: (N, 3)
        return torch.floor(points / self.cell_size).long()

    def _get_hash_idx(self, grid_coords: torch.Tensor) -> torch.Tensor:
        """
        使用空间哈希函数将3D网格坐标映射到哈希表索引。
        Hash(x,y,z) = (x*p1 ^ y*p2 ^ z*p3) % table_size
        """
        # 确保输入是 LongTensor
        coords = grid_coords.long()
        
        # 计算哈希值 (利用广播机制)
        # 注意：负数坐标的取模在 Python/PyTorch 中结果为正，符合预期
        hash_val = (coords[:, 0] * self.primes[0]) ^ \
                   (coords[:, 1] * self.primes[1]) ^ \
                   (coords[:, 2] * self.primes[2])
                   
        return hash_val % self.table_size

    def build_static_index(self, static_items_bounds: torch.Tensor):
        """
        为静态几何体（如3D包围盒）构建一个持久化的哈希索引。
        Args:
            static_items_bounds (torch.Tensor): 静态物体的AABB，形状 (num_items, 2, 3) for (min, max).
        """
        num_items = static_items_bounds.shape[0]
        if num_items == 0:
            self.static_cell_starts = torch.zeros(self.table_size + 1, dtype=torch.long, device=self.device)
            return

        item_min_bounds = static_items_bounds[:, 0] # (N, 3)
        item_max_bounds = static_items_bounds[:, 1] # (N, 3)
        
        start_cells = self.get_grid_coords(item_min_bounds)
        end_cells = self.get_grid_coords(item_max_bounds)

        item_ids = torch.arange(num_items, device=self.device)
        all_pairs = []
        
        # CPU 循环构建对
        # 对于无限网格，我们直接遍历 min 到 max 的整数范围
        # 使用 tolist() 转换为 Python list[list[int]]，避免 Tensor 在 range() 中报错，并加速访问
        start_cells_list = start_cells.cpu().tolist()
        end_cells_list = end_cells.cpu().tolist()
        
        for i in range(num_items):
            sx, sy, sz = start_cells_list[i]
            ex, ey, ez = end_cells_list[i]
            
            for x in range(sx, ex + 1):
                for y in range(sy, ey + 1):
                    for z in range(sz, ez + 1):
                        # 这里临时存储 grid 坐标，稍后统一 hash
                        all_pairs.append([item_ids[i].item(), x, y, z])
        
        if not all_pairs:
            item_cell_pairs = torch.empty((0, 2), dtype=torch.long, device=self.device)
        else:
            # 转换为 Tensor
            pairs_tensor = torch.tensor(all_pairs, dtype=torch.long, device=self.device)
            # 计算 Hash
            grid_coords = pairs_tensor[:, 1:] # (N, 3)
            hash_indices = self._get_hash_idx(grid_coords)
            # 组合 [item_id, hash_idx]
            item_cell_pairs = torch.stack([pairs_tensor[:, 0], hash_indices], dim=1)

        if item_cell_pairs.numel() == 0:
            self.static_sorted_items = torch.empty(0, dtype=torch.long, device=self.device)
            self.static_cell_starts = torch.zeros(self.table_size + 1, dtype=torch.long, device=self.device)
            return

        # 按 hash_idx 排序
        sorted_pairs = item_cell_pairs[item_cell_pairs[:, 1].argsort()]
        self.static_sorted_items = sorted_pairs[:, 0].contiguous()
        
        # 构建 CSR 索引指针
        self.static_cell_starts = torch.zeros(self.table_size + 1, dtype=torch.long, device=self.device)
        unique_hashes, counts = torch.unique_consecutive(sorted_pairs[:, 1], return_counts=True)
        
        self.static_cell_starts[unique_hashes + 1] = counts
        self.static_cell_starts = self.static_cell_starts.cumsum_(0)

    def query_points(self, points: torch.Tensor) -> torch.Tensor:
        """
        批量查询点，返回每个点对应的候选静态物体ID。
        注意：由于哈希冲突，返回的可能是无关物体（False Positive），通常需要进一步精确检测。
        Args:
            points: (N, 3) 查询点坐标
        Returns:
            (M, 2) tensor, 每行是 [point_idx, item_idx]
        """
        if self.static_sorted_items.numel() == 0:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)
            
        grid_coords = self.get_grid_coords(points)
        hash_indices = self._get_hash_idx(grid_coords)
        
        starts = self.static_cell_starts[hash_indices]
        ends = self.static_cell_starts[hash_indices + 1]
        num_candidates_per_point = ends - starts
        
        if num_candidates_per_point.sum() == 0:
            return torch.empty((0, 2), dtype=torch.long, device=self.device)
            
        point_indices_out = torch.arange(len(points), device=self.device).repeat_interleave(num_candidates_per_point)
        
        # 向量化获取 item_indices
        max_candidates = num_candidates_per_point.max().item()
        if max_candidates == 0:
             return torch.empty((0, 2), dtype=torch.long, device=self.device)

        offsets = torch.arange(max_candidates, device=self.device).unsqueeze(0)
        starts_expanded = starts.unsqueeze(1) + offsets
        valid_mask = offsets < num_candidates_per_point.unsqueeze(1)
        valid_starts = starts_expanded[valid_mask]
        item_indices_out = self.static_sorted_items[valid_starts]

        return torch.stack([point_indices_out, item_indices_out], dim=1)

    def query_dynamic_pairs(self, B: int, M: int, active_mask: torch.Tensor,
                            verts_t0: torch.Tensor, verts_t1: torch.Tensor,
                            max_neighbors: int, debug: bool = False, debug_env_idx: int = 0) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        为一批动态物体执行高效的临近对查询 (3D无限哈希版)。
        Args:
            verts_t0, verts_t1: (B, M, K, 3) 顶点的世界坐标
        """
        # 1. 计算每个Agent的AABB
        all_verts = torch.cat([verts_t0, verts_t1], dim=-2) # (B, M, 2K, 3)
        min_coords, _ = torch.min(all_verts, dim=-2) # (B, M, 3)
        max_coords, _ = torch.max(all_verts, dim=-2) # (B, M, 3)
        
        # 2. 获取涉及的网格范围
        min_grid = self.get_grid_coords(min_coords.view(-1, 3)).view(B, M, 3)
        max_grid = self.get_grid_coords(max_coords.view(-1, 3)).view(B, M, 3)
        
        span = (max_grid - min_grid + 1).clamp(min=1) # (B, M, 3)
        max_cells_per_agent = (span[..., 0] * span[..., 1] * span[..., 2]).max().item()

        # 3. 生成所有覆盖的 cell hash indices
        span_x = span[..., 0].view(-1)
        span_y = span[..., 1].view(-1)
        span_z = span[..., 2].view(-1)
        
        ranges = torch.arange(max_cells_per_agent, device=self.device)
        
        span_z_exp = span_z.unsqueeze(1)
        span_y_exp = span_y.unsqueeze(1)
        
        # 广播计算相对坐标
        dz = ranges % span_z_exp
        dy = (ranges // span_z_exp) % span_y_exp
        dx = ranges // (span_z_exp * span_y_exp)
        
        valid_mask = (dx < span_x.unsqueeze(1)) & active_mask.view(-1, 1)
        
        # 计算绝对网格坐标
        grid_x = min_grid[..., 0].view(-1, 1) + dx
        grid_y = min_grid[..., 1].view(-1, 1) + dy
        grid_z = min_grid[..., 2].view(-1, 1) + dz
        
        # 合并为 (TotalPoints, 3)
        grid_coords_flat = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)
        
        # 计算哈希值
        hash_indices_flat = self._get_hash_idx(grid_coords_flat)
        
        # 过滤无效项
        valid_flat_mask = valid_mask.flatten()
        cell_indices = hash_indices_flat[valid_flat_mask]
        
        batch_indices = torch.arange(B, device=self.device).view(B, 1).expand(-1, M).flatten().unsqueeze(-1).expand(-1, max_cells_per_agent).flatten()[valid_flat_mask]
        agent_indices = torch.arange(M, device=self.device).view(1, M).expand(B, -1).flatten().unsqueeze(-1).expand(-1, max_cells_per_agent).flatten()[valid_flat_mask]

        debug_info = None
        
        # 后续逻辑与 2D 版本完全一致（基于 (Batch, CellHash) 分组）
        # 注意：这里使用 cell_indices (其实是 hash indices) 作为分组键
        
        num_neighbors = min(max_neighbors, M - 1)
        if num_neighbors <= 0:
            return torch.full((B, M, 0), -1, dtype=torch.long, device=self.device), debug_info

        # 1) 分组
        group_keys = batch_indices.long() * self.table_size + cell_indices.long()
        sort_idx = torch.argsort(group_keys)
        sorted_keys = group_keys[sort_idx]
        sorted_batches = batch_indices[sort_idx]
        sorted_agents = agent_indices[sort_idx]

        if sorted_keys.numel() == 0:
            return torch.full((B, M, num_neighbors), -1, dtype=torch.long, device=self.device), debug_info

        all_keys, all_counts = torch.unique_consecutive(sorted_keys, return_counts=True)
        valid_groups_mask = all_counts >= 2
        if not valid_groups_mask.any():
            return torch.full((B, M, num_neighbors), -1, dtype=torch.long, device=self.device), debug_info

        group_counts = all_counts[valid_groups_mask]
        group_starts_all = torch.cumsum(torch.nn.functional.pad(all_counts, (1, 0)), dim=0)[:-1]
        group_starts = group_starts_all[valid_groups_mask]

        # 2) 生成对
        max_group_size = int(group_counts.max().item())
        tri = torch.triu_indices(max_group_size, max_group_size, offset=1, device=self.device)
        tri_i, tri_j = tri[0], tri[1]
        valid_pair_mask = (tri_i.unsqueeze(0) < group_counts.unsqueeze(1)) & (tri_j.unsqueeze(0) < group_counts.unsqueeze(1))
        
        if not valid_pair_mask.any():
            return torch.full((B, M, num_neighbors), -1, dtype=torch.long, device=self.device), debug_info

        pos_i = group_starts.unsqueeze(1) + tri_i.unsqueeze(0)
        pos_j = group_starts.unsqueeze(1) + tri_j.unsqueeze(0)
        flat_mask = valid_pair_mask.flatten()
        flat_i = pos_i.flatten()[flat_mask]
        flat_j = pos_j.flatten()[flat_mask]

        pair_batches = sorted_batches[flat_i]
        agents_i = sorted_agents[flat_i]
        agents_j = sorted_agents[flat_j]

        # 3) 去重
        a_min = torch.minimum(agents_i, agents_j)
        a_max = torch.maximum(agents_i, agents_j)
        pair_key = pair_batches.long() * (M * M) + a_min.long() * M + a_max.long()
        
        perm_pairs = torch.argsort(pair_key)
        sorted_pair_key = pair_key[perm_pairs]
        _, counts_pairs = torch.unique_consecutive(sorted_pair_key, return_counts=True)
        starts_pairs = torch.cumsum(torch.nn.functional.pad(counts_pairs, (1, 0)), dim=0)[:-1]
        unique_idx_sorted = perm_pairs[starts_pairs]
        
        pair_batches = pair_batches[unique_idx_sorted]
        a_min = a_min[unique_idx_sorted]
        a_max = a_max[unique_idx_sorted]

        # 4) 输出
        neigh_batches = torch.cat([pair_batches, pair_batches], dim=0)
        src_agents = torch.cat([a_min, a_max], dim=0)
        dst_agents = torch.cat([a_max, a_min], dim=0)

        # 5) Top-K 选择
        group_id = neigh_batches.long() * M + src_agents.long()
        lex_key = group_id * M + dst_agents.long()
        perm = torch.argsort(lex_key)
        gid_sorted = group_id[perm]
        dst_sorted = dst_agents[perm]

        uniq_gid, gid_counts = torch.unique_consecutive(gid_sorted, return_counts=True)
        gid_starts = torch.cumsum(torch.nn.functional.pad(gid_counts, (1, 0)), dim=0)[:-1]
        repeated_starts = gid_starts.repeat_interleave(gid_counts)
        positions = torch.arange(dst_sorted.numel(), device=self.device) - repeated_starts
        keep = positions < num_neighbors

        sel_gid = gid_sorted[keep]
        sel_pos = positions[keep].long()
        sel_dst = dst_sorted[keep]

        candidate_pairs = torch.full((B, M, num_neighbors), -1, dtype=torch.long, device=self.device)
        sel_batch = sel_gid // M
        sel_src = sel_gid % M
        candidate_pairs[sel_batch, sel_src, sel_pos] = sel_dst

        return candidate_pairs, debug_info
