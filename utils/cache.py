import os
import torch
import time
import shutil
from typing import Dict, Any, Optional, Tuple, Union

class LFUCache:
    """
    Least Frequently Used (LFU) Cache
    针对GPU优化的双层缓存系统：
    L1: GPU显存 (活跃数据)
    L2: CPU内存 (所有已生成的历史数据)
    Disk: 单个文件持久化 (world_map.pt)
    """
    def __init__(self, capacity: int = 100, map_path: str = "map/world_map.pt", device: Union[str, torch.device] = 'cpu'):
        """
        Args:
            capacity: L1(GPU)缓存容量
            map_path: 地图持久化文件路径
            device: L1数据的目标设备
        """
        self.capacity = capacity
        self.device = device
        
        # 确保保存目录存在
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.map_path = os.path.join(base_path, map_path)
        save_dir = os.path.dirname(self.map_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # L1 Cache (GPU): (cx, cz) -> {'data': Tensor/TensorDict, 'freq': int, 'last_time': float}
        #   - 要求 data 尽量是 tensor 或 由 tensor 组成的字典（例如 Chunk.compress() 的返回值）
        #   - 允许额外的派生数据（如 vis_data），但这些不会被持久化到磁盘
        self.l1_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        
        # L2 Cache (CPU): (cx, cz) -> 纯 tensor / tensor 字典（适合直接 torch.save / torch.load）
        #   - 对于地图 Chunk，约定结构为: {'compressed_data': <tensor dict from Chunk.compress()>}
        self.l2_cache: Dict[Tuple[int, int], Any] = {}
        
        # 初始化时尝试加载持久化地图
        self._load_from_disk()

    def get(self, key: Tuple[int, int]) -> Optional[Any]:
        """
        获取数据：L1 -> L2 -> None
        返回的数据已经被移动到 self.device 上（如果需要的话），并存入 L1。
        """
        # 1. 检查 L1 (GPU / 目标 device)
        if key in self.l1_cache:
            self.l1_cache[key]['freq'] += 1
            self.l1_cache[key]['last_time'] = time.time()
            return self.l1_cache[key]['data']
            
        # 2. 检查 L2 (CPU)
        if key in self.l2_cache:
            try:
                # 从 CPU 加载到目标 device，并升级到 L1
                data_cpu = self.l2_cache[key]
                data_dev = self._move_data_to_device(data_cpu, self.device, clone=True)

                # 直接写入 L1（不改变结构，只改变 device）
                self._put_l1_only(key, data_dev)
                return data_dev
            except Exception as e:
                print(f"Error moving cached item {key} to device {self.device}: {e}")
                return None
        
        return None

    def put(self, key: Tuple[int, int], data: Any):
        """
        存入缓存：更新 L1（在 self.device 上）并同步到 L2（CPU tensor / tensor dict）。
        要求 data 尽量是 tensor 或 由 tensor 组成的结构；
        若含有额外派生字段（如 vis_data 的 numpy 数组），这些字段不会被写入 L2/Disk。
        """
        # 1. 确保写入 L1 的数据在目标 device 上
        data_dev = self._move_data_to_device(data, self.device, clone=True)
        self._put_l1_only(key, data_dev)

        # 2. 同步到 L2 (CPU, tensor-only)
        try:
            # 先深拷贝一份到 CPU
            data_cpu = self._move_data_to_device(data_dev, 'cpu', clone=True)

            # 只将「tensor 相关核心数据」写入 L2：
            # - 若是 {'compressed_data': <tensor dict>}，只保留这一项
            # - 否则直接整个结构（默认认为是 tensor / tensor dict）
            if isinstance(data_cpu, dict) and 'compressed_data' in data_cpu:
                compressed_cpu = self._move_data_to_device(data_cpu['compressed_data'], 'cpu', clone=True)
                self.l2_cache[key] = {'compressed_data': compressed_cpu}
            else:
                self.l2_cache[key] = data_cpu
        except Exception as e:
            print(f"Warning: Failed to cache key {key} to L2 (CPU): {e}")

    def _put_l1_only(self, key: Tuple[int, int], data_dev: Any):
        """仅更新 L1，不触碰 L2。内部辅助函数。"""
        if key in self.l1_cache:
            self.l1_cache[key]['data'] = data_dev
            self.l1_cache[key]['freq'] += 1
            self.l1_cache[key]['last_time'] = time.time()
        else:
            if len(self.l1_cache) >= self.capacity:
                self._evict()
            self.l1_cache[key] = {
                'data': data_dev,
                'freq': 1,
                'last_time': time.time()
            }

    def _evict(self):
        """L1 -> L2 驱逐逻辑"""
        if not self.l1_cache:
            return
            
        # 找到 LFU
        lfu_key = min(self.l1_cache.keys(), key=lambda k: (self.l1_cache[k]['freq'], self.l1_cache[k]['last_time']))
        
        # 从 L1 移除
        self.l1_cache.pop(lfu_key)
        # L2 中已经有了（在put时压缩存入），所以无需操作

    def save_to_disk(self):
        """将当前所有数据(L1+L2)持久化到单个文件"""
        print(f"正在保存地图数据到磁盘: {self.map_path}")
        start_time = time.time()
        
        # 为稳健起见，将 L1 中仍未同步的 key 同步到 L2，且仅写入 tensor 友好的核心数据
        for key, item in self.l1_cache.items():
            try:
                data_cpu = self._move_data_to_device(item['data'], 'cpu', clone=True)

                if isinstance(data_cpu, dict) and 'compressed_data' in data_cpu:
                    compressed_cpu = self._move_data_to_device(data_cpu['compressed_data'], 'cpu', clone=True)
                    self.l2_cache[key] = {'compressed_data': compressed_cpu}
                else:
                    self.l2_cache[key] = data_cpu
            except Exception as e:
                 print(f"Error syncing key {key} to L2 for save: {e}")
        
        if not self.l2_cache:
            print("没有数据需要保存。")
            return

        try:
            os.makedirs(os.path.dirname(self.map_path), exist_ok=True)
            torch.save(self.l2_cache, self.map_path)
            try:
                os.sync()
            except:
                pass
                
            print(f"地图保存成功: {self.map_path} (包含 {len(self.l2_cache)} 个区块, 耗时 {time.time()-start_time:.2f}s)")
            
            if os.path.exists(self.map_path):
                size_mb = os.path.getsize(self.map_path) / (1024 * 1024)
                print(f"验证成功: 文件存在，大小 {size_mb:.2f} MB")
            else:
                print(f"严重警告: torch.save 没有报错，但文件 {self.map_path} 不存在！")
                
        except Exception as e:
            print(f"地图保存失败: {e}")

    def _load_from_disk(self):
        """从磁盘加载全量地图到 L2"""
        if os.path.exists(self.map_path):
            try:
                print(f"正在加载地图: {self.map_path}...")
                loaded_data = torch.load(self.map_path, map_location='cpu', weights_only=False)
                if isinstance(loaded_data, dict):
                    self.l2_cache = loaded_data
                    print(f"已加载 {len(self.l2_cache)} 个区块到内存")
            except Exception as e:
                print(f"加载地图失败: {e}")
        else:
            print(f"地图文件不存在: {self.map_path}，将创建新地图。")

    def _move_data_to_device(self, data: Any, device: Union[str, torch.device], clone: bool = False) -> Any:
        """递归移动数据"""
        if hasattr(data, 'to'):
            if clone and hasattr(data, 'clone'):
                try:
                    return data.to(device).clone()
                except Exception:
                    return data.to(device)
            return data.to(device)
        elif isinstance(data, dict):
            return {k: self._move_data_to_device(v, device, clone) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._move_data_to_device(v, device, clone) for v in data]
        return data

    def clear_disk_cache(self):
        """删除地图文件"""
        if os.path.exists(self.map_path):
            try:
                os.remove(self.map_path)
                print(f"已删除地图文件: {self.map_path}")
            except Exception as e:
                print(f"删除地图文件失败: {e}")
