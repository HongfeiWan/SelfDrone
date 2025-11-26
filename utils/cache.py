import os
import torch
import time
import shutil
from typing import Dict, Any, Optional, Tuple, Union

class LFUCache:
    """
    Least Frequently Used (LFU) Cache
    当缓存满时，移除访问频率最低的元素，并将其保存到磁盘。
    针对GPU优化：驱逐时移至CPU，加载时移回GPU。
    """
    def __init__(self, capacity: int = 100, save_dir: str = "map/chunks_cache", device: Union[str, torch.device] = 'cpu'):
        """
        Args:
            capacity: 缓存容量
            save_dir: 驱逐数据的保存目录
            device: 数据加载时的目标设备 (通常是 'cuda')
        """
        self.capacity = capacity
        self.device = device
        
        # 确保保存目录存在 (相对于项目根目录)
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.save_dir = os.path.join(base_path, save_dir)
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # 缓存结构: (cx, cz) -> {'data': Any, 'freq': int, 'last_time': float}
        self.cache: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def get(self, key: Tuple[int, int]) -> Optional[Any]:
        """获取缓存项，如果不在内存但在磁盘，则加载并移至目标设备"""
        # 1. 检查内存
        if key in self.cache:
            self.cache[key]['freq'] += 1
            self.cache[key]['last_time'] = time.time()
            return self.cache[key]['data']
            
        # 2. 检查磁盘
        filename = self._get_filename(key)
        if os.path.exists(filename):
            try:
                # 直接加载到目标设备
                data = torch.load(filename, map_location=self.device)
                
                # 如果数据包含对象且有 .to() 方法（如 Chunk），确保其内部状态也更新
                # 对于字典结构（如 {'chunk': chunk, ...}），尝试递归处理
                self._move_data_to_device(data, self.device)
                
                # 重新放入内存
                self.put(key, data)
                return data
            except Exception as e:
                print(f"Failed to load chunk {key} from disk: {e}")
                return None
        
        return None

    def put(self, key: Tuple[int, int], data: Any):
        """存入缓存"""
        if key in self.cache:
            self.cache[key]['data'] = data
            self.cache[key]['freq'] += 1
            self.cache[key]['last_time'] = time.time()
            return

        # 如果满了，进行驱逐
        if len(self.cache) >= self.capacity:
            self._evict()
            
        self.cache[key] = {
            'data': data,
            'freq': 1,
            'last_time': time.time()
        }

    def _evict(self):
        """驱逐逻辑：找到频率最低的 -> 移至CPU -> 保存到磁盘"""
        if not self.cache:
            return
            
        # 排序键：(频率, 上次访问时间)
        lfu_key = min(self.cache.keys(), key=lambda k: (self.cache[k]['freq'], self.cache[k]['last_time']))
        
        item = self.cache.pop(lfu_key)
        data = item['data']
        
        # 移至 CPU 以释放显存
        data_cpu = self._move_data_to_device(data, 'cpu', clone=False)
        
        # 保存到磁盘
        filename = self._get_filename(lfu_key)
        try:
            torch.save(data_cpu, filename)
        except Exception as e:
            print(f"Failed to save evicted chunk {lfu_key}: {e}")

    def _move_data_to_device(self, data: Any, device: Union[str, torch.device], clone: bool = False) -> Any:
        """辅助函数：递归将数据移动到指定设备"""
        if hasattr(data, 'to'):
            return data.to(device)
        elif isinstance(data, dict):
            # 对字典中的每个值尝试移动，返回新字典或修改原字典
            # 为了安全起见，这里修改原字典（如果不需要保留原件）或者创建新的
            # 考虑到缓存场景，我们通常希望修改对象本身
            for k, v in data.items():
                data[k] = self._move_data_to_device(v, device, clone)
            return data
        elif isinstance(data, list):
            return [self._move_data_to_device(v, device, clone) for v in data]
        return data

    def _get_filename(self, key: Tuple[int, int]) -> str:
        return os.path.join(self.save_dir, f"chunk_{key[0]}_{key[1]}.pt")

    def clear_disk_cache(self):
        """清空磁盘缓存文件"""
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
            os.makedirs(self.save_dir)
