import numpy as np
from typing import Dict, Any, Optional
from vispy import app

class KeyboardController:
    """
    键盘控制器：用于控制Agent的移动
    """
    def __init__(self, move_speed: float = 0.5):
        """
        Args:
            move_speed: 移动速度 (米/次)
        """
        self.move_speed = move_speed
        self.keys_pressed = set()
        self.agent = None
        self.visualizer = None
        self.camera_follow = False
        
        # 用于chunk更新
        self.chunk_class = None
        self.horizon = None
        self.device = None
        self.last_chunk_idx = None
        # 当前视野内的 Chunk 索引张量 (N, 2)，由 visualizer.update_chunks 返回
        self.visible_chunk_indices = None
        
    def bind_agent(self, agent: Any):
        """绑定要控制的Agent"""
        self.agent = agent
        
    def bind_visualizer(self, visualizer: Any, camera_follow: bool = True):
        """
        绑定可视化器，以便更新画面
        Args:
            visualizer: MapVisualizer实例
            camera_follow: 是否让相机跟随Agent
        """
        self.visualizer = visualizer
        self.camera_follow = camera_follow
        
    def bind_chunk_updater(self, chunk_class: Any, horizon: Any, device: Any):
        """绑定Chunk更新所需的参数"""
        self.chunk_class = chunk_class
        self.horizon = horizon
        self.device = device
        if self.agent and chunk_class:
            self.last_chunk_idx = chunk_class.world_to_chunk_idx(self.agent.x, self.agent.z)
    
    def on_key_press(self, event):
        """记录按下的键"""
        if hasattr(event, 'key'):
            self.keys_pressed.add(event.key.name)
            self._process_movement()

    def on_key_release(self, event):
        """移除释放的键"""
        if hasattr(event, 'key') and event.key.name in self.keys_pressed:
            self.keys_pressed.remove(event.key.name)

    def _process_movement(self):
        """根据按键状态更新Agent位置"""
        if self.agent is None:
            return

        dx = 0.0
        dz = 0.0
        
        # 上下控制 x 轴 (上为正，下为负)
        if 'Up' in self.keys_pressed:
            dx += self.move_speed
        if 'Down' in self.keys_pressed:
            dx -= self.move_speed
            
        # 左右控制 z 轴 (右为正，左为负)
        if 'Right' in self.keys_pressed:
            dz += self.move_speed
        if 'Left' in self.keys_pressed:
            dz -= self.move_speed
            
        if dx != 0.0 or dz != 0.0:
            new_x = self.agent.x + dx
            new_z = self.agent.z + dz
            
            # 更新Agent位置
            if hasattr(self.agent, 'update_position'):
                self.agent.update_position(new_x, self.agent.y, new_z)
            else:
                self.agent.x = new_x
                self.agent.z = new_z
            
            print(f"Agent移动至: ({self.agent.x:.2f}, {self.agent.y:.2f}, {self.agent.z:.2f})")
            
            # 更新可视化
            if self.visualizer:
                # 1. 更新Chunk (如果跨越了Chunk边界)
                if self.chunk_class and self.horizon and self.device:
                    current_chunk_idx = self.chunk_class.world_to_chunk_idx(self.agent.x, self.agent.z)
                    if current_chunk_idx != self.last_chunk_idx:
                        print(f"Agent跨越Chunk边界: {self.last_chunk_idx} -> {current_chunk_idx}，更新地图...")
                        self.visible_chunk_indices = self.visualizer.update_chunks(
                            self.agent.x, self.agent.z, 
                            self.chunk_class, self.horizon, self.device
                        )
                        self.last_chunk_idx = current_chunk_idx

                # 2. 更新Agent本身的可视化（叠加当前可见 Section）
                self.visualizer.update_agent_visuals(self.agent, self.visible_chunk_indices)
                
                # 3. 相机跟随
                if self.camera_follow:
                    self.visualizer.set_camera_target(np.array([self.agent.x, self.agent.y, self.agent.z]))
                
                # 4. 触发重绘
                self.visualizer.canvas.update()
