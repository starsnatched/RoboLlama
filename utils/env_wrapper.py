import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import cv2
from typing import Tuple, Any

class HumanoidPOVEnv(gym.Wrapper):
    def __init__(self, render_size: Tuple[int, int] = (224, 224), render_mode: str = "rgb_array"):
        env = gym.make("Humanoid-v5", render_mode=render_mode)
        super().__init__(env)
        self.render_size = render_size
        
        self.observation_space = Box(
            low=0, 
            high=255,
            shape=(3, *render_size),
            dtype=np.uint8
        )
        
    def get_pov(self) -> np.ndarray:
        rgb_array = self.env.render()
        if rgb_array is None:
            raise RuntimeError("Failed to get RGB array from environment")
            
        if not isinstance(rgb_array, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(rgb_array)}")
            
        if len(rgb_array.shape) != 3:
            raise ValueError(f"Expected 3D array (H,W,C), got shape {rgb_array.shape}")
            
        resized = cv2.resize(rgb_array, self.render_size)
        return np.transpose(resized, (2, 0, 1))
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        pov = self.get_pov()
        return pov, info
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        _, reward, terminated, truncated, info = self.env.step(action)
        pov = self.get_pov()
        return pov, reward, terminated, truncated, info