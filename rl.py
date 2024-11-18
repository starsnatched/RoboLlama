from models.vision_llm import RoboticVisionLLM
from utils.env_wrapper import HumanoidPOVEnv
import torch
import numpy as np
import logging
from collections import deque
import time
import gymnasium as gym
from typing import Deque

class RLTrainer:
    def __init__(
        self,
        model_path: str = "llm",
        buffer_size: int = 5,
        uncertainty_threshold: float = 0.5,
        num_diffusion_steps: int = 20,
        episodes: int = 1000,
        render: bool = True
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RoboticVisionLLM(model_path).to(self.device)
        self.model.train()
        
        self.env = HumanoidPOVEnv(render_mode="rgb_array")
        if render:
            self.render_env = gym.make("Humanoid-v5", render_mode="human")
        self.render = render
        self.episodes = episodes
        
        self.uncertainty_threshold = uncertainty_threshold
        self.num_diffusion_steps = num_diffusion_steps
        
        self.action_buffer: Deque = deque(maxlen=buffer_size)
        self.uncertainty_buffer: Deque = deque(maxlen=buffer_size)
        self.reward_history: Deque = deque(maxlen=100)
        
        self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def train_episode(self) -> float:
        obs, _ = self.env.reset()
        if self.render:
            self.render_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            if self.render:
                self.render_env.render()
                time.sleep(0.01)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                actions, uncertainty, _ = self.model(
                    obs_tensor,
                    stage_idx=0,
                    temp=1.0,
                    num_steps=self.num_diffusion_steps
                )
            
            actions = actions.cpu().numpy()[0]
            uncertainty = uncertainty.cpu().numpy()[0]
            
            self.action_buffer.append(actions)
            self.uncertainty_buffer.append(uncertainty)
            
            obs, reward, terminated, truncated, _ = self.env.step(actions)
            if self.render:
                self.render_env.step(actions)

            done = terminated or truncated
            episode_reward += reward
            
            if uncertainty.mean() > self.uncertainty_threshold:
                self.logger.warning(f"High uncertainty: {uncertainty.mean():.3f}")
                
        return episode_reward
        
    def train(self):
        self.logger.info("Starting training...")
        
        try:
            for episode in range(self.episodes):
                start_time = time.time()
                
                episode_reward = self.train_episode()
                self.reward_history.append(episode_reward)
                
                episode_time = time.time() - start_time
                
                self.logger.info(
                    f"Episode {episode + 1}/{self.episodes} - "
                    f"Reward: {episode_reward:.2f} - "
                    f"Avg Reward: {np.mean(self.reward_history):.2f} - "
                    f"Time: {episode_time:.2f}s"
                )
                
        except KeyboardInterrupt:
            self.logger.info("Training interrupted...")
            
        finally:
            self.env.close()
            
if __name__ == "__main__":
    trainer = RLTrainer(render=True)
    trainer.train()