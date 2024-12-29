from models.vision_llm import RoboticVisionLLM
from utils.camera import Camera
import cv2
import numpy as np
import torch
from typing import Tuple, List
import time
import logging
from collections import deque
from PIL import Image
from contextlib import contextmanager
from torchvision import transforms

class RobotVisionSystem:
    def __init__(
        self,
        model_path: str = "meta-llama/Llama-3.2-1B",
        buffer_size: int = 5,
        fps: int = 30,
        uncertainty_threshold: float = 0.5,
        stages: List[int] = [224, 384, 512],
        adaptive_temp: bool = True,
        num_diffusion_steps: int = 20,
        online_batch_size: int = 8
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RoboticVisionLLM(model_path).to(self.device)
        self.model.eval()
        
        self.stages = stages
        self.current_stage = 0
        self.adaptive_temp = adaptive_temp
        self.base_temp = 1.0
        self.fps = fps
        self.uncertainty_threshold = uncertainty_threshold
        self.num_diffusion_steps = num_diffusion_steps
        
        self.action_buffer = deque(maxlen=buffer_size)
        self.uncertainty_buffer = deque(maxlen=buffer_size)
        self.online_batch_size = online_batch_size
        # self.feature_buffer = torch.zeros((self.online_batch_size, 768), device=self.device)
        # self.frame_buffer = torch.zeros((self.online_batch_size, 3, 224, 224), device=self.device)
        self.buffer_idx = 0
        self.perf_metrics = {
            'inference_time': deque(maxlen=100),
            'processing_time': deque(maxlen=100)
        }
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.num_diffusion_steps = num_diffusion_steps
        self.min_diffusion_steps = max(5, num_diffusion_steps // 4)
        
        # self.feature_buffer = []
        # self.frame_buffer = []
        self.feature_buffer = deque(maxlen=online_batch_size)
        self.frame_buffer = deque(maxlen=online_batch_size)
        self.action_buffer = deque(maxlen=buffer_size)
        self.uncertainty_buffer = deque(maxlen=buffer_size)
        self.max_buffer_size = online_batch_size
        
        self.camera = Camera()
        self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    @contextmanager
    def timer(self, name: str):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.perf_metrics[name].append(elapsed)
        
    def update_stage(self, uncertainty: float):
        """Progressive stage management based on uncertainty"""
        if uncertainty > 0.8 and self.current_stage > 0:
            self.current_stage -= 1
        elif uncertainty < 0.2 and self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            
    def calculate_diffusion_steps(self, uncertainty: float) -> int:
        if not uncertainty:
            return self.num_diffusion_steps
        
        steps = int(self.num_diffusion_steps * (1 - uncertainty))
        return max(self.min_diffusion_steps, min(steps, self.num_diffusion_steps))

            
    def adaptive_temperature(self, uncertainty: float) -> float:
        """Adaptive temperature scaling based on uncertainty"""
        if not self.adaptive_temp:
            return self.base_temp
        return self.base_temp * (1 + uncertainty)
        
    def smooth_predictions(self, actions: np.ndarray, uncertainty: np.ndarray, features: torch.Tensor):
        self.feature_buffer.append(features)
        self.action_buffer.append(actions)
        self.uncertainty_buffer.append(uncertainty)
        
        weights = np.exp(-np.array([u.mean() for u in self.uncertainty_buffer]))
        weights = weights / weights.sum()
        
        smoothed_actions = np.average(
            np.array(self.action_buffer),
            weights=weights,
            axis=0
        )
        smoothed_uncertainty = np.mean(self.uncertainty_buffer, axis=0)
        
        return smoothed_actions, smoothed_uncertainty
           
    def online_update(self):
        if len(self.feature_buffer) < 2:
            return
            
        try:
            self.model.train()
            
            features = list(self.feature_buffer)
            curr_features = torch.stack([f.detach().clone().requires_grad_(True) for f in features[1:]])
            prev_features = torch.stack([f.detach() for f in features[:-1]])
            
            self.model.online_optimizer.zero_grad()
            
            loss = self.model.compute_ssl_loss(curr_features, prev_features)
            
            loss.backward(retain_graph=True)
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.model.online_optimizer.step()
            
            self.model._momentum_update()
            
            for f in self.feature_buffer:
                if f.grad is not None:
                    f.grad.zero_()
            
        except Exception as e:
            self.logger.error(f"Online update failed: {str(e)}")
        finally:
            self.model.eval()

    def update_buffers(self, frame_tensor: torch.Tensor, features: torch.Tensor):
        """Store features with proper gradient handling"""
        features = features.view(1, -1)
        
        self.frame_buffer.append(frame_tensor.detach().clone())
        
        feature_copy = features.detach().clone().requires_grad_(True) 
        self.feature_buffer.append(feature_copy)
                        
    def get_training_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get properly formatted training batch"""
        if len(self.feature_buffer) < self.max_buffer_size:
            return None, None
            
        curr_features = torch.stack(self.feature_buffer[1:])
        prev_features = torch.stack(self.feature_buffer[:-1])
        
        return curr_features, prev_features
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess camera frame for model input"""
        if frame is None or frame.size == 0:
            raise ValueError("Invalid input frame")
            
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            pil_image = Image.fromarray(frame_rgb)
            
            tensor = self.transform(pil_image).unsqueeze(0)
            
            return tensor.to(self.device)
            
        except Exception as e:
            self.logger.error(f"Frame preprocessing failed: {str(e)}")
            raise RuntimeError(f"Frame preprocessing failed: {str(e)}")
        
    def visualize_predictions(
        self,
        frame: np.ndarray,
        actions: np.ndarray,
        uncertainty: np.ndarray,
        features: np.ndarray
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        
        center = (w // 2, h // 2)
        scale = 100
        
        for i, (color, factor) in enumerate([
            ((0,0,255), [1,0,0]),  # x - red
            ((0,255,0), [0,1,0]),  # y - green
            ((255,0,0), [0,0,1])   # z - blue
        ]):
            end_point = (
                int(center[0] + actions[i] * scale * factor[0]),
                int(center[1] + actions[i] * scale * factor[1])
            )
            thickness = max(1, int((1 - uncertainty[i]) * 5))
            cv2.arrowedLine(frame, center, end_point, color, thickness)
            
        status_text = [
            f"Stage: {self.current_stage}/{len(self.stages)-1}",
            f"Uncertainty: {uncertainty.mean():.3f}",
            f"Temp: {self.adaptive_temperature(uncertainty.mean()):.2f}",
            f"FPS: {1.0/np.mean(self.perf_metrics['processing_time']):.1f}"
        ]
        
        for i, text in enumerate(status_text):
            cv2.putText(
                frame,
                text,
                (10, 30 + i*30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            
        return frame
        
    def run(self):
        self.logger.info("Starting Robot Vision System...")
        frame_time = 1.0 / self.fps

        try:
            while True:
                with self.timer('processing_time'):
                    ret, frame = self.camera.read_frame()
                    if not ret:
                        self.logger.error("Failed to read camera frame")
                        continue
                        
                    with self.timer('inference_time'):
                        try:
                            input_tensor = self.preprocess_frame(frame)
                            
                            current_uncertainty = (self.uncertainty_buffer[-1].mean() 
                                                if len(self.uncertainty_buffer) > 0 
                                                else 0.5)
                            
                            current_steps = self.calculate_diffusion_steps(current_uncertainty)
                            current_temp = self.adaptive_temperature(current_uncertainty)

                            actions, uncertainty, features = self.model(
                                input_tensor,
                                stage_idx=self.current_stage,
                                temp=current_temp,
                                num_steps=current_steps
                            )
                            
                            self.feature_buffer.append(features)
            
                            actions_np = actions.cpu().detach().numpy()[0]
                            uncertainty_np = uncertainty.cpu().detach().numpy()[0]
                            
                            smoothed_actions, smoothed_uncertainty = self.smooth_predictions(
                                actions_np, uncertainty_np, features
                            )
                            
                            self.online_update()
                            
                            self.update_stage(smoothed_uncertainty.mean())
                            
                            vis_frame = self.visualize_predictions(
                                frame.copy(),
                                smoothed_actions,
                                smoothed_uncertainty,
                                features.cpu().detach().numpy()[0]
                            )
                            cv2.imshow('Robot Vision', vis_frame)
                            
                            if smoothed_uncertainty.mean() > self.uncertainty_threshold:
                                self.logger.warning(
                                    f"High uncertainty detected: {smoothed_uncertainty.mean():.3f}"
                                )
                                
                        except Exception as e:
                            self.logger.error(f"Inference error: {str(e)}")
                            continue
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        except KeyboardInterrupt:
            self.logger.info("Gracefully shutting down...")
            
        finally:
            self.camera.release()
            cv2.destroyAllWindows()
            
if __name__ == "__main__":
    system = RobotVisionSystem()
    system.run()