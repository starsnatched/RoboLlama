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
        model_path: str = "llm",
        buffer_size: int = 5,
        fps: int = 30,
        uncertainty_threshold: float = 0.5,
        stages: List[int] = [224, 384, 512],
        adaptive_temp: bool = True,
        num_diffusion_steps: int = 20
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
        self.feature_buffer = deque(maxlen=buffer_size)
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
            
    def adaptive_temperature(self, uncertainty: float) -> float:
        """Adaptive temperature scaling based on uncertainty"""
        if not self.adaptive_temp:
            return self.base_temp
        return self.base_temp * (1 + uncertainty)
        
    def smooth_predictions(
        self,
        actions: np.ndarray,
        uncertainty: np.ndarray,
        features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Uncertainty-aware prediction smoothing"""
        self.action_buffer.append(actions)
        self.uncertainty_buffer.append(uncertainty)
        self.feature_buffer.append(features)
        
        weights = np.exp(-np.array([u.mean() for u in self.uncertainty_buffer]))
        weights = weights / weights.sum()
        
        smoothed_actions = np.average(
            np.array(self.action_buffer),
            weights=weights,
            axis=0
        )
        smoothed_uncertainty = np.mean(self.uncertainty_buffer, axis=0)
        
        return smoothed_actions, smoothed_uncertainty
    
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
                        
                    with torch.no_grad(), self.timer('inference_time'):
                        try:
                            input_tensor = self.preprocess_frame(frame)
                            
                            current_steps = max(
                                5,
                                int(self.num_diffusion_steps * (
                                    1 - np.mean(self.uncertainty_buffer)
                                    if len(self.uncertainty_buffer) > 0
                                    else 1.0
                                ))
                            )
                            
                            actions, uncertainty, features = self.model(
                                input_tensor,
                                stage_idx=self.current_stage,
                                temp=self.adaptive_temperature(
                                    uncertainty.mean() if len(self.uncertainty_buffer) > 0 else 0.5
                                ),
                                num_steps=current_steps
                            )
                            
                            actions = actions.cpu().numpy()[0]
                            uncertainty = uncertainty.cpu().numpy()[0]
                            features = features.cpu().numpy()[0]
                            
                            self.update_stage(uncertainty.mean())
                            
                            smoothed_actions, smoothed_uncertainty = self.smooth_predictions(
                                actions, uncertainty, features
                            )
                            
                            vis_frame = self.visualize_predictions(
                                frame.copy(),
                                smoothed_actions,
                                smoothed_uncertainty,
                                features
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