
"""
Advanced Gesture-Based Volume Controller for Mac
===============================================

A comprehensive implementation featuring:
- Multi-gesture recognition
- Performance optimization with threading
- Configurable sensitivity and smoothing
- Visual feedback and GUI
- Error handling and logging
"""

import cv2
import mediapipe as mp
import numpy as np
import subprocess
import threading
import queue
import time
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum


class GestureType(Enum):
    """Enumeration of supported gestures"""
    VOLUME_CONTROL = "volume_control"
    MUTE_TOGGLE = "mute_toggle" 
    PLAY_PAUSE = "play_pause"
    IDLE = "idle"


@dataclass
class GestureConfig:
    """Configuration settings for gesture recognition"""
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7
    max_num_hands: int = 1
    volume_smoothing_factor: float = 0.3
    gesture_threshold: float = 0.8
    fps_target: int = 30


class VolumeController:
    """Main controller for gesture-based volume control"""

    def __init__(self, config: GestureConfig = None):
        self.config = config or GestureConfig()
        self.setup_logging()
        self.setup_mediapipe()
        self.setup_threading()
        self.current_volume = 50
        self.previous_volume = 50
        self.is_muted = False
        self.running = False

    def setup_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def setup_mediapipe(self):
        """Initialize MediaPipe components"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_num_hands,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence
        )

    def setup_threading(self):
        """Initialize threading components"""
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue()
        self.processing_thread = None

    def get_hand_landmarks(self, image: np.ndarray) -> Optional[Any]:
        """Extract hand landmarks from image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        return results.multi_hand_landmarks

    def calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    def recognize_gesture(self, landmarks, image_shape: Tuple[int, int]) -> Tuple[GestureType, Dict]:
        """Recognize gesture from hand landmarks"""
        h, w = image_shape[:2]

        # Get key landmark positions
        thumb_tip = landmarks[4]
        index_tip = landmarks[8] 
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        # Convert normalized coordinates to pixel coordinates
        thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
        index_pos = (int(index_tip.x * w), int(index_tip.y * h))
        middle_pos = (int(middle_tip.x * w), int(middle_tip.y * h))

        # Calculate distances
        thumb_index_dist = self.calculate_distance(thumb_pos, index_pos)

        # Volume control gesture (thumb-index distance)
        if 30 <= thumb_index_dist <= 350:
            volume = np.interp(thumb_index_dist, [30, 350], [0, 100])
            return GestureType.VOLUME_CONTROL, {
                "volume": int(volume),
                "distance": thumb_index_dist,
                "thumb_pos": thumb_pos,
                "index_pos": index_pos
            }

        # Mute gesture (closed fist)
        avg_dist = np.mean([
            self.calculate_distance(thumb_pos, index_pos),
            self.calculate_distance(thumb_pos, middle_pos)
        ])

        if avg_dist < 40:
            return GestureType.MUTE_TOGGLE, {"fist_detected": True}

        return GestureType.IDLE, {}

    def set_system_volume(self, volume: int):
        """Set macOS system volume using osascript"""
        try:
            volume = max(0, min(100, volume))  # Clamp between 0-100
            subprocess.run([
                "osascript", "-e", 
                f"set volume output volume {volume}"
            ], check=True, capture_output=True)
            self.current_volume = volume
            self.logger.info(f"Volume set to {volume}%")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to set volume: {e}")

    def toggle_mute(self):
        """Toggle system mute state"""
        try:
            if not self.is_muted:
                subprocess.run([
                    "osascript", "-e", 
                    "set volume output muted true"
                ], check=True, capture_output=True)
                self.is_muted = True
                self.logger.info("System muted")
            else:
                subprocess.run([
                    "osascript", "-e", 
                    "set volume output muted false"
                ], check=True, capture_output=True)
                self.is_muted = False
                self.logger.info("System unmuted")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to toggle mute: {e}")

    def draw_ui_elements(self, image: np.ndarray, gesture_data: Dict) -> np.ndarray:
        """Draw UI elements on the image"""
        h, w = image.shape[:2]

        # Draw volume bar
        bar_x, bar_y = 50, 150
        bar_w, bar_h = 35, 250

        # Background bar
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)

        # Volume level bar
        volume_height = int((self.current_volume / 100) * bar_h)
        cv2.rectangle(
            image, 
            (bar_x, bar_y + bar_h - volume_height), 
            (bar_x + bar_w, bar_y + bar_h), 
            (0, 255, 0), -1
        )

        # Volume percentage text
        cv2.putText(
            image, f"{self.current_volume}%", 
            (bar_x - 10, bar_y + bar_h + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        # Mute indicator
        if self.is_muted:
            cv2.putText(
                image, "MUTED", 
                (w - 150, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

        # FPS counter
        fps = getattr(self, 'current_fps', 0)
        cv2.putText(
            image, f"FPS: {fps:.1f}", 
            (w - 150, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )

        return image

    def process_frame_threaded(self, frame: np.ndarray):
        """Process frame in separate thread"""
        landmarks = self.get_hand_landmarks(frame)

        if landmarks:
            hand_landmarks = landmarks[0]
            gesture_type, gesture_data = self.recognize_gesture(
                hand_landmarks.landmark, frame.shape
            )

            self.result_queue.put({
                'landmarks': hand_landmarks,
                'gesture_type': gesture_type,
                'gesture_data': gesture_data,
                'frame': frame
            })
        else:
            self.result_queue.put({
                'landmarks': None,
                'gesture_type': GestureType.IDLE,
                'gesture_data': {},
                'frame': frame
            })

    def run(self):
        """Main execution loop"""
        self.logger.info("Starting gesture-based volume controller")
        self.running = True

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps_target)

        if not cap.isOpened():
            self.logger.error("Cannot open camera")
            return

        fps_counter = 0
        fps_timer = time.time()

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)  # Mirror the image

                # Process frame in thread if not already processing
                if self.processing_thread is None or not self.processing_thread.is_alive():
                    self.processing_thread = threading.Thread(
                        target=self.process_frame_threaded,
                        args=(frame.copy(),),
                        daemon=True
                    )
                    self.processing_thread.start()

                # Get results if available
                try:
                    result = self.result_queue.get_nowait()

                    if result['landmarks']:
                        # Draw hand landmarks
                        self.mp_drawing.draw_landmarks(
                            frame,
                            result['landmarks'],
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )

                        # Handle gestures
                        if result['gesture_type'] == GestureType.VOLUME_CONTROL:
                            target_volume = result['gesture_data']['volume']
                            # Apply smoothing
                            smoothed_volume = (
                                self.config.volume_smoothing_factor * target_volume +
                                (1 - self.config.volume_smoothing_factor) * self.current_volume
                            )
                            self.set_system_volume(int(smoothed_volume))

                            # Draw gesture indicators
                            thumb_pos = result['gesture_data']['thumb_pos']
                            index_pos = result['gesture_data']['index_pos']
                            cv2.circle(frame, thumb_pos, 10, (255, 0, 0), -1)
                            cv2.circle(frame, index_pos, 10, (255, 0, 0), -1)
                            cv2.line(frame, thumb_pos, index_pos, (0, 255, 0), 3)

                        elif result['gesture_type'] == GestureType.MUTE_TOGGLE:
                            # Implement debouncing for mute toggle
                            if not hasattr(self, 'last_mute_time'):
                                self.last_mute_time = time.time()
                                self.toggle_mute()
                            elif time.time() - self.last_mute_time > 1.0:  # 1 second debounce
                                self.last_mute_time = time.time()
                                self.toggle_mute()

                except queue.Empty:
                    pass

                # Draw UI elements
                frame = self.draw_ui_elements(frame, {})

                # Calculate FPS
                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    self.current_fps = fps_counter
                    fps_counter = 0
                    fps_timer = time.time()

                cv2.imshow('Advanced Gesture Volume Control', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    self.toggle_mute()
                elif key == ord('r'):
                    self.current_volume = 50
                    self.set_system_volume(50)

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            self.cleanup(cap)

    def cleanup(self, cap):
        """Clean up resources"""
        self.running = False
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        self.logger.info("Cleanup completed")


def main():
    """Main entry point"""
    config = GestureConfig(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        volume_smoothing_factor=0.2,
        fps_target=30
    )

    controller = VolumeController(config)

    try:
        controller.run()
    except Exception as e:
        logging.error(f"Application error: {e}")


if __name__ == "__main__":
    main()
