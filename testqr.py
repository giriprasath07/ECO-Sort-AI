# sxv5.py

#!/usr/bin/env python3
"""
AI Waste Sorter Application for Raspberry Pi
Integrates video playback, hand detection, waste classification, and sustainability tracking
Uses PyQt5, YOLOv8, OpenCV, and Google Gemini API
"""

import sys
import cv2
import numpy as np
import threading
import time
import json
import qrcode
import os
import Checking
import bin_level
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                           QWidget, QLabel, QPushButton, QTextEdit, QSizePolicy, QStackedWidget, QMessageBox, QGridLayout, QFrame)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QRect, QPropertyAnimation, QEasingCurve, QUrl
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPen, QBrush, QPainterPath, QLinearGradient
from PyQt5.QtWidgets import QGraphicsDropShadowEffect

from ultralytics import YOLO

# Using fake_rpi for compatibility on non-Raspberry Pi systems.
# For a real Raspberry Pi, remove 'fake_rpi.' and ensure 'RPi.GPIO' is installed.
try:
    from fake_rpi.RPi import GPIO
except ImportError:
    print("Warning: fake_rpi not found. GPIO functionality will be disabled. Install fake_rpi or run on Raspberry Pi.")
    class MockDistanceSensor:
        def __init__(self, echo, trigger):
            pass
        @property
        def distance(self):
            return 0.2
    class MockGPIO:
        def DistanceSensor(self, echo, trigger):
            print("MockGPIO: DistanceSensor called. Returning a dummy sensor.")
            return MockDistanceSensor(echo, trigger)
    GPIO = MockGPIO()

from PIL import Image
import requests # Import requests for API calls
import base64
from dotenv import load_dotenv
import google.generativeai as genai
import re # Import for regular expressions

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Firebase Configuration ---
FIREBASE_DATABASE_URL = "https://mysmartbinproject-8bcad-default-rtdb.asia-southeast1.firebasedatabase.app/"
BIN_1_PATH = "bins/1/percentage" # Path for Non-Recyclable bin (Black text, Red/Orange/Green circle)
BIN_2_PATH = "bins/2/percentage" # Path for Recyclable bin (Green text, Red/Orange/Green circle)

class VideoPlayerThread(QThread):
    """
    A dedicated thread for playing video using OpenCV.
    It reads frames from a video file and emits them as QPixmap signals.
    This avoids blocking the main GUI thread and is more reliable than QtMultimedia on some systems.
    """
    frame_ready = pyqtSignal(QPixmap)
    video_ended = pyqtSignal()

    def __init__(self, video_path, loop_video=False): # Added loop_video parameter
        super().__init__()
        self.video_path = video_path
        self._running = False
        self._paused = True
        self.frame_delay_ms = 33
        self.loop_video = loop_video # Store looping preference

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: [VideoPlayerThread] Could not open video file: {self.video_path}")
            self._running = False
            self.video_ended.emit()
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            self.frame_delay_ms = int(100 / fps)
            print(f"[VideoPlayerThread] Video FPS: {fps:.2f}, Calculated frame delay: {self.frame_delay_ms}ms")
        else:
            print("[VideoPlayerThread] Could not get video FPS, defaulting to 30 FPS.")

        while self._running:
            if self._paused:
                self.msleep(10)
                continue

            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                if self.loop_video:
                    # If video ends and looping is enabled, reset to start
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read() # Read the first frame again
                    if not ret: # If still no frame after reset, something is wrong
                        print(f"[VideoPlayerThread] Error: Could not read frame after loop reset for {self.video_path}.")
                        break
                else:
                    # Video has ended and not set to loop, so break
                    print(f"[VideoPlayerThread] Video {self.video_path} has finished and is not set to loop.")
                    break

            # Convert OpenCV's BGR frame to RGB for Qt
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888) # Corrected line
            pixmap = QPixmap.fromImage(qt_image)
            self.frame_ready.emit(pixmap)

            # Control the frame rate dynamically
            elapsed_time_ms = (time.time() - start_time) * 1000
            sleep_time_ms = max(1, self.frame_delay_ms - elapsed_time_ms)
            self.msleep(int(sleep_time_ms))

        cap.release()
        print(f"[VideoPlayerThread] Video thread for {self.video_path} has finished.")
        self.video_ended.emit() # Emit signal when video truly ends (either by breaking or non-looping end)

    def stop(self):
        """Stops the video thread."""
        self._running = False
        self.wait()

    def pause(self):
        """Pauses video playback."""
        self._paused = True
        print(f"[VideoPlayerThread] Paused {self.video_path}.")

    def resume(self):
        """Resumes video playback."""
        self._paused = False
        print(f"[VideoPlayerThread] Resumed {self.video_path}.")

class CameraThread(QThread):
    """Thread for camera capture and person detection"""
    frame_ready = pyqtSignal(np.ndarray) # Emit raw camera frames for display
    object_detected_for_classification = pyqtSignal(np.ndarray, tuple, str) # frame, bbox, yolo_class_name
    person_only_detected = pyqtSignal(np.ndarray)  # frame for person only
    proximity_changed = pyqtSignal(bool) # Renamed from proximity_detected for clarity

    # Define states for classification trigger logic
    STATE_IDLE = 0
    STATE_PERSON_DETECTED = 1
    STATE_OBJECT_PRESENTED = 2
    STATE_CLASSIFYING = 3

    def __init__(self):
        super().__init__()
        self.model_detect = None # Initialize here to ensure attribute always exists

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera (index 0). Please check camera connection or index.")
            self.cap = None
        self.running = False # Will be set to True when thread starts

        self.last_yolo_class_name = None # To store the YOLO class name for classification
        self.ultrasonic = None # Ultrasonic sensor removed
        
        self.detection_threshold_frames = 3 # Number of frames to confirm person detection (Reduced from 5)
        self.object_stability_threshold = 5 # Number of frames an object needs to be stable before classification (Reduced from 10)

        self.classification_trigger_state = self.STATE_IDLE # Current state of the classification trigger
        self.person_detected_stable_frames = 0 # Counter for stable person detection
        self.object_stable_frames = 0 # Counter for stable object presentation

        self.person_currently_detected = False # Track current stable detection state for UI mode switching

        # Load YOLOv8 model for person detection
        try:
            self.model_detect = YOLO(os.path.join(SCRIPT_DIR, 'yolov8n.pt')) # Use os.path.join
            
             #self.model_detect = YOLO("/home/srini/Code/bin/bin1/Recycle/best.pt") # Use os.path.join
             #print(f"YOLOv8n model loaded successfully")
            
            print(f"YOLOv8n model loaded successfully from: {os.path.join(SCRIPT_DIR, 'yolov8n.pt')}")
        except Exception as e:
            print(f"Error loading YOLOv8n model: {e}. Please ensure 'yolov8n.pt' is available. Detection will be disabled.")
            self.model_detect = None # Disable detection if model fails to load

    def check_proximity_and_objects(self, frame):
        """Check proximity using YOLOv8 detection and return detection info"""
        if self.model_detect is None:
            print("[CameraThread] No YOLOv8n detection model loaded. Skipping object detection.")
            return False, None, None, None # No model, no detection

        # Predict on the frame
        results = self.model_detect.predict(source=frame, stream=True, conf=0.25, iou=0.45, classes=None, agnostic_nms=False, max_det=10)

        if results is None:
            print("[CameraThread] Error: Model prediction returned None.")
            return False, None, None, None

        PROXIMITY_AREA_THRESHOLD = 15000 # Increased threshold to filter out smaller background objects
        
        # Define YOLOv8n class IDs that are likely to be waste items
        RELEVANT_OBJECT_CLASSES = {
            39, # bottle
            41, # cup
            46, # banana
            47, # apple
            48, # sandwich
            49, # orange
            50, # broccoli
            51, # carrot
            53, # pizza
            54, # donut
            55, # cake
            67, # cell phone (e-waste)
            60, # dining table
            62, # laptop
            63, # mouse
            64, # remote
            65, # keyboard
            73, # book
            74, # clock
            75, # vase
            76, # scissors
            77, # teddy bear
            78, # hair drier
            79, # toothbrush
        }

        detected_objects = [] # This list will now ONLY contain non-person and relevant objects
        person_detected_in_frame = False
        person_bbox = None

        for r in results:
            if not hasattr(r, 'boxes') or r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1
                area = width * height
                class_id = int(box.cls[0])
                class_name = r.names[class_id]
                confidence = box.conf[0].item()

                if area > PROXIMITY_AREA_THRESHOLD: # Consider objects larger than a certain area
                    if class_id == 0:  # Person class (YOLO class_id 0 is usually 'person')
                        person_detected_in_frame = True
                        person_bbox = (x1, y1, x2, y2)
                    elif class_id in RELEVANT_OBJECT_CLASSES: # Only add if it's a relevant waste object
                        detected_objects.append({
                            'class_name': class_name,
                            'class_id': class_id,
                            'area': area,
                            'confidence': confidence,
                            'bbox': (x1, y1, x2, y2)
                        })

        # Logic to determine if a person is near and if a relevant object is present for classification.
        # This logic is critical for triggering the classification process.
        if person_detected_in_frame:
            if detected_objects:
                # If a person and other *relevant* objects are detected, try to find the "best" object for classification
                best_object = self.find_best_object_for_classification(detected_objects, person_bbox, frame.shape)
                if best_object:
                    return True, "objects_for_classification", best_object['bbox'], best_object['class_name']
                else:
                    # Person detected, but no suitable *relevant* object for classification found
                    return True, "person_only", None, None
            else:
                # Only a person detected, no relevant objects
                return True, "person_only", None, None
        # If no person is detected, or no relevant objects are detected, do not trigger classification.
        return False, None, None, None 

    def find_best_object_for_classification(self, detected_objects, person_bbox, frame_shape):
        """Find the best object to classify based on proximity to person's hands"""
        h, w = frame_shape[:2]

        if person_bbox:
            px1, py1, px2, py2 = person_bbox
            # Define a region where hands might be (e.g., lower half of person's bbox)
            hand_y_min = py1 + int((py2 - py1) * 0.6)
            hand_y_max = py2
            hand_center_x = (px1 + px2) // 2
            hand_center_y = (hand_y_min + hand_y_max) // 2
        else:
            # If no person detected, assume center of frame for object proximity
            hand_center_x = w // 2
            hand_center_y = h // 2
            hand_y_min = int(h * 0.4)
            hand_y_max = h

        recyclable_item_classes = {
            39, 41, 67, # bottle, cup, cell phone (from RELEVANT_OBJECT_CLASSES)
            62, 63, 64, 65, # laptop, mouse, remote, keyboard
            73, # book
            75, # vase
        }

        best_object = None
        min_distance = float('inf')

        for obj in detected_objects:
            ox1, oy1, ox2, oy2 = obj['bbox']
            obj_center_x = (ox1 + ox2) // 2
            obj_center_y = (oy1 + oy2) // 2

            distance = ((obj_center_x - hand_center_x) ** 2 + (obj_center_y - hand_center_y) ** 2) ** 0.5
            in_hand_region = (oy1 < hand_y_max and oy2 > hand_y_min) # Check if object is within hand region
            is_recyclable_item = obj['class_id'] in recyclable_item_classes # Check if it's a known recyclable item

            # Prioritize objects in hand region and recyclable items
            if in_hand_region and is_recyclable_item:
                if distance < min_distance:
                    min_distance = distance
                    best_object = obj
            elif in_hand_region and distance < min_distance * 0.8: # If in hand region but not explicitly recyclable, still consider
                min_distance = distance
                best_object = obj
            elif is_recyclable_item and distance < min_distance * 1.2: # If recyclable but slightly further
                min_distance = distance
                best_object = obj
            elif not best_object and distance < min_distance * 1.5: # Fallback: just the closest object
                min_distance = distance
                best_object = obj

        return best_object

    def get_smart_crop_bbox(self, frame, object_bbox):
        """Get smart cropping bounding box for the detected object"""
        if not object_bbox:
            return None

        x1, y1, x2, y2 = object_bbox
        h, w = frame.shape[:2]

        # Add padding to the bounding box to ensure the whole object is captured
        padding_x = int((x2 - x1) * 0.2)
        padding_y = int((y2 - y1) * 0.2)

        crop_x1 = max(0, x1 - padding_x)
        crop_y1 = max(0, y1 - padding_y)
        crop_x2 = min(w, x2 + padding_x)
        crop_y2 = min(h, y2 + padding_y)

        # Ensure minimum size for cropping to avoid tiny images
        min_size = 100
        if crop_x2 - crop_x1 < min_size:
            center_x = (crop_x1 + crop_x2) // 2
            crop_x1 = max(0, center_x - min_size // 2)
            crop_x2 = min(w, center_x + min_size // 2)

        if crop_y2 - crop_y1 < min_size:
            center_y = (crop_y1 + crop_y2) // 2
            crop_y1 = max(0, center_y - min_size // 2)
            crop_y2 = min(h, center_y + min_size // 2)

        return (crop_x1, crop_y1, crop_x2, crop_y2)

    def run(self):
        """Main camera processing loop"""
        self.running = True
        if self.cap is None: # Exit if camera failed to open
            self.running = False
            print("CameraThread: Camera not available, thread exiting.")
            return

        print("CameraThread: Camera thread started.")
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Warning: Could not read frame from camera. Retrying...")
                self.msleep(50)
                continue

            # Always emit raw frame for live display.
            self.frame_ready.emit(frame)

            # Only perform detection if not awaiting disposal (i.e., classification result screen is active)
            parent_app = getattr(self, 'parent_app', None)
            if parent_app and parent_app.awaiting_disposal:
                self.msleep(100) # Reduce CPU usage while waiting for disposal confirmation
                continue

            proximity, detection_info, bbox, yolo_class_name = self.check_proximity_and_objects(frame)

            # --- State machine logic for classification trigger ---
            if self.classification_trigger_state == self.STATE_IDLE:
                if proximity and (detection_info == "person_only" or detection_info == "objects_for_classification"):
                    self.person_detected_stable_frames += 1
                    if self.person_detected_stable_frames >= self.detection_threshold_frames:
                        self.classification_trigger_state = self.STATE_PERSON_DETECTED
                        print("CameraThread: State changed to STATE_PERSON_DETECTED (person stably present).")
                        # Only emit proximity_changed when entering this state
                        if not self.person_currently_detected:
                            self.person_currently_detected = True
                            self.proximity_changed.emit(True)
                else: # No proximity or not enough stable frames for person
                    self.person_detected_stable_frames = 0
                    if self.person_currently_detected: # Only emit if state was previously true
                        self.person_currently_detected = False
                        self.proximity_changed.emit(False)
                        print("CameraThread: Emitted proximity_changed(False) - no person.")

            elif self.classification_trigger_state == self.STATE_PERSON_DETECTED:
                if proximity and detection_info == "objects_for_classification" and bbox:
                    self.object_stable_frames += 1
                    if self.object_stable_frames >= self.object_stability_threshold:
                        self.classification_trigger_state = self.STATE_CLASSIFYING
                        print("CameraThread: State changed to STATE_CLASSIFYING - object stably presented.")
                        smart_bbox = self.get_smart_crop_bbox(frame, bbox)
                        if smart_bbox:
                            self.object_detected_for_classification.emit(frame, smart_bbox, yolo_class_name)
                        else:
                            print("[CameraThread] Could not create smart crop bbox for stable object. Reverting state.")
                            self.classification_trigger_state = self.STATE_PERSON_DETECTED # Revert if bbox fails
                            self.object_stable_frames = 0 # Reset object stability
                elif proximity and detection_info == "person_only":
                    self.object_stable_frames = 0 # Reset object stability if only person is detected
                    # Stay in STATE_PERSON_DETECTED
                elif not proximity: # Person left
                    self.person_detected_stable_frames = 0
                    self.object_stable_frames = 0
                    self.classification_trigger_state = self.STATE_IDLE
                    if self.person_currently_detected:
                        self.person_currently_detected = False
                        self.proximity_changed.emit(False)
                        print("CameraThread: Emitted proximity_changed(False) - person lost.")

            elif self.classification_trigger_state == self.STATE_CLASSIFYING:
                # In this state, we wait for the classification result from WasteSorterApp.
                # WasteSorterApp will call reset_system_after_delay which will
                # transition the CameraThread's state back to IDLE via camera_thread.reset_state().
                pass # Do nothing, waiting for classification to complete

            self.msleep(50) # Control loop speed

    def stop(self):
        """Stop camera thread"""
        self.running = False
        if self.cap:
            self.cap.release()
            print("CameraThread: Camera released.")

    def reset_state(self):
        """Resets the camera thread's internal state for a new detection cycle."""
        self.classification_trigger_state = self.STATE_IDLE
        self.person_detected_stable_frames = 0
        self.object_stable_frames = 0
        self.person_currently_detected = False
        print("CameraThread: Internal state reset.")


class ClassificationThread(QThread):
    """Thread for YOLOv8 waste classification"""
    classification_ready = pyqtSignal(str, float, np.ndarray, str, str, str, str)  # Add yolo_class_name, full_response, explanation

    def __init__(self):
        super().__init__()
        try:
            self.model = YOLO(os.path.join(SCRIPT_DIR, r'best.pt'))
            print(f"YOLOv8 classification model loaded successfully from: {os.path.join(SCRIPT_DIR, 'best.pt')}")
        except Exception as e:
            print(f"Error loading YOLOv8 classification model: {e}. YOLO classification will be disabled.")
            self.model = None

        self.results_dir = os.path.join(SCRIPT_DIR, "classified_results")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def classify_waste(self, frame, bbox, yolo_class_name=None):
        """Classify cropped waste object"""
        if self.model is None:
            print("[ClassifyWaste] YOLO model not loaded. Skipping YOLO classification.")
            self.classification_ready.emit("Unknown", 0.0, frame, "Unknown", yolo_class_name, "YOLO model not loaded.", "YOLO model not loaded.")
            return

        x_min, y_min, x_max, y_max = bbox
        h, w, _ = frame.shape
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)

        if x_max <= x_min or y_max <= y_min:
            print(f"[ClassifyWaste] Error: Invalid bounding box for cropping: ({x_min},{y_min},{x_max},{y_max}) in frame of size {w}x{h}.")
            self.classification_ready.emit("Error", 0.0, np.zeros((100,100,3), dtype=np.uint8), "Error", yolo_class_name, "Invalid crop area.", "Invalid crop area.")
            return

        cropped = frame[y_min:y_max, x_min:x_max]

        if cropped.size > 0:
            results = self.model(cropped, verbose=False)

            if results and hasattr(results[0], 'probs') and results[0].probs is not None:
                top_class = results[0].probs.top1
                confidence = results[0].probs.top1conf.item()
                class_name = results[0].names[top_class]

                print(f"[ClassifyWaste] YOLO Result: Class='{class_name}', Confidence={confidence:.2f}")

                if confidence < 0.80: 
                    print(f"[ClassifyWaste] Low confidence ({confidence:.2f}) for {class_name}. Classifying as Non-Recyclable.")
                    self.classification_ready.emit("Non-Recyclable", confidence, cropped, "Unknown Item", yolo_class_name, "Low confidence.", "Low confidence.")
                    return

                # Map specific yolo classes to Recyclable/Non-Recyclable
                if "plastic" in class_name.lower() or "metal" in class_name.lower() or "glass" in class_name.lower() or "paper" in class_name.lower() or "cardboard" in class_name.lower() or "e-waste" in class_name.lower() or "electronics" in class_name.lower():
                     classification = "Recyclable"
                else:
                     classification = "Non-Recyclable"

                self.save_classified_image(cropped, classification, confidence, class_name)
                self.classification_ready.emit(classification, confidence, cropped, class_name, yolo_class_name, class_name, class_name)
            else:
                print("[ClassifyWaste] Error: No valid predictions found in YOLOv8 results.")
                self.classification_ready.emit("Unknown", 0.0, cropped, "Unknown", yolo_class_name, "No YOLO predictions.", "No YOLO predictions.")
        else:
            print("[ClassifyWaste] Error: Cropped image is empty.")
            self.classification_ready.emit("Unknown", 0.0, cropped, "Unknown", yolo_class_name, "Empty cropped image.", "Empty cropped image.")

    def save_classified_image(self, cropped_image, classification, confidence, class_name):
        """Save the classified image with metadata"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{timestamp}_{classification}_{confidence:.2f}_{class_name}.jpg"
            filepath = os.path.join(self.results_dir, filename)
            cv2.imwrite(filepath, cropped_image)

            metadata = {
                "timestamp": datetime.now().isoformat(),
                "classification": classification,
                "confidence": confidence,
                "class_name": class_name,
                "image_path": filepath
            }

            metadata_file = filepath.replace('.jpg', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved classified image: {filepath}")
        except Exception as e:
            print(f"Error saving classified image: {e}")

class SustainabilityCalculator:
    """Calculate CO2 savings and manage statistics"""

    def __init__(self):
        self.stats = {
            'total_items': 0,
            'recyclable_items': 0,
            'non_recyclable_items': 0,
            'plastic_items': 0,
            'paper_items': 0,
            'glass_items': 0,
            'metal_items': 0,
            'e_waste_items': 0,
            'organic_items': 0,
            'other_items': 0,
            'total_co2_saved': 0.0
        }

        self.material_mapping = {
            'plastic': ['plastic', 'bottle', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'plastic bag', 'plastic wrap', 'plastic container', 'plastic bottle', 'plastic cup', 'plastic fork', 'plastic knife', 'plastic spoon', 'plastic bowl'],
            'paper': ['paper', 'book', 'pizza', 'coffee cup', 'magazine', 'newspaper', 'cardboard box','cardboard', 'paper bag', 'paper cup', 'paper plate', 'paper towel', 'paper napkin'],
            'glass': ['glass', 'wine glass', 'vase', 'glass jar','glass bottle', 'glass cup', 'glass plate', 'glass container'],
            'metal': ['metal', 'can', 'aluminium can', 'tin can', 'aluminum foil', 'metal can', 'metal bottle', 'metal container', 'metal lid', 'soda can', 'steel can', 'aluminum can', 'tin foil', 'metal cutlery'], # Added more metal keywords
            'e_waste': ['e-waste', 'e_waste', 'cell phone', 'mobile phone', 'laptop', 'mouse', 'remote', 'keyboard', 'smartphone', 'phone', 'electronics', 'battery', 'calculator','electronic device', 'tv', 'electronic gadget', 'electronic equipment', 'electronic item'],
            'organic': ['organic', 'food scraps', 'banana peel', 'apple core', 'donut', 'cake','vegetable scraps', 'fruit scraps', 'food waste', 'compostable', 'compost'],
            'other': ['other', 'styrofoam', 'ceramic plate', 'ceramic bowl', 'ceramic cup', 'ceramic mug', 'ceramic vase', 'ceramic jar', 'ceramic container', 'ceramic tile', 'ceramic figurine', 'ceramic dish', 'ceramic pot', 'ceramic tile']
        }

        self.emission_factors = {
            'plastic': 0.15,
            'paper': 0.05,
            'glass': 0.08,
            'metal': 0.09,
            'e_waste': 0.50,
            'organic': 0.00,
            'other': 0.00
        }

    def add_item(self, classification, classified_object_name='general_recyclable'):
        """Add classified item and calculate CO2 savings, updating specific material counts."""
        self.stats['total_items'] += 1

        item_type = 'other'
        classified_object_name_lower = classified_object_name.lower()

        for material, keywords in self.material_mapping.items():
            for keyword in keywords:
                if keyword in classified_object_name_lower:
                    item_type = material
                    break
            if item_type != 'other':
                break

        print(f"SustainabilityCalculator: Adding item - Classification: {classification}, Object Name: {classified_object_name}, Mapped Type: {item_type}")

        if classification == "Recyclable":
            co2_to_add = self.emission_factors.get(item_type, 0.1) 
            self.stats['recyclable_items'] += 1
            self.stats['total_co2_saved'] += co2_to_add
            print(f"SustainabilityCalculator: Added {co2_to_add:.2f} kg CO2 for {item_type} ({classified_object_name}).")

            if f'{item_type}_items' in self.stats:
                self.stats[f'{item_type}_items'] += 1
            else:
                self.stats['other_items'] += 1
        else: # Non-Recyclable
            self.stats['non_recyclable_items'] += 1
            if f'{item_type}_items' in self.stats:
                self.stats[f'{item_type}_items'] += 1
            else:
                self.stats['other_items'] += 1
        print(f"SustainabilityCalculator: Current stats: {self.stats}")


    def get_sustainability_report(self):
        """Generate sustainability report"""
        return {
            'total_items_processed': self.stats['total_items'],
            'items_recycled': self.stats['recyclable_items'],
            'items_landfilled': self.stats['non_recyclable_items'],
            'recycling_rate': f"{(self.stats['recyclable_items']/max(1, self.stats['total_items'])*100):.1f}%",
            'total_co2_saved_kg': f"{self.stats['total_co2_saved']:.2f}",
            'equivalent_trees_planted': f"{(self.stats['total_co2_saved']/21.77):.1f}",
            'breakdown': {
                'plastic': self.stats['plastic_items'],
                'paper_cardboard': self.stats['paper_items'],
                'organic': self.stats['organic_items'],
                'others': self.stats['other_items'],
                'e_waste': self.stats['e_waste_items'],
                'metal': self.stats['metal_items'],
                'glass': self.stats['glass_items']
            }
        }

class BinLevelWidget(QWidget):
    """
    A custom widget to display bin fill level with a circular progress bar and percentage.
    """
    def __init__(self, bin_name, bin_type_color, parent=None):
        super().__init__(parent)
        self.percentage = 0
        self.bin_name = bin_name # Store bin name for display below the circle
        self.bin_type_color = bin_type_color # 'green' for Recyclable, 'black' for Non-Recyclable

        # Adjusted size for the widget to provide more space for the text below
        # Increased width to accommodate "Non-Recyclable" fully
        # Increased height from 220 to 240 to give more vertical space
        self.setFixedSize(180, 240) 

        # Set up fonts
        # Reduced percentage font size from 24 to 20 and font weight from QFont.Bold to QFont.DemiBold
        self.percentage_font = QFont("Montserrat", 20, QFont.DemiBold) 
        self.bin_name_font = QFont("Montserrat", 14, QFont.Bold) # Font for bin name below

        # Initial color setup (will be updated by set_percentage)
        self.active_color_start = QColor(100, 100, 100) # Default grey
        self.active_color_end = QColor(150, 150, 150) # Default grey

        # Add a border and subtle shadow to the BinLevelWidget itself
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(5) # Reduced blur radius for a subtle effect
        shadow.setColor(QColor(0, 0, 0, 50)) # Reduced opacity for a subtle effect
        shadow.setOffset(2, 2) # Reduced offset for a subtle effect
        self.setGraphicsEffect(shadow)
        self.setStyleSheet("border: 2px solid #ccc; border-radius: 15px; background-color: white;") # Added background-color

    def set_percentage(self, percentage):
        self.percentage = percentage
        self.update_color_based_on_percentage() # Update colors based on percentage
        self.update() # Trigger repaint

    def update_color_based_on_percentage(self):
        if self.percentage >= 80:
            self.active_color_start = QColor(255, 60, 60) # Red start
            self.active_color_end = QColor(200, 0, 0)   # Red end
        elif self.percentage >= 50:
            self.active_color_start = QColor(255, 180, 60) # Orange start
            self.active_color_end = QColor(200, 120, 0)   # Orange end
        else:
            self.active_color_start = QColor(60, 255, 60) # Green start
            self.active_color_end = QColor(0, 200, 0)   # Green end

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        ring_width = 20 
        # Calculate outer radius to fit the new fixed size, leaving some margin
        # Increased margin from 5 to 10 to make the circle visually smaller
        # Adjusted outer_radius calculation to fit the new height and provide more space
        outer_radius = min(rect.width(), rect.height() - self.bin_name_font.pointSize() * 2) / 2 - 15 
        inner_radius = outer_radius - ring_width
        
        center_x = rect.width() / 2
        # Adjusted vertical center to place the circle higher, making space for the bin name
        # Moved slightly to the top from 0.4 to 0.35 to give more space below
        center_y_circle = int(rect.height() * 0.35) 

        arc_rect = QRect(int(center_x - outer_radius), int(center_y_circle - outer_radius),
                         int(outer_radius * 2), int(outer_radius * 2))

        # --- Draw the background track of the circle (light grey) ---
        painter.setPen(QPen(QColor(230, 230, 230, 150), ring_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)
        painter.drawArc(arc_rect, 90 * 16, -360 * 16) 

        # --- Draw the active progress arc with gradient ---
        if self.percentage > 0:
            gradient = QLinearGradient(arc_rect.topLeft(), arc_rect.bottomRight())
            gradient.setColorAt(0, self.active_color_start)
            gradient.setColorAt(1, self.active_color_end)
            
            painter.setPen(QPen(gradient, ring_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            
            span_angle = int(-self.percentage * 3.6 * 16)
            painter.drawArc(arc_rect, 90 * 16, span_angle)

        # --- Draw the inner white circle (background for text) ---
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(int(center_x - inner_radius), int(center_y_circle - inner_radius),
                            int(inner_radius * 2), int(inner_radius * 2))

        # --- Draw percentage text in the center ---
        painter.setPen(QColor(50, 50, 50))
        painter.setFont(self.percentage_font)
        percentage_text = f"{self.percentage}%"
        text_rect = painter.fontMetrics().boundingRect(percentage_text)
        # Position percentage text exactly in the center of the circle
        painter.drawText(int(center_x - text_rect.width() / 2), int(center_y_circle + text_rect.height() / 4), percentage_text)

        # --- Draw bin name text below the circle ---
        painter.setFont(self.bin_name_font)
        if self.bin_type_color == 'green':
            painter.setPen(QColor(0, 128, 0)) # Green color for Recyclable bin name
        else:
            painter.setPen(QColor(0, 0, 0)) # Black color for Non-Recyclable bin name
        
        bin_name_text_rect = painter.fontMetrics().boundingRect(self.bin_name)
        # Position the bin name text below the circle, centered horizontally, and with a fixed offset from the circle
        # Increased the vertical offset from 40 to 50 to add more space
        painter.drawText(int(center_x - bin_name_text_rect.width() / 2), int(center_y_circle + outer_radius + 50), self.bin_name) 

        painter.end()


class FirebaseDataFetcher(QThread):
    """Thread to fetch Firebase data periodically."""
    bin_data_ready = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._running = False
        self.polling_interval = 5 # seconds

    def run(self):
        self._running = True
        while self._running:
            bin1_level = self._get_bin_percentage(BIN_1_PATH)
            bin2_level = self._get_bin_percentage(BIN_2_PATH)
            
            self.bin_data_ready.emit({
                'bin1': bin1_level,
                'bin2': bin2_level
            })
            time.sleep(self.polling_interval)

    def _get_bin_percentage(self, bin_path):
        """Helper to fetch a single bin's percentage."""
        try:
            url = f"{FIREBASE_DATABASE_URL}{bin_path}.json"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if data is not None:
                return data
            return None
        except requests.exceptions.RequestException as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Network error fetching data for '{bin_path}': {e}")
            return None
        except json.JSONDecodeError:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error decoding JSON response for '{bin_path}'.")
            return None
        except Exception as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] An unexpected error occurred for '{bin_path}': {e}")
            return None

    def stop(self):
        self._running = False
        self.wait()


class WasteSorterApp(QMainWindow):
    """Main application window"""

    gemini_classification_result = pyqtSignal(str, float, np.ndarray, str, str, str, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Waste Sorter - SortyxNet")
        self.setFixedSize(1920,1080)
        self.setWindowFlags(Qt.FramelessWindowHint)

        # Application state
        self.current_mode = "intro_video" # Initial mode is intro video
        self.current_classification = None
        self.awaiting_disposal = False
        self.last_captured_frame = None
        self.last_captured_bbox = None
        self.last_yolo_class_name = None
        self.classification_method = "Gemini"

        # Initialize components
        self.camera_thread = CameraThread()
        self.classification_thread = ClassificationThread()
        self.sustainability_calc = SustainabilityCalculator()
        self.firebase_fetcher = FirebaseDataFetcher()
        self.firebase_fetcher.bin_data_ready.connect(self.update_bin_level_ui)
        self.firebase_fetcher.start()
        

        # --- Video Player Threads ---
        intro_video_path = os.path.join(SCRIPT_DIR, 'images', 'Sortyx_intro_video.mp4')
        self.intro_video_player_thread = VideoPlayerThread(intro_video_path, loop_video=False)

        instructional_video_path = os.path.join(SCRIPT_DIR, 'images', 'SortyxVideo.mp4')
        self.instructional_video_player_thread = VideoPlayerThread(instructional_video_path, loop_video=True)

        # --- Load classification images ---
        self.green_bin_person_pixmap = QPixmap(os.path.join(SCRIPT_DIR, 'images', 'green_bin_person.png')).scaled(500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.black_bin_person_pixmap = QPixmap(os.path.join(SCRIPT_DIR, 'images', 'black_bin_person.png')).scaled(500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.left_side_recyclable_pixmap = QPixmap(os.path.join(SCRIPT_DIR, 'images', 'left_image.png')).scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.right_side_non_recyclable_pixmap = QPixmap(os.path.join(SCRIPT_DIR, 'images', 'right_image.png')).scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.sustainability_summary_display_label = QLabel()

        # --- Stacked widget for mode switching ---
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # --- Screen 0: Intro Video Playback ---
        self.intro_video_screen = QWidget()
        self.setup_full_screen_video_screen(self.intro_video_screen)
        self.stacked_widget.addWidget(self.intro_video_screen)

        # --- Screen 1: Instructional Video Playback (Looping) ---
        self.instructional_video_screen = QWidget()
        self.setup_full_screen_video_screen(self.instructional_video_screen)
        self.stacked_widget.addWidget(self.instructional_video_screen)

        # --- Screen 2: Live Camera Feed with Classify Button and Breakdown ---
        self.live_camera_screen = QWidget()
        self.setup_live_camera_screen(self.live_camera_screen)
        self.stacked_widget.addWidget(self.live_camera_screen)

        # --- Screen 3: Classification Result Screen ---
        self.classification_result_screen = QWidget()
        self.setup_classification_result_screen(self.classification_result_screen)
        self.stacked_widget.addWidget(self.classification_result_screen)

        self.connect_signals()
        
        # Start threads that are always running or managed
        if not self.camera_thread.isRunning():
            self.camera_thread.start()
            print("WasteSorterApp: Camera thread started in __init__.")
        
        # Start the Firebase data fetcher thread
        if not self.firebase_fetcher.isRunning():
            self.firebase_fetcher.start()
            print("WasteSorterApp: Firebase data fetcher thread started.")
            
        self.bin_monitor_thread = threading.Thread(target=bin_level.start_monitoring, daemon=True)
        self.bin_monitor_thread.start()
        print("WasteSorterApp: Bin monitoring thread started.")

        # Start with intro video
        self.start_intro_video_mode()

    def _clear_layout(self, layout):
        """Helper function to clear all items from a layout."""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
                elif item.layout():
                    item.layout().setParent(None)
                elif item.spacerItem():
                    layout.removeItem(item)

    def create_header_widget(self):
        """Creates a reusable header widget with logo and date/time."""
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(20, 10, 20, 10)
        header_layout.setSpacing(0)

        self.sortyx_logo_label = QLabel()
        logo_path = os.path.join(SCRIPT_DIR, 'images', 'Sortyx_logo.png')
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            scaled_pixmap = pixmap.scaled(200, 70, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.sortyx_logo_label.setPixmap(scaled_pixmap)
        else:
            self.sortyx_logo_label.setText("SORTУХ")
            self.sortyx_logo_label.setFont(QFont("Montserrat", 24, QFont.Bold))
            self.sortyx_logo_label.setStyleSheet("color: #185a9d;")

        header_layout.addWidget(self.sortyx_logo_label, alignment=Qt.AlignLeft)
        header_layout.addStretch()

        self.datetime_label = QLabel()
        self.datetime_label.setFont(QFont("Montserrat", 12, QFont.Bold))
        self.datetime_label.setStyleSheet("color: #34495e;")
        header_layout.addWidget(self.datetime_label, alignment=Qt.AlignRight)

        self.datetime_timer = QTimer(self)
        self.datetime_timer.timeout.connect(self.update_datetime_label)
        self.datetime_timer.start(1000)
        self.update_datetime_label()

        return header_widget

    def update_datetime_label(self):
        """Updates the date and time label in the header."""
        current_datetime = datetime.now().strftime("%d/%m/%Y %H:%M")
        self.datetime_label.setText(current_datetime)

    def setup_full_screen_video_screen(self, parent_widget):
        """Setup a full-screen video playback screen using a QLabel for OpenCV frames."""
        main_layout = QVBoxLayout(parent_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        parent_widget.setStyleSheet("background-color: black;")

        video_display_label = QLabel()
        video_display_label.setAlignment(Qt.AlignCenter)
        video_display_label.setStyleSheet("background-color: black;")
        video_display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(video_display_label)
        
        if parent_widget == self.intro_video_screen:
            self.intro_video_display_label = video_display_label
        elif parent_widget == self.instructional_video_screen:
            self.instructional_video_display_label = video_display_label


    def setup_live_camera_screen(self, parent_widget):
        """Setup the live camera display with sustainability summary, breakdown, and method toggle using a grid layout."""
        main_layout = QVBoxLayout(parent_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        parent_widget.setStyleSheet("background-color: white;")

        header = self.create_header_widget()
        main_layout.addWidget(header, alignment=Qt.AlignTop)

        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken)
        line1.setStyleSheet("color: #ddd;")
        main_layout.addWidget(line1)

        content_grid_layout = QGridLayout()
        content_grid_layout.setContentsMargins(20, 20, 20, 20)
        content_grid_layout.setSpacing(20) # Spacing between major grid cells

        # Camera Feed (Row 0, Column 0)
        self.live_display_label = QLabel()
        self.live_display_label.setAlignment(Qt.AlignCenter)
        self.live_display_label.setStyleSheet("background-color: black; border-radius: 15px;")
        self.live_display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_grid_layout.addWidget(self.live_display_label, 0, 0, 1, 1) # Row 0, Col 0, Span 1 row, 1 col

        # Right Column Layout (for Item Breakdown, Bin Levels, Your Impact)
        right_column_vertical_layout = QVBoxLayout()
        right_column_vertical_layout.setContentsMargins(0, 0, 0, 0)
        right_column_vertical_layout.setSpacing(30) # Increased spacing between sections in right column (e.g., between bin levels and sustainability)

        # Item Breakdown (Top of Right Column)
        item_breakdown_container = QVBoxLayout()
        breakdown_caption_label = QLabel("ITEM BREAKDOWN")
        breakdown_caption_label.setFont(QFont("Montserrat", 16, QFont.Bold))
        breakdown_caption_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        breakdown_caption_label.setStyleSheet("color: #185a9d; margin-bottom: 5px;")
        item_breakdown_container.addWidget(breakdown_caption_label)

        self.item_breakdown_widget = QWidget()
        breakdown_layout = QGridLayout(self.item_breakdown_widget)
        breakdown_layout.setContentsMargins(10,10,10,10)
        breakdown_layout.setSpacing(5)

        self.breakdown_labels = {}
        categories = ["E-waste", "Paper/Cardboard", "Organic", "Metal", "Others", "Plastic"]
        for i, category in enumerate(categories):
            cat_label = QLabel(category)
            cat_label.setFont(QFont("Montserrat", 12))
            cat_label.setStyleSheet("color: #333;")
            count_label = QLabel("0")
            count_label.setFont(QFont("Montserrat", 12, QFont.Bold))
            count_label.setStyleSheet("color: #185a9d;")
            self.breakdown_labels[category] = count_label
            breakdown_layout.addWidget(cat_label, i, 0, Qt.AlignLeft)
            breakdown_layout.addWidget(count_label, i, 1, Qt.AlignRight)

        # Added border to item_breakdown_widget
        self.item_breakdown_widget.setStyleSheet("background: rgba(255,255,255,0.7); border-radius: 15px; padding: 5px; border: 2px solid #ccc;")
        self.item_breakdown_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        item_breakdown_container.addWidget(self.item_breakdown_widget)
        item_breakdown_container.addStretch(1)
        right_column_vertical_layout.addLayout(item_breakdown_container) # Add to right column layout

        # Bin Level Widgets (Below Item Breakdown)
        bin_level_container = QVBoxLayout()
        bin_level_caption_label = QLabel("BIN LEVELS")
        bin_level_caption_label.setFont(QFont("Montserrat", 16, QFont.Bold))
        bin_level_caption_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        bin_level_caption_label.setStyleSheet("color: #185a9d; margin-bottom: 5px;")
        bin_level_container.addWidget(bin_level_caption_label)

        bin_levels_layout = QHBoxLayout()
        bin_levels_layout.setContentsMargins(0, 0, 0, 0)
        bin_levels_layout.setSpacing(10) # Spacing between the two bin widgets

        self.recyclable_bin_widget = BinLevelWidget("Recyclable", "green") # Recyclable bin first
        self.non_recyclable_bin_widget = BinLevelWidget("Non-Recyclable", "black") # Non-Recyclable bin second
        
        bin_levels_layout.addWidget(self.recyclable_bin_widget, alignment=Qt.AlignCenter)
        bin_levels_layout.addWidget(self.non_recyclable_bin_widget, alignment=Qt.AlignCenter)
        bin_level_container.addLayout(bin_levels_layout)
        right_column_vertical_layout.addLayout(bin_level_container) # Add to right column layout

        # Sustainability Summary (Below Bin Levels)
        self.sustainability_summary_widget = QWidget()
        summary_layout = QVBoxLayout(self.sustainability_summary_widget)
        summary_layout.setContentsMargins(10, 10, 10, 10) # Increased padding for the summary widget
        summary_layout.setSpacing(5) # Increased spacing within the summary widget

        summary_caption_label = QLabel("YOUR IMPACT")
        summary_caption_label.setFont(QFont("Montserrat", 16, QFont.Bold))
        summary_caption_label.setAlignment(Qt.AlignCenter)
        summary_caption_label.setStyleSheet("color: #185a9d; margin-bottom: 5px;") # Adjusted margin
        summary_layout.addWidget(summary_caption_label, alignment=Qt.AlignCenter)

        self.sustainability_summary_display_label.setFont(QFont("Montserrat", 14, QFont.Bold))
        self.sustainability_summary_display_label.setStyleSheet("color: #11998e; background: rgba(255,255,255,0.8); padding: 5px; border-radius: 10px;") # Adjusted padding
        self.sustainability_summary_display_label.setAlignment(Qt.AlignCenter)
        summary_layout.addWidget(self.sustainability_summary_display_label)

        self.sustainability_summary_widget.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #e3fcec, stop:1 #d0e7ff); border-radius: 15px; padding: 5px;") # Adjusted padding
        self.sustainability_summary_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        right_column_vertical_layout.addWidget(self.sustainability_summary_widget) # Add to right column layout

        # Add the entire right column layout to the main grid
        content_grid_layout.addLayout(right_column_vertical_layout, 0, 1, 2, 1) # Row 0, Col 1, Span 2 rows, 1 col (adjusted span)

        # Classification Controls (Below Camera Feed, Left Column)
        classification_controls_container = QVBoxLayout()
        self.toggle_button = QPushButton("Switch to YOLO Classification")
        self.toggle_button.setFont(QFont("Montserrat", 10, QFont.Bold))
        self.toggle_button.setStyleSheet(
            """
            QPushButton {
                background-color: #34495e;
                color: #ecf0f1;
                border-radius: 8px;
                padding: 6px 12px;
                border: 1px solid #ecf0f1;
            }
            QPushButton:hover {
                background-color: #2c3e50;
                border-color: #43cea2;
            }
            """
        )
        self.toggle_button.clicked.connect(self.toggle_classification_method)
        classification_controls_container.addWidget(self.toggle_button, alignment=Qt.AlignLeft)

        self.method_label = QLabel("<b>Current: Gemini LLM</b>")
        self.method_label.setFont(QFont("Montserrat", 10))
        self.method_label.setStyleSheet("color: #ecf0f1; background: rgba(0,0,0,0.5); padding: 1px 4px; border-radius: 4px;")
        classification_controls_container.addWidget(self.method_label, alignment=Qt.AlignLeft)
        classification_controls_container.addStretch(1)
        content_grid_layout.addLayout(classification_controls_container, 1, 0, 1, 1) # Row 1, Col 0, Span 1 row, 1 col (adjusted span)


        # Set column and row stretches for proper resizing
        content_grid_layout.setColumnStretch(0, 4) # Camera column
        content_grid_layout.setColumnStretch(1, 2) # Right column

        content_grid_layout.setRowStretch(0, 6) # Camera/Item Breakdown row
        content_grid_layout.setRowStretch(1, 1) # Controls/Bin Levels/Your Impact row

        main_layout.addLayout(content_grid_layout)
        main_layout.setStretchFactor(content_grid_layout, 1)

        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        line2.setStyleSheet("color: #ddd;")
        main_layout.addWidget(line2)

        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(0, 20, 0, 20)
        self.footer_label = QLabel("Powered by <b>Sortyx</b> © 2025")
        self.footer_label.setFont(QFont("Montserrat", 9))
        self.footer_label.setAlignment(Qt.AlignCenter)
        self.footer_label.setStyleSheet("""QLabel{color: #888888; background-color: transparent;}""")
        footer_layout.addStretch()
        footer_layout.addWidget(self.footer_label, alignment=Qt.AlignCenter)
        footer_layout.addStretch()
        main_layout.addLayout(footer_layout)

        self.update_item_breakdown_labels()


    def setup_classification_result_screen(self, parent_widget):
        """Setup the screen to display classification results and directions"""
        main_layout = QVBoxLayout(parent_widget)
        main_layout.setContentsMargins(50, 20, 50, 50)
        main_layout.setSpacing(15)
        parent_widget.setStyleSheet("background-color: #f0f8ff;")

        header = self.create_header_widget()
        main_layout.addWidget(header) 

        line_result_header = QFrame()
        line_result_header.setFrameShape(QFrame.HLine)
        line_result_header.setFrameShadow(QFrame.Sunken)
        line_result_header.setStyleSheet("color: #ddd;")
        main_layout.addWidget(line_result_header)

        content_grid_layout = QGridLayout()
        content_grid_layout.setContentsMargins(0, 0, 0, 0)
        content_grid_layout.setSpacing(15)

        item_bin_layout = QVBoxLayout()
        item_bin_layout.setSpacing(15)

        self.item_name_label = QLabel("Item: Detecting...")
        self.item_name_label.setFont(QFont("Montserrat", 32, QFont.Bold))
        self.item_name_label.setAlignment(Qt.AlignCenter)
        self.item_name_label.setStyleSheet("color: #185a9d; background: #fff; border-radius: 20px; padding: 25px;")
        shadow_item = QGraphicsDropShadowEffect()
        shadow_item.setBlurRadius(15)
        shadow_item.setColor(QColor(0, 0, 0, 80))
        shadow_item.setOffset(5, 5)
        self.item_name_label.setGraphicsEffect(shadow_item)
        item_bin_layout.addWidget(self.item_name_label)

        self.bin_instruction_label = QLabel("Bin: Waiting for classification...")
        self.bin_instruction_label.setFont(QFont("Montserrat", 28, QFont.Bold))
        self.bin_instruction_label.setAlignment(Qt.AlignCenter)
        self.bin_instruction_label.setStyleSheet("color: #333; background: #fff; border-radius: 20px; padding: 25px;")
        shadow_bin = QGraphicsDropShadowEffect()
        shadow_bin.setBlurRadius(15)
        shadow_bin.setColor(QColor(0, 0, 0, 80))
        shadow_bin.setOffset(5, 5)
        self.bin_instruction_label.setGraphicsEffect(shadow_bin)
        item_bin_layout.addWidget(self.bin_instruction_label)
        
        content_grid_layout.addLayout(item_bin_layout, 0, 0, 1, 1)

        self.main_bin_person_image_label = QLabel() 
        self.main_bin_person_image_label.setFixedSize(450, 450)
        self.main_bin_person_image_label.setAlignment(Qt.AlignCenter)
        self.main_bin_person_image_label.setStyleSheet("background-color: transparent; border-radius: 15px;")
        content_grid_layout.addWidget(self.main_bin_person_image_label, 0, 1, 2, 1, alignment=Qt.AlignCenter)

        self.message_label = QLabel("")
        self.message_label.setFont(QFont("Montserrat", 20, QFont.Bold))
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setWordWrap(True)
        shadow_msg = QGraphicsDropShadowEffect()
        shadow_msg.setBlurRadius(10)
        shadow_msg.setColor(QColor(0, 0, 0, 60))
        shadow_msg.setOffset(3, 3)
        self.message_label.setGraphicsEffect(shadow_msg)
        content_grid_layout.addWidget(self.message_label, 1, 0, 1, 1)

        bottom_images_qr_layout = QHBoxLayout()
        bottom_images_qr_layout.setContentsMargins(0, 0, 0, 0)
        bottom_images_qr_layout.setSpacing(15)

        self.left_side_image_label = QLabel()
        self.left_side_image_label.setFixedSize(400, 400)
        self.left_side_image_label.setAlignment(Qt.AlignCenter)
        self.left_side_image_label.setStyleSheet("background-color: transparent; border-radius: 15px;") 
        bottom_images_qr_layout.addWidget(self.left_side_image_label, alignment=Qt.AlignBottom | Qt.AlignLeft)
        
        # Removed QR code elements from this screen
        # self.result_qr_label = QLabel()
        # self.result_qr_label.setFixedSize(150, 150)
        # self.result_qr_label.setAlignment(Qt.AlignCenter)
        # self.result_qr_label.setStyleSheet("background-color: transparent; border: none;") 
        # 
        # self.qr_message_label = QLabel("Scan for offers")
        # self.qr_message_label.setFont(QFont("Montserrat", 12))
        # self.qr_message_label.setStyleSheet("color: #555; margin-top: 5px; background-color: transparent; border: none;")
        # self.qr_message_label.setAlignment(Qt.AlignCenter)
        # 
        # qr_and_text_vertical_layout = QVBoxLayout()
        # qr_and_text_vertical_layout.setAlignment(Qt.AlignCenter)
        # qr_and_text_vertical_layout.addWidget(self.result_qr_label)
        # qr_and_text_vertical_layout.addWidget(self.qr_message_label)
        # 
        # bottom_images_qr_layout.addLayout(qr_and_text_vertical_layout) 

        self.bottom_right_image_label = QLabel()
        self.bottom_right_image_label.setFixedSize(400, 400)
        self.bottom_right_image_label.setAlignment(Qt.AlignCenter)
        self.bottom_right_image_label.setStyleSheet("background-color: transparent; border-radius: 15px;")
        bottom_images_qr_layout.addWidget(self.bottom_right_image_label, alignment=Qt.AlignBottom | Qt.AlignRight)

        content_grid_layout.addLayout(bottom_images_qr_layout, 2, 0, 1, 2)

        content_grid_layout.setColumnStretch(0, 1)
        content_grid_layout.setColumnStretch(1, 1)

        content_grid_layout.setRowStretch(0, 1)
        content_grid_layout.setRowStretch(1, 1)
        content_grid_layout.setRowStretch(2, 1)

        main_layout.addLayout(content_grid_layout)

        line_result_footer = QFrame()
        line_result_footer.setFrameShape(QFrame.HLine)
        line_result_footer.setFrameShadow(QFrame.Sunken)
        line_result_footer.setStyleSheet("color: #ddd;")
        main_layout.addWidget(line_result_footer)

        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(0, 5, 0, 5)
        self.footer_label = QLabel("Powered by <b>Sortyx</b> © 2025")
        self.footer_label.setFont(QFont("Montserrat", 9))
        self.footer_label.setAlignment(Qt.AlignCenter)
        self.footer_label.setStyleSheet("""QLabel{color: #888888; background-color: transparent;}""")
        footer_layout.addStretch()
        footer_layout.addWidget(self.footer_label, alignment=Qt.AlignCenter)
        footer_layout.addStretch()
        main_layout.addLayout(footer_layout)

        self.image_animation_main_bin = QPropertyAnimation(self.main_bin_person_image_label, b"windowOpacity")
        self.image_animation_main_bin.setDuration(600)
        self.image_animation_main_bin.setLoopCount(-1)
        self.image_animation_main_bin.setStartValue(1.0)
        self.image_animation_main_bin.setEndValue(0.8)
        self.image_animation_main_bin.setEasingCurve(QEasingCurve.InOutSine)


    def connect_signals(self):
        """Connect thread signals to UI updates"""
        self.intro_video_player_thread.frame_ready.connect(self.update_intro_video_frame)
        self.intro_video_player_thread.video_ended.connect(self.on_intro_video_ended)

        self.instructional_video_player_thread.frame_ready.connect(self.update_instructional_video_frame)

        self.camera_thread.parent_app = self
        self.camera_thread.frame_ready.connect(self.update_camera_display)
        self.camera_thread.object_detected_for_classification.connect(self.auto_classify_object)
        self.camera_thread.person_only_detected.connect(self.on_person_only_detected)
        self.camera_thread.proximity_changed.connect(self.on_proximity_changed)
        
        self.classification_thread.classification_ready.connect(self.on_classification_ready)
        self.gemini_classification_result.connect(self.on_classification_ready)

        # Connect Firebase data ready signal to a new slot
        self.firebase_fetcher.bin_data_ready.connect(self.update_bin_level_ui)


    def start_intro_video_mode(self):
        """Switch to full-screen intro video playback mode."""
        print("WasteSorterApp: Switching to intro video playback mode.")
        self.current_mode = "intro_video"
        self.stacked_widget.setCurrentWidget(self.intro_video_screen)
        
        if not self.intro_video_player_thread.isRunning():
            self.intro_video_player_thread.start()
        self.intro_video_player_thread.resume()

        self.instructional_video_player_thread.pause()


    def on_intro_video_ended(self):
        """Called when the introductory video finishes playing."""
        print("WasteSorterApp: Introductory video ended. Transitioning to instructional video mode.")
        self.intro_video_player_thread.stop()
        self.start_instructional_video_mode()


    def start_instructional_video_mode(self):
        """Switch to main instructional video playback mode (looping)."""
        print("WasteSorterApp: Switching to instructional video mode.")
        self.current_mode = "instructional_video"
        self.stacked_widget.setCurrentWidget(self.instructional_video_screen)
        
        if not self.instructional_video_player_thread.isRunning():
            self.instructional_video_player_thread.start()
        self.instructional_video_player_thread.resume()

        if not self.camera_thread.isRunning():
            self.camera_thread.start()
        self.camera_thread.reset_state()

        self.update_sustainability_summary_label()
        self.update_item_breakdown_labels()


    def start_camera_detection_mode(self):
        """Switch to live camera display mode (person detected)."""
        if self.current_mode == "camera_detection" and self.stacked_widget.currentWidget() == self.live_camera_screen:
            return

        print("WasteSorterApp: Switching to camera detection mode.")
        self.current_mode = "camera_detection"
        self.stacked_widget.setCurrentWidget(self.live_camera_screen)
        
        self.instructional_video_player_thread.pause()

        self.update_sustainability_summary_label()
        self.update_item_breakdown_labels()


    def update_intro_video_frame(self, pixmap):
        """Slot to receive and display a new intro video frame."""
        if self.current_mode == "intro_video":
            scaled_pixmap = pixmap.scaled(self.intro_video_display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.intro_video_display_label.setPixmap(scaled_pixmap)


    def update_instructional_video_frame(self, pixmap):
        """Slot to receive and display a new instructional video frame."""
        if self.current_mode == "instructional_video":
            scaled_pixmap = pixmap.scaled(self.instructional_video_display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.instructional_video_display_label.setPixmap(scaled_pixmap)


    def update_camera_display(self, frame):
        """Update the live camera feed display."""
        if self.current_mode == "camera_detection":
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

            scaled_pixmap = pixmap.scaled(self.live_display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            painter = QPainter(scaled_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)

            if self.camera_thread.classification_trigger_state == self.camera_thread.STATE_PERSON_DETECTED:
                message = "Please present a waste item for sorting."
                font = QFont("Montserrat", 15, QFont.Bold)
                painter.setFont(font)
                painter.setPen(QColor(255, 255, 255))
                
                text_size = painter.fontMetrics().boundingRect(message).width()
                text_x = (scaled_pixmap.width() - text_size) // 2
                text_y = scaled_pixmap.height() - 15

                bg_rect_y = scaled_pixmap.height() - 50
                painter.fillRect(0, bg_rect_y, scaled_pixmap.width(), 50, QColor(0, 0, 0, 150))

                painter.drawText(text_x, text_y, message)

            painter.end()

            self.live_display_label.setPixmap(scaled_pixmap)

    def auto_classify_object(self, frame, bbox, yolo_class_name):
        """Automatically trigger classification when an object is detected."""
        print("Auto-classify triggered by object detection.")
        if self.awaiting_disposal:
            print("Already awaiting disposal, ignoring auto-classify.")
            return

        if not self.camera_thread.cap or not self.camera_thread.cap.isOpened():
            print("Auto-classify: Camera not available, cannot classify.")
            return
        if not self.camera_thread.person_currently_detected:
            print("Auto-classify: No person detected, skipping classification.")
            return

        proximity, detection_info, _, _ = self.camera_thread.check_proximity_and_objects(frame)
        if not proximity or detection_info != "objects_for_classification" or not bbox:
            print("Auto-classify: No valid object for classification detected. Skipping.")
            self.camera_thread.reset_state()
            return

        self.last_captured_frame = frame
        self.last_captured_bbox = bbox
        self.last_yolo_class_name = yolo_class_name

        self.stacked_widget.setCurrentWidget(self.classification_result_screen)
        self.awaiting_disposal = True

        self.item_name_label.setText("Item: Classifying...")
        self.bin_instruction_label.setText("Bin: Please wait...")
        self.message_label.clear()
        
        self.left_side_image_label.setPixmap(QPixmap())
        self.bottom_right_image_label.setPixmap(QPixmap())
        self.main_bin_person_image_label.setPixmap(QPixmap())
        self.image_animation_main_bin.stop() 

        if self.classification_method == 'Gemini':
            print("Starting Gemini classification (auto-triggered)...")
            threading.Thread(target=self._run_gemini_classification).start()
        else:
            print("Starting YOLO classification (auto-triggered)...")
            self.classification_thread.classify_waste(self.last_captured_frame, self.last_captured_bbox, self.last_yolo_class_name)

    def _run_gemini_classification(self):
        """Helper to run Gemini classification in a separate thread."""
        print("_run_gemini_classification: Attempting to classify.")
        if self.last_captured_frame is None or self.last_captured_bbox is None:
            print("Error: No frame or bbox captured for Gemini classification.")
            self.gemini_classification_result.emit("Error", 0.0, np.zeros((100,100,3), dtype=np.uint8), "Error", "", "No image captured.", "No image captured.")
            return

        x1, y1, x2, y2 = self.last_captured_bbox
        h, w, _ = self.last_captured_frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            print(f"Error: Invalid bounding box for cropping: ({x1},{y1},{x2},{y2}) in frame of size {w}x{h}.")
            self.gemini_classification_result.emit("Error", 0.0, np.zeros((100,100,3), dtype=np.uint8), "Error", "", "Invalid crop area.", "Invalid crop area.")
            return

        cropped_image = self.last_captured_frame[y1:y2, x1:x2]
        if cropped_image.size == 0:
            print("Error: Cropped image is empty after bounding box check.")
            self.gemini_classification_result.emit("Error", 0.0, np.zeros((100,100,3), dtype=np.uint8), "Error", "", "Empty cropped image.", "Empty cropped image.")
            return
        
        print(f"Cropped image size for Gemini: {cropped_image.shape}")

        result, explanation, item_name_from_gemini = classify_with_gemini(cropped_image)
        self.gemini_classification_result.emit(result, 1.0, cropped_image, item_name_from_gemini, self.last_yolo_class_name, explanation, explanation)


    def on_object_detected_for_classification(self, frame, bbox, yolo_class_name):
        pass

    def on_person_only_detected(self, frame):
        pass

    def on_proximity_changed(self, is_near):
        """Handle proximity detection changes to switch between instructional video and camera detection."""
        print(f"on_proximity_changed: is_near={is_near}, current_mode={self.current_mode}, awaiting_disposal={self.awaiting_disposal}")
        
        if self.awaiting_disposal:
            print("Proximity change ignored: Awaiting disposal confirmation.")
            return

        if is_near and self.current_mode == "instructional_video":
            print("Switching to camera detection mode due to proximity.")
            self.start_camera_detection_mode()
        elif not is_near and self.current_mode == "camera_detection":
            print("Switching to instructional video mode due to person lost.")
            self.start_instructional_video_mode()


    def on_classification_ready(self, classification, confidence, cropped_image, class_name_from_classifier, yolo_class_name=None, full_response=None, explanation=None):
        """Handle classification results and update the classification result screen"""
        print(f"on_classification_ready: Classification received: {classification}, Item: {class_name_from_classifier}")

        final_classification = classification
        final_confidence = confidence
        final_class_name = class_name_from_classifier

        if self.classification_method == 'YOLO':
            if final_classification == "Unknown" or final_confidence < 0.80:
                 final_classification = "Non-Recyclable"
                 final_class_name = "Unknown Item"

        self.current_classification = final_classification
        self.classified_object_name = final_class_name

        self.left_side_image_label.setPixmap(QPixmap())
        self.bottom_right_image_label.setPixmap(QPixmap())
        self.main_bin_person_image_label.setPixmap(QPixmap())
        self.image_animation_main_bin.stop() 
        # Removed QR code hide calls from here
        # self.result_qr_label.hide()
        # self.qr_message_label.hide()

        if final_classification == "Unknown" or "Error" in final_classification or final_class_name == "No Waste Object":
            item_text = "Item: <b>UNKNOWN ITEM</b>"
            bin_text = "Bin: <span style='color: #888;'><b>PLEASE TRY AGAIN</b></span>"
            message_text = ""
            self.bin_instruction_label.setStyleSheet("color: #888; background: #f0f0f0; border-radius: 20px; padding: 25px;")
            self.item_name_label.setStyleSheet("color: #185a9d; background: #fff; border-radius: 20px; padding: 25px;")
            self.message_label.setStyleSheet("color: transparent;")

            QTimer.singleShot(1000, self.reset_system_after_delay)
        else:
            item_text = f"Item: <b>{final_class_name.upper()}</b>"
            
            material_type = next((k for k, v in self.sustainability_calc.material_mapping.items() if final_class_name.lower() in v), 'other')
            co2_val_for_display = self.sustainability_calc.emission_factors.get(material_type, 0.0) if final_classification == "Recyclable" else 0.0

            if final_classification == "Recyclable":
                threading.Thread(target=Checking.recycle_bin).start()
                bin_text = "Bin: <span style='color: #28a745;'><b>GREEN BIN</b></span>"
                message_text = f"🌍 You just saved {co2_val_for_display:.2f} kg CO2 🌍"
                self.bin_instruction_label.setStyleSheet("color: #28a745; background: #e0f7fa; border-radius: 20px; padding: 25px;")
                self.item_name_label.setStyleSheet("color: #185a9d; background: #e3fcec; border-radius: 20px; padding: 25px;")
                self.message_label.setStyleSheet("color: #28a745; margin-top: 10px; background: transparent; border-radius: 15px; padding: 15px;")
                self.left_side_image_label.setPixmap(self.left_side_recyclable_pixmap)
                self.main_bin_person_image_label.setPixmap(self.green_bin_person_pixmap)
                self.image_animation_main_bin.start()
                self.bottom_right_image_label.setPixmap(QPixmap()) 
               

            else: # Non-Recyclable
                bin_text = "Bin: <span style='color: #dc3545;'><b>BLACK BIN</b></span>"
                threading.Thread(target=Checking.non_recycle_bin).start()
                message_text = "❌ This item can't be recycled... ❌"
                self.bin_instruction_label.setStyleSheet("color: #dc3545; background: #ffe0e0; border-radius: 20px; padding: 25px;")
                self.item_name_label.setStyleSheet("color: #185a9d; background: #ffe0e0; border-radius: 20px; padding: 25px;")
                self.message_label.setStyleSheet("color: #dc3545; margin-top: 10px; background: transparent; border-radius: 15px; padding: 15px;")
                self.left_side_image_label.setPixmap(QPixmap())
                self.main_bin_person_image_label.setPixmap(self.black_bin_person_pixmap)
                self.image_animation_main_bin.start()
                self.bottom_right_image_label.setPixmap(self.right_side_non_recyclable_pixmap)
                

            # Removed QR code display from here, it will be in thank you message
            # self.show_qr_code()
            # self.result_qr_label.show()
            # self.qr_message_label.show()
            
            QTimer.singleShot(4000, self.auto_confirm_disposal)

        self.item_name_label.setText(item_text)
        self.bin_instruction_label.setText(bin_text)
        self.message_label.setText(message_text)

    def auto_confirm_disposal(self):
        """Automatically confirm disposal after classification display."""
        print("Auto-confirm disposal triggered.")
        if self.current_classification and self.current_classification not in ["Unknown", "Error", "No Waste Object"]:
            self.sustainability_calc.add_item(self.current_classification, self.classified_object_name)
            self.update_sustainability_summary_label()
            self.update_item_breakdown_labels()
            self.display_thank_you_message("Thank You!", "Your item has been recorded. Helping to build a sustainable future!")
        else:
            self.reset_system_after_delay()


    def display_thank_you_message(self, title, message, duration_ms=8000):
        """Displays a temporary, buttonless message overlay with a QR code."""
        self.message_overlay = QWidget(self)
        self.message_overlay.setGeometry(self.rect())
        self.message_overlay.setStyleSheet("background-color: rgba(0, 0, 0, 0.7);")
        self.message_overlay.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)

        overlay_layout = QVBoxLayout(self.message_overlay)
        overlay_layout.setAlignment(Qt.AlignCenter)

        # Create a central container widget for the message and QR code
        message_box_widget = QWidget()
        message_box_layout = QVBoxLayout(message_box_widget)
        message_box_layout.setAlignment(Qt.AlignCenter) # Align content of the box to center
        message_box_widget.setStyleSheet("background-color: #43cea2; border-radius: 20px; padding: 30px;")
        
        # Message Label
        msg_label = QLabel(f"<b>{title}</b><br><br>{message}")
        msg_label.setFont(QFont("Montserrat", 24, QFont.Bold))
        msg_label.setAlignment(Qt.AlignCenter)
        msg_label.setStyleSheet("color: white; background-color: transparent; border: none;") # Transparent background
        msg_label.setWordWrap(True)
        msg_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        
        message_box_layout.addWidget(msg_label, alignment=Qt.AlignCenter)

        # QR code container
        qr_label = QLabel()
        qr_label.setFixedSize(150, 150)
        qr_label.setAlignment(Qt.AlignCenter) # This centers the pixmap within its label
        qr_label.setStyleSheet("background-color: white; border-radius: 10px; padding: 5px;") # White background for QR

        qr_message_label = QLabel("Scan for offers")
        qr_message_label.setFont(QFont("Montserrat", 12, QFont.Bold))
        qr_message_label.setStyleSheet("color: white; margin-top: 10px; background-color: transparent;") # Transparent background
        qr_message_label.setAlignment(Qt.AlignCenter)

        # ✅ Set the QR image here
        self._generate_and_set_qr_code(qr_label)

        qr_and_text_vertical_layout = QVBoxLayout()
        qr_and_text_vertical_layout.setAlignment(Qt.AlignCenter) # This centers the QR label and text within its layout
        qr_and_text_vertical_layout.addWidget(qr_label)
        qr_and_text_vertical_layout.addWidget(qr_message_label)
        
        # Create a horizontal layout to center the qr_and_text_vertical_layout
        centered_qr_horizontal_layout = QHBoxLayout()
        centered_qr_horizontal_layout.addStretch() # Push content to center from left
        centered_qr_horizontal_layout.addLayout(qr_and_text_vertical_layout)
        centered_qr_horizontal_layout.addStretch() # Push content to center from right

        message_box_layout.addLayout(centered_qr_horizontal_layout) # Add the new centered horizontal layout to the main message box layout

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(8, 8)
        message_box_widget.setGraphicsEffect(shadow)

        overlay_layout.addWidget(message_box_widget, alignment=Qt.AlignCenter) # This centers the message box widget on the overlay
        self.message_overlay.show()

        # Generate and set the QR code
        #self._generate_and_set_qr_code("/home/srini/Code/bin/bin1/Recycle/images/My_Coupon.jpg")

        QTimer.singleShot(duration_ms, self.hide_temporary_message)

    def hide_temporary_message(self):
        """Hides the temporary message overlay and triggers system reset."""
        if hasattr(self, 'message_overlay') and self.message_overlay:
            self.message_overlay.hide()
            self.message_overlay.deleteLater()
            self.message_overlay = None
        
        self.reset_system_after_delay()


    def get_bin_fill_level(self):
        """Mock bin fill level as sensor is removed. Returns None."""
        return None

    # def _generate_and_set_qr_code(self, target_qr_label):
    #     """Generate and display QR code for points collection on the given QLabel."""
    #     qr_data = {
    #         "user_action": "waste_disposal",
    #         "classification": self.current_classification,
    #         "object_name": self.classified_object_name,
    #         "timestamp": datetime.now().isoformat(),
    #         "points_earned": 10 if self.current_classification == "Recyclable" else 5
    #     }
    #     qr = qrcode.QRCode(version=1, box_size=5, border=2)
    #     qr.add_data(json.dumps(qr_data))
    #     qr.make(fit=True)
    #     temp_qr_path = os.path.join(SCRIPT_DIR, 'images', "temp_qr.png")
    #     qr_img = qr.make_image(fill_color="black", back_color="white")
    #     qr_img.save(temp_qr_path)
        
    #     pixmap = QPixmap(temp_qr_path)
    #     scaled_pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    #     target_qr_label.setPixmap(scaled_pixmap)
    #     os.remove(temp_qr_path)

    def _generate_and_set_qr_code(self, target_qr_label, image_path="/home/guna/Desktop/Sortyx/Smartbin/images/My_Coupon.jpg"):
        """Display an existing QR code image in the GUI."""
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        target_qr_label.setPixmap(scaled_pixmap)


    def update_sustainability_summary_label(self):
        """Update the sustainability summary labels."""
        report = self.sustainability_calc.get_sustainability_report()
        summary_text = (
            f"♻️ Recycled: {report['items_recycled']} items | "
            f"🌳 CO₂ Saved: {report['total_co2_saved_kg']} kg"
        )
        if self.sustainability_summary_display_label:
            self.sustainability_summary_display_label.setText(summary_text)
        print(f"Sustainability Summary Updated: {summary_text}")


    def update_item_breakdown_labels(self):
        """Update the item breakdown labels on the live camera screen."""
        report = self.sustainability_calc.get_sustainability_report()
        breakdown = report['breakdown']
        print(f"Updating breakdown labels with: {breakdown}")
        self.breakdown_labels["E-waste"].setText(str(breakdown.get('e_waste', 0)))
        self.breakdown_labels["Paper/Cardboard"].setText(str(breakdown.get('paper_cardboard', 0)))
        self.breakdown_labels["Organic"].setText(str(breakdown.get('organic', 0)))
        self.breakdown_labels["Others"].setText(str(breakdown.get('others', 0)))
        self.breakdown_labels["Metal"].setText(str(breakdown.get('metal', 0)))
        self.breakdown_labels["Plastic"].setText(str(breakdown.get('plastic', 0)))

    def update_bin_level_ui(self, bin_data):
        """
        Slot to receive updated bin data from FirebaseDataFetcher and update UI.
        """
        bin1_level = bin_data.get('bin1') # Non-Recyclable
        bin2_level = bin_data.get('bin2') # Recyclable

        if bin1_level is not None:
            self.non_recyclable_bin_widget.set_percentage(bin1_level)
            print(f"UI Updated: Non-Recyclable Bin Level: {bin1_level}%")
        else:
            self.non_recyclable_bin_widget.set_percentage(0) # Or some default/error state
            # Removed status_text update for N/A as status text is removed
            print("UI Update: Non-Recyclable Bin data not available.")

        if bin2_level is not None:
            self.recyclable_bin_widget.set_percentage(bin2_level)
            print(f"UI Updated: Recyclable Bin Level: {bin2_level}%")
        else:
            self.recyclable_bin_widget.set_percentage(0) # Or some default/error state
            # Removed status_text update for N/A as status text is removed
            print("UI Update: Recyclable Bin data not available.")


    def reset_system_after_delay(self):
        """Reset the system to initial state, potentially switching back to video."""
        print("System reset initiated.")
        self.current_classification = None
        self.classified_object_name = None
        self.awaiting_disposal = False
        self.last_captured_frame = None
        self.last_captured_bbox = None
        self.last_yolo_class_name = None
        
        self.camera_thread.reset_state()

        self.item_name_label.clear()
        self.bin_instruction_label.clear()
        self.message_label.clear()
        # Removed QR code hide calls from here
        # self.result_qr_label.hide()
        # self.qr_message_label.hide()
        self.left_side_image_label.setPixmap(QPixmap())
        self.bottom_right_image_label.setPixmap(QPixmap())
        self.main_bin_person_image_label.setPixmap(QPixmap())
        self.image_animation_main_bin.stop()

        if self.camera_thread.person_currently_detected:
            print("Person still detected after reset. Switching to camera detection mode.")
            self.start_camera_detection_mode()
        else:
            print("No person detected after reset. Switching to instructional video mode.")
            self.start_instructional_video_mode()


    def toggle_classification_method(self):
        """Toggle between Gemini LLM and YOLO classification methods"""
        if self.classification_method == "Gemini":
            self.classification_method = "YOLO"
            self.toggle_button.setText("Switch to LLM Classification")
            self.method_label.setText("<b>Current: YOLO</b>")
            print("Switched to YOLO Classification")
        else:
            self.classification_method = "Gemini"
            self.toggle_button.setText("Switch to YOLO Classification")
            self.method_label.setText("<b>Current: Gemini LLM</b>")
            print("Switched to Gemini LLM Classification")

    '''def closeEvent(self, event):
        """Clean shutdown of all threads"""
        print("Closing application, stopping threads...")
        self.camera_thread.stop()
        self.intro_video_player_thread.stop()
        self.instructional_video_player_thread.stop()
        self.firebase_fetcher.stop() # Stop the Firebase fetcher thread
        event.accept()'''
        
    """Ensure all threads are stopped cleanly when the window is closed."""
    
    def closeEvent(self, event):
    #"""Ensure all threads are stopped cleanly when the window is closed."""
        try:
            if self.camera_thread.isRunning():
                self.camera_thread.stop()
                self.camera_thread.wait()

            if self.classification_thread.isRunning():
                self.classification_thread.quit()
                self.classification_thread.wait()

            if self.intro_video_player_thread.isRunning():
                self.intro_video_player_thread.stop()
                self.intro_video_player_thread.wait()

            if self.instructional_video_player_thread.isRunning():
                self.instructional_video_player_thread.stop()
                self.instructional_video_player_thread.wait()

            if self.firebase_fetcher.isRunning():
                self.firebase_fetcher.stop()
                self.firebase_fetcher.wait()

        except Exception as e:
            print(f"[Cleanup Error] {e}")

    # Let Qt continue closing the window
        event.accept()
  

def classify_with_gemini(image_np):
    """
    Use Gemini GenerativeModel client to classify the image as 'Recyclable' or 'Non-Recyclable'.
    Returns (classification, explanation, item_name)
    """
    
    load_dotenv()
    #api_key = os.getenv("GOOGLE_API_KEY")
    api_key = "AIzaSyDLM9AFv8YSKTyMQrlQR8dk0B2zaSRhI_k"
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        return "Error", "API key not found.", "API key not found."

    genai.configure(api_key=api_key)
    # Reverted to gemini-2.0-flash for consistency with the working file
    model = genai.GenerativeModel('gemini-2.0-flash')

    _, buffer = cv2.imencode('.jpg', image_np)
    img_bytes = buffer.tobytes()

    # Reverted to the simpler, more robust prompt from oldfile.py
    prompt = """
        You are an expert in waste management. Your primary task is to identify a clear waste object in the image and classify it.
        
        1.  **Classification**: Classify the object as 'Recyclable' or 'Non-Recyclable'.
        2.  **Explanation**: Provide a brief, one-sentence explanation for your classification.
        3.  **E-Waste**: Be specific. Items like phones, batteries, and cables are 'Recyclable' as e-waste.
        4.  **No Object**: If no clear waste object is present, respond ONLY with 'No Waste Object Detected.'

        **Examples:**
        -   'Recyclable: Plastic Bottle. This is recyclable as it's typically made of PET plastic.'
        -   'Recyclable: Mobile Phone. Electronics should be recycled at a designated e-waste facility.'
        -   'Recyclable: Aluminum Can. Metal cans are widely recyclable and can be melted down and reused.' # Added specific metal example
        -   'Non-Recyclable: Food Scraps. Organic waste should be composted, not recycled.'

        **Recyclable materials include**: paper, plastic bottles, glass, metal, cardboard, electronics (e-waste).
        **Non-recyclable materials include**: food waste, organic matter, ceramics, plastic bags, styrofoam, soiled paper.

        Analyze the image and provide your classification.
    """
    try:
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": img_bytes}
        ])
        text = response.text.strip()
        print("[Gemini LLM Full Response]:\n" + text)
        text_lower = text.lower()

        classification = "Unknown"
        explanation = text
        item_name = "Unknown Item"

        if "no waste object detected" in text_lower:
            return "Unknown", "No clear waste object detected in the image.", "No Waste Object"

        if "non-recyclable" in text_lower or "non recyclable" in text_lower:
            classification = "Non-Recyclable"
        elif "recyclable" in text_lower:
            classification = "Recyclable"

        # Restored flexible parsing logic from oldfile.py
        if ":" in text:
            try:
                _, item_and_explanation = text.split(":", 1)
                item_and_explanation = item_and_explanation.strip()
                explanation = item_and_explanation

                first_sentence_end = item_and_explanation.find(".")
                if first_sentence_end != -1:
                    item_name = item_and_explanation[:first_sentence_end].strip()
                else:
                    item_name = item_and_explanation
            except ValueError:
                pass

        if item_name == "Unknown Item" or len(item_name.split()) > 5:
            common_objects = ["plastic bottle", "glass jar", "paper cup", "cardboard box", "metal can",
                              "food scraps", "banana peel", "apple core", "plastic bag", "styrofoam",
                              "ceramic plate", "coffee cup", "magazine", "newspaper", "tin can", "aluminum foil",
                              "mobile phone", "cell phone", "smartphone", "battery"]
            for obj in common_objects:
                if obj in text_lower:
                    item_name = obj.title()
                    break

        if item_name == "Unknown Item":
             item_name = ' '.join(text.split()[:4]).replace(":", "").replace(".", "").strip()
             if len(item_name) > 25: item_name = item_name[:25] + "..."

        return classification, explanation, item_name

    except Exception as e:
        print(f"[GeminiAPI] Error: {e}")
        return "Error", str(e), "Error"

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    window = WasteSorterApp()
    window.show()
    window.showFullScreen()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
