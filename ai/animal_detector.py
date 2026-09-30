"""
100% Offline Wildlife & Farm Intrusion Animal Detection Engine
Powered by models from D:\animal_detection_model (YOLO11 / YOLOv8)
Seamlessly integrates with AgroEye's Unified Vision Pipeline.
"""

import os
import sys
import time
import types
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import cv2
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    torch = None
    TORCH_AVAILABLE = False

# Ensure clean torchvision shim to bypass Windows WDAC DLL issues on AppData
def _ensure_torchvision_shim():
    if not TORCH_AVAILABLE:
        return
    if "torchvision" not in sys.modules or not hasattr(sys.modules.get("torchvision"), "ops"):
        def pytorch_nms(boxes, scores, iou_threshold):
            if boxes.numel() == 0:
                return torch.empty((0,), dtype=torch.long, device=boxes.device)
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)
            order = scores.argsort(descending=True)
            keep = []
            while order.numel() > 0:
                i = order[0].item()
                keep.append(i)
                if order.numel() == 1:
                    break
                xx1 = torch.maximum(x1[i], x1[order[1:]])
                yy1 = torch.maximum(y1[i], y1[order[1:]])
                xx2 = torch.minimum(x2[i], x2[order[1:]])
                yy2 = torch.minimum(y2[i], y2[order[1:]])
                w = torch.clamp(xx2 - xx1, min=0)
                h = torch.clamp(yy2 - yy1, min=0)
                inter = w * h
                ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
                inds = (ovr <= iou_threshold).nonzero(as_tuple=False).view(-1)
                if inds.numel() == 0:
                    break
                order = order[inds + 1]
            return torch.tensor(keep, dtype=torch.long, device=boxes.device)

        fake_tv = types.ModuleType("torchvision")
        fake_tv.ops = types.ModuleType("torchvision.ops")
        fake_tv.ops.nms = pytorch_nms
        sys.modules["torchvision"] = fake_tv
        sys.modules["torchvision.ops"] = fake_tv.ops

_ensure_torchvision_shim()

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    YOLO_AVAILABLE = False

if TORCH_AVAILABLE and hasattr(torch, "inference_mode"):
    TORCH_INFERENCE = torch.inference_mode
elif TORCH_AVAILABLE and hasattr(torch, "no_grad"):
    TORCH_INFERENCE = torch.no_grad
else:
    import contextlib
    TORCH_INFERENCE = contextlib.nullcontext

from utils.logger import logger

# Standard High-Accuracy Supervised Animal Classes (COCO Dataset IDs)
COCO_ANIMAL_MAP = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe"
}

# Agricultural Crop-Threat Species (animals that damage farm crops)
CROP_THREAT_SPECIES = {
    "cow", "bull", "elephant", "wild boar", "monkey", "deer", "nilgai",
    "pig", "goat", "sheep", "horse", "bear", "onager", "zebra"
}

# Distinct High-Visibility Color Palette for HUD (BGR)
ANIMAL_SPECIES_COLORS = {
    "cow": (0, 165, 255),       # Vibrant Orange
    "bull": (0, 120, 255),      # Deep Orange-Red
    "dog": (0, 240, 255),       # Bright Yellow
    "cat": (255, 105, 180),     # Hot Pink
    "elephant": (255, 120, 0),  # Bright Sky Blue
    "horse": (19, 69, 139),     # Saddle Brown
    "sheep": (240, 255, 255),   # Ivory White
    "goat": (255, 191, 0),      # Deep Cyan / Azure
    "bird": (255, 255, 0),      # Cyan
    "bear": (128, 128, 0),      # Olive / Teal
    "zebra": (220, 220, 220),   # Silver Grey
    "giraffe": (0, 215, 255),   # Gold
    "wild boar": (0, 50, 255),  # Crimson Red
    "monkey": (0, 200, 255),    # Golden Amber
    "deer": (34, 139, 34),      # Forest Green
    "nilgai": (128, 0, 128),    # Royal Purple
    "pig": (203, 192, 255),     # Light Pink
    "onager": (50, 205, 50)     # Lime Green
}


class AnimalDetectionResult:
    """Standardized result container for animal and wildlife detections."""
    def __init__(
        self,
        has_animals: bool = False,
        has_threat: bool = False,
        total_animals: int = 0,
        counts: Optional[Dict[str, int]] = None,
        top_animal: str = "None",
        top_confidence: float = 0.0,
        display_confidence: int = 0,
        threat_level: str = "Clear",
        boxes: Optional[List[Dict[str, Any]]] = None,
        latency_ms: int = 0,
        model_name: str = "YOLO11 Animal Model"
    ):
        self.has_animals = has_animals
        self.has_threat = has_threat
        self.total_animals = total_animals
        self.counts = counts or {}
        self.top_animal = top_animal
        self.top_confidence = top_confidence
        self.display_confidence = display_confidence
        self.threat_level = threat_level
        self.boxes = boxes or []
        self.latency_ms = latency_ms
        self.model_name = model_name


class AnimalDetector:
    """
    Unified Offline Wildlife & Farm Intrusion Detector.
    Runs YOLO11 / YOLOv8 animal detection models on live video frames.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45
    ):
        self.conf_thresh = confidence_threshold
        self.iou_thresh = iou_threshold
        self.model_path = model_path
        self.animal_model = None
        self.lock = threading.Lock()
        self.model_loaded = False
        self.loaded_path = ""
        self.num_classes = 0
        self.target_class_ids: Optional[List[int]] = None
        self.class_names: Dict[int, str] = {}
        self.last_latency_ms = 0
        self.last_detection_time = 0.0
        self.persistence_sec = 1.5  # Bounding box stability window

        # Persisted state for smooth rendering
        self.persisted_boxes: List[Dict[str, Any]] = []
        self.persisted_result: Optional[AnimalDetectionResult] = None

        if YOLO_AVAILABLE:
            self._load_model()

    def _load_model(self):
        """Discovers and initializes animal model weights across Windows, Linux, and Raspberry Pi."""
        candidates = []
        if self.model_path:
            candidates.append(Path(self.model_path))
        
        base_dir = Path(__file__).resolve().parent.parent

        candidates.extend([
            # 1. Project local models directory (Standard 80-class models)
            base_dir / "models" / "yolo11m.pt",
            base_dir / "models" / "yolo11n.pt",
            base_dir / "models" / "yolov8n.pt",
            base_dir / "models" / "yolov8s.pt",
            base_dir / "models" / "yolov8m.pt",

            # 2. Windows absolute paths
            Path(r"D:\animal_detection_model\yolo11m.pt"),
            Path(r"D:\animal_detection_model\yolov8s-worldv2.pt"),
            Path(r"D:\animal_detection_model\yolov8m-worldv2.pt"),
            Path(r"C:\animal_detection_model\yolo11m.pt"),

            # 3. Pi / Linux home paths
            Path.home() / "PS180_2026" / "animal_detection_model" / "yolo11m.pt",
            Path.home() / "animal_detection_model" / "yolo11m.pt",
            Path("/home/gullu/PS180_2026/animal_detection_model/yolo11m.pt"),

            # 4. Sibling / parent directories
            base_dir.parent / "animal_detection_model" / "yolo11m.pt",

            # 5. Fallback custom / fine-tuned models
            base_dir / "models" / "animal_best.pt",
            base_dir / "models" / "animal_model.pt",
            base_dir / "models" / "best.pt",
            base_dir.parent / "animal_detection_model" / "best.pt",
            Path(r"D:\animal_detection_model\best.pt"),
            Path.home() / "PS180_2026" / "animal_detection_model" / "best.pt",
            Path("/home/gullu/PS180_2026/animal_detection_model/best.pt"),
        ])

        seen_paths = set()
        for cand_path in candidates:
            cand_resolved = str(cand_path)
            if cand_resolved in seen_paths:
                continue
            seen_paths.add(cand_resolved)

            if cand_path.exists():
                try:
                    loaded_m = YOLO(str(cand_path))
                    with self.lock:
                        self.animal_model = loaded_m
                        self.model_loaded = True
                        self.loaded_path = str(cand_path)
                        self.class_names = loaded_m.names if hasattr(loaded_m, "names") else {}
                        self.num_classes = len(self.class_names)

                        # If standard 80-class COCO model, filter to animal class IDs
                        if self.num_classes >= 80:
                            self.target_class_ids = [cid for cid in COCO_ANIMAL_MAP.keys() if cid in self.class_names]
                        else:
                            # Custom dataset (e.g. MMLA 4-class or fine-tuned) -> use all classes
                            self.target_class_ids = None

                    logger.info(f"Animal & Wildlife Detection Model loaded: {cand_path} ({self.num_classes} total classes)")
                    return
                except Exception as e:
                    logger.warning(f"Failed to initialize animal model at {cand_path}: {e}")

        logger.warning("No animal detection model found. Checked candidate paths.")

    def adjust_threshold(self, delta: float):
        """Allows live keyboard adjustment of animal detection sensitivity."""
        self.conf_thresh = max(0.10, min(0.90, round(self.conf_thresh + delta, 2)))
        logger.info(f"Animal detection confidence threshold adjusted to: {self.conf_thresh:.2f}")

    def detect(self, frame: np.ndarray) -> AnimalDetectionResult:
        """
        Runs animal detection on the full camera frame.
        Returns:
            AnimalDetectionResult with bounding boxes, species counts, and threat analysis.
        """
        t0 = time.time()
        now = time.time()

        with self.lock:
            model = self.animal_model
            target_ids = self.target_class_ids

        if model is None or frame is None or frame.size == 0:
            return AnimalDetectionResult()

        h, w = frame.shape[:2]
        boxes_out = []
        counts: Dict[str, int] = {}

        try:
            kwargs = {
                "source": frame,
                "conf": self.conf_thresh,
                "iou": self.iou_thresh,
                "imgsz": 640,
                "verbose": False
            }
            if target_ids is not None:
                kwargs["classes"] = target_ids

            with TORCH_INFERENCE():
                results = model(**kwargs)
            self.last_latency_ms = int((time.time() - t0) * 1000)

            if results and len(results) > 0:
                r = results[0]
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                    # Clamp coordinates to frame
                    x1 = max(0, min(w - 1, x1))
                    y1 = max(0, min(h - 1, y1))
                    x2 = max(0, min(w, x2))
                    y2 = max(0, min(h, y2))
                    bw = max(1, x2 - x1)
                    bh = max(1, y2 - y1)

                    raw_name = r.names.get(cls_id, f"Class_{cls_id}").strip()
                    clean_name = raw_name.lower().replace("_", " ")
                    is_threat = clean_name in CROP_THREAT_SPECIES
                    color = ANIMAL_SPECIES_COLORS.get(clean_name, (0, 220, 255))
                    disp_conf = max(1, min(99, int(round(conf * 100))))

                    counts[raw_name.title()] = counts.get(raw_name.title(), 0) + 1

                    boxes_out.append({
                        "x": x1,
                        "y": y1,
                        "w": bw,
                        "h": bh,
                        "x2": x2,
                        "y2": y2,
                        "class_id": cls_id,
                        "class_name": raw_name.title(),
                        "clean_name": clean_name,
                        "conf": conf,
                        "display_conf": disp_conf,
                        "is_threat": is_threat,
                        "color": color
                    })

        except Exception as e:
            logger.error(f"Error during animal detection inference: {e}")
            self.last_latency_ms = int((time.time() - t0) * 1000)

        # Build result
        if len(boxes_out) > 0:
            top_box = max(boxes_out, key=lambda b: b["conf"])
            has_threat = any(b["is_threat"] for b in boxes_out)
            threat_level = "🚨 CRITICAL INTRUSION" if has_threat else "⚠️ ANIMAL PRESENT"

            result = AnimalDetectionResult(
                has_animals=True,
                has_threat=has_threat,
                total_animals=len(boxes_out),
                counts=counts,
                top_animal=top_box["class_name"],
                top_confidence=top_box["conf"],
                display_confidence=top_box["display_conf"],
                threat_level=threat_level,
                boxes=boxes_out,
                latency_ms=self.last_latency_ms,
                model_name=Path(self.loaded_path).name if self.loaded_path else "YOLO11 Animal Model"
            )

            self.last_detection_time = now
            self.persisted_boxes = boxes_out
            self.persisted_result = result
            return result

        elif (now - self.last_detection_time) <= self.persistence_sec and self.persisted_result is not None:
            # Short persistence hold to prevent flicker
            res = self.persisted_result
            res.latency_ms = self.last_latency_ms
            return res

        else:
            self.persisted_boxes = []
            self.persisted_result = None
            return AnimalDetectionResult(
                has_animals=False,
                has_threat=False,
                total_animals=0,
                counts={},
                top_animal="None",
                top_confidence=0.0,
                display_confidence=0,
                threat_level="Perimeter Clear",
                boxes=[],
                latency_ms=self.last_latency_ms,
                model_name=Path(self.loaded_path).name if self.loaded_path else "YOLO11 Animal Model"
            )
