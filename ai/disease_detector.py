import os
import sys
import time
import warnings
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union, List

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    torch = None
    TORCH_AVAILABLE = False

# Ultralytics YOLO import (100% offline)
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

# =====================================================================
# EXACT 17-CLASS AGRONOMIC PATHOLOGY KNOWLEDGE BASE
# =====================================================================
AGRONOMIC_KNOWLEDGE_BASE = {
    "apple_scab_leaf": {
        "crop": "Apple (Malus domestica)",
        "disease": "Apple Scab",
        "pathogen": "Fungal (Venturia inaequalis)",
        "symptoms": "Olive-green to dark brown velvety circular spots on leaf surfaces.",
        "organic_remedy": "Apply liquid copper fungicide or sulfur spray before rain; rake & destroy fallen leaves.",
        "chemical_remedy": "Captan 50% WP (2.5g/L), Mancozeb 75% WP, or Difenoconazole 25% EC."
    },
    "apple_rust_leaf": {
        "crop": "Apple (Malus domestica)",
        "disease": "Cedar Apple Rust",
        "pathogen": "Fungal (Gymnosporangium juniperi-virginianae)",
        "symptoms": "Bright yellow-orange or rust-colored circular spots on upper leaf surfaces.",
        "organic_remedy": "Apply sulfur or neem oil sprays at pink bud stage; remove nearby red cedar trees.",
        "chemical_remedy": "Myclobutanil (Rally 40WSP), Propiconazole, or Mancozeb sprays."
    },
    "bell_pepper_leaf_spot": {
        "crop": "Bell Pepper (Capsicum annuum)",
        "disease": "Bacterial Leaf Spot",
        "pathogen": "Bacterial (Xanthomonas campestris)",
        "symptoms": "Small, water-soaked, blistering dark lesions with yellow halos on leaves.",
        "organic_remedy": "Copper soap or Bacillus subtilis foliar spray; avoid overhead sprinkling.",
        "chemical_remedy": "Copper Oxychloride 50% WP (3g/L) mixed with Streptocycline (0.5g/L)."
    },
    "corn_gray_leaf_spot": {
        "crop": "Corn / Maize (Zea mays)",
        "disease": "Gray Leaf Spot",
        "pathogen": "Fungal (Cercospora zeae-maydis)",
        "symptoms": "Narrow rectangular, tan-to-gray lesions running parallel to leaf veins.",
        "organic_remedy": "Crop rotation with non-host crops, deep tillage of crop residues, resistant hybrids.",
        "chemical_remedy": "Azoxystrobin + Difenoconazole (Amistar Top) or Pyraclostrobin."
    },
    "corn_leaf_blight": {
        "crop": "Corn / Maize (Zea mays)",
        "disease": "Northern Leaf Blight",
        "pathogen": "Fungal (Exserohilum turcicum)",
        "symptoms": "Long, elliptical cigar-shaped grayish-green to tan lesions on leaves.",
        "organic_remedy": "Trichoderma bio-fungicide seed treatment; remove infected debris after harvest.",
        "chemical_remedy": "Mancozeb 75% WP (2.5g/L) or Propiconazole 25% EC (1ml/L)."
    },
    "corn_rust_leaf": {
        "crop": "Corn / Maize (Zea mays)",
        "disease": "Common Corn Rust",
        "pathogen": "Fungal (Puccinia sorghi)",
        "symptoms": "Golden-brown to cinnamon-brown powdery pustules scattered on both leaf surfaces.",
        "organic_remedy": "Sulfur dusting, plant early to avoid peak spore dispersal, plant resistant cultivars.",
        "chemical_remedy": "Tebuconazole 25.9% EC (1.5ml/L) or Azoxystrobin."
    },
    "potato_leaf_early_blight": {
        "crop": "Potato (Solanum tuberosum)",
        "disease": "Early Blight",
        "pathogen": "Fungal (Alternaria solani)",
        "symptoms": "Dark brown circular spots with distinctive concentric target rings on older foliage.",
        "organic_remedy": "Neem oil 2%, copper hydroxide spray, prune lower diseased foliage.",
        "chemical_remedy": "Chlorothalonil 75% WP (2g/L) or Mancozeb 75% WP."
    },
    "potato_leaf_late_blight": {
        "crop": "Potato (Solanum tuberosum)",
        "disease": "Late Blight",
        "pathogen": "Oomycete (Phytophthora infestans)",
        "symptoms": "Water-soaked irregular dark lesions surrounded by light yellow-green halo with white mold.",
        "organic_remedy": "Fixed copper fungicides, eliminate cull piles, ensure good field drainage.",
        "chemical_remedy": "Metalaxyl + Mancozeb (Ridomil MZ 2.5g/L) or Cymoxanil."
    },
    "squash_powdery_mildew_leaf": {
        "crop": "Squash / Cucurbits (Cucurbita spp.)",
        "disease": "Powdery Mildew",
        "pathogen": "Fungal (Podosphaera xanthii)",
        "symptoms": "Talcum powder-like white fungal coating on upper and lower leaf surfaces.",
        "organic_remedy": "Potassium bicarbonate spray, horticultural neem oil 2%, diluted whey spray.",
        "chemical_remedy": "Hexaconazole 5% EC (1ml/L) or Myclobutanil."
    },
    "tomato_early_blight_leaf": {
        "crop": "Tomato (Solanum lycopersicum)",
        "disease": "Early Blight",
        "pathogen": "Fungal (Alternaria solani)",
        "symptoms": "Brown-black concentric target rings on leaves, causing lower leaf yellowing and defoliation.",
        "organic_remedy": "Copper soap fungicide, mulching around plants, drip irrigation only.",
        "chemical_remedy": "Mancozeb 75% WP (2g/L) or Chlorothalonil."
    },
    "tomato_septoria_leaf_spot": {
        "crop": "Tomato (Solanum lycopersicum)",
        "disease": "Septoria Leaf Spot",
        "pathogen": "Fungal (Septoria lycopersici)",
        "symptoms": "Numerous small circular spots with dark brown margins and gray-tan centers.",
        "organic_remedy": "Copper hydroxide spray, remove lower infected leaves, sanitize staking tools.",
        "chemical_remedy": "Copper Oxychloride 50% WP (3g/L) or Chlorothalonil 75% WP."
    },
    "tomato_leaf_bacterial_spot": {
        "crop": "Tomato (Solanum lycopersicum)",
        "disease": "Bacterial Spot",
        "pathogen": "Bacterial (Xanthomonas perforans)",
        "symptoms": "Small, dark, water-soaked greasy spots that turn necrotic with translucent borders.",
        "organic_remedy": "Liquid copper octanoate, avoid working with wet plants, crop rotation.",
        "chemical_remedy": "Copper Hydroxide + Streptomycin sulfate (Plantomycin 0.5g/L)."
    },
    "tomato_leaf_late_blight": {
        "crop": "Tomato (Solanum lycopersicum)",
        "disease": "Late Blight",
        "pathogen": "Oomycete (Phytophthora infestans)",
        "symptoms": "Rapidly expanding irregular greasy gray-green water-soaked lesions on leaves and stems.",
        "organic_remedy": "Copper-based fungicides, destroy severely diseased plants immediately.",
        "chemical_remedy": "Metalaxyl 8% + Mancozeb 64% WP (Ridomil 2g/L) or Dimethomorph."
    },
    "tomato_leaf_mosaic_virus": {
        "crop": "Tomato (Solanum lycopersicum)",
        "disease": "Mosaic Virus (ToMV)",
        "pathogen": "Viral (Tomato Mosaic Tobamovirus)",
        "symptoms": "Mottled light and dark green mosaic patterns, leaf curling, blistering, and stunting.",
        "organic_remedy": "Wash hands with skim milk/soap before handling; disinfect shears with 10% bleach.",
        "chemical_remedy": "No direct virucide; control aphid & thrips vectors with insecticidal soap."
    },
    "tomato_leaf_yellow_virus": {
        "crop": "Tomato (Solanum lycopersicum)",
        "disease": "Yellow Leaf Curl (TYLCV)",
        "pathogen": "Viral (Begomovirus transmitted by Whiteflies)",
        "symptoms": "Upward curling and cupping of leaves with pronounced yellowing of leaf margins.",
        "organic_remedy": "Install yellow sticky traps, spray neem oil 3%, use reflective silver mulches.",
        "chemical_remedy": "Control whitefly vector: Imidacloprid 17.8% SL (0.5ml/L) or Acetamiprid 20% SP."
    },
    "tomato_mold_leaf": {
        "crop": "Tomato (Solanum lycopersicum)",
        "disease": "Leaf Mold",
        "pathogen": "Fungal (Passalora fulva)",
        "symptoms": "Pale yellow chlorotic spots on upper leaf surface with olive-brown velvety mold underneath.",
        "organic_remedy": "Increase greenhouse ventilation, reduce relative humidity below 85%, copper spray.",
        "chemical_remedy": "Chlorothalonil, Mancozeb, or Difenoconazole foliar application."
    },
    "grape_leaf_black_rot": {
        "crop": "Grape (Vitis vinifera)",
        "disease": "Black Rot",
        "pathogen": "Fungal (Guignardia bidwellii)",
        "symptoms": "Small reddish-brown circular spots with dark margins and tiny black spore dots (pycnidia).",
        "organic_remedy": "Lime-sulfur dormant spray, liquid copper, prune diseased shoots and mummified berries.",
        "chemical_remedy": "Mancozeb 75% WP (2.5g/L) or Myclobutanil (Rally) spray."
    }
}


def lookup_agronomic_info(raw_class_name: str, confidence: float = 0.0) -> Dict[str, Any]:
    """Resolves raw YOLO class name to detailed agronomic metadata and remedies."""
    norm = raw_class_name.lower().strip().replace("-", "_").replace(" ", "_").replace("___", "_").replace("__", "_")

    matched_key = None
    for key in AGRONOMIC_KNOWLEDGE_BASE:
        if key == norm or key in norm or norm in key:
            matched_key = key
            break

    if not matched_key:
        for key in AGRONOMIC_KNOWLEDGE_BASE:
            parts = key.split("_")
            if sum(1 for p in parts if p in norm) >= 2:
                matched_key = key
                break

    if matched_key:
        info = AGRONOMIC_KNOWLEDGE_BASE[matched_key].copy()
    else:
        clean_title = raw_class_name.replace("_", " ").title()
        info = {
            "crop": "Crop / Plant",
            "disease": clean_title,
            "pathogen": "Pathogenic Agent",
            "symptoms": f"Visible leaf lesion / symptom detected ({clean_title}).",
            "organic_remedy": "Apply 2% cold-pressed neem oil spray, isolate infected leaves, avoid wetting foliage.",
            "chemical_remedy": "Broad-spectrum fungicide: Copper Oxychloride 50% WP or Mancozeb 75% WP."
        }

    info["is_diseased"] = True
    info["clean_name"] = info.get("disease", raw_class_name.replace("_", " ").title())

    # Genuine confidence percentage from YOLO model
    calibrated_pct = max(1, min(99, int(round(confidence * 100))))
    info["display_confidence"] = calibrated_pct

    if calibrated_pct >= 65:
        info["severity"] = "High / Acute"
    elif calibrated_pct >= 35:
        info["severity"] = "Moderate"
    else:
        info["severity"] = "Early Stage / Low"

    return info


class LocalPlantDetector:
    """Multi-Scale 100% Offline YOLOv8 Plant Disease Inference Engine."""
    def __init__(self, model_path: Optional[str] = None, conf_thresh: float = 0.50):
        self.conf_thresh = conf_thresh
        self.model_path = model_path
        self.plant_model = None
        self.lock = threading.Lock()
        self.model_loaded = False
        self.loaded_path = ""
        self.num_classes = 0

        if YOLO_AVAILABLE:
            self._load_model()

    def _load_model(self):
        base_dir = Path(__file__).resolve().parent.parent
        candidates = []
        if self.model_path:
            candidates.append(self.model_path)
        candidates.extend([
            "models/disease_model.pt",
            "models/plant_disease_best.pt",
            "runs/detect/train/weights/best.pt",
            "models/best.pt",
            "disease_model.pt"
        ])

        for p in candidates:
            candidate_path = Path(p) if Path(p).is_absolute() else (base_dir / p)
            if candidate_path.exists():
                try:
                    loaded_p = YOLO(str(candidate_path))
                    with self.lock:
                        self.plant_model = loaded_p
                        self.model_loaded = True
                        self.loaded_path = str(candidate_path)
                        self.num_classes = len(loaded_p.names) if hasattr(loaded_p, "names") else 0
                    logger.info(f"Offline YOLOv8 Plant Disease Model loaded: {candidate_path} ({self.num_classes} classes)")
                    return
                except Exception as e:
                    logger.warning(f"Error initializing YOLO model at {candidate_path}: {e}")

        logger.warning("No local .pt model found in models/disease_model.pt.")

    @staticmethod
    def _is_plant_foliage(crop: np.ndarray) -> bool:
        """
        Botanical foliage verification gate:
        1. Explicitly rejects human skin tones (YCrCb color space).
        2. Measures Excess Green Index (ExG = 2G - R - B) to ensure chlorophyll reflection.
        3. Checks leaf green, chlorotic yellow, and necrotic lesion HSV spectrum.
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
            return False

        h, w = crop.shape[:2]
        total_pixels = float(h * w)

        # 1. Human Skin Tone Rejection in YCrCb: Cr in [133, 173], Cb in [77, 127]
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
        skin_ratio = np.count_nonzero(skin_mask) / total_pixels
        if skin_ratio > 0.25:
            return False

        # 2. Excess Green Botanical Index: ExG = 2*G - R - B
        b, g, r = cv2.split(crop.astype(np.float32))
        exg = 2 * g - r - b

        # 3. True Plant Chlorophyll Spectrum in HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Leaf green foliage: Hue 28 to 90, Sat >= 30, Val >= 30
        green_mask = cv2.inRange(hsv, np.array([28, 30, 30]), np.array([90, 255, 255]))
        # Chlorotic / yellow diseased leaf: Hue 18 to 28, Sat >= 45, Val >= 40
        yellow_mask = cv2.inRange(hsv, np.array([18, 45, 40]), np.array([28, 255, 255]))
        # Necrotic brown leaf lesion: Hue 8 to 18, Sat >= 35, Val in [25, 210]
        brown_mask = cv2.inRange(hsv, np.array([8, 35, 25]), np.array([18, 255, 210]))

        plant_mask = (exg > 0) | ((green_mask > 0) | (yellow_mask > 0) | (brown_mask > 0))
        foliage_ratio = np.count_nonzero(plant_mask) / total_pixels

        # Require genuine foliage/lesion presence
        if foliage_ratio < 0.20:
            return False

        return True

    def detect_multiscale(self, frame: np.ndarray) -> Tuple[List[dict], Tuple[int, int, int, int]]:
        """
        Runs high-speed single-pass YOLOv8 inference with Botanical Foliage Verification Gate.
        """
        h, w = frame.shape[:2]
        plant_boxes = []

        # Center Target ROI (70% center zone)
        cx, cy = w // 2, h // 2
        size_w = int(w * 0.70)
        size_h = int(h * 0.70)
        rx1 = max(0, cx - size_w // 2)
        ry1 = max(0, cy - size_h // 2)
        rx2 = min(w, cx + size_w // 2)
        ry2 = min(h, cy + size_h // 2)
        target_roi = (rx1, ry1, rx2, ry2)

        with self.lock:
            p_model = self.plant_model

        if p_model is None or frame is None or frame.size == 0 or np.std(frame) < 14.0:
            return plant_boxes, target_roi

        with TORCH_INFERENCE():
            try:
                # Single high-speed forward pass on frame with resolution 640
                results = p_model(frame, verbose=False, conf=self.conf_thresh, imgsz=640)
                if results and len(results) > 0:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0].item())
                        name = results[0].names.get(cls_id, f"Class_{cls_id}")
                        conf = float(box.conf[0].item())
                        if conf < self.conf_thresh:
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        x1 = max(0, min(w - 1, x1))
                        y1 = max(0, min(h - 1, y1))
                        x2 = max(0, min(w, x2))
                        y2 = max(0, min(h, y2))
                        w_box = max(4, x2 - x1)
                        h_box = max(4, y2 - y1)

                        if w_box > 16 and h_box > 16:
                            box_crop = frame[y1:y1+h_box, x1:x1+w_box]
                            if self._is_plant_foliage(box_crop):
                                info = lookup_agronomic_info(name, conf)
                                plant_boxes.append({
                                    "x": x1, "y": y1, "w": w_box, "h": h_box,
                                    "conf": conf,
                                    "display_conf": info["display_confidence"],
                                    "label": info["clean_name"],
                                    "raw_label": name,
                                    "crop": info["crop"],
                                    "is_diseased": True,
                                    "source": "Vision AI"
                                })
            except Exception as e:
                logger.debug(f"Plant inference error: {e}")

        # Non-Maximum Suppression
        if len(plant_boxes) > 1:
            plant_boxes = sorted(plant_boxes, key=lambda b: b["conf"], reverse=True)
            clean_boxes = []
            for b in plant_boxes:
                keep = True
                bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
                for cb in clean_boxes:
                    cx1, cy1, cx2, cy2 = cb["x"], cb["y"], cb["x"] + cb["w"], cb["y"] + cb["h"]
                    ix1, iy1 = max(bx1, cx1), max(by1, cy1)
                    ix2, iy2 = min(bx2, cx2), min(by2, cy2)
                    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                    inter_area = iw * ih
                    union_area = (b["w"] * b["h"]) + (cb["w"] * cb["h"]) - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    if iou > 0.35:
                        keep = False
                        break
                if keep:
                    clean_boxes.append(b)
            plant_boxes = clean_boxes[:2]

        return plant_boxes, target_roi


class PlantPathologyHUD:
    """Renders unified dual-model real-time HUD overlays (Plant Pathology + Wildlife Intrusion)."""

    @staticmethod
    def draw(
        frame: np.ndarray,
        fps: float,
        latency_ms: int,
        conf_thresh: float,
        detection_data: Optional[Dict[str, Any]],
        local_yolo_boxes: Optional[List[dict]],
        target_roi: Tuple[int, int, int, int],
        animal_boxes: Optional[List[dict]] = None,
        animal_result: Optional[Any] = None,
        animal_thresh: float = 0.35,
        hud_expanded: bool = True,
        sensor_overlay_text: Optional[str] = None
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        canvas = frame.copy()
        detection_data = detection_data or {}
        local_yolo_boxes = local_yolo_boxes or []
        animal_boxes = animal_boxes or []

        # --- 1. DRAW TARGET SCAN RETICLE IF NO PLANT DISEASE IN VIEW ---
        rx1, ry1, rx2, ry2 = target_roi
        if len(local_yolo_boxes) == 0:
            c_len = 24
            bracket_color = (0, 180, 240)
            # Reticle corners
            cv2.line(canvas, (rx1, ry1), (rx1 + c_len, ry1), bracket_color, 2)
            cv2.line(canvas, (rx1, ry1), (rx1, ry1 + c_len), bracket_color, 2)
            cv2.line(canvas, (rx2, ry1), (rx2 - c_len, ry1), bracket_color, 2)
            cv2.line(canvas, (rx2, ry1), (rx2, ry1 + c_len), bracket_color, 2)
            cv2.line(canvas, (rx1, ry2), (rx1 + c_len, ry2), bracket_color, 2)
            cv2.line(canvas, (rx1, ry2), (rx1, ry2 - c_len), bracket_color, 2)
            cv2.line(canvas, (rx2, ry2), (rx2 - c_len, ry2), bracket_color, 2)
            cv2.line(canvas, (rx2, ry2), (rx2, ry2 - c_len), bracket_color, 2)

            reticle_lbl = "ALIGN LEAF IN TARGET ZONE"
            (rtw, rth), _ = cv2.getTextSize(reticle_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.putText(canvas, reticle_lbl, (rx1 + (rx2 - rx1 - rtw)//2, ry1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, bracket_color, 1)

        # --- 2. DRAW ANIMAL INTRUSION BOUNDING BOXES ---
        for ab in animal_boxes:
            ax1, ay1, abw, abh = ab["x"], ab["y"], ab["w"], ab["h"]
            aclass = ab.get("class_name", "Animal")
            aconf = ab.get("display_conf", 80)
            is_threat = ab.get("is_threat", False)
            box_color = ab.get("color", (0, 200, 255))

            # Bounding box
            cv2.rectangle(canvas, (ax1, ay1), (ax1 + abw, ay1 + abh), box_color, 2)

            # High-tech corner brackets
            c_len = min(22, min(abw, abh) // 4)
            if c_len > 4:
                cv2.line(canvas, (ax1, ay1), (ax1 + c_len, ay1), (255, 255, 255), 3)
                cv2.line(canvas, (ax1, ay1), (ax1, ay1 + c_len), (255, 255, 255), 3)
                cv2.line(canvas, (ax1 + abw, ay1 + abh), (ax1 + abw - c_len, ay1 + abh), (255, 255, 255), 3)
                cv2.line(canvas, (ax1 + abw, ay1 + abh), (ax1 + abw, ay1 + abh - c_len), (255, 255, 255), 3)

            prefix = "THREAT" if is_threat else "ANIMAL"
            tag = f"{prefix}: {aclass.upper()} [{aconf}%]"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 2)
            box_top = max(0, ay1 - th - 8)
            cv2.rectangle(canvas, (ax1, box_top), (ax1 + tw + 10, ay1), box_color, -1)
            cv2.putText(canvas, tag, (ax1 + 5, ay1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 2)

        # --- 3. DRAW GENUINE PLANT DISEASE BOUNDING BOXES ---
        for b in local_yolo_boxes:
            x, y, bw, bh, disp_conf, label = b["x"], b["y"], b["w"], b["h"], b.get("display_conf", 85), b["label"]
            box_color = (0, 30, 255)  # Crimson Red for plant disease

            cv2.rectangle(canvas, (x, y), (x + bw, y + bh), box_color, 2)

            # High-tech corner highlights
            corner_len = min(22, min(bw, bh) // 4)
            if corner_len > 4:
                cv2.line(canvas, (x, y), (x + corner_len, y), (255, 255, 255), 3)
                cv2.line(canvas, (x, y), (x, y + corner_len), (255, 255, 255), 3)
                cv2.line(canvas, (x + bw, y + bh), (x + bw - corner_len, y + bh), (255, 255, 255), 3)
                cv2.line(canvas, (x + bw, y + bh), (x + bw, y + bh - corner_len), (255, 255, 255), 3)

            tag = f"DISEASE: {label} [{disp_conf}%]"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
            box_top = max(0, y - th - 8)
            cv2.rectangle(canvas, (x, box_top), (x + tw + 10, y), box_color, -1)
            cv2.putText(canvas, tag, (x + 5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

        # --- 4. TOP HEADER STATUS BAR (Two clean rows, zero overlap) ---
        top_bar_h = 58
        cv2.rectangle(canvas, (0, 0), (w, top_bar_h), (18, 24, 28), -1)
        cv2.line(canvas, (0, top_bar_h), (w, top_bar_h), (50, 70, 80), 1)

        # Row 1: System Title & Dual Telemetry
        pulse = int((time.time() * 3) % 2)
        live_color = (0, 255, 0) if pulse == 0 else (0, 200, 50)
        cv2.circle(canvas, (18, 18), 6, live_color, -1)
        cv2.putText(canvas, "AGRO-EYE DUAL VISION AI", (32, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

        telemetry_text = f"FPS: {fps:04.1f} | Lat: {latency_ms}ms | DisThresh: {conf_thresh:.2f} | AnimThresh: {animal_thresh:.2f}"
        cv2.putText(canvas, telemetry_text, (max(260, w - 460), 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 230, 255), 1)

        # Row 2: Dual Detection Status Bar (Plant Disease + Animal Intrusion)
        has_disease = detection_data.get("has_disease", False)
        top_disease = detection_data.get("disease_name", "None")
        top_crop = detection_data.get("plant_name", "Crop")
        conf_pct = detection_data.get("display_confidence", 0)

        if has_disease and top_disease != "None":
            plant_badge = f"🌿 DISEASE: {top_disease.upper()} on {top_crop.upper()} [{conf_pct}%]"
            plant_color = (0, 70, 255)
        else:
            plant_badge = "🌿 LEAF SCAN: Target Zone Active"
            plant_color = (0, 220, 100)

        cv2.putText(canvas, plant_badge, (18, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.40, plant_color, 2)

        # Animal Status on right of Row 2
        has_anim = animal_result is not None and getattr(animal_result, "has_animals", False)
        if has_anim:
            tot = getattr(animal_result, "total_animals", 1)
            top_a = getattr(animal_result, "top_animal", "Animal")
            a_conf = getattr(animal_result, "display_confidence", 80)
            is_threat = getattr(animal_result, "has_threat", False)

            if is_threat:
                anim_badge = f"🚨 THREAT: {top_a.upper()} ({tot}) [{a_conf}%]"
                anim_color = (0, 50, 255)
            else:
                anim_badge = f"🐾 ANIMAL: {top_a.upper()} ({tot}) [{a_conf}%]"
                anim_color = (0, 220, 255)
        else:
            anim_badge = "🐾 PERIMETER: Clear"
            anim_color = (0, 200, 120)

        (abw_t, _), _ = cv2.getTextSize(anim_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 2)
        cv2.putText(canvas, anim_badge, (max(w // 2, w - abw_t - 20), 46), cv2.FONT_HERSHEY_SIMPLEX, 0.40, anim_color, 2)

        # Draw Sensor Overlay below top bar if available
        if sensor_overlay_text:
            cv2.putText(canvas, sensor_overlay_text, (15, top_bar_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        # --- 5. SIDE DIAGNOSTICS TELEMETRY PANEL ---
        if hud_expanded and w >= 760:
            panel_w = 400
            panel_x = w - panel_w - 15
            panel_y = top_bar_h + 10
            panel_h = min(h - top_bar_h - 25, 590)

            cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (15, 20, 25), -1)
            cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (50, 70, 80), 1)

            cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + 32), (25, 35, 45), -1)
            cv2.putText(canvas, "DUAL VISION AI DIAGNOSTICS", (panel_x + 15, panel_y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 240, 200), 2)

            curr_y = panel_y + 52

            # Section 1: Plant Pathology
            plant_name = detection_data.get("plant_name", "None")
            pathogen_type = detection_data.get("pathogen_type", "None")
            severity = detection_data.get("severity", "None")

            cv2.putText(canvas, "🌿 CROP PATHOLOGY STATUS", (panel_x + 15, curr_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 255, 150), 2)
            curr_y += 18
            cv2.putText(canvas, f"Target Crop: {plant_name}", (panel_x + 20, curr_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1)
            curr_y += 18
            cv2.putText(canvas, f"Pathogen: {pathogen_type}", (panel_x + 20, curr_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 220, 240), 1)
            curr_y += 22

            health_status = f"Infected ({top_disease})" if (has_disease and top_disease != "None") else "Monitoring (No Disease)"
            dis_color = (0, 50, 255) if has_disease else (0, 230, 90)
            cv2.putText(canvas, f"Health: {health_status}", (panel_x + 20, curr_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, dis_color, 1)
            curr_y += 20

            org_rem = detection_data.get("organic_remedy", "")
            if org_rem:
                cv2.putText(canvas, "Bio-Remedy:", (panel_x + 20, curr_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 240, 255), 1)
                curr_y += 16
                for line in [org_rem[i:i+44] for i in range(0, min(len(org_rem), 44), 44)]:
                    cv2.putText(canvas, f"{line}", (panel_x + 20, curr_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 245, 255), 1)
                    curr_y += 16

            curr_y += 4
            cv2.line(canvas, (panel_x + 15, curr_y), (panel_x + panel_w - 15, curr_y), (45, 55, 65), 1)
            curr_y += 18

            # Section 2: Wildlife & Animal Intrusion
            cv2.putText(canvas, "🐾 WILDLIFE & PERIMETER GUARD", (panel_x + 15, curr_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 200, 50), 2)
            curr_y += 20

            if has_anim:
                counts = getattr(animal_result, "counts", {})
                count_str = ", ".join([f"{k}: {v}" for k, v in counts.items()]) if counts else "Detected"
                threat_lvl = getattr(animal_result, "threat_level", "Intrusion")

                cv2.putText(canvas, f"Intrusion: {threat_lvl}", (panel_x + 20, curr_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 100, 255) if "CRITICAL" in threat_lvl else (0, 220, 255), 1)
                curr_y += 18
                cv2.putText(canvas, f"Animals: {count_str}", (panel_x + 20, curr_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1)
                curr_y += 18
            else:
                cv2.putText(canvas, "Perimeter Status: SECURE (No animals in field)", (panel_x + 20, curr_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 230, 100), 1)
                curr_y += 18

            curr_y = panel_y + panel_h - 22
            cv2.putText(canvas, "Engines: YOLOv8 Plant + YOLO11 Animal AI", (panel_x + 15, curr_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (130, 150, 160), 1)

        instructions = "[Q] Quit  |  [C] Switch Cam  |  [S] Save  |  [H] HUD  |  [A] Animal AI  |  [+] / [-] Sens"
        cv2.putText(canvas, instructions, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 220, 230), 1)

        return canvas

        return canvas


class DetectionResult:
    """Standardized result container for plant disease detections."""
    def __init__(
        self,
        has_disease: bool = False,
        top_class: str = "None",
        confidence: float = 0.0,
        display_confidence: int = 0,
        severity: str = "None",
        organic_remedy: str = "",
        chemical_remedy: str = "",
        plant_name: str = "Crop / Plant",
        model_name: str = "YOLOv8 disease_model.pt",
        raw_response: Optional[Dict[str, Any]] = None
    ):
        self.has_disease = has_disease
        self.top_class = top_class
        self.confidence = confidence
        self.display_confidence = display_confidence
        self.severity = severity
        self.organic_remedy = organic_remedy
        self.chemical_remedy = chemical_remedy
        self.plant_name = plant_name
        self.model_name = model_name
        self.raw_response = raw_response or {}


class DiseaseDetector:
    """
    100% Offline Plant Disease Detection System:
      - Native Multi-Scale YOLOv8 Neural Network (models/disease_model.pt)
      - Exact 17-Class Agricultural Pathology & Remedy Knowledge Base
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.50,
        persistence_sec: float = 2.0,
        **kwargs
    ):
        self.confidence_threshold = confidence_threshold
        self.box_persistence_sec = persistence_sec
        self.local_detector = LocalPlantDetector(
            model_path=model_path,
            conf_thresh=confidence_threshold
        )
        self.last_latency_ms = 0
        self.last_log_time = 0.0
        self.last_target_roi = (100, 50, 540, 430)
        self.last_detection_time = 0.0
        self.persisted_boxes: List[dict] = []
        self.persisted_result: Optional[DetectionResult] = None
        self.persisted_data: Optional[Dict[str, Any]] = None

    def adjust_threshold(self, delta: float):
        """Allows live dynamic adjustment of confidence threshold."""
        self.confidence_threshold = max(0.02, min(0.90, round(self.confidence_threshold + delta, 2)))
        self.local_detector.conf_thresh = self.confidence_threshold
        logger.info(f"Confidence threshold adjusted to: {self.confidence_threshold:.2f}")

    def process_frame(self, frame: np.ndarray) -> Tuple[DetectionResult, List[dict], List[dict], Dict[str, Any]]:
        """
        Runs multi-scale offline YOLOv8 inference on the input frame.
        Maintains a 2.0-second persistence window so bounding boxes stay firmly locked on target.
        Returns:
          (DetectionResult, active_yolo_boxes, instant_leaf_boxes, detection_data)
        """
        t0 = time.time()
        now = time.time()

        # Multi-scale YOLOv8 Disease Model Inference
        raw_yolo_boxes, target_roi = self.local_detector.detect_multiscale(frame)
        self.last_target_roi = target_roi
        self.last_latency_ms = int((time.time() - t0) * 1000)

        if len(raw_yolo_boxes) > 0:
            has_disease = True
            top_box = max(raw_yolo_boxes, key=lambda b: b["conf"])
            top_disease = top_box["label"]
            confidence = top_box["conf"]
            raw_name = top_box.get("raw_label", top_disease)
            disease_info = lookup_agronomic_info(raw_name, confidence)
            display_conf = disease_info.get("display_confidence", 85)

            detection_data = {
                "has_disease": has_disease,
                "disease_name": top_disease,
                "confidence": confidence,
                "display_confidence": display_conf,
                "plant_name": disease_info.get("crop", "Crop / Plant"),
                "pathogen_type": disease_info.get("pathogen", "None"),
                "severity": disease_info.get("severity", "None"),
                "symptoms": disease_info.get("symptoms", ""),
                "organic_remedy": disease_info.get("organic_remedy", ""),
                "chemical_remedy": disease_info.get("chemical_remedy", ""),
                "latency_ms": self.last_latency_ms
            }

            result = DetectionResult(
                has_disease=has_disease,
                top_class=top_disease,
                confidence=confidence,
                display_confidence=display_conf,
                severity=disease_info.get("severity", "None"),
                organic_remedy=disease_info.get("organic_remedy", ""),
                chemical_remedy=disease_info.get("chemical_remedy", ""),
                plant_name=disease_info.get("crop", "Crop / Plant"),
                model_name="YOLOv8 disease_model.pt",
                raw_response=detection_data
            )

            # Refresh 2-second persistence cache with newest box coordinates
            self.last_detection_time = time.time()
            self.persisted_boxes = raw_yolo_boxes
            self.persisted_result = result
            self.persisted_data = detection_data

            if now - self.last_log_time >= 2.5:
                self.last_log_time = now
                logger.info(f"Offline YOLOv8 -> 🚨 DISEASE DETECTED: {top_disease} [{display_conf}%] on {disease_info.get('crop')} | {self.last_latency_ms}ms")

            return result, raw_yolo_boxes, [], detection_data

        elif (now - self.last_detection_time) <= self.box_persistence_sec and self.persisted_result is not None:
            # Maintain 2-second persistence lock-on
            result = self.persisted_result
            active_boxes = self.persisted_boxes
            detection_data = self.persisted_data.copy()
            detection_data["latency_ms"] = self.last_latency_ms
            return result, active_boxes, [], detection_data

        else:
            # 2 seconds elapsed with no target in view
            self.persisted_boxes = []
            self.persisted_result = None
            self.persisted_data = None

            disease_info = {
                "crop": "Crop / Plant",
                "disease": "None",
                "pathogen": "None",
                "symptoms": "Scanning foliage... Hold leaf in target zone to diagnose.",
                "organic_remedy": "",
                "chemical_remedy": "",
                "is_diseased": False,
                "clean_name": "None",
                "severity": "None",
                "display_confidence": 0
            }

            detection_data = {
                "has_disease": False,
                "disease_name": "None",
                "confidence": 0.0,
                "display_confidence": 0,
                "plant_name": "Crop / Plant",
                "pathogen_type": "None",
                "severity": "None",
                "symptoms": disease_info["symptoms"],
                "organic_remedy": "",
                "chemical_remedy": "",
                "latency_ms": self.last_latency_ms
            }

            result = DetectionResult(
                has_disease=False,
                top_class="None",
                confidence=0.0,
                display_confidence=0,
                severity="None",
                organic_remedy="",
                chemical_remedy="",
                plant_name="Crop / Plant",
                model_name="YOLOv8 disease_model.pt",
                raw_response=detection_data
            )

            return result, [], [], detection_data

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        detection_data: Dict[str, Any],
        local_yolo_boxes: List[dict],
        target_roi: Optional[Tuple[int, int, int, int]] = None,
        animal_boxes: Optional[List[dict]] = None,
        animal_result: Optional[Any] = None,
        animal_thresh: float = 0.35,
        hud_expanded: bool = True,
        sensor_overlay_text: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """Renders the comprehensive offline agronomic and wildlife HUD."""
        return PlantPathologyHUD.draw(
            frame=frame,
            fps=fps,
            latency_ms=self.last_latency_ms,
            conf_thresh=self.confidence_threshold,
            detection_data=detection_data,
            local_yolo_boxes=local_yolo_boxes,
            target_roi=target_roi or self.last_target_roi,
            animal_boxes=animal_boxes,
            animal_result=animal_result,
            animal_thresh=animal_thresh,
            hud_expanded=hud_expanded,
            sensor_overlay_text=sensor_overlay_text
        )

    def stop(self):
        """No background threads to stop in pure offline mode."""
        pass
