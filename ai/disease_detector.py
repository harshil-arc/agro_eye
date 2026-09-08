import os
import sys
import time
import json
import re
import warnings
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union, List

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
from PIL import Image

# Google GenAI SDK imports
GENAI_SDK_TYPE = None
try:
    from google import genai
    GENAI_SDK_TYPE = "google.genai"
except ImportError:
    try:
        import google.generativeai as genai_legacy
        GENAI_SDK_TYPE = "google.generativeai"
    except ImportError:
        GENAI_SDK_TYPE = None

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from utils.logger import logger

DEFAULT_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")



class FastFoliageTracker:
    """Ultra-fast (5ms) computer vision leaf & plant contour locator for instant 0ms bounding boxes."""
    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def extract_leaf_boxes(self, frame: np.ndarray) -> List[dict]:
        """Detects active plant foliage / leaf regions instantly from frame color and salience."""
        h, w = frame.shape[:2]
        scale = 320.0 / max(w, h)
        sw, sh = int(w * scale), int(h * scale)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # Green foliage spectrum + yellow/brown lesion spectrum
        mask_green = cv2.inRange(hsv, (24, 25, 25), (95, 255, 255))
        mask_lesion = cv2.inRange(hsv, (8, 35, 35), (24, 255, 255))
        mask = cv2.bitwise_or(mask_green, mask_lesion)

        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        inv_scale = 1.0 / scale

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 450:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                fx = int(bx * inv_scale)
                fy = int(by * inv_scale)
                fw = int(bw * inv_scale)
                fh = int(bh * inv_scale)

                boxes.append({
                    "x": max(0, fx),
                    "y": max(0, fy),
                    "w": min(w - fx, fw),
                    "h": min(h - fy, fh),
                    "conf": 0.88,
                    "label": "Active Leaf ROI",
                    "is_diseased": False,
                    "source": "Instant Tracker"
                })

        return sorted(boxes, key=lambda b: b["w"] * b["h"], reverse=True)[:3]


class GeminiPlantPathologist:
    """Asynchronous background engine analyzing live frames & generating 2D spatial bounding boxes."""

    ANALYSIS_PROMPT = """Expert Agricultural Plant Pathologist & Computer Vision AI.
Diagnose plant health, disease, and locate bounding boxes for any leaf, plant, or crop in frame.

Return raw JSON only:
{
  "plant_detected": true or false,
  "plant_name": "Plant species name or Unknown",
  "health_status": "Healthy" | "Diseased" | "Pest Infested" | "Nutrient Deficient" | "No Plant Detected",
  "disease_name": "Specific Disease or Pest Name or None",
  "pathogen_type": "Fungal" | "Bacterial" | "Viral" | "Pest/Insect" | "Nutritional" | "None",
  "severity": "None" | "Low" | "Moderate" | "Severe",
  "confidence": 0 to 100,
  "symptoms": "Description of visible symptoms or None",
  "organic_remedy": "Organic/biological treatment or None",
  "chemical_remedy": "Chemical fungicide/pesticide or None",
  "bounding_boxes": [
    {
      "box_2d": [ymin, xmin, ymax, xmax],
      "label": "Specific Disease or Leaf Name",
      "is_diseased": true or false
    }
  ]
}"""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.5-flash-lite", interval: float = 0.8):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or DEFAULT_GEMINI_API_KEY
        self.model_name = model_name
        self.interval = max(0.4, interval)
        self.running = False
        self.paused = False
        self.trigger_instant_scan = False

        self.latest_result: Dict[str, Any] = {
            "plant_detected": False,
            "plant_name": "Scanning for plants...",
            "health_status": "Scanning",
            "disease_name": "None",
            "pathogen_type": "None",
            "severity": "None",
            "confidence": 0,
            "symptoms": "Point camera at plant/leaf to diagnose...",
            "organic_remedy": "None",
            "chemical_remedy": "None",
            "bounding_boxes": [],
            "last_updated": datetime.now().strftime("%H:%M:%S"),
            "latency_ms": 0,
            "status": "INITIALIZING"
        }

        self.current_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.client = None
        self.legacy_model = None
        self._init_gemini()

    def _init_gemini(self):
        if GENAI_SDK_TYPE == "google.genai":
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"Initialized Google GenAI Client with model: '{self.model_name}'")
            except Exception as e:
                logger.warning(f"Failed to initialize google.genai Client: {e}")
        elif GENAI_SDK_TYPE == "google.generativeai":
            try:
                genai_legacy.configure(api_key=self.api_key)
                self.legacy_model = genai_legacy.GenerativeModel(self.model_name)
                logger.info(f"Initialized Google GenerativeAI model: '{self.model_name}'")
            except Exception as e:
                logger.warning(f"Failed to initialize google.generativeai: {e}")
        else:
            logger.warning("No Google GenAI SDK available in Python environment.")

    def update_frame(self, frame: np.ndarray):
        with self.lock:
            self.current_frame = frame

    def request_instant_scan(self):
        self.trigger_instant_scan = True

    def get_latest_result(self) -> Dict[str, Any]:
        with self.lock:
            return self.latest_result.copy()

    def start(self):
        self.running = True
        thread = threading.Thread(target=self._worker_loop, daemon=True)
        thread.start()
        return self

    def stop(self):
        self.running = False

    def _worker_loop(self):
        """Continuous asynchronous loop sending frame samples to Gemini."""
        candidate_models = [self.model_name, "gemini-3.5-flash-lite", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-flash-latest"]

        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            frame_to_process = None
            with self.lock:
                if self.current_frame is not None:
                    frame_to_process = self.current_frame.copy()

            if frame_to_process is None:
                time.sleep(0.05)
                continue

            h, w = frame_to_process.shape[:2]
            target_w = 480
            target_h = int(h * (target_w / w))
            resized = cv2.resize(frame_to_process, (target_w, target_h), interpolation=cv2.INTER_AREA)
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            t0 = time.time()
            success = False

            for model_cand in candidate_models:
                try:
                    raw_text = None
                    if self.client is not None:
                        res = self.client.models.generate_content(
                            model=model_cand,
                            contents=[pil_image, self.ANALYSIS_PROMPT],
                            config={"temperature": 0.1, "max_output_tokens": 350}
                        )
                        raw_text = res.text
                    elif self.legacy_model is not None:
                        model_obj = genai_legacy.GenerativeModel(model_cand)
                        res = model_obj.generate_content(
                            [pil_image, self.ANALYSIS_PROMPT],
                            generation_config={"temperature": 0.1, "max_output_tokens": 350}
                        )
                        raw_text = res.text

                    if raw_text:
                        latency_ms = int((time.time() - t0) * 1000)
                        clean_json = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.IGNORECASE)
                        clean_json = re.sub(r"\s*```$", "", clean_json)
                        parsed = json.loads(clean_json)

                        parsed["last_updated"] = datetime.now().strftime("%H:%M:%S")
                        parsed["latency_ms"] = latency_ms
                        parsed["status"] = "OK"
                        parsed["active_model"] = model_cand

                        if "bounding_boxes" not in parsed or not isinstance(parsed["bounding_boxes"], list):
                            parsed["bounding_boxes"] = []

                        with self.lock:
                            self.latest_result = parsed
                        success = True
                        self.model_name = model_cand

                        plant_st = parsed.get("health_status", "N/A")
                        plant_nm = parsed.get("plant_name", "Unknown")
                        dis_nm = parsed.get("disease_name", "None")
                        bboxes = parsed.get("bounding_boxes", [])
                        logger.info(f"Gemini AI -> {plant_nm} ({plant_st}) | Disease: {dis_nm} | BBoxes: {len(bboxes)} | {latency_ms}ms")
                        break
                except Exception:
                    continue

            if not success:
                with self.lock:
                    self.latest_result["status"] = "Retrying AI connection..."
                    self.latest_result["latency_ms"] = int((time.time() - t0) * 1000)

            for _ in range(int(self.interval * 10)):
                if not self.running or self.trigger_instant_scan:
                    self.trigger_instant_scan = False
                    break
                time.sleep(0.05)


class LocalPlantDetector:
    """High-speed local detector for real-time 60 FPS plant disease bounding boxes."""
    def __init__(self, conf_thresh: float = 0.25):
        self.conf_thresh = conf_thresh
        self.plant_model = None
        self.lock = threading.Lock()

        if YOLO_AVAILABLE:
            threading.Thread(target=self._load_model_background, daemon=True).start()

    def _load_model_background(self):
        plant_candidates = ["models/plant_disease_best.pt", "runs/detect/train/weights/best.pt", "models/best.pt"]
        for p in plant_candidates:
            if Path(p).exists():
                try:
                    loaded_p = YOLO(p)
                    with self.lock:
                        self.plant_model = loaded_p
                    logger.info(f"Local Plant Disease YOLO loaded: {p} ({len(loaded_p.names)} classes)")
                    break
                except Exception as e:
                    logger.warning(f"Error loading plant model {p}: {e}")

    def detect(self, frame: np.ndarray) -> List[dict]:
        plant_boxes = []

        with self.lock:
            p_model = self.plant_model

        if p_model is not None:
            try:
                results = p_model(frame, verbose=False, conf=self.conf_thresh, imgsz=416)
                if results and len(results) > 0:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0].item())
                        name = results[0].names.get(cls_id, "")
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        w, h = x2 - x1, y2 - y1

                        is_healthy = "healthy" in name.lower()
                        clean_name = name.replace("_", " ")

                        plant_boxes.append({
                            "x": x1, "y": y1, "w": w, "h": h,
                            "conf": conf, "label": clean_name,
                            "is_diseased": not is_healthy,
                            "source": "Local YOLO"
                        })
            except Exception:
                pass

        return plant_boxes


class PlantPathologyHUD:
    """Renders cyber-agronomic heads-up display on live camera frames."""

    @staticmethod
    def draw(frame: np.ndarray,
             fps: float,
             gemini_data: Dict[str, Any],
             local_yolo_boxes: List[dict],
             instant_leaf_boxes: List[dict],
             hud_expanded: bool = True,
             conf_thresh: float = 0.25,
             sensor_overlay_text: Optional[str] = None) -> np.ndarray:
        h, w = frame.shape[:2]
        canvas = frame.copy()

        final_boxes = []
        gemini_bboxes = gemini_data.get("bounding_boxes", [])
        dis_name_global = gemini_data.get("disease_name", "Diseased Foliage")
        plant_name_global = gemini_data.get("plant_name", "Plant")
        health_global = gemini_data.get("health_status", "Scanning")
        global_conf = gemini_data.get("confidence", 85)

        # 1. Add Local YOLO Boxes
        for b in local_yolo_boxes:
            final_boxes.append(b)

        # 2. Add Gemini Spatial Boxes
        gemini_parsed_boxes = []
        for g_box in gemini_bboxes:
            try:
                box_2d = g_box.get("box_2d", [])
                if len(box_2d) == 4:
                    ymin, xmin, ymax, xmax = box_2d
                    bx1 = int(xmin * w / 1000.0)
                    by1 = int(ymin * h / 1000.0)
                    bx2 = int(xmax * w / 1000.0)
                    by2 = int(ymax * h / 1000.0)
                    bw = max(12, bx2 - bx1)
                    bh = max(12, by2 - by1)

                    is_dis = g_box.get("is_diseased", health_global in ["Diseased", "Pest Infested", "Nutrient Deficient"])
                    label_text = g_box.get("label", "")
                    if not label_text:
                        label_text = dis_name_global if (is_dis and dis_name_global != "None") else f"{plant_name_global} (Healthy)"

                    gemini_parsed_boxes.append({
                        "x": bx1, "y": by1, "w": bw, "h": bh,
                        "conf": global_conf / 100.0,
                        "label": label_text,
                        "is_diseased": is_dis,
                        "source": "Gemini AI"
                    })
            except Exception:
                pass

        if len(gemini_parsed_boxes) > 0:
            final_boxes.extend(gemini_parsed_boxes)

        # 3. Instant Foliage ROI Framing
        if len(final_boxes) == 0 and len(instant_leaf_boxes) > 0:
            for ib in instant_leaf_boxes:
                is_dis = health_global in ["Diseased", "Pest Infested", "Nutrient Deficient"]
                active_lbl = dis_name_global if (is_dis and dis_name_global != "None") else (f"{plant_name_global} (Healthy)" if health_global == "Healthy" else "Active Leaf [Analyzing...]")
                final_boxes.append({
                    "x": ib["x"], "y": ib["y"], "w": ib["w"], "h": ib["h"],
                    "conf": 0.85 if health_global != "Scanning" else 0.70,
                    "label": active_lbl,
                    "is_diseased": is_dis,
                    "source": "Instant Tracker"
                })

        # --- DRAW BOUNDING BOXES ---
        for b in final_boxes:
            x, y, bw, bh, conf, label = b["x"], b["y"], b["w"], b["h"], b["conf"], b["label"]
            is_diseased = b.get("is_diseased", False)

            if is_diseased:
                box_color = (0, 50, 255) # Red
            elif "analyzing" in label.lower():
                box_color = (0, 220, 255) # Gold
            else:
                box_color = (0, 230, 90) # Green

            cv2.rectangle(canvas, (x, y), (x + bw, y + bh), box_color, 2)

            corner_len = min(22, min(bw, bh) // 4)
            if corner_len > 4:
                cv2.line(canvas, (x, y), (x + corner_len, y), (255, 255, 255), 3)
                cv2.line(canvas, (x, y), (x, y + corner_len), (255, 255, 255), 3)
                cv2.line(canvas, (x + bw, y + bh), (x + bw - corner_len, y + bh), (255, 255, 255), 3)
                cv2.line(canvas, (x + bw, y + bh), (x + bw, y + bh - corner_len), (255, 255, 255), 3)

            prefix = "[DISEASE]" if is_diseased else ("[SCANNING]" if "analyzing" in label.lower() else "[HEALTHY]")
            tag = f"{prefix} {label} [{int(conf * 100)}%]"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
            cv2.rectangle(canvas, (x, max(0, y - th - 8)), (x + tw + 10, y), box_color, -1)
            cv2.putText(canvas, tag, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

        # --- TOP HEADER STATUS BAR ---
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, 52), (15, 20, 24), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

        pulse = int((time.time() * 3) % 2)
        live_color = (0, 255, 0) if pulse == 0 else (0, 200, 50)
        cv2.circle(canvas, (24, 26), 7, live_color, -1)
        cv2.putText(canvas, "REAL-TIME PLANT AI", (40, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        latency = gemini_data.get("latency_ms", 0)
        fps_text = f"FPS: {fps:04.1f} | Latency: {latency}ms | Boxes: {len(final_boxes)}"
        cv2.putText(canvas, fps_text, (270, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 255), 2)

        # Status Badge
        health = gemini_data.get("health_status", "Scanning")
        dis_name = gemini_data.get("disease_name", "None")

        if len(final_boxes) > 0 and any(p["is_diseased"] for p in final_boxes):
            top_dis = [p["label"] for p in final_boxes if p["is_diseased"]][0]
            badge_text = f"ALERT: {top_dis.upper()}"
            h_color = (0, 60, 255)
        elif health in ["Diseased", "Pest Infested", "Nutrient Deficient"]:
            badge_text = f"ALERT: {health.upper()} ({dis_name if dis_name != 'None' else 'Detected'})"
            h_color = (0, 60, 255)
        elif health == "Healthy" or (len(final_boxes) > 0 and all(not p["is_diseased"] for p in final_boxes)):
            badge_text = "PLANT: 100% HEALTHY"
            h_color = (0, 230, 90)
        elif len(instant_leaf_boxes) > 0:
            badge_text = "LEAF DETECTED: ANALYZING PATHOLOGY..."
            h_color = (0, 220, 255)
        else:
            badge_text = "PLANT: SCANNING FOLIAGE..."
            h_color = (180, 180, 180)

        cv2.putText(canvas, badge_text, (max(10, w - 520), 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, h_color, 2)

        # Draw Sensor Overlay in Upper Area if available
        if sensor_overlay_text:
            cv2.putText(canvas, sensor_overlay_text, (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # --- SIDE DIAGNOSTICS DASHBOARD ---
        if hud_expanded and w >= 760:
            panel_w = 400
            panel_x = w - panel_w - 15
            panel_y = 65
            panel_h = min(h - 80, 590)

            card_overlay = canvas.copy()
            cv2.rectangle(card_overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (12, 16, 20), -1)
            cv2.addWeighted(card_overlay, 0.90, canvas, 0.10, 0, canvas)
            cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (50, 70, 80), 1)

            cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + 35), (25, 35, 45), -1)
            cv2.putText(canvas, "PLANT PATHOLOGY TELEMETRY", (panel_x + 15, panel_y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 240, 200), 2)

            curr_y = panel_y + 58

            plant_name = gemini_data.get("plant_name", "Scanning...")
            pathogen_type = gemini_data.get("pathogen_type", "None")
            severity = gemini_data.get("severity", "None")
            conf = gemini_data.get("confidence", 0)

            cv2.putText(canvas, "CROP IDENTIFICATION", (panel_x + 15, curr_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 255, 150), 2)
            curr_y += 20
            cv2.putText(canvas, f"Crop Species: {plant_name}", (panel_x + 20, curr_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1)
            curr_y += 20
            cv2.putText(canvas, f"Pathogen Type: {pathogen_type}", (panel_x + 20, curr_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 220, 240), 1)
            curr_y += 26

            cv2.line(canvas, (panel_x + 15, curr_y - 6), (panel_x + panel_w - 15, curr_y - 6), (45, 55, 65), 1)

            cv2.putText(canvas, "PATHOLOGY DIAGNOSIS", (panel_x + 15, curr_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 180, 80), 2)
            curr_y += 20

            dis_color = (0, 230, 90) if health == "Healthy" else ((0, 60, 255) if health in ["Diseased", "Pest Infested", "Nutrient Deficient"] else (200, 200, 200))
            cv2.putText(canvas, f"Condition: {health}", (panel_x + 20, curr_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, dis_color, 2)
            curr_y += 20

            if dis_name and dis_name != "None":
                cv2.putText(canvas, f"Diagnosis: {dis_name}", (panel_x + 20, curr_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 90, 255), 2)
                curr_y += 20

            sev_color = (0, 220, 100)
            if severity in ["Moderate", "Medium"]:
                sev_color = (0, 180, 255)
            elif severity in ["Severe", "High"]:
                sev_color = (0, 0, 255)

            cv2.putText(canvas, f"Severity: {severity} | Confidence: {conf}%", (panel_x + 20, curr_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, sev_color, 1)
            curr_y += 24

            symptoms = gemini_data.get("symptoms", "None")
            if symptoms and symptoms != "None":
                cv2.putText(canvas, "Observed Symptoms:", (panel_x + 20, curr_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170, 200, 200), 1)
                curr_y += 16
                sym_lines = [symptoms[i:i+44] for i in range(0, min(len(symptoms), 88), 44)]
                for line in sym_lines:
                    cv2.putText(canvas, f"  * {line}", (panel_x + 20, curr_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 220, 220), 1)
                    curr_y += 16

            curr_y += 6
            cv2.line(canvas, (panel_x + 15, curr_y - 4), (panel_x + panel_w - 15, curr_y - 4), (45, 55, 65), 1)

            org_rem = gemini_data.get("organic_remedy", "")
            chem_rem = gemini_data.get("chemical_remedy", "")

            if org_rem and org_rem != "None":
                cv2.putText(canvas, "ORGANIC REMEDY / BIOLOGICAL:", (panel_x + 15, curr_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 240, 255), 2)
                curr_y += 18
                org_lines = [org_rem[i:i+44] for i in range(0, min(len(org_rem), 88), 44)]
                for line in org_lines:
                    cv2.putText(canvas, f"{line}", (panel_x + 20, curr_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 245, 255), 1)
                    curr_y += 16

            if chem_rem and chem_rem != "None":
                curr_y += 4
                cv2.putText(canvas, "CHEMICAL / FUNGICIDE:", (panel_x + 15, curr_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 160, 200), 2)
                curr_y += 18
                chem_lines = [chem_rem[i:i+44] for i in range(0, min(len(chem_rem), 88), 44)]
                for line in chem_lines:
                    cv2.putText(canvas, f"{line}", (panel_x + 20, curr_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 210, 230), 1)
                    curr_y += 16

            curr_y = panel_y + panel_h - 22
            last_sync = gemini_data.get("last_updated", "N/A")
            active_m = gemini_data.get("active_model", "Gemini Cloud")
            cv2.putText(canvas, f"AI Sync: {last_sync} ({active_m})", (panel_x + 15, curr_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 140, 150), 1)

        instructions = "[Q/ESC] Quit  |  [SPACE] Instant AI Re-Scan  |  [S] Save Snapshot  |  [H] Toggle HUD"
        cv2.putText(canvas, instructions, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 220, 230), 1)

        return canvas


class DetectionResult:
    """Unified result container for system-wide consumption."""
    def __init__(
        self,
        has_disease: bool = False,
        top_class: str = "",
        confidence: float = 0.0,
        severity: str = "None",
        organic_remedy: str = "",
        chemical_remedy: str = "",
        plant_name: str = "Unknown",
        raw_response: Optional[Dict[str, Any]] = None
    ):
        self.has_disease = has_disease
        self.top_class = top_class
        self.confidence = confidence
        self.severity = severity
        self.organic_remedy = organic_remedy
        self.chemical_remedy = chemical_remedy
        self.plant_name = plant_name
        self.raw_response = raw_response or {}


class DiseaseDetector:
    """
    Multi-Tiered High-Speed Dual Vision Engine:
      1. Local Fast Foliage Saliency Tracker (0-5ms)
      2. Local YOLO Plant Disease Model (10ms)
      3. Continuous Async Gemini Multimodal Pathologist
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.5-flash-lite",
        ai_interval: float = 0.8,
        confidence_threshold: float = 0.25
    ):
        self.confidence_threshold = confidence_threshold
        self.foliage_tracker = FastFoliageTracker()
        self.local_detector = LocalPlantDetector(conf_thresh=confidence_threshold)
        self.pathologist = GeminiPlantPathologist(
            api_key=api_key,
            model_name=model_name,
            interval=ai_interval
        ).start()

    def update_frame(self, frame: np.ndarray):
        """Streams newest camera frame into background Gemini pathologist."""
        self.pathologist.update_frame(frame)

    def request_instant_scan(self):
        """Forces an immediate Gemini AI re-scan."""
        self.pathologist.request_instant_scan()

    def process_frame(self, frame: np.ndarray) -> Tuple[DetectionResult, List[dict], List[dict], Dict[str, Any]]:
        """
        Executes instant local inference + merges with background Gemini diagnosis.
        Returns (DetectionResult, local_yolo_boxes, instant_leaf_boxes, gemini_data).
        """
        self.update_frame(frame)

        # 1. Instant Leaf Saliency Tracker (3ms)
        instant_leaf_boxes = self.foliage_tracker.extract_leaf_boxes(frame)

        # 2. Local YOLO Plant Disease Detector (10ms)
        local_yolo_boxes = self.local_detector.detect(frame)

        # 3. Retrieve Latest Background Gemini Pathology
        gemini_data = self.pathologist.get_latest_result()

        # Determine disease state across all tiers
        has_disease = False
        top_disease = "None"
        confidence = 0.0

        # Tier A: Local YOLO detection
        diseased_yolo = [b for b in local_yolo_boxes if b.get("is_diseased", False)]
        if len(diseased_yolo) > 0:
            has_disease = True
            top_disease = diseased_yolo[0]["label"]
            confidence = diseased_yolo[0]["conf"]

        # Tier B: Gemini Deep Diagnosis
        elif gemini_data.get("health_status") in ["Diseased", "Pest Infested", "Nutrient Deficient"]:
            has_disease = True
            dis_name = gemini_data.get("disease_name", "None")
            top_disease = dis_name if dis_name != "None" else gemini_data.get("health_status", "Diseased Foliage")
            confidence = float(gemini_data.get("confidence", 85)) / 100.0

        result = DetectionResult(
            has_disease=has_disease,
            top_class=top_disease,
            confidence=confidence,
            severity=gemini_data.get("severity", "None"),
            organic_remedy=gemini_data.get("organic_remedy", ""),
            chemical_remedy=gemini_data.get("chemical_remedy", ""),
            plant_name=gemini_data.get("plant_name", "Unknown"),
            raw_response=gemini_data
        )

        return result, local_yolo_boxes, instant_leaf_boxes, gemini_data

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        gemini_data: Dict[str, Any],
        local_yolo_boxes: List[dict],
        instant_leaf_boxes: List[dict],
        hud_expanded: bool = True,
        sensor_overlay_text: Optional[str] = None
    ) -> np.ndarray:
        """Renders comprehensive cyber-agronomic HUD."""
        return PlantPathologyHUD.draw(
            frame=frame,
            fps=fps,
            gemini_data=gemini_data,
            local_yolo_boxes=local_yolo_boxes,
            instant_leaf_boxes=instant_leaf_boxes,
            hud_expanded=hud_expanded,
            conf_thresh=self.confidence_threshold,
            sensor_overlay_text=sensor_overlay_text
        )

    def stop(self):
        """Stops background Gemini pathologist threads."""
        self.pathologist.stop()
