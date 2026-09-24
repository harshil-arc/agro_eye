import requests
import json
import time
from typing import Optional, Dict, Any, List
from config import FIREBASE_DATABASE_URL, WEBRTC_SESSION_PATH
from firebase.firebase_client import FirebaseClient
from utils.logger import logger

class FirebaseWebRTCSignaling:
    """
    Zero-configuration WebRTC Signaling via Firebase Realtime Database.
    Exchanges SDP Offers, SDP Answers, and ICE candidates between Raspberry Pi and Farmer Mobile App.
    Works with both Firebase Admin SDK and high-performance direct REST API fallback.
    """
    def __init__(self, session_path: str = WEBRTC_SESSION_PATH, db_url: Optional[str] = None):
        self.fb = FirebaseClient()
        self.db_url = (db_url or FIREBASE_DATABASE_URL or "").rstrip("/")
        self.session_path = session_path.strip("/")
        self.session = requests.Session()

    def _get_node(self, subpath: str = "") -> Optional[Any]:
        """Fetches data from Firebase RTDB node."""
        path = f"{self.session_path}/{subpath}".strip("/") if subpath else self.session_path
        
        # 1. Try Firebase Admin SDK
        if self.fb.is_ready or self.fb.initialize():
            try:
                from firebase_admin import db
                ref = db.reference(path)
                return ref.get()
            except Exception as e:
                logger.debug(f"Admin SDK read error at /{path}: {e}")

        # 2. REST API Fallback
        if not self.db_url:
            return None
        url = f"{self.db_url}/{path}.json"
        try:
            resp = self.session.get(url, timeout=5.0)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug(f"REST GET error at /{path}: {e}")
        return None

    def _set_node(self, subpath: str, data: Any) -> bool:
        """Sets data on Firebase RTDB node."""
        path = f"{self.session_path}/{subpath}".strip("/") if subpath else self.session_path
        
        # 1. Try Firebase Admin SDK
        if self.fb.is_ready or self.fb.initialize():
            try:
                from firebase_admin import db
                ref = db.reference(path)
                ref.set(data)
                return True
            except Exception as e:
                logger.debug(f"Admin SDK write error at /{path}: {e}")

        # 2. REST API Fallback
        if not self.db_url:
            return False
        url = f"{self.db_url}/{path}.json"
        try:
            resp = self.session.put(url, json=data, timeout=5.0)
            return resp.status_code in (200, 201, 204)
        except Exception as e:
            logger.debug(f"REST PUT error at /{path}: {e}")
        return False

    def _push_node(self, subpath: str, data: Any) -> bool:
        """Pushes a child to a Firebase RTDB list node."""
        path = f"{self.session_path}/{subpath}".strip("/")
        
        # 1. Try Firebase Admin SDK
        if self.fb.is_ready or self.fb.initialize():
            try:
                from firebase_admin import db
                ref = db.reference(path)
                ref.push(data)
                return True
            except Exception as e:
                logger.debug(f"Admin SDK push error at /{path}: {e}")

        # 2. REST API Fallback
        if not self.db_url:
            return False
        url = f"{self.db_url}/{path}.json"
        try:
            resp = self.session.post(url, json=data, timeout=5.0)
            return resp.status_code in (200, 201, 204)
        except Exception as e:
            logger.debug(f"REST POST error at /{path}: {e}")
        return False

    def get_offer(self) -> Optional[Dict[str, Any]]:
        """Retrieves active SDP Offer from Farmer's app if present and pending."""
        offer_data = self._get_node("offer")
        if offer_data and isinstance(offer_data, dict) and offer_data.get("sdp"):
            return offer_data
        return None

    def send_answer(self, sdp: str, sdp_type: str = "answer") -> bool:
        """Sends Raspberry Pi's generated SDP Answer to the cloud."""
        payload = {
            "sdp": sdp,
            "type": sdp_type,
            "timestamp": int(time.time() * 1000),
            "device": "Raspberry Pi (AgroEye)"
        }
        return self._set_node("answer", payload)

    def send_ice_candidate(self, candidate_dict: Dict[str, Any]) -> bool:
        """Publishes a local Pi ICE candidate."""
        return self._push_node("pi_candidates", candidate_dict)

    def get_client_ice_candidates(self) -> List[Dict[str, Any]]:
        """Fetches ICE candidates uploaded by the Farmer's app."""
        candidates = self._get_node("client_candidates")
        if not candidates:
            return []
        if isinstance(candidates, dict):
            return list(candidates.values())
        if isinstance(candidates, list):
            return [c for c in candidates if c is not None]
        return []

    def update_status(self, status: str, extra: Optional[Dict[str, Any]] = None):
        """Updates connection state on the signaling node."""
        data = {
            "status": status,
            "last_active": int(time.time() * 1000),
            **(extra or {})
        }
        self._set_node("status", data)

    def clear_session(self):
        """Cleans up session signaling channel."""
        self._set_node("offer", None)
        self._set_node("answer", None)
        self._set_node("pi_candidates", None)
        self._set_node("client_candidates", None)
        self.update_status("idle")
