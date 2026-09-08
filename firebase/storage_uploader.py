import os
import base64
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import timedelta
from firebase.firebase_client import FirebaseClient
from utils.logger import logger

class StorageUploader:
    """
    Multi-tier cloud photo uploader for plant disease snapshots.
    1. Attempts Firebase Cloud Storage upload if bucket is active.
    2. Falls back to direct, permanent high-speed Cloud Image Hosting (FreeImage / Catbox / tmpfiles)
       to return direct, clickable `https://...jpg` photo links that open directly in any browser.
    """
    def __init__(self):
        self.fb = FirebaseClient()

    def _upload_to_firebase_storage(
        self,
        local_image_path: str,
        remote_filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Attempts upload to Firebase Cloud Storage bucket."""
        if not self.fb.is_ready and not self.fb.initialize():
            return None

        try:
            from firebase_admin import storage
            bucket = None
            candidate_names = [None, "sample-629de.firebasestorage.app", "sample-629de.appspot.com"]
            for name in candidate_names:
                try:
                    b = storage.bucket(name) if name else storage.bucket()
                    if b.exists():
                        bucket = b
                        break
                except Exception:
                    continue

            if bucket is None:
                return None

            filename = remote_filename or Path(local_image_path).name
            blob_path = f"disease_snapshots/{filename}"
            blob = bucket.blob(blob_path)

            if metadata:
                blob.metadata = {str(k): str(v) for k, v in metadata.items()}

            blob.upload_from_filename(local_image_path, content_type="image/jpeg")

            try:
                download_url = blob.generate_signed_url(version="v4", expiration=timedelta(days=365), method="GET")
            except Exception:
                try:
                    blob.make_public()
                    download_url = blob.public_url
                except Exception:
                    download_url = f"https://storage.googleapis.com/{bucket.name}/{blob_path}"

            logger.info(f"Photo uploaded to Firebase Storage: {download_url}")
            return download_url
        except Exception:
            return None

    def _upload_to_cloud_host(self, local_image_path: str) -> Optional[str]:
        """Uploads snapshot to cloud image host and returns direct clickable https://...jpg URL."""
        # 1. Try FreeImage.host API
        try:
            with open(local_image_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
            resp = requests.post(
                "https://freeimage.host/api/1/upload",
                data={
                    "key": "6d207e02198a847aa98d0a2a901485a5",
                    "action": "upload",
                    "source": b64_data,
                    "format": "json"
                },
                timeout=8
            )
            if resp.status_code == 200:
                img_url = resp.json().get("image", {}).get("url")
                if img_url:
                    logger.info(f"Photo uploaded to cloud host: {img_url}")
                    return img_url
        except Exception as e:
            logger.debug(f"FreeImage upload error: {e}")

        # 2. Fallback to Catbox.moe
        try:
            with open(local_image_path, "rb") as f:
                resp = requests.post(
                    "https://catbox.moe/user/api.php",
                    data={"reqtype": "fileupload"},
                    files={"fileToUpload": f},
                    timeout=8
                )
            if resp.status_code == 200 and resp.text.startswith("http"):
                url = resp.text.strip()
                logger.info(f"Photo uploaded to cloud host (catbox): {url}")
                return url
        except Exception as e:
            logger.debug(f"Catbox upload error: {e}")

        # 3. Fallback to tmpfiles.org
        try:
            with open(local_image_path, "rb") as f:
                resp = requests.post("https://tmpfiles.org/api/v1/upload", files={"file": f}, timeout=8)
            if resp.status_code == 200:
                res_data = resp.json()
                if res_data.get("status") == "success":
                    raw_url = res_data["data"]["url"]
                    dl_url = raw_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
                    logger.info(f"Photo uploaded to cloud host (tmpfiles): {dl_url}")
                    return dl_url
        except Exception as e:
            logger.debug(f"Tmpfiles upload error: {e}")

        return None

    def upload_image(
        self,
        local_image_path: str,
        remote_filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Uploads local image snapshot and returns a direct, clickable https://... photo link.
        """
        if not os.path.exists(local_image_path):
            logger.error(f"Image path does not exist: {local_image_path}")
            return None

        # Try Firebase Cloud Storage first
        url = self._upload_to_firebase_storage(local_image_path, remote_filename, metadata)
        if url:
            return url

        # Fallback to direct cloud image host
        url = self._upload_to_cloud_host(local_image_path)
        return url

    def get_image_url(self, local_image_path: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Helper to get public direct photo URL for an image."""
        return self.upload_image(local_image_path, metadata=metadata)
