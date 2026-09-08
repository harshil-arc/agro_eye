import os
from pathlib import Path
from typing import Optional
from config import BASE_DIR, FIREBASE_CREDENTIALS_PATH, FIREBASE_DATABASE_URL, FIREBASE_STORAGE_BUCKET
from utils.logger import logger

class FirebaseClient:
    _instance = None
    _initialized = False
    _warned_missing = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FirebaseClient, cls).__new__(cls)
        return cls._instance

    def initialize(self) -> bool:
        """Initializes the Firebase Admin SDK app instance."""
        if self._initialized:
            return True

        # Resolve credentials path (supports relative to project root or absolute)
        cred_path = Path(FIREBASE_CREDENTIALS_PATH)
        if not cred_path.is_absolute():
            cred_path = (BASE_DIR / cred_path).resolve()

        if not cred_path.exists():
            if not FirebaseClient._warned_missing:
                logger.warning(
                    f"Firebase credentials file not found at: {cred_path}. "
                    "Cloud sync will run in offline mode (buffering to local SQLite) until service_account.json is added."
                )
                FirebaseClient._warned_missing = True
            return False

        try:
            import firebase_admin
            from firebase_admin import credentials

            if not firebase_admin._apps:
                cred = credentials.Certificate(str(cred_path))
                firebase_admin.initialize_app(cred, {
                    'databaseURL': FIREBASE_DATABASE_URL,
                    'storageBucket': FIREBASE_STORAGE_BUCKET
                })
            self._initialized = True
            logger.info("Firebase Admin SDK initialized successfully. Cloud sync is active!")
            return True
        except Exception as e:
            logger.error(f"Error initializing Firebase Admin SDK: {e}")
            self._initialized = False
            return False

    @property
    def is_ready(self) -> bool:
        return self._initialized
