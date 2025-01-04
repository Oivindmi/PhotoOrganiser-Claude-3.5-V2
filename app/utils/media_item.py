import os
from datetime import datetime
import imagehash
from PIL import Image
import cv2
import numpy as np

class MediaItem:
    def __init__(self, file_path):
        self.file_path = file_path
        self.is_video = self._check_if_video()
        self.creation_time = self._get_creation_time()
        self.perceptual_hash = None
        self.keyframes = []
        self.audio_fingerprint = None

    def _check_if_video(self):
        _, ext = os.path.splitext(self.file_path)
        return ext.lower() in ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv']

    def _get_creation_time(self):
        return datetime.fromtimestamp(os.path.getctime(self.file_path))

    def compute_perceptual_hash(self):
        if self.is_video:
            # For video, compute hash of the first frame
            cap = cv2.VideoCapture(self.file_path)
            ret, frame = cap.read()
            if ret:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.perceptual_hash = imagehash.phash(img)
            cap.release()
        else:
            # For image, compute hash directly
            self.perceptual_hash = imagehash.phash(Image.open(self.file_path))

    def extract_keyframes(self, interval=1):
        if not self.is_video:
            return

        cap = cv2.VideoCapture(self.file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % (fps * interval) == 0:
                self.keyframes.append(frame)

            frame_count += 1

        cap.release()

    def compute_audio_fingerprint(self):
        if not self.is_video:
            return
        # Implement audio fingerprinting here
        # This is a placeholder and should be replaced with actual audio fingerprinting logic
        self.audio_fingerprint = "audio_fingerprint_placeholder"