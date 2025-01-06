import os
import hashlib
import logging
import cv2
import numpy as np

class FileScanner:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.photo_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.heic', '.heif', '.raw', '.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef', '.srw'}
        self.video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v', '.mpg', '.mpeg', '.3gp', '.3g2', '.mts', '.m2ts', '.ts'}
        self.allowed_extensions = self.photo_extensions.union(self.video_extensions)
        self.frames_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'video_frames')
        os.makedirs(self.frames_dir, exist_ok=True)

    def is_media_file(self, file_path):
        _, extension = os.path.splitext(file_path)
        return extension.lower() in self.allowed_extensions

    def is_video(self, file_path: str) -> bool:
        _, ext = os.path.splitext(file_path)
        return ext.lower() in self.video_extensions

    def scan_folders(self, folder_paths):
        all_files = []
        file_info_dict = {}
        for folder_path in folder_paths:
            self.logger.info(f"Scanning folder: {folder_path}")
            for root, _, files in os.walk(folder_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    if self.is_media_file(full_path):
                        all_files.append(full_path)
                        file_info = self.get_file_info(full_path)
                        file_info_dict[file_info] = full_path
                    else:
                        self.logger.debug(f"Skipping non-media file: {full_path}")

        self.logger.info(f"Found {len(all_files)} media files.")
        return all_files, file_info_dict

    @staticmethod
    def get_file_info(file_path):
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        parent_dir = os.path.basename(os.path.dirname(file_path))
        unique_id = hashlib.md5(f"{file_name}_{file_size}_{parent_dir}".encode()).hexdigest()
        return unique_id

    def clean_old_frames(self):
        """Remove frame files that are no longer referenced in the database"""
        if not os.path.exists(self.frames_dir):
            return

        try:
            # This would need database access to check which frames are still needed
            # For now, we'll keep all frames
            pass
        except Exception as e:
            self.logger.error(f"Error cleaning old frames: {str(e)}")

    def verify_video_frames(self, frame_paths: list[str]) -> list[str]:
        """Verify frames exist and return only valid paths"""
        if not frame_paths:
            return []

        valid_frames = []
        for path in frame_paths:
            if os.path.isfile(path):
                valid_frames.append(path)
            else:
                self.logger.warning(f"Missing video frame: {path}")

        return valid_frames

    def extract_video_frames(self, video_path: str) -> list[str]:
        if not self.is_video(video_path):
            return []

        video_hash = hashlib.md5(video_path.encode()).hexdigest()
        frame_dir = os.path.join(self.frames_dir, video_hash)
        os.makedirs(frame_dir, exist_ok=True)

        try:
            # Handle Unicode paths for video
            stream = open(video_path, 'rb')
            bytes_array = bytes(stream.read())
            numpy_array = np.asarray(bytearray(bytes_array), dtype=np.uint8)
            cap = cv2.VideoCapture()
            cap.open(video_path)

            if not cap.isOpened():
                self.logger.error(f"Could not open video: {video_path}")
                return []

            frames_data = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            if duration <= 0:
                self.logger.error(f"Invalid duration for video: {video_path}")
                return []

            time_positions = [
                0,
                duration * 0.25,
                duration * 0.50,
                duration * 0.75,
                max(0, duration - 1)
            ]

            for idx, pos in enumerate(time_positions):
                cap.set(cv2.CAP_PROP_POS_MSEC, pos * 1000)
                ret, frame = cap.read()
                if ret:
                    frame_path = os.path.join(frame_dir, f"frame_{idx}.jpg")
                    # Use imencode/imdecode for saving
                    _, buf = cv2.imencode('.jpg', frame)
                    with open(frame_path, 'wb') as f:
                        f.write(buf)
                    frames_data.append(frame_path)
                    self.logger.info(f"Extracted frame {idx} at {pos:.2f}s from {video_path}")
                else:
                    self.logger.warning(f"Failed to read frame at {pos:.2f}s from {video_path}")

            return frames_data

        except Exception as e:
            self.logger.error(f"Error extracting frames from {video_path}: {str(e)}")
            return []

        finally:
            if 'cap' in locals():
                cap.release()
            if 'stream' in locals():
                stream.close()