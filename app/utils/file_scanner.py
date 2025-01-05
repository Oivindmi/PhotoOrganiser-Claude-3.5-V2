import os
import hashlib
import logging

class FileScanner:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.photo_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.heic', '.heif', '.raw', '.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef', '.srw'}
        self.video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v', '.mpg', '.mpeg', '.3gp', '.3g2', '.mts', '.m2ts', '.ts'}
        self.allowed_extensions = self.photo_extensions.union(self.video_extensions)

    def is_media_file(self, file_path):
        _, extension = os.path.splitext(file_path)
        return extension.lower() in self.allowed_extensions


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