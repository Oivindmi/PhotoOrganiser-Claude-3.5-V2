import subprocess
import json
import os
import tempfile
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from dateutil import parser
import re

class MetadataExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def create_arg_file(file_paths):
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            for path in file_paths:
                f.write(f'-charset\nfilename=utf8\n{path}\n')
        return f.name

    def process_batch_with_exiftool(self, file_paths):
        try:
            arg_file = self.create_arg_file(file_paths)
            cmd = ['exiftool', '-j', '-a', '-G', '-time:all', '-EXIF:Make', '-EXIF:Model', '-IFD0:Make', '-IFD0:Model',
                   '-QuickTime:Model', '-XMP:Model', '-MakerNotes:Model', '-XML:Model', '-charset', 'filename=utf8',
                   '-@', arg_file]
            self.logger.debug(f"ExifTool command: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

            self.logger.debug(f"ExifTool stdout: {result.stdout}")
            self.logger.debug(f"ExifTool stderr: {result.stderr}")

            if result.returncode != 0:
                self.logger.warning(f"ExifTool warning (return code {result.returncode}):")
                self.logger.warning(f"Standard error: {result.stderr}")

            try:
                metadata = json.loads(result.stdout)
                self.logger.debug(f"Parsed metadata: {json.dumps(metadata, indent=2)}")
                return metadata
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON Decode error: {str(e)}")
                self.logger.error(f"ExifTool output: {result.stdout}")
                return []
        except Exception as e:
            self.logger.error(f"Error processing batch: {str(e)}")
            return []
        finally:
            if 'arg_file' in locals():
                os.remove(arg_file)

    def process_files_in_batches(self, file_paths, batch_size=100):
        results = []
        with ProcessPoolExecutor() as executor:
            futures = []
            for i in range(0, len(file_paths), batch_size):
                batch = file_paths[i:i + batch_size]
                futures.append(executor.submit(self.process_batch_with_exiftool, batch))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                result = future.result()
                results.extend(result)

        return results

    def extract_relevant_metadata(self, metadata):
        relevant_metadata = []
        for item in metadata:
            file_path = item.get('SourceFile')
            if not file_path:
                self.logger.warning(f"Skipping item without SourceFile: {item}")
                continue

            camera_make = item.get('EXIF:Make') or item.get('IFD0:Make') or ''
            camera_model = (
                item.get('EXIF:Model') or
                item.get('IFD0:Model') or
                item.get('QuickTime:Model') or
                item.get('XMP:Model') or
                item.get('MakerNotes:Model') or
                item.get('XML:Model')
            )

            if camera_make and camera_model:
                camera_info = f"{camera_make} {camera_model}"
            elif camera_model:
                camera_info = camera_model
            elif camera_make:
                camera_info = camera_make
            else:
                camera_info = "Unknown"

            file_metadata = {
                'file_path': file_path,
                'file_info': None,  # To be filled by the caller
                'camera_model': camera_info,
                'file_size': os.path.getsize(file_path) / (1024 * 1024) if os.path.exists(file_path) else 0,
            }

            extra_metadata = {}
            for key, value in item.items():
                if 'Date' in key or 'Time' in key:
                    cleaned_value = self._clean_date_string(key, value)
                    if cleaned_value:
                        extra_metadata[key.replace(':', '_')] = cleaned_value

            file_metadata['extra_metadata'] = extra_metadata
            relevant_metadata.append(file_metadata)

        self.logger.info(f"Extracted metadata for {len(relevant_metadata)} files")
        if len(relevant_metadata) > 0:
            self.logger.debug(f"Sample metadata: {relevant_metadata[0]}")
        return relevant_metadata

    def _clean_date_string(self, date_string):
        if not isinstance(date_string, str):
            return str(date_string)

        # Remove all letters
        cleaned = re.sub(r'[a-zA-Z]', '', date_string)

        # Remove timezone information (anything after '+' or '-' near the end of the string)
        cleaned = re.sub(r'[+-]\d{2}:?\d{2}$', '', cleaned)

        # Replace ':' with '-' in date part (assuming date comes before time)
        parts = cleaned.split()
        if len(parts) > 0:
            parts[0] = parts[0].replace(':', '-')
        cleaned = ' '.join(parts)

        return cleaned.strip()

    def _clean_date_string(self, field_name, date_string):
        if not isinstance(date_string, str):
            return str(date_string)

        # Remove all letters and leading/trailing whitespace
        cleaned = re.sub(r'[a-zA-Z]', '', date_string).strip()

        # Remove timezone information (anything after '+' or '-' near the end of the string)
        cleaned = re.sub(r'[+-]\d{2}:?\d{2}$', '', cleaned)

        # Check if it's likely a time-only field
        is_time_only = 'Time' in field_name and 'Date' not in field_name and len(cleaned) <= 12

        if not is_time_only:
            # Replace ':' with '-' in date part only if it's not a time-only string
            parts = cleaned.split()
            if len(parts) > 0:
                parts[0] = parts[0].replace(':', '-')
            cleaned = ' '.join(parts)

        return cleaned.strip()