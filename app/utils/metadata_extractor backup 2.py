import subprocess
import json
import os
import tempfile
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


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
            cmd = ['exiftool', '-j', '-a', '-G', '-ALL', '-charset', 'filename=utf8', '-@', arg_file]
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

    def process_files_in_batches(self, file_paths, batch_size=500):
        results = []
        with ThreadPoolExecutor() as executor:
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

            # Expanded camera model extraction
            camera_model = (
                    item.get('EXIF:Model') or
                    item.get('QuickTime:Model') or
                    item.get('XMP:Model') or
                    item.get('IFD0:Model') or
                    item.get('MakerNotes:Model') or
                    item.get('XML:Model') or
                    item.get('File:FileType')
            )

            file_metadata = {
                'file_path': file_path,
                'file_info': None,  # This will be filled by the caller (file_scanner)
                'camera_model': camera_model,
                'file_size': os.path.getsize(file_path) / (1024 * 1024) if os.path.exists(file_path) else 0,
            }

            # Extract all time-related fields and other metadata
            extra_metadata = {}
            for key, value in item.items():
                if 'Date' in key or 'Time' in key or key not in file_metadata:
                    extra_metadata[key.replace(':', '_')] = value

            file_metadata['extra_metadata'] = extra_metadata
            relevant_metadata.append(file_metadata)
            """
            extra_metadata = {}
            for key, value in item.items():
                if key not in ['SourceFile', 'File:FileSize']:
                    extra_metadata[key.replace(':', '_')] = value

            file_metadata['extra_metadata'] = extra_metadata
            relevant_metadata.append(file_metadata)
            """
        return relevant_metadata


