import os
from collections import defaultdict
from datetime import datetime, timedelta
import re

class FileGrouper:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.next_group_id = 1

    def group_files(self, time_window_minutes=10):
        session = self.db_manager.Session()
        try:
            all_files = session.query(self.db_manager.FileMetadata).all()
            groups = defaultdict(lambda: defaultdict(list))
            group_id_map = {}

            for file in all_files:
                group_key = self._generate_group_key(file)
                creation_time = self._get_creation_time(file)
                time_window = creation_time.replace(
                    minute=creation_time.minute - creation_time.minute % time_window_minutes,
                    second=0, microsecond=0
                )
                groups[group_key][time_window].append(file.id)

                if group_key not in group_id_map:
                    group_id_map[group_key] = self.next_group_id
                    self.next_group_id += 1

            # Assign group IDs and keys
            for group_key, time_windows in groups.items():
                group_id = group_id_map[group_key]
                for time_window, file_ids in time_windows.items():
                    session.query(self.db_manager.FileMetadata).filter(
                        self.db_manager.FileMetadata.id.in_(file_ids)
                    ).update({"group_id": group_id, "group_key": group_key}, synchronize_session='fetch')

            session.commit()
            return dict(groups), group_id_map
        finally:
            session.close()

    def _generate_group_key(self, file):
        parent_folder = os.path.basename(os.path.dirname(file.file_path))
        if file.camera_model:
            return f"{file.camera_model}_{parent_folder}"
        else:
            file_type = os.path.splitext(file.file_path)[1].lower()
            name_pattern = self._get_name_pattern(os.path.basename(file.file_path))
            return f"Unknown_{parent_folder}_{file_type}_{name_pattern}"

    def _get_name_pattern(self, filename):
        pattern = re.sub(r'\d+', '#', filename)
        return pattern

    def _get_creation_time(self, file):
        return file.creation_time if hasattr(file, 'creation_time') else datetime.fromtimestamp(os.path.getctime(file.file_path))

    def get_group_info(self):
        session = self.db_manager.Session()
        try:
            groups = defaultdict(lambda: {'count': 0, 'sample_file': None, 'group_key': None})
            for file in session.query(self.db_manager.FileMetadata).all():
                groups[file.group_id]['count'] += 1
                groups[file.group_id]['group_key'] = file.group_key
                if groups[file.group_id]['sample_file'] is None:
                    groups[file.group_id]['sample_file'] = file
            return groups
        finally:
            session.close()