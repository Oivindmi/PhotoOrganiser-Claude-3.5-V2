import json
import logging
from dateutil import parser
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QObject, pyqtSignal
from app.utils.file_grouper import FileGrouper
from app.utils.image_comparison import ImageComparison
from app.views.synchronization_dialog import SynchronizationDialog
import os
from datetime import datetime, timedelta
import traceback
import re

"""test 2025"""

logger = logging.getLogger(__name__)

class TimeSynchronizer(QObject):
    database_updated = pyqtSignal()

    def __init__(self, db_manager, main_window):
        super().__init__()
        self.db_manager = db_manager
        self.main_window = main_window
        self.file_grouper = FileGrouper(db_manager)
        self.similarity_threshold = 0.5
        self.evaluated_groups = set()
        self.max_recursion_depth = 1000  # Add a recursion limit

    def start_synchronization(self):
        logger.info(f"Starting synchronization with threshold {self.similarity_threshold}")
        groups, group_id_map = self.file_grouper.group_files()
        sorted_groups = sorted(groups.items(), key=lambda x: sum(len(files) for files in x[1].values()), reverse=True)

        matches_confirmed = 0
        for i in range(len(sorted_groups)):
            for j in range(i + 1, len(sorted_groups)):
                base_group_key = sorted_groups[i][0]
                compare_group_key = sorted_groups[j][0]
                base_group_id = group_id_map[base_group_key]
                compare_group_id = group_id_map[compare_group_key]
                if base_group_id not in self.evaluated_groups or compare_group_id not in self.evaluated_groups:
                    match_confirmed = self.synchronize_groups(base_group_id, compare_group_id, base_group_key,
                                                              compare_group_key, sorted_groups[i][1],
                                                              sorted_groups[j][1])
                    if match_confirmed:
                        matches_confirmed += 1

        if matches_confirmed == 0:
            QMessageBox.warning(self.main_window, "No Matches Confirmed",
                                "No matches were confirmed between any of the groups.")
        else:
            QMessageBox.information(self.main_window, "Synchronization Complete",
                                    f"Completed synchronization. Confirmed matches: {matches_confirmed}")

    def synchronize_groups(self, base_group_id, compare_group_id, base_group_key, compare_group_key, base_time_windows,
                           compare_time_windows):
        session = self.db_manager.Session()
        try:
            best_matches = self.find_best_matches_across_windows(session, base_group_id, compare_group_id,
                                                                 base_time_windows, compare_time_windows)

            if not best_matches:
                logger.info(f"No matches found between groups {base_group_key} and {compare_group_key}")
                return False

            dialog = SynchronizationDialog(self.main_window)

            for base_file, compare_file, similarity in best_matches:
                logger.info(f"Comparing {base_file.file_path} with {compare_file.file_path}, similarity: {similarity}")
                dialog.set_images(base_file.file_path, compare_file.file_path)

                base_field = base_file.original_time_field if base_group_id in self.evaluated_groups else None
                compare_field = compare_file.original_time_field if compare_group_id in self.evaluated_groups else None

                dialog.set_metadata(base_file.extra_metadata, compare_file.extra_metadata, base_field, compare_field)
                dialog.set_group_info(f"Group {base_group_key} (ID: {base_group_id})",
                                      f"Group {compare_group_key} (ID: {compare_group_id})")
                dialog.set_file_names(os.path.basename(base_file.file_path), os.path.basename(compare_file.file_path))

                choice, left_field, right_field = dialog.get_user_choice()
                if choice == 'reject':
                    continue
                elif choice in ['left', 'right']:
                    self.update_group_times(session, base_group_id, compare_group_id, base_file, compare_file,
                                            left_field, right_field, choice)
                    return True

            return False  # If all matches were rejected
        finally:
            session.close()

    def _find_datetime_in_metadata(self, metadata, chosen_field):
        if not metadata:
            return None

        try:
            return self._parse_datetime_string(metadata.get(chosen_field))
        except ValueError as e:
            logger.warning(f"Failed to parse. Error: {str(e)}")

            logger.error(f"Failed to find valid datetime in metadata: {metadata}")
            return None

        except Exception as e:
            logger.error(f"Error in _find_datetime_in_metadata: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def update_group_times(self, session, base_group_id, compare_group_id, base_file, compare_file,
                           left_field, right_field, choice):
        try:
            logger.info(
                f"Starting update_group_times for base_group_id: {base_group_id}, compare_group_id: {compare_group_id}")

            base_metadata = self._ensure_dict(base_file.extra_metadata)
            compare_metadata = self._ensure_dict(compare_file.extra_metadata)

            #base_time = base_metadata.get(left_field)
            #compare_time = compare_metadata.get(right_field)
            base_time = self._find_datetime_in_metadata(base_metadata,left_field)
            compare_time = self._find_datetime_in_metadata(compare_metadata,right_field)

            if base_time and compare_time:
                base_time = self._make_naive(base_time)
                compare_time = self._make_naive(compare_time)

                time_delta = base_time - compare_time

                logger.info(f"Base time: {base_time}, Compare time: {compare_time}, Time delta: {time_delta}")

                if base_group_id not in self.evaluated_groups:
                    self.update_group(session, base_group_id, left_field,
                                      time_delta if choice == 'left' else timedelta())
                    self.evaluated_groups.add(base_group_id)

                if compare_group_id not in self.evaluated_groups:
                    self.update_group(session, compare_group_id, right_field,
                                      timedelta() if choice == 'left' else -time_delta)
                    self.evaluated_groups.add(compare_group_id)

                session.commit()
                logger.info(f"Committed changes to database")
                self.database_updated.emit()
                logger.info(f"Emitted database_updated signal")
            else:
                logger.error(f"Failed to parse datetime for fields: {left_field}, {right_field}")

        except Exception as e:
            logger.error(f"Error in update_group_times: {str(e)}")
            logger.error(traceback.format_exc())
            session.rollback()

    def update_group(self, session, group_id, chosen_field, time_delta):
        try:
            logger.info(f"Starting update_group for group_id: {group_id}, chosen_field: {chosen_field}")

            files = session.query(self.db_manager.FileMetadata).filter_by(group_id=group_id).all()
            logger.info(f"Found {len(files)} files in group {group_id}")

            for file in files:
                file_metadata = self._ensure_dict(file.extra_metadata)
                original_time = self._find_datetime_in_metadata(file_metadata,chosen_field)
                if original_time:
                    file.correct_time = original_time + time_delta
                    file.original_time_field = chosen_field
                    logger.info(
                        f"Updating file {file.file_path}: correct_time = {file.correct_time}, original_time_field = {file.original_time_field}")
                else:
                    logger.warning(f"Could not find original time for file {file.file_path}")

        except Exception as e:
            logger.error(f"Error in update_group: {str(e)}")
            logger.error(traceback.format_exc())
            session.rollback()

    def apply_time_delta(self, session, group_id, time_delta):
        files = session.query(self.db_manager.FileMetadata).filter_by(group_id=group_id).all()
        for file in files:
            if file.correct_time:
                file.correct_time += time_delta
                logger.info(f"Applied time delta to file {file.file_path}: new correct_time = {file.correct_time}")

    def find_best_matches_across_windows(self, session, base_group_id, compare_group_id, base_time_windows,
                                         compare_time_windows):
        all_matches = []
        for base_window, base_file_ids in base_time_windows.items():
            compare_window_start = base_window - timedelta(minutes=30)
            compare_window_end = base_window + timedelta(minutes=30)

            compare_file_ids = []
            for window, file_ids in compare_time_windows.items():
                if compare_window_start <= window <= compare_window_end:
                    compare_file_ids.extend(file_ids)

            base_files = session.query(self.db_manager.FileMetadata).filter(
                self.db_manager.FileMetadata.id.in_(base_file_ids)).all()
            compare_files = session.query(self.db_manager.FileMetadata).filter(
                self.db_manager.FileMetadata.id.in_(compare_file_ids)).all()

            matches = self.find_best_matches(base_files, compare_files)
            all_matches.extend(matches)

        all_matches.sort(key=lambda x: x[2], reverse=True)
        return all_matches[:5]  # Return top 5 matches across all time windows

    def find_best_matches(self, base_files, compare_files):
        matches = []
        for base_file in base_files:
            for compare_file in compare_files:
                similarity = ImageComparison.compare_media(base_file.file_path, compare_file.file_path)
                if similarity >= self.similarity_threshold:
                    matches.append((base_file, compare_file, similarity))
        return matches

    def _ensure_dict(self, metadata):
        if isinstance(metadata, str):
            try:
                return json.loads(metadata)
            except json.JSONDecodeError:
                return {}
        return metadata if isinstance(metadata, dict) else {}

    def _parse_datetime_string(self, date_string):
        try:
            # Remove timezone information
            date_string = re.sub(r'[+-]\d{2}:\d{2}$', '', date_string)

            # Parse the datetime
            dt = parser.parse(date_string)

            # Return a naive datetime object
            return dt.replace(tzinfo=None)

        except Exception as e:
            logger.error(f"Error in _parse_datetime_string: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    """
    def _parse_datetime(self, date_string):
        if not date_string:
            return None

        # Preprocess the date string
        date_string = self._preprocess_date_string(date_string)

        # List of date formats to try
        formats = [
            "%Y:%m:%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%d.%m.%Y %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",  # ISO format
            "%Y:%m:%d %H:%M:%S.%f",  # With microseconds
            "%Y%m%d_%H%M%S",  # Compact format
            "%Y:%m:%d",  # Date only
            "%d/%m/%Y",  # European date format
            "%m/%d/%Y",  # American date format
            "%b %d %Y %H:%M:%S",  # Month abbreviation format
            "%Y-%m-%d %H:%M:%S%z",  # With timezone
            # Add more formats as needed
        ]

        logger.info(f"Attempting to parse date string: {date_string}")

        for fmt in formats:
            try:
                dt = datetime.datetime.strptime(date_string, fmt)
                logger.info(f"Successfully parsed datetime: {dt} using format: {fmt}")
                return dt
            except ValueError:
                continue

        logger.error(f"Failed to parse datetime: {date_string}. No matching format found.")
        return None

    def _preprocess_date_string(self, date_string):
        # Remove any leading/trailing whitespace
        date_string = date_string.strip()

        # Replace multiple spaces with a single space
        date_string = re.sub(r'\s+', ' ', date_string)

        # Replace commas with spaces
        date_string = date_string.replace(',', ' ')

        # If the string starts with a day of the week, remove it
        date_string = re.sub(r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*,?\s*', '', date_string, flags=re.IGNORECASE)

        # If there's a timezone abbreviation at the end, remove it
        date_string = re.sub(r'\s+[A-Z]{3,4}$', '', date_string)

        # Replace '.' with ':' in time part (if present)
        time_part = date_string.split()[-1]
        if ':' not in time_part and '.' in time_part:
            time_part_new = time_part.replace('.', ':')
            date_string = date_string.replace(time_part, time_part_new)

        return date_string

    """

    def _parse_datetime(self, date_string):
        if not date_string:
            return None
        try:
            logger.info(f"date sent to parser.parse {date_string}")
            # dt=datetime.datetime.strptime(date_string, "%Y:%m:%d %H:%M:%S")
            dt = parser.parse(date_string)
            logger.info(f"date returned by parser.parse {dt}")
            return dt
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to parse datetime: {date_string}. Error: {str(e)}")
            return None

    """

    I think the main problem is with timezones for the date and time metadata. I am not really interested in the timezones and am considering to remove them altogether. Could we remove them when importing the data into the database?


    def _parse_datetime(self, date_string):
        if not date_string:
            return None
        try:
            # Log the input string
            logger.info(f"Attempting to parse date string: {date_string}")

            # Split the date string
            parts = date_string.replace(':', ' ').split()
            if len(parts) >= 5:  # Ensure we have at least YYYY MM DD HH MM
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
                hour = int(parts[3])
                minute = int(parts[4])
                second = int(parts[5]) if len(parts) > 5 else 0

                dt = datetime(year, month, day, hour, minute, second)
                logger.info(f"Parsed datetime: {dt}")
                return dt
            else:
                logger.error(f"Invalid date string format: {date_string}")
                return None
        except (ValueError, TypeError, IndexError) as e:
            logger.error(f"Failed to parse datetime: {date_string}. Error: {str(e)}")
            return None
    """

    def _make_naive(self, dt):
        if dt.tzinfo is None:
            return dt
        return dt.astimezone(pytz.UTC).replace(tzinfo=None)

    def set_similarity_threshold(self, threshold):
        self.similarity_threshold = threshold
        logger.info(f"Similarity threshold set to {threshold}")