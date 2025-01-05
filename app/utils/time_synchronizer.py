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

""" 04.01.2025 10:00"""

logger = logging.getLogger(__name__)

class TimeSynchronizer(QObject):
    database_updated = pyqtSignal()

    def __init__(self, db_manager, main_window):
        super().__init__()
        self.db_manager = db_manager
        self.main_window = main_window
        self.file_grouper = FileGrouper(db_manager)
        self.similarity_threshold = 0.5
        self.combined_groups = []  # List of lists of group_ids that are combined
        self.evaluated_groups = set()  # Now only tracks which groups have chosen time fields

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
            logger.info(f"Base group ID: {base_group_id} has been evaluated: {base_group_id in self.evaluated_groups}")
            logger.info(
                f"Compare group ID: {compare_group_id} has been evaluated: {compare_group_id in self.evaluated_groups}")

            best_matches = self.find_best_matches_across_windows(session, base_group_id, compare_group_id,
                                                                 base_time_windows, compare_time_windows)

            if not best_matches:
                logger.info(f"No matches found between groups {base_group_key} and {compare_group_key}")
                return False

            dialog = SynchronizationDialog(self.main_window)

            for base_file, compare_file, similarity in best_matches:
                try:
                    logger.info(f"Processing match: {base_file.file_path} with {compare_file.file_path}")

                    dialog.set_images(base_file.file_path, compare_file.file_path)
                    logger.info("Images set in dialog")

                    base_evaluated = base_group_id in self.evaluated_groups
                    compare_evaluated = compare_group_id in self.evaluated_groups
                    logger.info(f"Evaluation status - Base: {base_evaluated}, Compare: {compare_evaluated}")

                    base_field = base_file.original_time_field if base_evaluated else None
                    compare_field = compare_file.original_time_field if compare_evaluated else None
                    logger.info(f"Time fields - Base: {base_field}, Compare: {compare_field}")

                    # Log metadata before setting
                    logger.info(f"Base metadata: {base_file.extra_metadata}")
                    logger.info(f"Compare metadata: {compare_file.extra_metadata}")
                    logger.info(f"Base correct_time: {base_file.correct_time}")
                    logger.info(f"Compare correct_time: {compare_file.correct_time}")

                    try:
                        dialog.set_metadata(
                            base_file.extra_metadata,
                            compare_file.extra_metadata,
                            base_evaluated,
                            compare_evaluated,
                            base_field,
                            compare_field,
                            base_file.correct_time,
                            compare_file.correct_time
                        )
                        logger.info("Metadata set in dialog")
                    except Exception as e:
                        logger.error(f"Error setting metadata: {str(e)}")
                        raise

                    # dialog.set_group_info(f"Group {base_group_key} (ID: {base_group_id})",
                    #                      f"Group {compare_group_key} (ID: {compare_group_id})")
                    logger.info("Group info set in dialog")

                    dialog.set_file_names(os.path.basename(base_file.file_path),
                                          os.path.basename(compare_file.file_path))
                    logger.info("File names set in dialog")

                    choice, left_field, right_field = dialog.get_user_choice()
                    logger.info(f"User choice received: {choice}, {left_field}, {right_field}")

                    if choice == 'reject':
                        continue
                    elif choice in ['left', 'right']:
                        self.update_group_times(session, base_group_id, compare_group_id, base_file, compare_file,
                                                left_field, right_field, choice)
                        return True

                except Exception as e:
                    logger.error(f"Error processing match: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue

            return False
        except Exception as e:
            logger.error(f"Error in synchronize_groups: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        finally:
            try:
                session.close()
            except Exception as e:
                logger.error(f"Error closing session: {str(e)}")

    def _find_datetime_in_metadata(self, metadata, chosen_field):
        if not metadata:
            return None
        logging.info(f"\nLooking for datetime in field: {chosen_field}")
        logging.info(f"Available metadata fields: {json.dumps(metadata, indent=2)}")
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

            base_combined_group = self.get_combined_group_for_id(base_group_id)
            logger.info(f"Base combined group: {base_combined_group}")

            if choice == 'right':
                # Using right (compare) time as reference
                if right_field == "Updated Time":
                    compare_time = compare_file.correct_time
                else:
                    compare_metadata = self._ensure_dict(compare_file.extra_metadata)
                    compare_time = self._find_datetime_in_metadata(compare_metadata, right_field)

                if compare_time:
                    logger.info(f"Using compare time as reference: {compare_time}")
                    # Set compare group without adjustment
                    self.update_group(session, compare_group_id, right_field, timedelta())

                    # Calculate time delta for base files
                    if left_field == "Updated Time":
                        left_base_time = base_file.correct_time
                    else:
                        left_metadata = self._ensure_dict(base_file.extra_metadata)
                        left_base_time = self._find_datetime_in_metadata(left_metadata, left_field)

                    if left_base_time:
                        time_delta = compare_time - left_base_time
                        logger.info(f"Calculated time delta: {time_delta}")

                        # Update all files in the combined group with the same delta
                        for group_id in base_combined_group:
                            logger.info(f"Updating group {group_id} with time delta")
                            files = session.query(self.db_manager.FileMetadata).filter_by(group_id=group_id).all()
                            for file in files:
                                if file.original_time_field and file.correct_time:
                                    # Update existing correct_time
                                    file.correct_time += time_delta
                                    logger.info(f"Updated existing time for {file.file_path} to {file.correct_time}")
                                else:
                                    # For files not yet evaluated, set initial time using same field as base
                                    file_metadata = self._ensure_dict(file.extra_metadata)
                                    original_time = self._find_datetime_in_metadata(file_metadata, left_field)
                                    if original_time:
                                        file.correct_time = original_time + time_delta
                                        file.original_time_field = left_field
                                        logger.info(f"Set initial time for {file.file_path} to {file.correct_time}")
            else:
                # Using left (base) time as reference
                if left_field == "Updated Time":
                    base_time = base_file.correct_time
                else:
                    base_metadata = self._ensure_dict(base_file.extra_metadata)
                    base_time = self._find_datetime_in_metadata(base_metadata, left_field)

                if base_time:
                    logger.info(f"Using base time as reference: {base_time}")
                    # Keep base times as they are
                    time_delta = timedelta()
                    for group_id in base_combined_group:
                        logger.info(f"Maintaining existing times for group {group_id}")
                        self.update_group(session, group_id, left_field, time_delta)

                    # Calculate and apply delta to compare group
                    if right_field == "Updated Time":
                        compare_time = compare_file.correct_time
                    else:
                        compare_metadata = self._ensure_dict(compare_file.extra_metadata)
                        compare_time = self._find_datetime_in_metadata(compare_metadata, right_field)

                    if compare_time:
                        time_delta = base_time - compare_time
                        logger.info(f"Calculated time delta for compare group: {time_delta}")
                        self.update_group(session, compare_group_id, right_field, time_delta)

            # Mark all groups as evaluated and merge them
            self.evaluated_groups.add(compare_group_id)
            for group_id in base_combined_group:
                self.evaluated_groups.add(group_id)

            # Merge the groups
            self.merge_groups(base_group_id, compare_group_id)

            session.commit()
            logger.info("Committed changes to database")
            self.database_updated.emit()
            logger.info("Emitted database_updated signal")
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

                if chosen_field == "Updated Time" and file.correct_time:
                    # Just apply the delta to the existing correct_time
                    file.correct_time += time_delta
                    logger.info(f"Updated correct_time with delta for {file.file_path}: new time = {file.correct_time}")
                else:
                    # Find the original time from metadata and apply delta
                    original_time = self._find_datetime_in_metadata(file_metadata, chosen_field)
                    if original_time:
                        file.correct_time = original_time + time_delta
                        file.original_time_field = chosen_field
                        logger.info(f"Setting correct_time for {file.file_path} to {file.correct_time}, "
                                    f"original_time_field = {chosen_field}")
                    else:
                        logger.warning(f"Could not find original time for file {file.file_path}")

            logger.info(f"Completed update_group for group {group_id}")
        except Exception as e:
            logger.error(f"Error in update_group: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def apply_time_delta(self, session, group_id, time_delta):
        files = session.query(self.db_manager.FileMetadata).filter_by(group_id=group_id).all()
        for file in files:
            if file.correct_time:
                file.correct_time += time_delta
                logger.info(f"Applied time delta to file {file.file_path}: new correct_time = {file.correct_time}")

    def find_best_matches_across_windows(self, session, base_group_id, compare_group_id,
                                         base_time_windows, compare_time_windows):
        all_matches = []
        base_files = {}
        compare_files = {}
        image_pairs = []

        # Get all base files
        base_file_ids = [id for window_files in base_time_windows.values() for id in window_files]
        base_query = session.query(self.db_manager.FileMetadata).filter(
            self.db_manager.FileMetadata.id.in_(base_file_ids)
        ).all()
        base_files = {f.id: f for f in base_query}

        # Get all compare files
        compare_file_ids = [id for window_files in compare_time_windows.values() for id in window_files]
        compare_query = session.query(self.db_manager.FileMetadata).filter(
            self.db_manager.FileMetadata.id.in_(compare_file_ids)
        ).all()
        compare_files = {f.id: f for f in compare_query}

        # Create pairs for parallel processing
        for base_window, base_window_ids in base_time_windows.items():
            compare_window_start = base_window - timedelta(minutes=30)
            compare_window_end = base_window + timedelta(minutes=30)

            relevant_compare_ids = []
            for compare_window, compare_window_ids in compare_time_windows.items():
                if compare_window_start <= compare_window <= compare_window_end:
                    relevant_compare_ids.extend(compare_window_ids)

            for base_id in base_window_ids:
                if base_id in base_files:
                    base_file = base_files[base_id]
                    for compare_id in relevant_compare_ids:
                        if compare_id in compare_files:
                            compare_file = compare_files[compare_id]
                            image_pairs.append(((base_file.file_path, compare_file.file_path),
                                                (base_file, compare_file)))

        if not image_pairs:
            return []

        # Split the pairs
        file_pairs, file_objects = zip(*image_pairs)

        # Process all comparisons in parallel
        similarities = ImageComparison.batch_compare_media(file_pairs)

        # Create match tuples with similarity scores
        for (base_file, compare_file), similarity in zip(file_objects, similarities):
            if similarity >= self.similarity_threshold:
                all_matches.append((base_file, compare_file, similarity))

        all_matches.sort(key=lambda x: x[2], reverse=True)
        return all_matches[:5]

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

    def get_combined_group_for_id(self, group_id):
        for combined_group in self.combined_groups:
            if group_id in combined_group:
                return combined_group
        return [group_id]

    def merge_groups(self, group_id1, group_id2):
        """Merge two groups and their associated groups"""
        logger.info(f"Merging groups {group_id1} and {group_id2}")

        # Get existing combined groups
        group1 = self.get_combined_group_for_id(group_id1)
        group2 = self.get_combined_group_for_id(group_id2)

        # Remove existing combined groups if they exist
        if group1 in self.combined_groups:
            self.combined_groups.remove(group1)
        if group2 in self.combined_groups and group1 != group2:
            self.combined_groups.remove(group2)

        # Create new combined group
        merged_group = list(set(group1 + group2))
        self.combined_groups.append(merged_group)

        logger.info(f"Created new combined group: {merged_group}")
        return merged_group
