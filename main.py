import os
import sys
import logging
from PyQt5.QtWidgets import QApplication
from app.controllers.main_controller import MainController
from app.utils.file_scanner import FileScanner
from app.utils.metadata_extractor import MetadataExtractor
from app.models.database_manager import DatabaseManager
from app.utils.file_grouper import FileGrouper
from app.utils.time_synchronizer import TimeSynchronizer


def setup_logging():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename='photo_organizer.log',
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def normalize_path(path):
    return os.path.normpath(path).replace('\\', '/')

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    db_path = 'photo_organizer.db'
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.info(f"Deleted existing database: {db_path}")

    app = QApplication(sys.argv)

    file_scanner = FileScanner()
    metadata_extractor = MetadataExtractor()
    db_manager = DatabaseManager(db_path)

    folders_to_process = [r"C:\TEST FOLDER FOR PHOTO APP\TEST ARE RUN ON THIS FOLDER"]
    all_files, file_info_dict = file_scanner.scan_folders(folders_to_process)
    all_files_set = set(normalize_path(path) for path in all_files)

    logger.info(f"Total media files found: {len(all_files)}")
    if len(all_files) > 0:
        logger.debug(f"Sample file path: {all_files[0]}")

    metadata = metadata_extractor.process_files_in_batches(all_files)
    logger.info(f"Processed metadata for {len(metadata)} files")

    relevant_metadata = metadata_extractor.extract_relevant_metadata(metadata)
    logger.info(f"Extracted relevant metadata for {len(relevant_metadata)} files")

    filtered_metadata = [item for item in relevant_metadata if normalize_path(item['file_path']) in all_files_set]

    logger.info(f"Filtered metadata items: {len(filtered_metadata)}")
    if len(filtered_metadata) > 0:
        logger.debug(f"Sample filtered metadata: {filtered_metadata[0]}")
    else:
        logger.warning("No metadata passed the filter. Check file paths and metadata extraction.")

    if not filtered_metadata:
        logger.warning("No metadata to add to the database. Check file paths and metadata extraction.")
    else:
        for item in filtered_metadata:
            item['file_info'] = file_scanner.get_file_info(item['file_path'])

        if db_manager.add_file_metadata_bulk(filtered_metadata):
            logger.info(f"Successfully added/updated {len(filtered_metadata)} entries in the database.")
        else:
            logger.error("Failed to add/update metadata in the database.")

    # Group files with time-based subgrouping
    file_grouper = FileGrouper(db_manager)
    groups = file_grouper.group_files(time_window_minutes=10)  # 10-minute time windows
    logger.info(f"Grouped files into {len(groups)} camera groups with time-based subgroups.")

    # Initialize main controller and show main window
    controller = MainController(db_manager)
    main_window = controller.show_main_window()

    # Initialize time synchronization
    time_synchronizer = TimeSynchronizer(db_manager, main_window)
    main_window.set_time_synchronizer(time_synchronizer)

    # Connect the start synchronization button
    main_window.sync_button.clicked.connect(controller.start_synchronization)

    # Connect the similarity threshold slider
    main_window.threshold_slider.valueChanged.connect(lambda value: controller.set_similarity_threshold(value / 100.0))

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()