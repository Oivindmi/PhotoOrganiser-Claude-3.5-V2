import os
import sys
import logging
import time
from PyQt5.QtWidgets import QApplication
from app.controllers.main_controller import MainController
from app.utils.file_scanner import FileScanner
from app.utils.metadata_extractor import MetadataExtractor
from app.models.database_manager import DatabaseManager
from app.utils.file_grouper import FileGrouper
from app.utils.time_synchronizer import TimeSynchronizer
from app.utils.image_comparison import ImageComparison


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
    total_start = time.time()
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting photo organizer")
    db_path = 'photo_organizer.db'

    # Database initialization
    init_start = time.time()
    if os.path.exists(db_path):
        os.remove(db_path)
        logger.info(f"Deleted existing database: {db_path}")

    file_scanner = FileScanner()
    metadata_extractor = MetadataExtractor(file_scanner)
    db_manager = DatabaseManager(db_path)
    ImageComparison.set_db_manager(db_manager)
    logger.info(f"Initialization took: {time.time() - init_start:.2f} seconds")

    # File scanning
    scan_start = time.time()
    folders_to_process = [r"C:\TEST FOLDER FOR PHOTO APP\TEST ARE RUN ON THIS FOLDER"]
    all_files, file_info_dict = file_scanner.scan_folders(folders_to_process)
    all_files_set = set(normalize_path(path) for path in all_files)
    logger.info(f"File scanning took: {time.time() - scan_start:.2f} seconds")
    logger.info(f"Found {len(all_files)} media files")

    # Metadata extraction
    meta_start = time.time()
    metadata = metadata_extractor.process_files_in_batches(all_files)
    logger.info(f"Metadata extraction took: {time.time() - meta_start:.2f} seconds")
    logger.info(f"Processed metadata for {len(metadata)} files")

    # Metadata processing
    process_start = time.time()
    relevant_metadata = metadata_extractor.extract_relevant_metadata(metadata)
    logger.info(f"Metadata processing took: {time.time() - process_start:.2f} seconds")

    # Database operations
    db_start = time.time()
    filtered_metadata = [item for item in relevant_metadata if normalize_path(item['file_path']) in all_files_set]
    for item in filtered_metadata:
        item['file_info'] = file_scanner.get_file_info(item['file_path'])
    db_manager.add_file_metadata_bulk(filtered_metadata)
    logger.info(f"Database operations took: {time.time() - db_start:.2f} seconds")


    # Grouping
    group_start = time.time()
    file_grouper = FileGrouper(db_manager)
    groups = file_grouper.group_files(time_window_minutes=10)
    logger.info(f"File grouping took: {time.time() - group_start:.2f} seconds")

    # UI setup
    ui_start = time.time()
    app = QApplication(sys.argv)
    controller = MainController(db_manager)
    main_window = controller.show_main_window()
    time_synchronizer = TimeSynchronizer(db_manager, main_window)
    main_window.set_time_synchronizer(time_synchronizer)
    main_window.sync_button.clicked.connect(controller.start_synchronization)
    main_window.threshold_slider.valueChanged.connect(lambda value: controller.set_similarity_threshold(value / 100.0))
    logger.info(f"UI setup took: {time.time() - ui_start:.2f} seconds")

    logger.info(f"Total setup time: {time.time() - total_start:.2f} seconds")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()