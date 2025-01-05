from dateutil import parser
import os
import json
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QTextEdit, QPushButton,
                             QHBoxLayout, QComboBox, QLabel, QCheckBox, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
import logging


class QTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.widget = parent
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.append(msg)


class DebugView(QWidget):
    clear_db_signal = pyqtSignal()

    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self.init_ui()
        self.setup_logging()

    def init_ui(self):
        layout = QVBoxLayout()

        # Controls
        controls_layout = QHBoxLayout()

        # Log Level Selector
        self.log_level = QComboBox()
        self.log_level.addItems(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
        self.log_level.setCurrentText('INFO')
        self.log_level.currentTextChanged.connect(self.change_log_level)
        controls_layout.addWidget(QLabel('Log Level:'))
        controls_layout.addWidget(self.log_level)

        # Auto-scroll checkbox
        self.auto_scroll = QCheckBox('Auto-scroll')
        self.auto_scroll.setChecked(True)
        controls_layout.addWidget(self.auto_scroll)

        # Clear button
        clear_btn = QPushButton('Clear Log')
        clear_btn.clicked.connect(self.clear_log)
        controls_layout.addWidget(clear_btn)

        # Database controls
        db_btn = QPushButton('Clear Database')
        db_btn.clicked.connect(self.clear_database)
        controls_layout.addWidget(db_btn)

        # Test sync button
        test_sync_btn = QPushButton('Test Sync')
        test_sync_btn.clicked.connect(self.test_sync)
        controls_layout.addWidget(test_sync_btn)

        # Show sync summary button
        summary_btn = QPushButton('Show Sync Summary')
        summary_btn.clicked.connect(self.show_sync_summary)
        controls_layout.addWidget(summary_btn)

        # Clear cache button
        clear_cache_btn = QPushButton('Clear Cache')
        clear_cache_btn.clicked.connect(self.clear_image_cache)
        controls_layout.addWidget(clear_cache_btn)

        # Add stretch to push controls to the left
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Log display
        self.log_display = QTextEdit()
        self.log_display.setLineWrapMode(QTextEdit.NoWrap)
        layout.addWidget(self.log_display)

        self.setLayout(layout)

    def setup_logging(self):
        self.logTextBox = QTextEditLogger(self.log_display)
        self.logTextBox.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(self.logTextBox)
        self.change_log_level(self.log_level.currentText())

    def change_log_level(self, level):
        logging.getLogger().setLevel(getattr(logging, level))
        self.logTextBox.setLevel(getattr(logging, level))

    def clear_log(self):
        self.log_display.clear()

    def clear_database(self):
        confirm = QMessageBox.question(
            self, 'Confirm Clear',
            'Are you sure you want to clear the database?',
            QMessageBox.Yes | QMessageBox.No
        )

        if confirm == QMessageBox.Yes:
            self.clear_db_signal.emit()
            logging.info("Database cleared by user request")

    def test_sync(self):
        logging.info("Starting synchronization test...")

        session = self.db_manager.Session()
        try:
            # Get all metadata for inspection
            metadata_entries = session.query(self.db_manager.FileMetadata).all()

            # Log available time fields for each file
            for entry in metadata_entries:
                logging.info(f"\nFile: {os.path.basename(entry.file_path)}")
                logging.info(f"Camera Model: {entry.camera_model}")
                logging.info(f"Group ID: {entry.group_id}")
                logging.info(f"Group Key: {entry.group_key}")

                if isinstance(entry.extra_metadata, str):
                    metadata = json.loads(entry.extra_metadata)
                else:
                    metadata = entry.extra_metadata

                time_fields = {k: v for k, v in metadata.items() if 'Date' in k or 'Time' in k}
                logging.info("Available time fields:")
                for field, value in time_fields.items():
                    logging.info(f"  {field}: {value}")

        except Exception as e:
            logging.error(f"Error during sync test: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
        finally:
            session.close()

    def clear_image_cache(self):
        confirm = QMessageBox.question(
            self, 'Confirm Clear Cache',
            'Are you sure you want to clear the image comparison cache? This will require recalculating all image similarities.',
            QMessageBox.Yes | QMessageBox.No
        )

        if confirm == QMessageBox.Yes:
            from app.utils.image_comparison import ImageComparison
            ImageComparison.clear_cache()
            logging.info("Image comparison cache cleared by user request")
            QMessageBox.information(self, "Cache Cleared", "Image comparison cache has been cleared.")

    def show_sync_summary(self):
        logging.info("\n=== SYNCHRONIZATION SUMMARY ===")

        session = self.db_manager.Session()
        try:
            # Get all files grouped by group_id
            groups = {}
            entries = session.query(self.db_manager.FileMetadata).order_by(
                self.db_manager.FileMetadata.group_id,
                self.db_manager.FileMetadata.file_path
            ).all()

            for entry in entries:
                if entry.group_id not in groups:
                    groups[entry.group_id] = {
                        'files': [],
                        'camera_model': entry.camera_model,
                        'group_key': entry.group_key
                    }
                groups[entry.group_id]['files'].append(entry)

            # Print summary for each group
            for group_id, group_data in groups.items():
                logging.info(f"\nGroup {group_id} ({group_data['camera_model']})")
                logging.info(f"Group Key: {group_data['group_key']}")
                logging.info("Files:")

                for file in group_data['files']:
                    filename = os.path.basename(file.file_path)
                    metadata = file.extra_metadata
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)

                    original_time = metadata.get(file.original_time_field) if file.original_time_field else "Not set"
                    corrected_time = file.correct_time.strftime(
                        '%Y-%m-%d %H:%M:%S') if file.correct_time else "Not adjusted"

                    logging.info(f"\nFile: {filename}")
                    logging.info(f"  Original Time Field: {file.original_time_field}")
                    logging.info(f"  Original Time: {original_time}")
                    logging.info(f"  Corrected Time: {corrected_time}")

                    if file.correct_time:
                        try:
                            from dateutil import parser
                            orig_dt = parser.parse(original_time)
                            time_difference = file.correct_time - orig_dt
                            logging.info(f"  Time Adjustment: {time_difference}")
                        except Exception as e:
                            logging.info(f"  Time Adjustment: Could not calculate ({str(e)})")

        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
        finally:
            session.close()

        logging.info("\n=== END OF SUMMARY ===\n")

    def show_gpu_info(self):
        from app.utils.image_comparison import ImageComparison

        if ImageComparison.has_cuda():
            gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
            device_name = cv2.cuda.getDevice()
            info = f"GPU Acceleration: Enabled\n"
            info += f"CUDA Devices: {gpu_count}\n"
            info += f"Current Device: {device_name}\n"
        else:
            info = "GPU Acceleration: Not Available (using CPU)"

        logging.info(info)
        QMessageBox.information(self, "GPU Status", info)