import logging
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QMessageBox, QTableWidget, QTableWidgetItem, QAbstractItemView)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import json

class SynchronizationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        self.choice = None
        self.left_field = None
        self.right_field = None

    def initUI(self):
        layout = QVBoxLayout()

        # Image displays
        image_layout = QHBoxLayout()
        self.left_image = QLabel()
        self.right_image = QLabel()
        image_layout.addWidget(self.left_image)
        image_layout.addWidget(self.right_image)
        layout.addLayout(image_layout)

        # File names
        name_layout = QHBoxLayout()
        self.left_name = QLabel()
        self.right_name = QLabel()
        name_layout.addWidget(self.left_name)
        name_layout.addWidget(self.right_name)
        layout.addLayout(name_layout)

        # Group info
        self.group_info = QLabel()
        layout.addWidget(self.group_info)

        # Metadata tables
        tables_layout = QHBoxLayout()
        self.left_table = QTableWidget()
        self.right_table = QTableWidget()
        self.left_table.setColumnCount(2)
        self.right_table.setColumnCount(2)
        self.left_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.right_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.left_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.right_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.left_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.right_table.setSelectionMode(QAbstractItemView.SingleSelection)
        tables_layout.addWidget(self.left_table)
        tables_layout.addWidget(self.right_table)
        layout.addLayout(tables_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.use_left_button = QPushButton("Use Left Time")
        self.use_right_button = QPushButton("Use Right Time")
        self.reject_button = QPushButton("Reject Match")
        button_layout.addWidget(self.use_left_button)
        button_layout.addWidget(self.use_right_button)
        button_layout.addWidget(self.reject_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.use_left_button.clicked.connect(self.use_left)
        self.use_right_button.clicked.connect(self.use_right)
        self.reject_button.clicked.connect(self.reject)

    def set_images(self, left_path, right_path):
        self.left_image.setPixmap(QPixmap(left_path).scaled(400, 300, Qt.KeepAspectRatio))
        self.right_image.setPixmap(QPixmap(right_path).scaled(400, 300, Qt.KeepAspectRatio))

    def set_file_names(self, left_name, right_name):
        self.left_name.setText(left_name)
        self.right_name.setText(right_name)

    def set_group_info(self, left_group, right_group):
        self.group_info.setText(f"Comparing {left_group} with {right_group}")

    def set_metadata(self, left_metadata, right_metadata, left_evaluated=False, right_evaluated=False,
                    left_field=None, right_field=None, left_correct_time=None, right_correct_time=None):
        logger = logging.getLogger(__name__)
        try:
            logger.info("Starting set_metadata")
            logger.info(f"Setting metadata - left_evaluated: {left_evaluated}, right_evaluated: {right_evaluated}")
            logger.info(f"Left correct time: {left_correct_time}")
            logger.info(f"Right correct time: {right_correct_time}")

            left_metadata = self._ensure_dict(left_metadata)
            right_metadata = self._ensure_dict(right_metadata)

            self.left_table.setRowCount(0)
            self.right_table.setRowCount(0)

            if left_evaluated:
                logger.info("Processing left evaluated side")
                row_position = self.left_table.rowCount()
                self.left_table.insertRow(row_position)
                self.left_table.setItem(row_position, 0, QTableWidgetItem("Updated Time"))
                time_str = str(left_correct_time) if left_correct_time is not None else "Not set"
                logger.info(f"Setting left time string: {time_str}")
                self.left_table.setItem(row_position, 1, QTableWidgetItem(time_str))
            else:
                logger.info("Processing left unevaluated side")
                self._populate_table(self.left_table, left_metadata, left_field)

            if right_evaluated:
                logger.info("Processing right evaluated side")
                row_position = self.right_table.rowCount()
                self.right_table.insertRow(row_position)
                self.right_table.setItem(row_position, 0, QTableWidgetItem("Updated Time"))
                time_str = str(right_correct_time) if right_correct_time is not None else "Not set"
                logger.info(f"Setting right time string: {time_str}")
                self.right_table.setItem(row_position, 1, QTableWidgetItem(time_str))
            else:
                logger.info("Processing right unevaluated side")
                self._populate_table(self.right_table, right_metadata, right_field)

            logger.info("Completed set_metadata")
        except Exception as e:
            logger.error(f"Error in set_metadata: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def _populate_table(self, table, metadata, selected_field=None):
        for key, value in metadata.items():
            if 'date' in key.lower() or 'time' in key.lower():
                row_position = table.rowCount()
                table.insertRow(row_position)
                field_item = QTableWidgetItem(key)
                value_item = QTableWidgetItem(str(value))

                if key == selected_field:
                    field_item.setBackground(Qt.yellow)
                    value_item.setBackground(Qt.yellow)

                table.setItem(row_position, 0, field_item)
                table.setItem(row_position, 1, value_item)

        table.resizeColumnsToContents()

    def _ensure_dict(self, metadata):
        if isinstance(metadata, str):
            try:
                return json.loads(metadata)
            except json.JSONDecodeError:
                return {}
        return metadata if isinstance(metadata, dict) else {}

    def use_left(self):
        if self.validate_selection():
            self.choice = 'left'
            self.accept()

    def use_right(self):
        if self.validate_selection():
            self.choice = 'right'
            self.accept()

    def reject(self):
        self.choice = 'reject'
        self.accept()

    def validate_selection(self):
        left_selected = self.left_table.selectedItems()
        right_selected = self.right_table.selectedItems()

        if not left_selected and not right_selected:
            QMessageBox.warning(self, "Incomplete Selection",
                                "Please select a date/time field for at least one image before confirming.")
            return False
        return True

    def get_user_choice(self):
        self.exec_()
        if self.choice in ['left', 'right']:
            left_selected = self.left_table.selectedItems()
            right_selected = self.right_table.selectedItems()

            self.left_field = left_selected[0].text() if left_selected else None
            self.right_field = right_selected[0].text() if right_selected else None

        return self.choice, self.left_field, self.right_field
