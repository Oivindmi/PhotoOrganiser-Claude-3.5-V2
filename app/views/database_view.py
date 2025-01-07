from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QLabel, QHeaderView, QAbstractItemView
from PyQt5.QtCore import Qt
import json
import os
import logging


class DatabaseView(QWidget):
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.init_ui()

    def load_data(self):
        self.db_manager.Session.close_all()
        session = self.db_manager.Session()
        try:
            data = session.query(self.db_manager.FileMetadata).all()
            total_entries = len(data)
            displayed_entries = min(5000, total_entries)

            self.info_label.setText(f"Displaying {displayed_entries} out of {total_entries} entries")

            # Clear existing table data
            self.table.setRowCount(0)

            # Determine all unique time/date metadata fields
            time_date_fields = set()
            for item in data[:displayed_entries]:
                extra_metadata = self._ensure_dict(item.extra_metadata)
                time_date_fields.update(
                    key for key in extra_metadata.keys() if 'Date' in key or 'Time' in key)

            # Sort the time/date fields
            time_date_fields = sorted(time_date_fields)

            # Set up the table
            columns = ['File Name', 'File Type', 'Group ID', 'Group Key', 'Camera Model', 'File Size (MB)',
                       'Updated Time', 'Original Time Field'] + list(time_date_fields) + ['Full Path']
            self.table.setColumnCount(len(columns))
            self.table.setHorizontalHeaderLabels(columns)


            # Enable column resizing and sorting
            self.table.horizontalHeader().setSectionsMovable(True)
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
            self.table.setSortingEnabled(True)

            # Freeze the first column
            self.table.setColumnCount(len(columns) + 1)  # Add an extra column
            self.table.setHorizontalHeaderItem(len(columns), QTableWidgetItem(""))  # Empty header for the extra column
            self.table.horizontalHeader().setSectionResizeMode(len(columns), QHeaderView.Fixed)
            self.table.horizontalHeader().resizeSection(len(columns), 0)  # Hide the last column
            self.table.setColumnWidth(0, 200)  # Set width for the frozen column


            # Populate the table
            self.table.setRowCount(displayed_entries)
            for row, item in enumerate(data[:displayed_entries]):
                file_name = os.path.basename(item.file_path)
                file_type = os.path.splitext(file_name)[1][1:].upper()
                parent_folder = os.path.basename(os.path.dirname(item.file_path))

                self.table.setItem(row, 0, QTableWidgetItem(file_name))
                self.table.setItem(row, 1, QTableWidgetItem(file_type))
                self.table.setItem(row, 2, QTableWidgetItem(str(item.group_id)))
                self.table.setItem(row, 3, QTableWidgetItem(item.group_key or ""))
                self.table.setItem(row, 4, QTableWidgetItem(item.camera_model or "Unknown"))
                self.table.setItem(row, 5, QTableWidgetItem(f"{item.file_size:.2f}"))

                # Updated Time column
                updated_time = str(item.correct_time) if item.correct_time else ""
                self.table.setItem(row, 6, QTableWidgetItem(updated_time))
                self.logger.debug(
                    f"Setting Updated Time for {file_name}: '{updated_time}', raw value: {item.correct_time}")

                # Original Time Field column
                original_time_field = item.original_time_field or ""
                self.table.setItem(row, 7, QTableWidgetItem(original_time_field))
                self.logger.info(f"Setting Original Time Field for {file_name}: '{original_time_field}'")

                # Add time/date metadata
                extra_metadata = self._ensure_dict(item.extra_metadata)
                for col, field in enumerate(time_date_fields, start=8):
                    value = extra_metadata.get(field, "")
                    self.table.setItem(row, col, QTableWidgetItem(str(value)))

                # Add full path as the last column
                self.table.setItem(row, len(columns) - 1, QTableWidgetItem(item.file_path))

                # Log all attributes of the item for debugging
                self.logger.debug(f"All attributes for {file_name}:")
                for attr, value in vars(item).items():
                    self.logger.debug(f"  {attr}: {value}")

            # Set the first column to freeze
            self.table.setColumnWidth(0, 200)
            self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
            self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
            self.table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)

            # Hide the grid
            self.table.setShowGrid(False)

            # Set alternate row colors
            self.table.setAlternatingRowColors(True)

            # Resize columns to content
            self.table.resizeColumnsToContents()

            # Set a minimum width for columns
            min_column_width = 100
            for i in range(1, self.table.columnCount() - 1):  # Skip the first (frozen) and last (hidden) columns
                if self.table.columnWidth(i) < min_column_width:
                    self.table.setColumnWidth(i, min_column_width)

        finally:
            session.close()

    def _ensure_dict(self, data):
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse JSON: {data}")
                return {}
        elif isinstance(data, dict):
            return data
        else:
            self.logger.error(f"Unexpected data type for extra_metadata: {type(data)}")
            return {}

    def init_ui(self):
        layout = QVBoxLayout()
        self.info_label = QLabel()
        layout.addWidget(self.info_label)
        self.table = QTableWidget()
        layout.addWidget(self.table)
        self.setLayout(layout)

        self.load_data()

    def refresh_view(self):
        self.logger.info("Refreshing DatabaseView")
        self.load_data()

    def resizeEvent(self, event):
        # Resize the table to fit the window
        self.table.setGeometry(0, self.info_label.height(), self.width(), self.height() - self.info_label.height())
        super().resizeEvent(event)