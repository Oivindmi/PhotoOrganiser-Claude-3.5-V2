from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem
from app.utils.file_grouper import FileGrouper

class GroupsView(QWidget):
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self.file_grouper = FileGrouper(db_manager)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.groups_table = QTableWidget()
        layout.addWidget(self.groups_table)
        self.setLayout(layout)
        self.update_groups_table()

    def update_groups_table(self):
        group_info = self.file_grouper.get_group_info()
        self.groups_table.setColumnCount(4)
        self.groups_table.setHorizontalHeaderLabels(["Group ID", "File Count", "Sample File", "Camera Model"])
        self.groups_table.setRowCount(len(group_info))

        for row, (group_id, info) in enumerate(group_info.items()):
            self.groups_table.setItem(row, 0, QTableWidgetItem(str(group_id)))
            self.groups_table.setItem(row, 1, QTableWidgetItem(str(info['count'])))
            self.groups_table.setItem(row, 2, QTableWidgetItem(info['sample_file'].file_path))
            self.groups_table.setItem(row, 3, QTableWidgetItem(info['sample_file'].camera_model or "Unknown"))

        self.groups_table.resizeColumnsToContents()