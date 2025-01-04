from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget, QTabWidget, QSlider, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt
from app.views.database_view import DatabaseView
from app.views.groups_view import GroupsView
from app.utils.time_synchronizer import TimeSynchronizer

class MainWindow(QMainWindow):
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self.setWindowTitle("Photo Organizer")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)

        self.database_view = DatabaseView(self.db_manager)
        self.groups_view = GroupsView(self.db_manager)

        self.tab_widget.addTab(self.database_view, "Database")
        self.tab_widget.addTab(self.groups_view, "Groups")

        # Add similarity threshold slider
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Similarity Threshold:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(50)  # Default to 0.5
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        threshold_layout.addWidget(self.threshold_slider)
        self.threshold_value_label = QLabel("0.50")
        threshold_layout.addWidget(self.threshold_value_label)
        self.layout.addLayout(threshold_layout)

        self.sync_button = QPushButton("Start Synchronization")
        self.layout.addWidget(self.sync_button)

        # Connect slider to update function
        self.threshold_slider.valueChanged.connect(self.update_threshold_value)

        self.time_synchronizer = TimeSynchronizer(self.db_manager, self)
        self.time_synchronizer.database_updated.connect(self.refresh_database_view)

    def refresh_database_view(self):
        self.database_view.refresh_view()
    def update_threshold_value(self):
        value = self.threshold_slider.value() / 100.0
        self.threshold_value_label.setText(f"{value:.2f}")
        # Update the TimeSynchronizer's threshold
        if hasattr(self, 'time_synchronizer'):
            self.time_synchronizer.set_similarity_threshold(value)

    def set_time_synchronizer(self, time_synchronizer):
        self.time_synchronizer = time_synchronizer