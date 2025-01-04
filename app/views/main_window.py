from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget, QTabWidget, QSlider, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt
from app.views.database_view import DatabaseView
from app.views.groups_view import GroupsView
from app.views.debug_view import DebugView
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

        # Initialize views
        self.database_view = DatabaseView(self.db_manager)
        self.groups_view = GroupsView(self.db_manager)
        print("Creating debug view...")
        try:
            self.debug_view = DebugView(self.db_manager)
            print("Debug view created successfully")
        except Exception as e:
            print(f"Error creating debug view: {str(e)}")
            import traceback
            traceback.print_exc()

        # Add tabs
        self.tab_widget.addTab(self.database_view, "Database")
        self.tab_widget.addTab(self.groups_view, "Groups")
        self.tab_widget.addTab(self.debug_view, "Debug")

        # Similarity threshold controls
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Similarity Threshold:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(50)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        threshold_layout.addWidget(self.threshold_slider)
        self.threshold_value_label = QLabel("0.50")
        threshold_layout.addWidget(self.threshold_value_label)
        self.layout.addLayout(threshold_layout)

        # Start synchronization button
        self.sync_button = QPushButton("Start Synchronization")
        self.layout.addWidget(self.sync_button)

        # Connect signals
        self.threshold_slider.valueChanged.connect(self.update_threshold_value)
        self.debug_view.clear_db_signal.connect(self.clear_database)

        # Initialize time synchronizer
        self.time_synchronizer = TimeSynchronizer(self.db_manager, self)
        self.time_synchronizer.database_updated.connect(self.refresh_views)

    def refresh_views(self):
        self.database_view.refresh_view()
        self.groups_view.update_groups_table()

    def update_threshold_value(self):
        value = self.threshold_slider.value() / 100.0
        self.threshold_value_label.setText(f"{value:.2f}")
        if hasattr(self, 'time_synchronizer'):
            self.time_synchronizer.set_similarity_threshold(value)

    def set_time_synchronizer(self, time_synchronizer):
        self.time_synchronizer = time_synchronizer

    def clear_database(self):
        self.db_manager.create_tables()  # This will drop and recreate tables
        self.refresh_views()