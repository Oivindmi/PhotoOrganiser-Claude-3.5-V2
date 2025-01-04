from PyQt5.QtWidgets import QMainWindow
from app.views.main_window import MainWindow
from app.utils.time_synchronizer import TimeSynchronizer

class MainController:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.main_window = None
        self.time_synchronizer = None

    def show_main_window(self):
        self.main_window = MainWindow(self.db_manager)
        self.main_window.show()
        self.time_synchronizer = self.main_window.time_synchronizer
        return self.main_window

    def start_synchronization(self):
        if self.time_synchronizer:
            self.time_synchronizer.start_synchronization()

    def set_similarity_threshold(self, threshold):
        if self.time_synchronizer:
            self.time_synchronizer.set_similarity_threshold(threshold)