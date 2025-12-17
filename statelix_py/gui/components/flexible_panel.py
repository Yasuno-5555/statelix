from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSplitter, QDialog, QFrame
from PySide6.QtCore import Qt, Signal, Slot, QEvent

class FlexiblePanel(QWidget):
    """
    A wrapper widget that acts like a Dock:
    - Title Bar with Drag/Popout controls (simulated)
    - Content Area
    - Supports 'Popping Out' to a separate window
    """
    def __init__(self, title, content_widget, parent_splitter=None):
        super().__init__()
        self.title_text = title
        self.content = content_widget
        self.parent_splitter = parent_splitter
        self.is_popped_out = False
        self.floating_window = None
        
        # Layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Title Bar
        self.title_bar = QFrame()
        self.title_bar.setObjectName("PanelTitleBar")
        # Style will be applied via main stylesheet QFrame#PanelTitleBar
        self.title_layout = QHBoxLayout(self.title_bar)
        self.title_layout.setContentsMargins(8, 4, 8, 4)
        
        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet("font-weight: bold;")
        
        self.btn_detach = QPushButton("❐") # Emoji for PopOut
        self.btn_detach.setFixedWidth(24)
        self.btn_detach.setToolTip("Detach Panel")
        self.btn_detach.clicked.connect(self.toggle_popout)
        
        self.title_layout.addWidget(self.lbl_title)
        self.title_layout.addStretch()
        self.title_layout.addWidget(self.btn_detach)
        
        # Content Container
        self.content_container = QWidget()
        self.content_layout = QVBoxLayout(self.content_container)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.addWidget(self.content)
        
        self.layout.addWidget(self.title_bar)
        self.layout.addWidget(self.content_container)
        
        # Apply border style (can be done in styles.py)
        # self.setStyleSheet("border: 1px solid #444;")

    def toggle_popout(self):
        if self.is_popped_out:
            self.dock_back()
        else:
            self.pop_out()
            
    def pop_out(self):
        if self.is_popped_out: return
        
        # Create Floating Window
        self.floating_window = QDialog(self)
        self.floating_window.setWindowTitle(self.title_text)
        self.floating_window.setLayout(QVBoxLayout())
        self.floating_window.layout().setContentsMargins(0,0,0,0)
        
        # Move content
        self.content.setParent(self.floating_window)
        self.floating_window.layout().addWidget(self.content)
        
        self.floating_window.resize(600, 400)
        self.floating_window.show()
        self.floating_window.finished.connect(self.dock_back) # If closed, dock back
        
        # Hide internal placeholder
        self.content_container.hide()
        self.btn_detach.setText("sw_down") # Change icon to 'Dock Back'
        self.is_popped_out = True
    
    def dock_back(self, result=None):
        if not self.is_popped_out: return
        
        # Move content back
        self.content.setParent(self.content_container)
        self.content_layout.addWidget(self.content)
        self.content.show()
        self.content_container.show()
        
        if self.floating_window:
            self.floating_window.close()
            self.floating_window = None
            
        self.btn_detach.setText("❐")
        self.is_popped_out = False
