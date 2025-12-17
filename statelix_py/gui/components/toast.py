from PySide6.QtWidgets import QLabel, QWidget, QVBoxLayout, QGraphicsOpacityEffect
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint

class Toast(QLabel):
    def __init__(self, parent, text, duration=3000):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QLabel {
                background-color: #333333;
                color: white;
                padding: 12px 24px;
                border-radius: 4px;
                font-size: 14px;
                border: 1px solid #555;
            }
        """)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.adjustSize()
        
        # Effect for fading
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        
        # Animation
        self.anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.anim.setDuration(500)
        self.anim.setStartValue(0.0)
        self.anim.setEndValue(1.0)
        self.anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        
        # Timer to hide
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.fade_out)
        self.duration = duration

    def show_toast(self):
        self.show()
        # Center bottom position or user defined?
        # Let's rely on parent to move us or center us.
        if self.parent():
            parent_rect = self.parent().rect()
            x = (parent_rect.width() - self.width()) // 2
            y = parent_rect.height() - self.height() - 50
            self.move(x, y)
            
        self.raise_()
        self.anim.setDirection(QPropertyAnimation.Direction.Forward)
        self.anim.start()
        self.timer.start(self.duration)

    def fade_out(self):
        self.anim.setDirection(QPropertyAnimation.Direction.Backward)
        self.anim.finished.connect(self.close)
        self.anim.start()

# Helper for MainWindow to manage multiple toasts if needed, 
# for now single instance is simpler.
