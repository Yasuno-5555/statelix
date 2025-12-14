import sys
from PySide6.QtWidgets import QApplication
from statelix_py.gui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion') # Modern look
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
