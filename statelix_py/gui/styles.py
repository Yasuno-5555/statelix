
import json
from pathlib import Path

class StatelixTheme:
    """
    Statelix Theme System with Dark/Light mode support
    """
    # Dark Palette
    DARK_BG_MAIN = "#1e1e1e"
    DARK_BG_PANEL = "#252526"
    DARK_BG_INPUT = "#333337"
    DARK_TEXT = "#d4d4d4"
    DARK_ACCENT = "#007acc"
    DARK_ACCENT_HOVER = "#0098ff"
    DARK_BORDER = "#3e3e42"
    DARK_HEADER = "#2d2d30"
    
    # Light Palette
    LIGHT_BG_MAIN = "#ffffff"
    LIGHT_BG_PANEL = "#f5f5f5"
    LIGHT_BG_INPUT = "#ffffff"
    LIGHT_TEXT = "#333333"
    LIGHT_ACCENT = "#0078d4"
    LIGHT_ACCENT_HOVER = "#106ebe"
    LIGHT_BORDER = "#e0e0e0"
    LIGHT_HEADER = "#f0f0f0"
    
    # Current theme state
    _current_theme = "dark"
    _observers = []

    # Dynamic Colors (Access as StatelixTheme.COLOR_NAME)
    COLOR_BG_MAIN = DARK_BG_MAIN
    COLOR_BG_PANEL = DARK_BG_PANEL
    COLOR_BG_INPUT = DARK_BG_INPUT
    COLOR_TEXT_MAIN = DARK_TEXT
    COLOR_ACCENT = DARK_ACCENT
    COLOR_BORDER = DARK_BORDER

    @classmethod
    def get_stylesheet(cls, theme: str = None) -> str:
        """Get stylesheet for specified theme (dark/light)."""
        if theme is None:
            theme = cls._current_theme
        return cls._generate_stylesheet(theme)
    
    @classmethod
    def set_theme(cls, theme: str):
        """Set current theme and save to settings."""
        cls._current_theme = theme
        
        # Update dynamic colors
        if theme == "dark":
            cls.COLOR_BG_MAIN = cls.DARK_BG_MAIN
            cls.COLOR_BG_PANEL = cls.DARK_BG_PANEL
            cls.COLOR_BG_INPUT = cls.DARK_BG_INPUT
            cls.COLOR_TEXT_MAIN = cls.DARK_TEXT
            cls.COLOR_ACCENT = cls.DARK_ACCENT
            cls.COLOR_BORDER = cls.DARK_BORDER
        else:
            cls.COLOR_BG_MAIN = cls.LIGHT_BG_MAIN
            cls.COLOR_BG_PANEL = cls.LIGHT_BG_PANEL
            cls.COLOR_BG_INPUT = cls.LIGHT_BG_INPUT
            cls.COLOR_TEXT_MAIN = cls.LIGHT_TEXT
            cls.COLOR_ACCENT = cls.LIGHT_ACCENT
            cls.COLOR_BORDER = cls.LIGHT_BORDER

        cls._save_theme_setting(theme)
        for callback in cls._observers:
            try:
                callback(theme)
            except Exception:
                pass
    
    @classmethod
    def get_theme(cls) -> str:
        """Get current theme name."""
        return cls._current_theme
    
    @classmethod
    def toggle_theme(cls) -> str:
        """Toggle between dark and light theme. Returns new theme name."""
        new_theme = "light" if cls._current_theme == "dark" else "dark"
        cls.set_theme(new_theme)
        return new_theme
    
    @classmethod
    def add_observer(cls, callback):
        """Add callback for theme changes."""
        cls._observers.append(callback)
    
    @classmethod
    def _save_theme_setting(cls, theme: str):
        """Save theme to settings file."""
        config_dir = Path.home() / ".statelix"
        config_dir.mkdir(exist_ok=True)
        config_file = config_dir / "settings.json"
        
        settings = {}
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    settings = json.load(f)
            except Exception:
                pass
        
        settings["theme"] = theme
        try:
            with open(config_file, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception:
            pass
    
    @classmethod
    def _load_theme_setting(cls):
        """Load theme from settings file."""
        config_file = Path.home() / ".statelix" / "settings.json"
        theme = "dark"
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    settings = json.load(f)
                    theme = settings.get("theme", "dark")
            except Exception:
                pass
        cls.set_theme(theme)
    
    @classmethod
    def _generate_stylesheet(cls, theme: str) -> str:
        """Generate QSS stylesheet for theme."""
        if theme == "light":
            BG_MAIN = cls.LIGHT_BG_MAIN
            BG_PANEL = cls.LIGHT_BG_PANEL
            BG_INPUT = cls.LIGHT_BG_INPUT
            TEXT = cls.LIGHT_TEXT
            ACCENT = cls.LIGHT_ACCENT
            ACCENT_HOVER = cls.LIGHT_ACCENT_HOVER
            BORDER = cls.LIGHT_BORDER
            HEADER = cls.LIGHT_HEADER
            TABLE_HEADER_BG = "#217346"
            TABLE_HEADER_TEXT = "#ffffff"
        else:
            BG_MAIN = cls.DARK_BG_MAIN
            BG_PANEL = cls.DARK_BG_PANEL
            BG_INPUT = cls.DARK_BG_INPUT
            TEXT = cls.DARK_TEXT
            ACCENT = cls.DARK_ACCENT
            ACCENT_HOVER = cls.DARK_ACCENT_HOVER
            BORDER = cls.DARK_BORDER
            HEADER = cls.DARK_HEADER
            TABLE_HEADER_BG = "#217346"
            TABLE_HEADER_TEXT = "#ffffff"
        
        return f"""
    QMainWindow, QDialog, QDockWidget {{
        background-color: {BG_MAIN};
        color: {TEXT};
    }}
    QWidget {{
        color: {TEXT};
        font-family: 'Segoe UI', 'Helvetica Neue', 'Arial', sans-serif;
        font-size: 10pt;
    }}
    .DataPanel, .ModelPanel, .ResultPanel, .PlotPanel, .InquiryPanel, .VariableInspector {{
         background-color: {BG_MAIN};
    }}
    
    /* --- Splitter & Layouts --- */
    QSplitter::handle {{
        background-color: {BORDER};
        width: 2px;
    }}
    
    /* --- Group Box --- */
    QGroupBox {{
        background-color: {BG_PANEL};
        border: 1px solid {BORDER};
        border-radius: 4px;
        margin-top: 1.5em;
        padding-top: 10px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 5px;
        color: {ACCENT};
        font-weight: bold;
    }}

    /* --- Tabs --- */
    QTabWidget::pane {{
        border: 1px solid {BORDER};
        background-color: {BG_PANEL};
        border-radius: 4px;
    }}
    QTabBar::tab {{
        background: {BG_MAIN};
        border: 1px solid {BORDER};
        padding: 8px 20px;
        margin-right: 2px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        color: {TEXT};
    }}
    QTabBar::tab:selected {{
        background: {BG_PANEL};
        border-bottom-color: {BG_PANEL};
        color: {ACCENT};
        font-weight: bold;
    }}
    QTabBar::tab:hover {{
        background: {HEADER};
    }}

    /* --- Inputs --- */
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background-color: {BG_INPUT};
        border: 1px solid {BORDER};
        border-radius: 3px;
        padding: 5px;
        color: {TEXT};
        selection-background-color: {ACCENT};
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{
        border: 1px solid {ACCENT};
    }}
    
    /* --- QListWidget --- */
    QListWidget {{
        background-color: {BG_INPUT};
        border: 1px solid {BORDER};
        border-radius: 3px;
    }}
    QListWidget::item {{
        padding: 4px;
    }}
    QListWidget::item:selected {{
        background-color: {ACCENT};
        color: white;
        border-radius: 2px;
    }}

    /* --- Buttons --- */
    QPushButton {{
        background-color: {ACCENT};
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: bold;
    }}
    QPushButton:hover {{
        background-color: {ACCENT_HOVER};
    }}
    QPushButton:pressed {{
        background-color: {HEADER};
    }}
    QPushButton:disabled {{
        background-color: {BORDER};
        color: #888888;
    }}

    /* --- Labels --- */
    QLabel {{
        color: {TEXT};
    }}
    
    /* --- ScrollBars --- */
    QScrollBar:vertical {{
        background: {BG_MAIN};
        width: 10px;
        margin: 0px;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER};
        min-height: 20px;
        border-radius: 5px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    /* --- Dock Widgets --- */
    QDockWidget > QWidget {{
         background-color: {BG_MAIN};
         border: 1px solid {BORDER};
    }}
    
    /* --- Table View (Excel-like) --- */
    QTableView {{
        background-color: {BG_MAIN};
        gridline-color: {BORDER};
        border: 1px solid {BORDER};
        selection-background-color: #0078d4;
        selection-color: #ffffff;
    }}
    QTableView::item {{
        padding: 4px;
        border-right: 1px solid {BORDER};
        border-bottom: 1px solid {BORDER};
    }}
    QTableView::item:selected {{
        background-color: #0078d4;
        color: #ffffff;
    }}
    QHeaderView::section {{
        background-color: {TABLE_HEADER_BG};
        color: {TABLE_HEADER_TEXT};
        padding: 6px 8px;
        border: 1px solid #1a5f38;
        font-weight: bold;
    }}
    QHeaderView::section:hover {{
        background-color: #28a05b;
    }}
    
    /* --- Menu Bar --- */
    QMenuBar {{
        background-color: {BG_PANEL};
        color: {TEXT};
    }}
    QMenuBar::item:selected {{
        background-color: {ACCENT};
    }}
    QMenu {{
        background-color: {BG_PANEL};
        border: 1px solid {BORDER};
    }}
    QMenu::item {{
        padding: 6px 30px;
    }}
    QMenu::item:selected {{
        background-color: {ACCENT};
    }}
    
    /* --- Tool Bar --- */
    QToolBar {{
        background-color: {BG_PANEL};
        border: none;
        spacing: 5px;
        padding: 3px;
    }}
    QToolButton {{
        background-color: transparent;
        border: none;
        padding: 5px;
        border-radius: 3px;
    }}
    QToolButton:hover {{
        background-color: {ACCENT};
    }}
    
    /* --- Progress Bar --- */
    QProgressBar {{
        border: 1px solid {BORDER};
        border-radius: 3px;
        text-align: center;
        background-color: {BG_INPUT};
    }}
    QProgressBar::chunk {{
        background-color: {ACCENT};
        border-radius: 2px;
    }}
    
    /* --- Status Bar --- */
    QStatusBar {{
        background-color: {BG_PANEL};
        border-top: 1px solid {BORDER};
    }}
    """

# After class definition, set legacy attribute
StatelixTheme.DARK_STYLESHEET = StatelixTheme._generate_stylesheet("dark")

# Load saved theme on import
StatelixTheme._load_theme_setting()


