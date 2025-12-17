
class StatelixTheme:
    """
    Statelix Modern Dark Theme
    """
    # Palette
    COLOR_BG_MAIN = "#1e1e1e"
    COLOR_BG_PANEL = "#252526"
    COLOR_BG_INPUT = "#333337"
    COLOR_TEXT = "#d4d4d4"
    COLOR_ACCENT = "#007acc"
    COLOR_ACCENT_HOVER = "#0098ff"
    COLOR_BORDER = "#3e3e42"
    COLOR_HEADER = "#2d2d30"

    DARK_STYLESHEET = f"""
    QMainWindow, QDialog, QDockWidget {{
        background-color: {COLOR_BG_MAIN};
        color: {COLOR_TEXT};
    }}
    QWidget {{
        color: {COLOR_TEXT};
        font-family: 'Segoe UI', 'Helvetica Neue', 'Arial', sans-serif;
        font-size: 10pt;
    }}
    /* Explicitly set background for top-level panels if needed */
    .DataPanel, .ModelPanel, .ResultPanel, .PlotPanel, .InquiryPanel, .VariableInspector {{
         background-color: {COLOR_BG_MAIN};
    }}
    
    /* --- Splitter & Layouts --- */
    QSplitter::handle {{
        background-color: {COLOR_BORDER};
        width: 2px;
    }}
    
    /* --- Group Box --- */
    QGroupBox {{
        background-color: {COLOR_BG_PANEL};
        border: 1px solid {COLOR_BORDER};
        border-radius: 4px;
        margin-top: 1.5em; /* leave space for title */
        padding-top: 10px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 5px;
        color: {COLOR_ACCENT};
        font-weight: bold;
    }}

    /* --- Tabs --- */
    QTabWidget::pane {{
        border: 1px solid {COLOR_BORDER};
        background-color: {COLOR_BG_PANEL};
        border-radius: 4px;
    }}
    QTabBar::tab {{
        background: {COLOR_BG_MAIN};
        border: 1px solid {COLOR_BORDER};
        padding: 8px 20px;
        margin-right: 2px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        color: {COLOR_TEXT};
    }}
    QTabBar::tab:selected {{
        background: {COLOR_BG_PANEL};
        border-bottom-color: {COLOR_BG_PANEL};
        color: {COLOR_ACCENT};
        font-weight: bold;
    }}
    QTabBar::tab:hover {{
        background: {COLOR_HEADER};
    }}

    /* --- Inputs --- */
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background-color: {COLOR_BG_INPUT};
        border: 1px solid {COLOR_BORDER};
        border-radius: 3px;
        padding: 5px;
        color: #ffffff;
        selection-background-color: {COLOR_ACCENT};
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{
        border: 1px solid {COLOR_ACCENT};
    }}
    
    /* --- QListWidget --- */
    QListWidget {{
        background-color: {COLOR_BG_INPUT};
        border: 1px solid {COLOR_BORDER};
        border-radius: 3px;
    }}
    QListWidget::item {{
        padding: 4px;
    }}
    QListWidget::item:selected {{
        background-color: {COLOR_ACCENT};
        color: white;
        border-radius: 2px;
    }}

    /* --- Buttons --- */
    QPushButton {{
        background-color: {COLOR_ACCENT};
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: bold;
    }}
    QPushButton:hover {{
        background-color: {COLOR_ACCENT_HOVER};
    }}
    QPushButton:pressed {{
        background-color: {COLOR_HEADER};
    }}
    QPushButton:disabled {{
        background-color: {COLOR_BORDER};
        color: #888888;
    }}

    /* --- Labels --- */
    QLabel {{
        color: {COLOR_TEXT};
    }}
    
    /* --- ScrollBars --- */
    QScrollBar:vertical {{
        background: {COLOR_BG_MAIN};
        width: 10px;
        margin: 0px 0px 0px 0px;
    }}
    QScrollBar::handle:vertical {{
        background: {COLOR_BORDER};
        min-height: 20px;
        border-radius: 5px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    /* --- Docks (Native Handling for Stability) --- */
    
    /* --- FlexiblePanel Styles --- */
    QFrame#PanelTitleBar {{
        background: {COLOR_HEADER};
        border-bottom: 1px solid {COLOR_BORDER};
    }}
    QFrame#PanelTitleBar QLabel {{
        color: {COLOR_TEXT};
        font-weight: bold;
    }}
    QFrame#PanelTitleBar QPushButton {{
        background: transparent;
        border: none;
        color: {COLOR_TEXT};
        font-weight: bold;
    }}
    QFrame#PanelTitleBar QPushButton:hover {{
        background: {COLOR_BG_PANEL};
        border-radius: 3px;
        color: #4db8ff;
    }}

    /* When docked, the dock widget content itself needs bg */
    QDockWidget > QWidget {{
         background-color: {COLOR_BG_MAIN};
         border: 1px solid {COLOR_BORDER};
    }}
    """
