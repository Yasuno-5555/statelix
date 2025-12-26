"""
Path Widget: Visualizing Model Is a Path

Interactive visualization of assumption paths, showing how estimates
change as assumptions are relaxed and where cliffs (breakdowns) occur.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QGroupBox, QComboBox, QToolTip, QFrame
)
from PySide6.QtCore import Qt, Signal, Slot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from statelix_py.core.assumption_path import AssumptionPath, PathPoint, CliffPoint


class AssumptionPathWidget(QWidget):
    """
    Visualizes the Assumption Path - Model Is a Path.
    
    Shows:
    - 2D/3D trajectory through assumption space
    - Estimate values along the path
    - Curvature heatmap highlighting unstable regions
    - Cliff markers where assumptions break down
    
    Signals:
        point_selected: Emitted when user clicks a path point
        cliff_selected: Emitted when user clicks a cliff marker
    """
    
    point_selected = Signal(dict)  # Emits state params of clicked point
    cliff_selected = Signal(int)   # Emits cliff index
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._path: Optional['AssumptionPath'] = None
        self._view_mode = '2d'
        self._show_curvature = True
        self._highlight_cliffs = True
        self._selected_dimensions = ('linearity', 'normality')
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(8)
        
        # Header
        header = QLabel("📈 Model Is a Path: Assumption Trajectory")
        header.setStyleSheet("""
            font-weight: bold; 
            font-size: 14px;
            color: #dcdcaa;
            padding: 5px;
        """)
        layout.addWidget(header)
        
        # Control Panel
        control_panel = self._create_control_panel()
        layout.addWidget(control_panel)
        
        # Matplotlib Figure
        self.figure = plt.figure(figsize=(7, 5))
        self.figure.patch.set_facecolor('#1e1e1e')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)
        
        # Info Panel
        info_panel = self._create_info_panel()
        layout.addWidget(info_panel)
        
        # Help Label
        help_label = QLabel("Click a point to inspect. Cliffs (🔴) indicate assumption breakdowns.")
        help_label.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        
        self.setLayout(layout)
        self.canvas.mpl_connect('button_press_event', self._on_click)
    
    def _create_control_panel(self) -> QWidget:
        """Create the control panel for visualization options."""
        panel = QGroupBox("Visualization Controls")
        panel.setStyleSheet("""
            QGroupBox { 
                color: #aaa; 
                border: 1px solid #3a3a3a; 
                border-radius: 4px; 
                margin-top: 8px; 
                padding-top: 8px;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 5px; 
            }
            QPushButton { 
                background: #2d2d30; 
                color: #ccc; 
                border: 1px solid #444; 
                border-radius: 3px; 
                padding: 4px 10px; 
            }
            QPushButton:hover { background: #3e3e42; }
            QPushButton:checked { background: #4ec9b0; color: #1e1e1e; }
            QComboBox { 
                background: #2d2d30; 
                color: #ccc; 
                border: 1px solid #444; 
                padding: 3px; 
            }
            QLabel { color: #aaa; }
        """)
        
        layout = QHBoxLayout()
        
        # View Mode
        self.view_toggle = QPushButton("3D View")
        self.view_toggle.setCheckable(True)
        self.view_toggle.clicked.connect(self._toggle_view_mode)
        layout.addWidget(self.view_toggle)
        
        # X-Axis Dimension
        layout.addWidget(QLabel("X:"))
        self.x_dim_combo = QComboBox()
        self.x_dim_combo.addItems([
            'linearity', 'independence', 'stationarity', 
            'normality', 'homoscedasticity', 'exogeneity'
        ])
        self.x_dim_combo.currentTextChanged.connect(self._on_dimension_change)
        layout.addWidget(self.x_dim_combo)
        
        # Y-Axis Dimension
        layout.addWidget(QLabel("Y:"))
        self.y_dim_combo = QComboBox()
        self.y_dim_combo.addItems([
            'linearity', 'independence', 'stationarity', 
            'normality', 'homoscedasticity', 'exogeneity'
        ])
        self.y_dim_combo.setCurrentIndex(3)  # normality
        self.y_dim_combo.currentTextChanged.connect(self._on_dimension_change)
        layout.addWidget(self.y_dim_combo)
        
        # Curvature Toggle
        self.curvature_toggle = QPushButton("Curvature")
        self.curvature_toggle.setCheckable(True)
        self.curvature_toggle.setChecked(True)
        self.curvature_toggle.clicked.connect(self._toggle_curvature)
        layout.addWidget(self.curvature_toggle)
        
        # Cliff Toggle
        self.cliff_toggle = QPushButton("Cliffs")
        self.cliff_toggle.setCheckable(True)
        self.cliff_toggle.setChecked(True)
        self.cliff_toggle.clicked.connect(self._toggle_cliffs)
        layout.addWidget(self.cliff_toggle)
        
        panel.setLayout(layout)
        return panel
    
    def _create_info_panel(self) -> QWidget:
        """Create the info panel showing path statistics."""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background: #252526;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 5px;
            }
            QLabel { color: #ccc; font-size: 11px; }
        """)
        
        layout = QHBoxLayout()
        layout.setSpacing(20)
        
        self.stability_label = QLabel("Stability: --")
        layout.addWidget(self.stability_label)
        
        self.path_length_label = QLabel("Path Length: --")
        layout.addWidget(self.path_length_label)
        
        self.cliff_count_label = QLabel("Cliffs: --")
        layout.addWidget(self.cliff_count_label)
        
        self.fragile_assumption_label = QLabel("Most Fragile: --")
        layout.addWidget(self.fragile_assumption_label)
        
        panel.setLayout(layout)
        return panel
    
    def set_path(self, path: 'AssumptionPath'):
        """Set the assumption path to visualize."""
        self._path = path
        self._update_info_panel()
        self._update_plot()
    
    def _update_info_panel(self):
        """Update the info panel with path statistics."""
        if self._path is None:
            return
        
        summary = self._path.summary()
        
        # Stability score with color coding
        stability = summary['stability_score']
        if stability > 0.8:
            color = '#4ec9b0'  # Green
            icon = '✓'
        elif stability > 0.5:
            color = '#dcdcaa'  # Yellow
            icon = '⚠'
        else:
            color = '#f44747'  # Red
            icon = '✗'
        
        self.stability_label.setText(f"Stability: {icon} {stability:.2f}")
        self.stability_label.setStyleSheet(f"color: {color};")
        
        self.path_length_label.setText(f"Path Length: {summary['path_length']:.3f}")
        self.cliff_count_label.setText(f"Cliffs: {summary['n_cliffs']}")
        
        fragile = summary.get('most_fragile_assumption')
        if fragile:
            self.fragile_assumption_label.setText(f"Most Fragile: {fragile}")
            self.fragile_assumption_label.setStyleSheet("color: #f44747;")
        else:
            self.fragile_assumption_label.setText("Most Fragile: None")
            self.fragile_assumption_label.setStyleSheet("color: #4ec9b0;")
    
    def _toggle_view_mode(self):
        """Toggle between 2D and 3D views."""
        self._view_mode = '3d' if self.view_toggle.isChecked() else '2d'
        self.view_toggle.setText("2D View" if self._view_mode == '3d' else "3D View")
        
        self.figure.clear()
        if self._view_mode == '3d':
            self.ax = self.figure.add_subplot(111, projection='3d')
        else:
            self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        
        self._update_plot()
    
    def _toggle_curvature(self):
        """Toggle curvature visualization."""
        self._show_curvature = self.curvature_toggle.isChecked()
        self._update_plot()
    
    def _toggle_cliffs(self):
        """Toggle cliff highlighting."""
        self._highlight_cliffs = self.cliff_toggle.isChecked()
        self._update_plot()
    
    def _on_dimension_change(self):
        """Handle dimension selection change."""
        self._selected_dimensions = (
            self.x_dim_combo.currentText(),
            self.y_dim_combo.currentText()
        )
        self._update_plot()
    
    def _get_dimension_value(self, point: 'PathPoint', dim_name: str) -> float:
        """Extract dimension value from a path point."""
        return getattr(point.state, dim_name, 0.0)
    
    def _update_plot(self):
        """Update the visualization."""
        if self._path is None or len(self._path.points) == 0:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "No path data", 
                        color='gray', ha='center', va='center',
                        transform=self.ax.transAxes)
            self.canvas.draw()
            return
        
        self.ax.clear()
        points = self._path.points
        
        # Get coordinates
        x_dim, y_dim = self._selected_dimensions
        x_vals = np.array([self._get_dimension_value(p, x_dim) for p in points])
        y_vals = np.array([self._get_dimension_value(p, y_dim) for p in points])
        estimates = np.array([p.estimate for p in points])
        
        # Filter out NaN estimates
        valid = ~np.isnan(estimates)
        x_vals = x_vals[valid]
        y_vals = y_vals[valid]
        estimates = estimates[valid]
        valid_points = [p for p, v in zip(points, valid.tolist()) if v]
        
        if len(valid_points) == 0:
            self.ax.text(0.5, 0.5, "No valid points", 
                        color='gray', ha='center', va='center',
                        transform=self.ax.transAxes)
            self.canvas.draw()
            return
        
        # Color by curvature if enabled
        if self._show_curvature:
            curvatures = np.array([p.curvature if p.curvature else 0.0 for p in valid_points])
            if curvatures.max() > 0:
                curv_norm = curvatures / curvatures.max()
            else:
                curv_norm = np.zeros_like(curvatures)
            colors = plt.cm.RdYlGn_r(curv_norm)  # Red=high curvature, Green=low
        else:
            colors = '#4ec9b0'
        
        if self._view_mode == '3d':
            # 3D plot: x_dim, y_dim, estimate
            self.ax.scatter(x_vals, y_vals, estimates, c=colors, s=50, alpha=0.8)
            
            # Draw path lines
            for i in range(len(x_vals) - 1):
                alpha = 0.6 if valid_points[i].is_stable and valid_points[i+1].is_stable else 0.3
                self.ax.plot([x_vals[i], x_vals[i+1]], 
                            [y_vals[i], y_vals[i+1]], 
                            [estimates[i], estimates[i+1]],
                            color='#888', alpha=alpha, linewidth=1)
            
            # Mark cliffs
            if self._highlight_cliffs and self._path.cliffs:
                for cliff in self._path.cliffs:
                    # Find closest point
                    idx = np.argmin([abs(p.t - cliff.t) for p in valid_points])
                    self.ax.scatter([x_vals[idx]], [y_vals[idx]], [estimates[idx]], 
                                   c='red', s=150, marker='X', zorder=10)
            
            self.ax.set_xlabel(x_dim.capitalize(), color='#aaa')
            self.ax.set_ylabel(y_dim.capitalize(), color='#aaa')
            self.ax.set_zlabel('Estimate', color='#aaa')
            self.ax.tick_params(colors='#666')
            
        else:
            # 2D plot: t vs estimate with assumption coloring
            t_vals = np.array([p.t for p in valid_points])
            
            # Draw path
            self.ax.plot(t_vals, estimates, color='#4ec9b0', linewidth=2, alpha=0.8)
            
            # Draw points colored by curvature
            scatter = self.ax.scatter(t_vals, estimates, c=colors, s=50, zorder=5)
            
            # Draw confidence bands
            std_errors = np.array([p.std_error for p in valid_points])
            self.ax.fill_between(t_vals, 
                                estimates - std_errors, 
                                estimates + std_errors,
                                color='#4ec9b0', alpha=0.15)
            
            # Mark cliffs
            if self._highlight_cliffs and self._path.cliffs:
                for i, cliff in enumerate(self._path.cliffs):
                    # Find closest point
                    idx = np.argmin([abs(p.t - cliff.t) for p in valid_points])
                    self.ax.scatter([t_vals[idx]], [estimates[idx]], 
                                   c='red', s=150, marker='X', zorder=10)
                    self.ax.annotate(f"Cliff: {cliff.broken_assumption}", 
                                    xy=(t_vals[idx], estimates[idx]),
                                    xytext=(10, 10), textcoords='offset points',
                                    color='#f44747', fontsize=9,
                                    arrowprops=dict(arrowstyle='->', color='#f44747'))
            
            # Mark unstable regions
            for i, p in enumerate(valid_points):
                if not p.is_stable:
                    self.ax.axvspan(t_vals[max(0, i-1)], t_vals[min(len(t_vals)-1, i+1)], 
                                   alpha=0.1, color='red')
            
            self.ax.set_xlabel('Path Parameter (t)', color='#aaa')
            self.ax.set_ylabel('Estimate', color='#aaa')
            self.ax.tick_params(colors='#666')
            
            # Add assumption relaxation annotation
            self.ax.text(0.02, 0.98, f"Classical\n(t=0)", transform=self.ax.transAxes,
                        va='top', ha='left', color='#4ec9b0', fontsize=9)
            self.ax.text(0.98, 0.98, f"Relaxed\n(t=1)", transform=self.ax.transAxes,
                        va='top', ha='right', color='#dcdcaa', fontsize=9)
        
        # Common styling
        self.ax.set_facecolor('#1e1e1e')
        for spine in self.ax.spines.values():
            spine.set_color('#333')
        self.ax.grid(True, alpha=0.1, color='white')
        
        self.canvas.draw()
    
    def _on_click(self, event):
        """Handle click events on the plot."""
        if event.inaxes != self.ax or self._path is None:
            return
        
        if self._view_mode == '2d':
            # Find closest point
            t_vals = np.array([p.t for p in self._path.points 
                              if not np.isnan(p.estimate)])
            if len(t_vals) == 0:
                return
            
            idx = np.argmin(np.abs(t_vals - event.xdata))
            point = [p for p in self._path.points if not np.isnan(p.estimate)][idx]
            
            # Emit signal
            params = {
                'linearity': point.state.linearity,
                'independence': point.state.independence,
                'normality': point.state.normality,
                't': point.t,
                'estimate': point.estimate
            }
            self.point_selected.emit(params)
            
            # Show tooltip
            QToolTip.showText(
                self.canvas.mapToGlobal(self.canvas.pos()),
                f"t={point.t:.2f}\n"
                f"Estimate: {point.estimate:.4f}±{point.std_error:.4f}\n"
                f"Linearity: {point.state.linearity:.2f}\n"
                f"Curvature: {point.curvature or 0:.4f}",
                self.canvas
            )
