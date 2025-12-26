
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout, QToolTip,
    QPushButton, QSlider, QGroupBox, QComboBox, QSpinBox
)
from PySide6.QtCore import Qt, Signal, Slot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from statelix_py.core.unified_space import CausalSpace

class CausalManifoldWidget(QWidget):
    """
    Visualizes the Causal Manifold (Sensitivity Plot).
    Shows how the estimate changes with hyperparameters.
    
    Enhanced with:
    - CausalSpace tensor visualization (3D point cloud)
    - Rotor rotation controls for testing invariance
    - Stability gradient visualization
    - Interactive vector operations
    """
    point_selected = Signal(dict)  # Emits params of the selected point
    vector_operation = Signal(str, dict)  # operation_name, params
    rotor_applied = Signal(float, int, int)  # angle, axis1, axis2
    
    def __init__(self):
        super().__init__()
        self._causal_space: Optional['CausalSpace'] = None
        self._view_mode = '2d'  # '2d' or '3d'
        self._show_stability = False
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        self.title = QLabel("Causal Manifold: Sensitivity Analysis")
        self.title.setStyleSheet("font-weight: bold; color: #ccc;")
        layout.addWidget(self.title)
        
        # Control Panel
        control_panel = self._create_control_panel()
        layout.addWidget(control_panel)
        
        # Matplotlib Figure
        self.figure = plt.figure(figsize=(6, 5))
        self.figure.patch.set_facecolor('#1e1e1e')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)
        
        # Stability Info Label
        self.stability_label = QLabel("")
        self.stability_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.stability_label)
        
        # Help Label
        help_label = QLabel("Click a point to adopt settings. Use controls to apply tensor operations.")
        help_label.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        
        self.setLayout(layout)
        self.canvas.mpl_connect('button_press_event', self.on_click)
    
    def _create_control_panel(self) -> QWidget:
        """Create the control panel for tensor operations."""
        panel = QGroupBox("Tensor Operations")
        panel.setStyleSheet("""
            QGroupBox { color: #aaa; border: 1px solid #444; border-radius: 4px; margin-top: 8px; padding-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background: #3a3a3a; color: #ccc; border: 1px solid #555; border-radius: 3px; padding: 4px 8px; }
            QPushButton:hover { background: #4a4a4a; }
            QLabel { color: #aaa; }
            QSlider::groove:horizontal { background: #444; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #4ec9b0; width: 14px; margin: -4px 0; border-radius: 7px; }
        """)
        
        layout = QHBoxLayout()
        
        # View Mode Toggle
        self.view_toggle = QPushButton("3D View")
        self.view_toggle.setCheckable(True)
        self.view_toggle.clicked.connect(self._toggle_view_mode)
        layout.addWidget(self.view_toggle)
        
        # Rotor Angle Slider
        layout.addWidget(QLabel("Rotor:"))
        self.rotor_slider = QSlider(Qt.Horizontal)
        self.rotor_slider.setRange(-180, 180)
        self.rotor_slider.setValue(0)
        self.rotor_slider.setMinimumWidth(80)
        self.rotor_slider.valueChanged.connect(self._on_rotor_change)
        layout.addWidget(self.rotor_slider)
        
        # Rotation Plane Selection
        self.plane_combo = QComboBox()
        self.plane_combo.addItems(["XY", "YZ", "XZ"])
        self.plane_combo.currentIndexChanged.connect(self._on_rotor_change)
        layout.addWidget(self.plane_combo)
        
        # Apply Rotor Button
        apply_btn = QPushButton("Apply Rotor")
        apply_btn.clicked.connect(self._apply_rotor)
        layout.addWidget(apply_btn)
        
        # Stability Toggle
        self.stability_toggle = QPushButton("Show Stability")
        self.stability_toggle.setCheckable(True)
        self.stability_toggle.clicked.connect(self._toggle_stability)
        layout.addWidget(self.stability_toggle)
        
        panel.setLayout(layout)
        return panel
    
    def set_causal_space(self, space: 'CausalSpace'):
        """Set the CausalSpace for tensor-backed operations."""
        self._causal_space = space
        self._update_3d_view()
    
    def _toggle_view_mode(self):
        """Toggle between 2D and 3D view."""
        self._view_mode = '3d' if self.view_toggle.isChecked() else '2d'
        self.view_toggle.setText("2D View" if self._view_mode == '3d' else "3D View")
        
        # Recreate axes
        self.figure.clear()
        if self._view_mode == '3d':
            self.ax = self.figure.add_subplot(111, projection='3d')
            self.ax.set_facecolor('#1e1e1e')
            self._update_3d_view()
        else:
            self.ax = self.figure.add_subplot(111)
            self.ax.set_facecolor('#1e1e1e')
            if hasattr(self, 'points') and self.points:
                self.update_manifold(self.points)
        
        self.canvas.draw()
    
    def _toggle_stability(self):
        """Toggle stability gradient visualization."""
        self._show_stability = self.stability_toggle.isChecked()
        if self._view_mode == '3d':
            self._update_3d_view()
        elif hasattr(self, 'points') and self.points:
            self.update_manifold(self.points)
    
    def _on_rotor_change(self):
        """Handle rotor slider/combo changes - preview only."""
        if self._view_mode == '3d' and self._causal_space is not None:
            self._update_3d_view(preview_rotor=True)
    
    def _apply_rotor(self):
        """Apply the rotor transformation to the CausalSpace."""
        if self._causal_space is None:
            return
        
        angle = np.radians(self.rotor_slider.value())
        plane_map = {"XY": (0, 1), "YZ": (1, 2), "XZ": (0, 2)}
        plane = plane_map[self.plane_combo.currentText()]
        
        self.rotor_applied.emit(angle, plane[0], plane[1])
        
        # Apply to internal space
        try:
            from statelix_py.core.unified_space import RotorTransform
            rotor = RotorTransform(angle, plane, self._causal_space.embedding_dim)
            self._causal_space = self._causal_space.apply_rotor(rotor)
            self._update_3d_view()
            
            # Check invariance
            pd_score = self._causal_space.topological_filter().structure_score()
            self.stability_label.setText(f"✓ Rotor applied. Topology Score: {pd_score:.4f}")
        except Exception as e:
            self.stability_label.setText(f"⚠ Rotor error: {str(e)[:50]}")
    
    def _update_3d_view(self, preview_rotor: bool = False):
        """Update the 3D tensor visualization."""
        if self._causal_space is None or self._view_mode != '3d':
            return
        
        self.ax.clear()
        
        points = self._causal_space.points
        if points.shape[0] == 0:
            self.ax.text(0, 0, 0, "No data", color='gray')
            self.canvas.draw()
            return
        
        # Apply preview rotor if requested
        if preview_rotor:
            try:
                from statelix_py.core.unified_space import RotorTransform
                angle = np.radians(self.rotor_slider.value())
                plane_map = {"XY": (0, 1), "YZ": (1, 2), "XZ": (0, 2)}
                plane = plane_map[self.plane_combo.currentText()]
                rotor = RotorTransform(angle, plane, self._causal_space.embedding_dim)
                points = rotor.apply(points)
            except ImportError:
                pass
        
        # Extract first 3 dimensions for visualization
        x = points[:, 0] if points.shape[1] > 0 else np.zeros(points.shape[0])
        y = points[:, 1] if points.shape[1] > 1 else np.zeros(points.shape[0])
        z = points[:, 2] if points.shape[1] > 2 else np.zeros(points.shape[0])
        
        # Color by stability gradient if enabled
        if self._show_stability:
            try:
                gradient = self._causal_space.compute_stability_gradient()
                gradient_norm = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8)
                colors = plt.cm.RdYlGn_r(gradient_norm)  # Red = unstable, Green = stable
            except Exception:
                colors = '#4ec9b0'
        else:
            colors = '#4ec9b0'
        
        self.ax.scatter(x, y, z, c=colors, s=50, alpha=0.8)
        
        # Draw edges from adjacency
        if self._causal_space.adjacency is not None:
            adj = self._causal_space.adjacency
            for i in range(adj.shape[0]):
                for j in range(adj.shape[1]):
                    if adj[i, j] > 0:
                        self.ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 
                                     color='#666', alpha=0.4, linewidth=0.8)
        
        # Node labels
        if self._causal_space.node_names:
            for i, name in enumerate(self._causal_space.node_names):
                self.ax.text(x[i], y[i], z[i], f" {name}", color='#aaa', fontsize=8)
        
        # Styling
        self.ax.set_xlabel('Dim 1', color='#666')
        self.ax.set_ylabel('Dim 2', color='#666')
        self.ax.set_zlabel('Dim 3', color='#666')
        self.ax.tick_params(colors='#555')
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        
        self.canvas.draw()
        
    def update_manifold(self, points, quivers=None):
        """Update 2D manifold visualization."""
        self.points = points
        
        if self._view_mode == '3d':
            return  # Skip 2D update in 3D mode
        
        self.ax.clear()
        
        if not points:
            self.ax.text(0.5, 0.5, "No manifold data available", 
                         color='gray', ha='center', va='center')
            self.canvas.draw()
            return
            
        estimates = [p.estimate for p in points]
        errors = [p.std_error for p in points]
        x_vals = np.arange(len(points))
        
        # Plot estimate line
        self.ax.plot(x_vals, estimates, color='#4ec9b0', linewidth=2, label='Estimate')
        
        # Confidence Band
        self.ax.fill_between(x_vals, 
                             np.array(estimates) - np.array(errors),
                             np.array(estimates) + np.array(errors),
                             color='#4ec9b0', alpha=0.1)
        
        # Plot points with stability coloring if enabled
        if self._show_stability and self._causal_space is not None:
            try:
                gradient = self._causal_space.compute_stability_gradient()
                if len(gradient) == len(points):
                    gradient_norm = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8)
                    colors = plt.cm.RdYlGn_r(gradient_norm)
                else:
                    colors = ['#4ec9b0' if p.is_stable else '#f44747' for p in points]
            except Exception:
                colors = ['#4ec9b0' if p.is_stable else '#f44747' for p in points]
        else:
            colors = ['#4ec9b0' if p.is_stable else '#f44747' for p in points]
        
        self.ax.scatter(x_vals, estimates, c=colors, s=30, zorder=5)
        
        # Plot Quivers (Instability Arrows)
        if quivers:
            for q in quivers:
                self.ax.annotate('', xy=(q['x'], q['y'] + q['v']), xytext=(q['x'], q['y']),
                                 arrowprops=dict(arrowstyle='->', color=q['color'], lw=1.5))
        
        # Styling
        param_name = list(points[0].params.keys())[0] if points else "Hyperparameter"
        self.ax.set_xlabel(f"Search Path ({param_name})", color='#888')
        self.ax.set_ylabel("Causal Effect", color='#888')
        self.ax.tick_params(colors='#666')
        for spine in self.ax.spines.values():
            spine.set_color('#333')
        
        self.ax.grid(True, alpha=0.1, color='white')
        self.canvas.draw()

    def update_hypothetical_manifold(self, points):
        """Overlay a dashed line for 'What-If' scenario."""
        if not points or not hasattr(self, 'ax'): return
        
        estimates = [p.estimate for p in points]
        x_vals = np.arange(len(points))
        
        # Plot hypothetical line
        self.ax.plot(x_vals, estimates, color='#dcdcaa', linestyle='--', linewidth=1.5, 
                     alpha=0.8, label='Hypothetical')
        self.ax.legend(facecolor='#252526', edgecolor='#333', labelcolor='#ccc')
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax or not hasattr(self, 'points'):
            return
        
        if self._view_mode == '2d':
            # Find closest point in 2D mode
            x = int(round(event.xdata))
            if 0 <= x < len(self.points):
                p = self.points[x]
                self.point_selected.emit(p.params)
                QToolTip.showText(self.canvas.mapToGlobal(self.canvas.pos()), 
                                  f"Selected: {p.params}\nEffect: {p.estimate:.4f}", self.canvas)

