
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QToolTip
from PySide6.QtCore import Qt, Signal, Slot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import numpy as np

class CausalManifoldWidget(QWidget):
    """
    Visualizes the Causal Manifold (Sensitivity Plot).
    Shows how the estimate changes with hyperparameters.
    """
    point_selected = Signal(dict) # Emits params of the selected point
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.title = QLabel("Causal Manifold: Sensitivity Analysis")
        self.title.setStyleSheet("font-weight: bold; color: #ccc;")
        layout.addWidget(self.title)
        
        # Matplotlib Figure
        self.figure, self.ax = plt.subplots(figsize=(5, 4))
        self.figure.patch.set_facecolor('#1e1e1e')
        self.ax.set_facecolor('#1e1e1e')
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)
        
        # Legend/Help
        help_label = QLabel("Arrows indicate areas of high sensitivity (instability). Click a point to adopt those settings.")
        help_label.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        
        self.setLayout(layout)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
    def update_manifold(self, points, quivers=None):
        self.points = points
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
        
        # Plot points
        colors = ['#4ec9b0' if p.is_stable else '#f44747' for p in points]
        self.ax.scatter(x_vals, estimates, c=colors, s=30, zorder=5)
        
        # Plot Quivers (Instability Arrows)
        if quivers:
            for q in quivers:
                # v is the gradient, we want to point where it's 'falling' or 'rising'
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
            
        # Find closest point
        x = int(round(event.xdata))
        if 0 <= x < len(self.points):
            p = self.points[x]
            self.point_selected.emit(p.params)
            QToolTip.showText(self.canvas.mapToGlobal(self.canvas.pos()), 
                              f"Selected: {p.params}\nEffect: {p.estimate:.4f}", self.canvas)
