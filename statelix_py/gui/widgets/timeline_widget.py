
from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PySide6.QtCore import Signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

class TimelineWidget(QWidget):
    """
    Visualizes DiagnosticHistory evolution over iterations.
    """
    iteration_selected = Signal(int)

    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.figure.patch.set_facecolor('#1e1e1e') # VSCode dark theme background
        
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        
        # Style axes
        self.ax.tick_params(colors='#888', labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color('#444')
            
        layout.addWidget(self.canvas)
        
        # Connect click event
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(200)

    def update_history(self, history_obj, strict_threshold=0.8):
        self.ax.clear()
        
        evolution = history_obj.get_evolution()
        if not evolution:
            self.ax.text(0.5, 0.5, "NO HISTORY DATA", color='#555', ha='center', transform=self.ax.transAxes)
            self.canvas.draw()
            return

        iters = [e['iteration'] for e in evolution]
        mci_scores = [e['mci'] for e in evolution]
        fit_scores = [e['fit_score'] for e in evolution]
        topo_scores = [e['topo_score'] for e in evolution]
        geo_scores = [e['geo_score'] for e in evolution]
        
        # Plot lines
        self.ax.plot(iters, mci_scores, label='MCI', color='#4ec9b0', linewidth=2, marker='o', markersize=4)
        self.ax.plot(iters, fit_scores, label='Fit', color='#569cd6', linestyle='--', alpha=0.6)
        self.ax.plot(iters, topo_scores, label='Topo', color='#ce9178', linestyle='--', alpha=0.6)
        self.ax.plot(iters, geo_scores, label='Geo', color='#dcdcaa', linestyle='--', alpha=0.6)
        
        # Threshold line
        self.ax.axhline(strict_threshold, color='#f44747', linestyle=':', alpha=0.5, label='Threshold')
        
        # Stagnation markers
        stagnant_iters = history_obj.get_stagnation_points()
        if stagnant_iters:
            stag_scores = [mci_scores[i] for i in stagnant_iters]
            self.ax.scatter(stagnant_iters, stag_scores, color='#d16969', marker='x', s=50, label='Stagnation', zorder=5)

        # Labels
        self.ax.set_xlabel("Iteration", color='#888', fontsize=8)
        self.ax.set_ylabel("Score", color='#888', fontsize=8)
        self.ax.legend(prop={'size': 7}, facecolor='#252526', edgecolor='#444', labelcolor='#ccc', loc='lower right')
        
        self.ax.set_ylim(-0.05, 1.05)
        self.ax.grid(True, color='#333', linestyle=':', alpha=0.5)
        
        self.figure.tight_layout()
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax: return
        # Find nearest iteration
        iters = self.ax.get_lines()[0].get_xdata()
        if len(iters) == 0: return
        
        idx = (np.abs(iters - event.xdata)).argmin()
        target_iter = int(iters[idx])
        
        # Draw vertical line to indicate selection
        for line in self.ax.get_lines():
            if getattr(line, '_selection_mark', False):
                line.remove()
        
        v_line = self.ax.axvline(target_iter, color='white', alpha=0.3)
        v_line._selection_mark = True
        self.canvas.draw()
        
        self.iteration_selected.emit(target_iter)
