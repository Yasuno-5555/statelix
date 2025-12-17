import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtWidgets import QWidget, QVBoxLayout, QComboBox, QLabel, QHBoxLayout
from statelix_py.gui.styles import StatelixTheme

class PlotPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.apply_dark_theme()
        self.init_ui()

    def apply_dark_theme(self):
        # Match StatelixTheme
        face_color = StatelixTheme.COLOR_BG_PANEL
        text_color = StatelixTheme.COLOR_TEXT
        
        self.figure.patch.set_facecolor(face_color)
        
        # Update rcParams for this instance naturally or manually set props
        # but since we are one app, updating global rcParams is okay-ish, 
        # but let's do it per plot or effectively globally.
        # Actually, let's just set the figure and axes defaults for this figure.
        import matplotlib.pyplot as plt
        plt.style.use('dark_background') # Base dark theme
        
        # Override specific colors
        item_color = text_color
        
        self.figure.patch.set_facecolor(face_color)
        # We also need to ensure newly created subplots use these.
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Toolbar / Controls
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Plot Type:"))
        self.plot_type = QComboBox()
        self.plot_type.addItems(["Residuals vs Fitted", "Residual Histogram", "Q-Q Plot"])
        ctrl_layout.addWidget(self.plot_type)
        ctrl_layout.addStretch()
        
        layout.addLayout(ctrl_layout)
        layout.addWidget(self.toolbar) # Add Toolbar
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        
        # Enable Mouse Wheel Zoom
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

    def on_scroll(self, event):
        """Mouse Wheel Zoom"""
        if event.inaxes is None: return
        
        ax = event.inaxes
        # Scale Factor: 1.1 for Zoom In (Up), 0.9 for Zoom Out (Down)
        # Event step > 0 is Up/Away from user (Zoom In)
        
        # User reported "Upside Down" previously.
        # Standard: Scroll UP (step > 0) = Zoom IN (View Range Decreases)
        # If user said it's reversed, maybe they want UP = Zoom OUT?
        # NO, "Upside Down" likely meant Up -> Out, which is unintuitive.
        # I will ensure Up -> In.
        
        # User Feedback: "Upside Down / Left Right Reversed"
        # Flipping Logic Reversely based on request.
        # Now: Scroll UP (step > 0) -> Zoom OUT (Larger View / Scale > 1)
        #      Scroll DOWN (step < 0) -> Zoom IN (Smaller View / Scale < 1)
        
        base_scale = 1.15
        if event.step > 0:
            scale_factor = base_scale      # Zoom Out (Expand view)
        else:
            scale_factor = 1 / base_scale  # Zoom In (Shrink view)
            
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Center of zoom
        xdata = event.xdata
        ydata = event.ydata
        
        # New Width/Height
        w = (xlim[1] - xlim[0]) * scale_factor
        h = (ylim[1] - ylim[0]) * scale_factor
        
        # Preserving center relative to mouse position? 
        # Simplified: Zoom keeping center mostly fixed
        # New limits
        new_xmin = xdata - (xdata - xlim[0]) * scale_factor
        new_xmax = xdata + (xlim[1] - xdata) * scale_factor
        new_ymin = ydata - (ydata - ylim[0]) * scale_factor
        new_ymax = ydata + (ylim[1] - ydata) * scale_factor
        
        ax.set_xlim([new_xmin, new_xmax])
        ax.set_ylim([new_ymin, new_ymax])
        self.canvas.draw()

    def _setup_ax(self):
        """Helper to create a styled subplot"""
        from statelix_py.gui.styles import StatelixTheme
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(StatelixTheme.COLOR_BG_MAIN) # Slightly darker for plot area
        
        # grid styling
        ax.grid(True, linestyle=':', alpha=0.3, color='white')
        
        # Spines
        for spine in ax.spines.values():
            spine.set_color(StatelixTheme.COLOR_BORDER)
            
        ax.tick_params(colors=StatelixTheme.COLOR_TEXT)
        ax.xaxis.label.set_color(StatelixTheme.COLOR_TEXT)
        ax.yaxis.label.set_color(StatelixTheme.COLOR_TEXT)
        ax.title.set_color(StatelixTheme.COLOR_TEXT)
        return ax

    def clear_plot(self):
        self.figure.clear()
        self.canvas.draw()

    def plot_ols_diagnostics(self, fitted, residuals):
        self.figure.clear()
        
        ptype = self.plot_type.currentText()
        ax = self._setup_ax()

        if ptype == "Residuals vs Fitted":
            ax.scatter(fitted, residuals, alpha=0.6, color='#00d1b2') # Light cyan
            ax.axhline(0, color='#ff3860', linestyle='--') # Red
            ax.set_xlabel("Fitted Values")
            ax.set_ylabel("Residuals")
            ax.set_title("Residuals vs Fitted")
            
        elif ptype == "Residual Histogram":
            ax.hist(residuals, bins=20, edgecolor='white', alpha=0.7, color='#3273dc')
            ax.set_xlabel("Residuals")
            ax.set_title("Histogram of Residuals")
            
        elif ptype == "Q-Q Plot":
            import numpy as np
            sorted_res = np.sort(residuals)
            n = len(sorted_res)
            theoretical = np.linspace(-3, 3, n)
            ax.scatter(theoretical, sorted_res, color='#3273dc')
            
            m, b = np.polyfit(theoretical, sorted_res, 1)
            ax.plot(theoretical, m*theoretical + b, color='#ff3860')
            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel("Sample Quantiles")
            ax.set_title("Normal Q-Q Plot")

        self.canvas.draw()

    def plot_clustering(self, X, labels, centroids=None):
        self.figure.clear()
        ax = self._setup_ax()
        
        x_data = X[:, 0]
        y_data = X[:, 1] if X.shape[1] > 1 else X[:, 0]
        
        ax.scatter(x_data, y_data, c=labels, cmap='viridis', alpha=0.8, edgecolors='none')
        
        if centroids is not None:
             cx = centroids[:, 0]
             cy = centroids[:, 1] if centroids.shape[1] > 1 else centroids[:, 0]
             ax.scatter(cx, cy, c='#ff3860', marker='x', s=100, linewidths=2, label='Centroids')
             ax.legend()
             
        ax.set_title("K-Means Clustering (First 2 Dimensions)")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        self.canvas.draw()

    def plot_boxplot(self, data, groups):
        self.figure.clear()
        ax = self._setup_ax()
        
        import numpy as np
        unique_groups = np.unique(groups)
        plot_data = [data[groups == g] for g in unique_groups]
        
        # Matplotlib boxplot is hard to style perfectly in one go, but let's try
        bp = ax.boxplot(plot_data, labels=unique_groups, patch_artist=True)
        
        for box in bp['boxes']:
            box.set(facecolor='#3273dc', alpha=0.5)
        
        ax.set_title("One-Way ANOVA: Group Comparison")
        ax.set_xlabel("Group")
        ax.set_ylabel("Value")
        self.canvas.draw()

    def plot_time_series(self, series, title="Time Series"):
        self.figure.clear()
        ax = self._setup_ax()
        
        ax.plot(series, label='Observed', color='#00d1b2')
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend(facecolor='#333333', edgecolor='white') # legend bg needs to be dark
        self.canvas.draw()

    def plot_hmc_trace(self, trace_data):
        self.figure.clear()
        n_params = trace_data.shape[1]
        n_plot = min(n_params, 4)
        
        ax = self._setup_ax()
        
        colors = ['#00d1b2', '#ff3860', '#3273dc', '#ffe08a']
        for i in range(n_plot):
            c = colors[i % len(colors)]
            ax.plot(trace_data[:, i], label=f"Param {i}", alpha=0.8, color=c)
            
        ax.set_title("HMC Trace Plot (MCMC Sampling)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Parameter Value")
        ax.legend(facecolor='#333333', edgecolor='white')
        
        self.canvas.draw()
