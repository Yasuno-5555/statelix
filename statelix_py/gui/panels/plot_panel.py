import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QComboBox, QLabel, QHBoxLayout

class PlotPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.init_ui()

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
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)

    def clear_plot(self):
        self.figure.clear()
        self.canvas.draw()

    def plot_ols_diagnostics(self, fitted, residuals):
        self.figure.clear()
        
        ptype = self.plot_type.currentText()
        ax = self.figure.add_subplot(111)

        if ptype == "Residuals vs Fitted":
            ax.scatter(fitted, residuals, alpha=0.6)
            ax.axhline(0, color='red', linestyle='--')
            ax.set_xlabel("Fitted Values")
            ax.set_ylabel("Residuals")
            ax.set_title("Residuals vs Fitted")
            
        elif ptype == "Residual Histogram":
            ax.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
            ax.set_xlabel("Residuals")
            ax.set_title("Histogram of Residuals")
            
        elif ptype == "Q-Q Plot":
            # Simple Q-Q implementation without scipy (to reduce deps if needed)
            # But normally we'd use scipy.stats.probplot
            import numpy as np
            sorted_res = np.sort(residuals)
            n = len(sorted_res)
            # Theoretical quantiles (approx)
            theoretical = np.linspace(-3, 3, n) # Simplification
            ax.scatter(theoretical, sorted_res)
            # Line (simple)
            m, b = np.polyfit(theoretical, sorted_res, 1)
            ax.plot(theoretical, m*theoretical + b, color='red')
            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel("Sample Quantiles")
            ax.set_title("Normal Q-Q Plot")

        self.canvas.draw()

    def plot_clustering(self, X, labels, centroids=None):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Plot only first 2 dimensions if high dim
        x_data = X[:, 0]
        y_data = X[:, 1] if X.shape[1] > 1 else X[:, 0] # Fallback if 1D
        
        scatter = ax.scatter(x_data, y_data, c=labels, cmap='viridis', alpha=0.6)
        
        if centroids is not None:
             cx = centroids[:, 0]
             cy = centroids[:, 1] if centroids.shape[1] > 1 else centroids[:, 0]
             ax.scatter(cx, cy, c='red', marker='x', s=100, linewidths=2, label='Centroids')
             ax.legend()
             
        ax.set_title("K-Means Clustering (First 2 Dimensions)")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        self.canvas.draw()

    def plot_boxplot(self, data, groups):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        import numpy as np
        unique_groups = np.unique(groups)
        plot_data = [data[groups == g] for g in unique_groups]
        
        ax.boxplot(plot_data, labels=unique_groups)
        ax.set_title("One-Way ANOVA: Group Comparison")
        ax.set_xlabel("Group")
        ax.set_ylabel("Value")
        self.canvas.draw()

    def plot_time_series(self, series, title="Time Series"):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        ax.plot(series, label='Observed')
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        self.canvas.draw()
