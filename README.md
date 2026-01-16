# Projects-for-Foundations-of-Machine-Learning-for-Chemistry

import sys
import os
import matplotlib

matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QSpinBox,
                             QPushButton, QSizePolicy, QColorDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

# Boiling point data for different hydrocarbon series
data = {
    'alkane': {
        'name': 'Linear Alkanes',
        'boiling_points': [-162, -89, -42, -0.5, 36, 69, 98, 126, 151, 174, 196, 216, 235, 254, 271]
    },
    'alkene': {
        'name': 'Linear Alkenes (1-ene)',
        'boiling_points': [-104, -47, -6, 30, 64, 94, 121, 146, 169, 191, 213, 233, 251, 268]
    },
    'alkyne': {
        'name': 'Linear Alkynes (1-yne)',
        'boiling_points': [-84, -23, 8, 40, 71, 100, 126, 151, 174, 196, 215, 234, 251]
    }
}


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class HydrocarbonPlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hydrocarbon Boiling Point Analyzer")
        self.setGeometry(100, 100, 1000, 700)

        # Store current background color
        self.bg_color = QColor(240, 240, 240)  # Light gray default

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.main_widget = main_widget
        layout = QVBoxLayout(main_widget)

        # Control panel
        control_layout = QHBoxLayout()

        # Hydrocarbon type selection
        control_layout.addWidget(QLabel("Hydrocarbon Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(['alkane', 'alkene', 'alkyne'])
        control_layout.addWidget(self.type_combo)

        # Number of carbons selection
        control_layout.addWidget(QLabel("Number of Carbons:"))
        self.carbons_spinbox = QSpinBox()
        self.carbons_spinbox.setMinimum(2)
        self.carbons_spinbox.setMaximum(15)
        self.carbons_spinbox.setValue(10)
        control_layout.addWidget(self.carbons_spinbox)

        # Update button
        self.update_btn = QPushButton("Update Plot")
        self.update_btn.clicked.connect(self.update_plot)
        control_layout.addWidget(self.update_btn)

        # Save button
        self.save_btn = QPushButton("Save Plot")
        self.save_btn.clicked.connect(self.save_plot)
        control_layout.addWidget(self.save_btn)

        # Save Excel button
        self.save_excel_btn = QPushButton("Save Data to Excel")
        self.save_excel_btn.clicked.connect(self.save_to_excel)
        control_layout.addWidget(self.save_excel_btn)

        # Color picker button
        self.color_btn = QPushButton("Change Interface Color")
        self.color_btn.clicked.connect(self.change_color)
        control_layout.addWidget(self.color_btn)

        control_layout.addStretch()
        layout.addLayout(control_layout)

        # Info label for equation and R²
        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 12pt; padding: 10px;")
        layout.addWidget(self.info_label)

        # Create matplotlib canvas
        self.canvas = MplCanvas(self, width=10, height=6, dpi=100)
        layout.addWidget(self.canvas)

        # Create annotation for hover tooltip
        self.annot = None

        # Connect mouse motion event
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)

        # Initial plot
        self.update_plot()

    def update_plot(self):
        # Get selected parameters
        hc_type = self.type_combo.currentText()
        num_carbons = self.carbons_spinbox.value()

        # Get data
        hc_data = data[hc_type]
        max_carbons = len(hc_data['boiling_points']) + 1

        # Adjust if requested carbons exceeds available data
        if num_carbons > max_carbons:
            num_carbons = max_carbons
            self.carbons_spinbox.setValue(num_carbons)

        # Prepare data based on hydrocarbon type
        if hc_type == 'alkane':
            carbons = list(range(1, num_carbons + 1))
            boiling_points = hc_data['boiling_points'][:num_carbons]
        else:  # alkenes and alkynes start from C2
            carbons = list(range(2, num_carbons + 1))
            boiling_points = hc_data['boiling_points'][:num_carbons - 1]

        # Store current data for Excel export
        self.current_carbons = carbons
        self.current_boiling_points = boiling_points
        self.current_hc_type = hc_type
        self.current_hc_name = hc_data['name']

        # Perform linear regression using numpy
        coefficients = np.polyfit(carbons, boiling_points, 1)
        slope = coefficients[0]
        intercept = coefficients[1]

        # Calculate R-squared
        y_pred = slope * np.array(carbons) + intercept
        ss_res = np.sum((np.array(boiling_points) - y_pred) ** 2)
        ss_tot = np.sum((np.array(boiling_points) - np.mean(boiling_points)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Store regression info for Excel export
        self.current_slope = slope
        self.current_intercept = intercept
        self.current_r_squared = r_squared
        self.current_predicted = y_pred

        # Create regression line
        x_line = np.array(carbons)
        y_line = slope * x_line + intercept

        # Clear and plot
        self.canvas.axes.clear()

        # Scatter plot - store the collection for hover interaction
        self.scatter = self.canvas.axes.scatter(carbons, boiling_points, s=100, c='blue',
                                                alpha=0.7, edgecolors='black', label='Data points', zorder=3)

        # Regression line
        self.canvas.axes.plot(x_line, y_line, 'r--', linewidth=2, label='Linear fit', zorder=2)

        # Labels and title
        self.canvas.axes.set_title(f'Boiling Points of {hc_data["name"]}',
                                   fontsize=16, fontweight='bold')
        self.canvas.axes.set_xlabel('Number of Carbon Atoms', fontsize=12)
        self.canvas.axes.set_ylabel('Boiling Point (°C)', fontsize=12)
        self.canvas.axes.grid(True, alpha=0.3, zorder=1)
        self.canvas.axes.legend()

        # Update canvas
        self.canvas.draw()

        # Create or update annotation
        if self.annot is None:
            self.annot = self.canvas.axes.annotate("", xy=(0, 0), xytext=(20, 20),
                                                   textcoords="offset points",
                                                   bbox=dict(boxstyle="round", fc="yellow", alpha=0.9),
                                                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"))
            self.annot.set_visible(False)

        # Update info label with equation and R²
        equation_text = f"Equation: y = {slope:.2f}x + {intercept:.2f}  |  R² = {r_squared:.4f}"
        self.info_label.setText(equation_text)

    def save_plot(self):
        # Save the plot to the AlkaneBoilingPoints directory
        output_dir = 'AlkaneBoilingPoints'
        os.makedirs(output_dir, exist_ok=True)

        hc_type = self.type_combo.currentText()
        num_carbons = self.carbons_spinbox.value()
        filename = f'{hc_type}_C{num_carbons}_boiling_points.png'
        output_path = os.path.abspath(os.path.join(output_dir, filename))

        self.canvas.figure.savefig(output_path, dpi=300, bbox_inches='tight')

        # Show confirmation with full path
        print(f"Plot saved to: {output_path}")  # Also print to console
        current_text = self.info_label.text()
        self.info_label.setText(f"Plot saved to: {output_path}")

    def change_color(self):
        # Open color picker dialog
        color = QColorDialog.getColor(self.bg_color, self, "Choose Interface Color")

        if color.isValid():
            self.bg_color = color
            # Apply color to main widget background
            self.main_widget.setStyleSheet(f"background-color: {color.name()};")

            # Adjust text color for readability (use white text on dark colors, black on light)
            brightness = (color.red() * 299 + color.green() * 587 + color.blue() * 114) / 1000
            text_color = "white" if brightness < 128 else "black"

            self.info_label.setStyleSheet(f"font-size: 12pt; padding: 10px; color: {text_color};")

    def save_to_excel(self):
        # Create output directory
        output_dir = 'AlkaneBoilingPoints'
        os.makedirs(output_dir, exist_ok=True)

        # Create filename with absolute path
        filename = f'{self.current_hc_type}_C{len(self.current_carbons)}_data.xlsx'
        output_path = os.path.abspath(os.path.join(output_dir, filename))

        # Create DataFrame with the data
        df = pd.DataFrame({
            'Number of Carbons': self.current_carbons,
            'Boiling Point (°C)': self.current_boiling_points,
            'Predicted BP (°C)': self.current_predicted.round(2)
        })

        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Write main data
            df.to_excel(writer, sheet_name='Data', index=False)

            # Create a summary sheet
            summary_df = pd.DataFrame({
                'Parameter': [
                    'Hydrocarbon Type',
                    'Number of Data Points',
                    'Regression Equation',
                    'Slope',
                    'Intercept',
                    'R² Value'
                ],
                'Value': [
                    self.current_hc_name,
                    len(self.current_carbons),
                    f'y = {self.current_slope:.2f}x + {self.current_intercept:.2f}',
                    f'{self.current_slope:.4f}',
                    f'{self.current_intercept:.4f}',
                    f'{self.current_r_squared:.6f}'
                ]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Show confirmation with full path
        print(f"Excel file saved to: {output_path}")  # Also print to console
        current_text = self.info_label.text()
        self.info_label.setText(f"Excel saved to: {output_path}")

    def on_hover(self, event):
        # Check if mouse is over the axes
        if event.inaxes == self.canvas.axes:
            # Check if mouse is over a data point
            cont, ind = self.scatter.contains(event)
            if cont:
                # Get the index of the point
                idx = ind["ind"][0]

                # Get data for this point
                carbon_count = self.current_carbons[idx]
                bp = self.current_boiling_points[idx]
                predicted_bp = self.current_predicted[idx]
                residual = bp - predicted_bp

                # Create compound name
                compound_names = {
                    'alkane': ['Methane', 'Ethane', 'Propane', 'Butane', 'Pentane',
                               'Hexane', 'Heptane', 'Octane', 'Nonane', 'Decane',
                               'Undecane', 'Dodecane', 'Tridecane', 'Tetradecane', 'Pentadecane'],
                    'alkene': ['', 'Ethene', 'Propene', 'Butene', 'Pentene',
                               'Hexene', 'Heptene', 'Octene', 'Nonene', 'Decene',
                               'Undecene', 'Dodecene', 'Tridecene', 'Tetradecene'],
                    'alkyne': ['', 'Ethyne', 'Propyne', 'Butyne', 'Pentyne',
                               'Hexyne', 'Heptyne', 'Octyne', 'Nonyne', 'Decyne',
                               'Undecyne', 'Dodecyne', 'Tridecyne']
                }

                compound_name = compound_names[self.current_hc_type][carbon_count - 1]

                # Update annotation text
                text = f"{compound_name}\n"
                text += f"Carbons: {carbon_count}\n"
                text += f"Actual BP: {bp}°C\n"
                text += f"Predicted BP: {predicted_bp:.2f}°C\n"
                text += f"Residual: {residual:.2f}°C"

                self.annot.xy = (carbon_count, bp)
                self.annot.set_text(text)
                self.annot.set_visible(True)
                self.canvas.draw_idle()
            else:
                # Hide annotation if not over a point
                if self.annot.get_visible():
                    self.annot.set_visible(False)
                    self.canvas.draw_idle()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HydrocarbonPlotter()
    window.show()
    sys.exit(app.exec_())
