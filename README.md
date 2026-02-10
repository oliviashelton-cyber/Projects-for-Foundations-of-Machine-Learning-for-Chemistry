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


    WEEK2
    import pubchempy as pcp
from PIL import Image
import requests
from io import BytesIO

# Fetch theobromine by name
compounds = pcp.get_compounds('theobromine', 'name')
compound = compounds[0]

# Print formatted output
print("=" * 60)
print("THEOBROMINE - COMPOUND DATA")
print("=" * 60)

print("\n📋 Basic Information:")
print(f"   Compound ID (CID): {compound.cid}")
print(f"   IUPAC Name: {compound.iupac_name}")

print("\n🧪 Molecular Properties:")
print(f"   Molecular Formula: {compound.molecular_formula}")
print(f"   Molecular Weight: {compound.molecular_weight} g/mol")
print(f"   Exact Mass: {compound.exact_mass}")

print("\n🔗 Structure Representations:")
print(f"   SMILES: {compound.smiles}")
print(f"   InChI: {compound.inchi}")
print(f"   InChIKey: {compound.inchikey}")

print("\n🖼️  Structure Images:")
print(f"   2D Structure: https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid={compound.cid}&t=l")
print(f"   3D Structure: https://pubchem.ncbi.nlm.nih.gov/compound/{compound.cid}#section=3D-Conformer")

# Download and display the 2D structure image
print("\n   Fetching 2D structure image...")
img_url = f"https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid={compound.cid}&t=l"
response = requests.get(img_url)
img = Image.open(BytesIO(response.content))
img.show()
print("   ✓ Structure image opened in default viewer")

print("\n⚛️  Chemical Properties:")
print(f"   Complexity: {compound.complexity}")
print(f"   TPSA: {compound.tpsa} Ų")
print(f"   H-Bond Donors: {compound.h_bond_donor_count}")
print(f"   H-Bond Acceptors: {compound.h_bond_acceptor_count}")
print(f"   Rotatable Bonds: {compound.rotatable_bond_count}")
print(f"   Heavy Atoms: {compound.heavy_atom_count}")
print(f"   Charge: {compound.charge}")

print("\n🏷️  Common Names:")
for i, synonym in enumerate(compound.synonyms[:5], 1):
    print(f"   {i}. {synonym}")

print("\n" + "=" * 60)

WEEK 2 PART 2

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import requests
from io import BytesIO

class MoleculeViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Molecule Explorer")
        self.root.geometry("900x800")
        self.root.configure(bg="#f0f4f8")
        self.root.resizable(True, True)
        
        # Molecule database
        self.molecules = [
            {
                "name": "Ethanol",
                "formula": "C₂H₆O",
                "smiles": "CCO",
                "weight": "46.07",
                "hbd": "1",
                "hba": "1",
                "tpsa": "20.23",
                "complexity": "2.00",
                "heavy_atoms": "3",
                "rotatable_bonds": "0",
                "description": "Common alcohol found in beverages",
                "cid": "702"
            },
            {
                "name": "Caffeine",
                "formula": "C₈H₁₀N₄O₂",
                "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "weight": "194.19",
                "hbd": "0",
                "hba": "6",
                "tpsa": "58.44",
                "complexity": "293",
                "heavy_atoms": "14",
                "rotatable_bonds": "0",
                "description": "Stimulant found in coffee and tea",
                "cid": "2519"
            },
            {
                "name": "Theobromine",
                "formula": "C₇H₈N₄O₂",
                "smiles": "CN1C=NC2=C1C(=O)NC(=O)N2C",
                "weight": "180.16",
                "hbd": "1",
                "hba": "6",
                "tpsa": "67.20",
                "complexity": "267",
                "heavy_atoms": "13",
                "rotatable_bonds": "0",
                "description": "Compound found in chocolate",
                "cid": "5429"
            },
            {
                "name": "Aspirin",
                "formula": "C₉H₈O₄",
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "weight": "180.16",
                "hbd": "1",
                "hba": "4",
                "tpsa": "63.60",
                "complexity": "212",
                "heavy_atoms": "13",
                "rotatable_bonds": "3",
                "description": "Common pain reliever and anti-inflammatory",
                "cid": "2244"
            },
            {
                "name": "Glucose",
                "formula": "C₆H₁₂O₆",
                "smiles": "C(C1C(C(C(C(O1)O)O)O)O)O",
                "weight": "180.16",
                "hbd": "5",
                "hba": "6",
                "tpsa": "110.38",
                "complexity": "130",
                "heavy_atoms": "12",
                "rotatable_bonds": "1",
                "description": "Simple sugar, primary energy source for cells",
                "cid": "5793"
            },
            {
                "name": "Acetaminophen",
                "formula": "C₈H₉NO₂",
                "smiles": "CC(=O)NC1=CC=C(C=C1)O",
                "weight": "151.16",
                "hbd": "2",
                "hba": "3",
                "tpsa": "49.33",
                "complexity": "153",
                "heavy_atoms": "11",
                "rotatable_bonds": "2",
                "description": "Pain reliever and fever reducer (Tylenol)",
                "cid": "1983"
            },
            {
                "name": "Dopamine",
                "formula": "C₈H₁₁NO₂",
                "smiles": "C1=CC(=C(C=C1CCN)O)O",
                "weight": "153.18",
                "hbd": "3",
                "hba": "3",
                "tpsa": "66.48",
                "complexity": "91.3",
                "heavy_atoms": "11",
                "rotatable_bonds": "2",
                "description": "Neurotransmitter associated with reward and pleasure",
                "cid": "681"
            },
            {
                "name": "Vitamin C",
                "formula": "C₆H₈O₆",
                "smiles": "C(C(C1C(=C(C(=O)O1)O)O)O)O",
                "weight": "176.12",
                "hbd": "4",
                "hba": "6",
                "tpsa": "107.22",
                "complexity": "232",
                "heavy_atoms": "12",
                "rotatable_bonds": "2",
                "description": "Essential nutrient and antioxidant",
                "cid": "54670067"
            }
        ]
        
        self.current_index = 0
        self.setup_ui()
        self.load_molecule()
    
    def setup_ui(self):
        # Title
        title_label = tk.Label(
            self.root,
            text="Molecule Explorer",
            font=("Arial", 22, "bold"),
            bg="#f0f4f8",
            fg="#1e3a8a"
        )
        title_label.pack(pady=10)
        
        # Subtitle
        subtitle_label = tk.Label(
            self.root,
            text="Learn about common molecules and their properties",
            font=("Arial", 11),
            bg="#f0f4f8",
            fg="#64748b"
        )
        subtitle_label.pack(pady=(0, 10))
        
        # Main frame
        main_frame = tk.Frame(self.root, bg="white", relief=tk.RAISED, borderwidth=2)
        main_frame.pack(padx=30, pady=5, fill=tk.BOTH, expand=True)
        
        # Navigation frame
        nav_frame = tk.Frame(main_frame, bg="white")
        nav_frame.pack(pady=10)
        
        # Previous button
        self.prev_btn = tk.Button(
            nav_frame,
            text="← Previous",
            command=self.previous_molecule,
            font=("Arial", 12, "bold"),
            bg="#6366f1",
            fg="#000000",
            activebackground="#4f46e5",
            activeforeground="#000000",
            padx=20,
            pady=10,
            relief=tk.FLAT,
            cursor="hand2"
        )
        self.prev_btn.pack(side=tk.LEFT, padx=10)
        
        # Molecule name
        self.name_label = tk.Label(
            nav_frame,
            text="",
            font=("Arial", 20, "bold"),
            bg="white",
            fg="#1e3a8a"
        )
        self.name_label.pack(side=tk.LEFT, padx=30)
        
        # Next button
        self.next_btn = tk.Button(
            nav_frame,
            text="Next →",
            command=self.next_molecule,
            font=("Arial", 12, "bold"),
            bg="#6366f1",
            fg="#000000",
            activebackground="#4f46e5",
            activeforeground="#000000",
            padx=20,
            pady=10,
            relief=tk.FLAT,
            cursor="hand2"
        )
        self.next_btn.pack(side=tk.LEFT, padx=10)
        
        # Description
        self.desc_label = tk.Label(
            main_frame,
            text="",
            font=("Arial", 11),
            bg="white",
            fg="#64748b"
        )
        self.desc_label.pack(pady=(0, 10))
        
        # Counter
        self.counter_label = tk.Label(
            main_frame,
            text="",
            font=("Arial", 10),
            bg="white",
            fg="#94a3b8"
        )
        self.counter_label.pack()
        
        # Image frame
        self.image_frame = tk.Frame(main_frame, bg="#f8fafc", relief=tk.SUNKEN, borderwidth=1)
        self.image_frame.pack(pady=10, padx=20)
        
        self.image_label = tk.Label(self.image_frame, bg="#f8fafc")
        self.image_label.pack(pady=10)
        
        # Properties frame
        prop_frame = tk.Frame(main_frame, bg="white")
        prop_frame.pack(pady=8, padx=20, fill=tk.X)
        
        # Create 3x2 grid for properties
        self.prop_frames = []
        colors = [("#dbeafe", "#1e40af"), ("#e9d5ff", "#6b21a8"), 
                  ("#d1fae5", "#065f46"), ("#fed7aa", "#c2410c"),
                  ("#fce7f3", "#9f1239"), ("#e0e7ff", "#3730a3")]
        labels = ["Molecular Formula", "Molecular Weight", "H-Bond Donors", 
                  "H-Bond Acceptors", "TPSA", "Complexity"]
        
        for i in range(6):
            row = i // 3
            col = i % 3
            
            frame = tk.Frame(prop_frame, bg=colors[i][0], relief=tk.RAISED, borderwidth=1)
            frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            label = tk.Label(
                frame,
                text=labels[i],
                font=("Arial", 10, "bold"),
                bg=colors[i][0],
                fg=colors[i][1]
            )
            label.pack(pady=(10, 5))
            
            value = tk.Label(
                frame,
                text="",
                font=("Arial", 16, "bold"),
                bg=colors[i][0],
                fg=colors[i][1]
            )
            value.pack(pady=(0, 10))
            
            self.prop_frames.append(value)
        
        prop_frame.columnconfigure(0, weight=1)
        prop_frame.columnconfigure(1, weight=1)
        prop_frame.columnconfigure(2, weight=1)
        
        # SMILES frame
        smiles_outer = tk.Frame(main_frame, bg="#f8fafc", relief=tk.SUNKEN, borderwidth=1)
        smiles_outer.pack(pady=8, padx=20, fill=tk.X)
        
        smiles_title = tk.Label(
            smiles_outer,
            text="SMILES String",
            font=("Arial", 10, "bold"),
            bg="#f8fafc",
            fg="#475569"
        )
        smiles_title.pack(pady=(10, 5))
        
        self.smiles_label = tk.Label(
            smiles_outer,
            text="",
            font=("Courier", 10),
            bg="#f8fafc",
            fg="#1e293b",
            wraplength=700
        )
        self.smiles_label.pack(pady=(0, 10), padx=10)
    
    def load_molecule(self):
        mol = self.molecules[self.current_index]
        
        # Update labels
        self.name_label.config(text=mol["name"])
        self.desc_label.config(text=mol["description"])
        self.counter_label.config(text=f"{self.current_index + 1} of {len(self.molecules)}")
        
        # Update properties
        self.prop_frames[0].config(text=mol["formula"])
        self.prop_frames[1].config(text=f"{mol['weight']} g/mol")
        self.prop_frames[2].config(text=mol["hbd"])
        self.prop_frames[3].config(text=mol["hba"])
        self.prop_frames[4].config(text=f"{mol['tpsa']} Ų")
        self.prop_frames[5].config(text=mol["complexity"])
        
        # Update SMILES
        self.smiles_label.config(text=mol["smiles"])
        
        # Load image
        self.load_image(mol["cid"])
    
    def load_image(self, cid):
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid={cid}&t=l"
            response = requests.get(url)
            img_data = Image.open(BytesIO(response.content))
            img_data = img_data.resize((350, 350), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img_data)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference
        except Exception as e:
            self.image_label.config(text=f"Error loading image: {e}")
    
    def next_molecule(self):
        self.current_index = (self.current_index + 1) % len(self.molecules)
        self.load_molecule()
    
    def previous_molecule(self):
        self.current_index = (self.current_index - 1) % len(self.molecules)
        self.load_molecule()

if __name__ == "__main__":
    root = tk.Tk()
    app = MoleculeViewer(root)
    root.mainloop()

  WEEK 3

  """
Titanic Dataset Analysis
Load and analyze the Titanic dataset from seaborn library.
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, confusion_matrix
import numpy as np

# Create directory for saving plots
save_dir = 'MakingDataWhole'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Created directory: {save_dir}")
else:
    print(f"Directory already exists: {save_dir}")

# Load the Titanic dataset
print("\n" + "🚢 " * 30)
print("=" * 60)
print("    TITANIC DATASET ANALYSIS & MACHINE LEARNING")
print("=" * 60)
print("🚢 " * 30)
print("\nLoading Titanic dataset...")
titanic = sns.load_dataset('titanic')
print("✓ Dataset loaded successfully!\n")

# Display basic information about the dataset
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Dataset Shape: {titanic.shape}")
print(f"Rows: {titanic.shape[0]}, Columns: {titanic.shape[1]}\n")

# Display column names and data types
print("=" * 60)
print("COLUMN INFORMATION")
print("=" * 60)
titanic.info()

# Display rows 5 to 10 (indices 4 to 9)
print("\n" + "=" * 60)
print("ROWS 5 TO 10")
print("=" * 60)
print(titanic.iloc[4:10])

# Check for missing values in the Age column (BEFORE filling)
print("\n" + "=" * 60)
print("AGE COLUMN ANALYSIS (BEFORE FILLING)")
print("=" * 60)

# Save original data for comparison
titanic_original = titanic.copy()
age_missing = titanic['age'].isnull().sum()
age_missing_mask = titanic['age'].isnull()  # Save which rows had missing ages
total_rows = len(titanic)
age_missing_percent = (age_missing / total_rows) * 100
print(f"Number of missing values: {age_missing}")
print(f"Total rows: {total_rows}")
print(f"Percentage missing: {age_missing_percent:.2f}%")

# Calculate mean age using only known ages
mean_age = titanic['age'].mean()  # pandas automatically ignores NaN values
print(f"\nMean age (using only known ages): {mean_age:.2f} years")
print(f"Number of known ages used: {titanic['age'].notna().sum()}")

# MEAN IMPUTATION
print("\n" + "=" * 60)
print("METHOD 1: MEAN IMPUTATION")
print("=" * 60)
print("Filling missing Age values with mean age...")
titanic_mean = titanic.copy()
titanic_mean['age'].fillna(mean_age, inplace=True)

# Verify no missing values remain in Age column
age_missing_after = titanic_mean['age'].isnull().sum()
print(f"Missing values in Age column (AFTER mean imputation): {age_missing_after}")
print("✓ Age column now has NO missing values!")

# Show updated statistics
print(f"\nUpdated Age statistics (Mean Imputation):")
print(f"  Mean: {titanic_mean['age'].mean():.2f} years")
print(f"  Std Dev: {titanic_mean['age'].std():.2f} years")
print(f"  Min: {titanic_mean['age'].min():.2f} years")
print(f"  Max: {titanic_mean['age'].max():.2f} years")

# KNN IMPUTATION
print("\n" + "=" * 60)
print("METHOD 2: KNN IMPUTATION")
print("=" * 60)

# Prepare data for KNN imputation
titanic_knn = titanic_original.copy()

# Select features for KNN imputation
# We'll use numerical features and encode categorical ones
print("\nPreparing features for KNN imputation...")

# Create a subset with relevant features
features_for_knn = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex', 'embarked']
knn_data = titanic_knn[features_for_knn].copy()

# Encode categorical variables
le_sex = LabelEncoder()
le_embarked = LabelEncoder()

# Handle sex
knn_data['sex_encoded'] = le_sex.fit_transform(knn_data['sex'].astype(str))

# Handle embarked (fill missing embarked first with mode)
knn_data['embarked_encoded'] = knn_data['embarked'].fillna(knn_data['embarked'].mode()[0])
knn_data['embarked_encoded'] = le_embarked.fit_transform(knn_data['embarked_encoded'].astype(str))

# Create feature matrix for KNN
knn_features = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_encoded', 'embarked_encoded']
X_knn = knn_data[knn_features].copy()

# Fill missing fare values with median (if any)
X_knn['fare'].fillna(X_knn['fare'].median(), inplace=True)

print(f"Using features: {knn_features}")
print(f"Shape of feature matrix: {X_knn.shape}")

# Apply KNN imputation
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
print("\nApplying KNN imputation (k=5 neighbors)...")
X_imputed = knn_imputer.fit_transform(X_knn)

# Extract imputed ages
titanic_knn['age'] = X_imputed[:, knn_features.index('age')]

# Verify no missing values
age_missing_knn = titanic_knn['age'].isnull().sum()
print(f"Missing values in Age column (AFTER KNN imputation): {age_missing_knn}")
print("✓ Age column now has NO missing values!")

print(f"\nUpdated Age statistics (KNN Imputation):")
print(f"  Mean: {titanic_knn['age'].mean():.2f} years")
print(f"  Std Dev: {titanic_knn['age'].std():.2f} years")
print(f"  Min: {titanic_knn['age'].min():.2f} years")
print(f"  Max: {titanic_knn['age'].max():.2f} years")

# COMPARISON OF METHODS
print("\n" + "=" * 60)
print("COMPARISON: MEAN vs KNN IMPUTATION")
print("=" * 60)

# Compare only the imputed values (not the original known ages)
mean_imputed_ages = titanic_mean.loc[age_missing_mask, 'age']
knn_imputed_ages = titanic_knn.loc[age_missing_mask, 'age']

print(f"\nStatistics for IMPUTED values only ({age_missing} values):")
print("\nMean Imputation:")
print(f"  Mean: {mean_imputed_ages.mean():.2f} years")
print(f"  Std Dev: {mean_imputed_ages.std():.2f} years")
print(f"  Min: {mean_imputed_ages.min():.2f} years")
print(f"  Max: {mean_imputed_ages.max():.2f} years")

print("\nKNN Imputation:")
print(f"  Mean: {knn_imputed_ages.mean():.2f} years")
print(f"  Std Dev: {knn_imputed_ages.std():.2f} years")
print(f"  Min: {knn_imputed_ages.min():.2f} years")
print(f"  Max: {knn_imputed_ages.max():.2f} years")

print("\nKey Observations:")
print(f"  • Mean imputation: All {age_missing} values set to {mean_age:.2f}")
print(f"  • KNN imputation: Values range from {knn_imputed_ages.min():.2f} to {knn_imputed_ages.max():.2f}")
print(f"  • KNN preserves more variance (Std: {knn_imputed_ages.std():.2f} vs {mean_imputed_ages.std():.2f})")

# Use KNN imputed data for rest of analysis
titanic = titanic_knn.copy()
print("\n✓ Using KNN-imputed dataset for remaining analysis")

# VALIDATE KNN IMPUTATION MODEL
print("\n" + "=" * 60)
print("KNN IMPUTATION MODEL VALIDATION")
print("=" * 60)

# To validate the KNN model, we'll use cross-validation on the KNOWN ages
# We'll hide some known ages, predict them with KNN, then compare
print("\nValidating KNN model using known ages...")

# Get rows with known ages
known_age_data = titanic_original[titanic_original['age'].notna()].copy()
print(f"Number of rows with known ages: {len(known_age_data)}")

# Prepare the same features used in KNN imputation
validation_data = known_age_data[features_for_knn].copy()
validation_data['sex_encoded'] = le_sex.transform(validation_data['sex'].astype(str))
validation_data['embarked_filled'] = validation_data['embarked'].fillna(validation_data['embarked'].mode()[0])
validation_data['embarked_encoded'] = le_embarked.transform(validation_data['embarked_filled'].astype(str))

X_validation = validation_data[knn_features].copy()
X_validation['fare'].fillna(X_validation['fare'].median(), inplace=True)

# Store actual ages
actual_ages = X_validation['age'].values.copy()

# Simulate missing ages by setting them to NaN
X_validation_test = X_validation.copy()
X_validation_test['age'] = np.nan

# Combine with some known data to give KNN context
# We'll use a subset of the data with known ages for KNN to learn from
n_train = int(len(known_age_data) * 0.7)
indices = np.random.RandomState(42).permutation(len(known_age_data))
train_idx = indices[:n_train]
test_idx = indices[n_train:]

# Create training and test sets
X_train = X_validation.iloc[train_idx].copy()
X_test = X_validation.iloc[test_idx].copy()

actual_ages_test = X_test['age'].values.copy()
X_test_missing = X_test.copy()
X_test_missing['age'] = np.nan

# Combine train and test for imputation
X_combined = pd.concat([X_train, X_test_missing])

# Apply KNN imputation
knn_validator = KNNImputer(n_neighbors=5, weights='distance')
X_imputed_validation = knn_validator.fit_transform(X_combined)

# Extract predicted ages for test set
predicted_ages = X_imputed_validation[n_train:, knn_features.index('age')]

# Calculate Mean Absolute Error
mae = np.mean(np.abs(actual_ages_test - predicted_ages))

print(f"\nModel Validation Results:")
print(f"  Test set size: {len(test_idx)} samples")
print(f"  Mean Absolute Error (MAE): {mae:.2f} years")
print(f"  Root Mean Squared Error (RMSE): {np.sqrt(np.mean((actual_ages_test - predicted_ages) ** 2)):.2f} years")
print(f"  Mean actual age: {actual_ages_test.mean():.2f} years")
print(f"  Mean predicted age: {predicted_ages.mean():.2f} years")

# Calculate additional metrics
residuals = actual_ages_test - predicted_ages
print(f"\nResidual Statistics:")
print(f"  Mean residual: {residuals.mean():.2f} years")
print(f"  Std of residuals: {residuals.std():.2f} years")
print(f"  Min residual: {residuals.min():.2f} years")
print(f"  Max residual: {residuals.max():.2f} years")

# Store for plotting
validation_results = {
    'actual': actual_ages_test,
    'predicted': predicted_ages,
    'mae': mae,
    'residuals': residuals
}

# FINAL SUMMARY: BEFORE AND AFTER KNN IMPUTATION
print("\n" + "=" * 60)
print("FINAL SUMMARY: AGE IMPUTATION RESULTS")
print("=" * 60)

# Calculate average age BEFORE imputation (using only known ages)
age_before_imputation = titanic_original['age'].mean()  # Mean of known ages only
num_known_ages = titanic_original['age'].notna().sum()

# Calculate average age AFTER KNN imputation (using all ages)
age_after_imputation = titanic_knn['age'].mean()  # Mean including imputed ages
total_ages = len(titanic_knn)

print(f"\nBEFORE KNN Imputation:")
print(f"  • Number of known ages: {num_known_ages}")
print(f"  • Number of missing ages: {age_missing}")
print(f"  • Average age (known only): {age_before_imputation:.2f} years")

print(f"\nAFTER KNN Imputation:")
print(f"  • Total ages (all imputed): {total_ages}")
print(f"  • Number of missing ages: 0")
print(f"  • Average age (all passengers): {age_after_imputation:.2f} years")

print(f"\nChange in Average Age:")
age_difference = age_after_imputation - age_before_imputation
print(f"  • Difference: {age_difference:+.2f} years")
if abs(age_difference) < 0.5:
    print(f"  • Interpretation: Minimal change - KNN preserved the age distribution well")
elif age_difference > 0:
    print(f"  • Interpretation: Slight increase - imputed passengers tend to be slightly older")
else:
    print(f"  • Interpretation: Slight decrease - imputed passengers tend to be slightly younger")

print(f"\nImputed Ages Statistics:")
print(f"  • Mean of imputed values: {knn_imputed_ages.mean():.2f} years")
print(f"  • Median of imputed values: {knn_imputed_ages.median():.2f} years")
print(f"  • Std Dev of imputed values: {knn_imputed_ages.std():.2f} years")
print(f"  • Range: {knn_imputed_ages.min():.2f} - {knn_imputed_ages.max():.2f} years")

print(f"\nModel Performance:")
print(f"  • Mean Absolute Error (MAE): {mae:.2f} years")
print(f"  • This means predictions are off by ~{mae:.2f} years on average")

# RANDOM FOREST MODEL FOR SURVIVAL PREDICTION
print("\n" + "=" * 60)
print("RANDOM FOREST MODEL: SURVIVAL PREDICTION")
print("=" * 60)

# Prepare features for Random Forest
print("\nPreparing data for Random Forest model...")

# Use the KNN-imputed dataset
rf_data = titanic.copy()

# Select features for prediction
feature_columns = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target_column = 'survived'

# Create feature matrix
X_rf = rf_data[feature_columns].copy()

# Encode categorical variables
print("Encoding categorical variables...")
le_sex_rf = LabelEncoder()
le_embarked_rf = LabelEncoder()

X_rf['sex'] = le_sex_rf.fit_transform(X_rf['sex'].astype(str))
X_rf['embarked'] = X_rf['embarked'].fillna(X_rf['embarked'].mode()[0])  # Fill missing embarked
X_rf['embarked'] = le_embarked_rf.fit_transform(X_rf['embarked'].astype(str))

# Handle any remaining missing values in fare
X_rf['fare'].fillna(X_rf['fare'].median(), inplace=True)

# Target variable
y_rf = rf_data[target_column]

print(f"Feature matrix shape: {X_rf.shape}")
print(f"Target variable shape: {y_rf.shape}")
print(f"Features used: {feature_columns}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_rf, y_rf, test_size=0.2, random_state=42, stratify=y_rf
)

print(f"\nData split:")
print(f"  • Training set: {len(X_train)} samples")
print(f"  • Test set: {len(X_test)} samples")
print(f"  • Survival rate in training: {y_train.mean():.1%}")
print(f"  • Survival rate in test: {y_test.mean():.1%}")

# Train Random Forest model
print("\n" + "-" * 60)
print("Training Random Forest Classifier...")
print("-" * 60)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("✓ Model training complete!")

# Make predictions
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)

# Display results
print("\n" + "=" * 60)
print("MODEL PERFORMANCE METRICS")
print("=" * 60)

print("\n📊 ACCURACY SCORES:")
print(f"  • Training Accuracy: {train_accuracy:.2%}")
print(f"  • Test Accuracy: {test_accuracy:.2%}")
print(f"  • Difference: {abs(train_accuracy - test_accuracy):.2%} ", end="")
if abs(train_accuracy - test_accuracy) < 0.05:
    print("(Good - minimal overfitting)")
elif abs(train_accuracy - test_accuracy) < 0.10:
    print("(Fair - slight overfitting)")
else:
    print("(Warning - possible overfitting)")

print("\n📏 MEAN ABSOLUTE ERROR (MAE):")
print(f"  • Training MAE: {train_mae:.4f}")
print(f"  • Test MAE: {test_mae:.4f}")
print(f"  • Interpretation: On average, predictions are off by {test_mae:.4f} (0=survived, 1=died)")

# Feature importance
print("\n" + "-" * 60)
print("FEATURE IMPORTANCE RANKING")
print("-" * 60)
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nMost influential features for survival prediction:")
for idx, row in feature_importance.iterrows():
    bar_length = int(row['Importance'] * 50)
    bar = '█' * bar_length
    print(f"  {row['Feature']:12s} │ {bar} {row['Importance']:.4f}")

# Confusion matrix
print("\n" + "-" * 60)
print("CONFUSION MATRIX (Test Set)")
print("-" * 60)
cm = confusion_matrix(y_test, y_pred_test)
print("\n                 Predicted")
print("               Dead  Survived")
print(f"Actual Dead      {cm[0, 0]:3d}     {cm[0, 1]:3d}")
print(f"       Survived  {cm[1, 0]:3d}     {cm[1, 1]:3d}")

# Classification report
print("\n" + "-" * 60)
print("DETAILED CLASSIFICATION REPORT")
print("-" * 60)
print(classification_report(y_test, y_pred_test,
                            target_names=['Died', 'Survived'],
                            digits=3))

# Store results for plotting
rf_results = {
    'model': rf_model,
    'feature_importance': feature_importance,
    'y_test': y_test,
    'y_pred_test': y_pred_test,
    'confusion_matrix': cm,
    'test_accuracy': test_accuracy,
    'test_mae': test_mae
}

# Display basic statistics
print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)
print(titanic.describe())

# Check for missing values in all columns
print("\n" + "=" * 60)
print("MISSING VALUES IN ALL COLUMNS")
print("=" * 60)
print(titanic.isnull().sum())

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

# Generate correlation matrix
print("\n" + "=" * 60)
print("CORRELATION MATRIX ANALYSIS")
print("=" * 60)

# Select only numeric columns for correlation
numeric_cols = titanic.select_dtypes(include=['float64', 'int64']).columns
print(f"\nNumeric columns used for correlation: {list(numeric_cols)}\n")

# Calculate correlation matrix
correlation_matrix = titanic[numeric_cols].corr()

# Display full correlation matrix
print("Full Correlation Matrix:")
print(correlation_matrix)

# Focus on Age correlations
print("\n" + "=" * 60)
print("AGE CORRELATIONS (Sorted by Strength)")
print("=" * 60)

# Get correlations with age and sort by absolute value
age_correlations = correlation_matrix['age'].sort_values(key=abs, ascending=False)
print("\nCorrelations with Age:")
print(age_correlations)

# Identify strongest positive and negative correlations
print("\n" + "-" * 60)
print("STRONGEST CORRELATIONS WITH AGE:")
print("-" * 60)

# Exclude age's correlation with itself
age_corr_filtered = age_correlations[age_correlations.index != 'age']

if len(age_corr_filtered) > 0:
    strongest_positive = age_corr_filtered[age_corr_filtered > 0].head(3)
    strongest_negative = age_corr_filtered[age_corr_filtered < 0].head(3)

    print("\nTop 3 Positive Correlations:")
    for variable, corr in strongest_positive.items():
        print(f"  {variable}: {corr:.4f}")

    print("\nTop 3 Negative Correlations:")
    for variable, corr in strongest_negative.items():
        print(f"  {variable}: {corr:.4f}")

print("\n" + "=" * 60)
print("INTERPRETATION GUIDE")
print("=" * 60)
print("Correlation values range from -1 to +1:")
print("  +1.0 = Perfect positive correlation")
print("  -1.0 = Perfect negative correlation")
print("   0.0 = No linear correlation")
print("\nRule of thumb:")
print("  0.0 - 0.3: Weak correlation")
print("  0.3 - 0.7: Moderate correlation")
print("  0.7 - 1.0: Strong correlation")

# Create and save visualizations
print("\n" + "=" * 60)
print("GENERATING AND SAVING PLOTS")
print("=" * 60)

# 1. Comparison of Imputation Methods
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Mean vs KNN Imputation Comparison', fontsize=16, fontweight='bold')

# Plot 1: Age distribution - Mean Imputation
axes[0, 0].hist(titanic_mean['age'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 0].axvline(mean_age, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_age:.2f}')
axes[0, 0].set_xlabel('Age', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('Age Distribution - Mean Imputation', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# Plot 2: Age distribution - KNN Imputation
axes[0, 1].hist(titanic_knn['age'], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
axes[0, 1].axvline(titanic_knn['age'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {titanic_knn["age"].mean():.2f}')
axes[0, 1].set_xlabel('Age', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('Age Distribution - KNN Imputation', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# Plot 3: Imputed values only - Mean
axes[1, 0].hist(mean_imputed_ages, bins=20, edgecolor='black', alpha=0.7, color='coral')
axes[1, 0].set_xlabel('Imputed Age Values', fontsize=11)
axes[1, 0].set_ylabel('Frequency', fontsize=11)
axes[1, 0].set_title(f'Mean Imputation - Imputed Values Only\n(Std: {mean_imputed_ages.std():.2f})',
                     fontsize=12, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Imputed values only - KNN
axes[1, 1].hist(knn_imputed_ages, bins=20, edgecolor='black', alpha=0.7, color='gold')
axes[1, 1].set_xlabel('Imputed Age Values', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontsize=11)
axes[1, 1].set_title(f'KNN Imputation - Imputed Values Only\n(Std: {knn_imputed_ages.std():.2f})',
                     fontsize=12, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
comparison_path = os.path.join(save_dir, 'imputation_comparison.png')
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {comparison_path}")
plt.close()

# 2. Box plot comparison
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
data_to_plot = [
    titanic_original['age'].dropna(),
    mean_imputed_ages,
    knn_imputed_ages
]
box_plot = ax.boxplot(data_to_plot, tick_labels=['Original\n(Known Ages)', 'Mean\nImputation', 'KNN\nImputation'],
                      patch_artist=True, showmeans=True)

# Color the boxes
colors = ['lightblue', 'coral', 'gold']
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)

ax.set_ylabel('Age (years)', fontsize=12)
ax.set_title('Age Distribution Comparison: Original vs Imputation Methods', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
boxplot_path = os.path.join(save_dir, 'imputation_boxplot.png')
plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {boxplot_path}")
plt.close()

# 3. Actual vs Predicted Ages (KNN Validation)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('KNN Model Validation: Actual vs Predicted Ages', fontsize=16, fontweight='bold')

# Scatter plot: Actual vs Predicted
axes[0].scatter(validation_results['actual'], validation_results['predicted'],
                alpha=0.6, edgecolors='black', s=50)
axes[0].plot([0, 80], [0, 80], 'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Age (years)', fontsize=12)
axes[0].set_ylabel('Predicted Age (years)', fontsize=12)
axes[0].set_title(f'Actual vs Predicted\nMAE = {validation_results["mae"]:.2f} years',
                  fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_xlim([0, 80])
axes[0].set_ylim([0, 80])

# Residual plot
axes[1].scatter(validation_results['predicted'], validation_results['residuals'],
                alpha=0.6, edgecolors='black', s=50, color='purple')
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
axes[1].set_xlabel('Predicted Age (years)', fontsize=12)
axes[1].set_ylabel('Residual (Actual - Predicted)', fontsize=12)
axes[1].set_title(
    f'Residual Plot\nMean = {validation_results["residuals"].mean():.2f}, Std = {validation_results["residuals"].std():.2f}',
    fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
validation_path = os.path.join(save_dir, 'knn_validation.png')
plt.savefig(validation_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {validation_path}")
plt.close()

# 4. Residual distribution
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.hist(validation_results['residuals'], bins=30, edgecolor='black', alpha=0.7, color='teal')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax.axvline(validation_results['residuals'].mean(), color='orange', linestyle='--', linewidth=2,
           label=f'Mean Residual = {validation_results["residuals"].mean():.2f}')
ax.set_xlabel('Residual (Actual - Predicted Age)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
residual_path = os.path.join(save_dir, 'residual_distribution.png')
plt.savefig(residual_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {residual_path}")
plt.close()

# 5. Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', square=True, linewidths=1)
plt.title('Correlation Matrix Heatmap - Titanic Dataset', fontsize=16, fontweight='bold')
plt.tight_layout()
heatmap_path = os.path.join(save_dir, 'correlation_heatmap.png')
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {heatmap_path}")
plt.close()

# 2. Age Correlations Bar Plot
plt.figure(figsize=(10, 6))
age_corr_plot = age_correlations[age_correlations.index != 'age']
colors = ['green' if x > 0 else 'red' for x in age_corr_plot.values]
age_corr_plot.plot(kind='barh', color=colors, edgecolor='black')
plt.xlabel('Correlation Coefficient', fontsize=12)
plt.ylabel('Variables', fontsize=12)
plt.title('Correlations with Age', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
age_corr_path = os.path.join(save_dir, 'age_correlations.png')
plt.savefig(age_corr_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {age_corr_path}")
plt.close()

# 3. Age Distribution (Before and After Imputation would need before data)
plt.figure(figsize=(10, 6))
plt.hist(titanic['age'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Age Distribution (After Mean Imputation)', fontsize=14, fontweight='bold')
plt.axvline(mean_age, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_age:.2f}')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
age_dist_path = os.path.join(save_dir, 'age_distribution.png')
plt.savefig(age_dist_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {age_dist_path}")
plt.close()

# 4. Scatter plots for strongest correlations with age
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Age vs Other Variables (Strongest Correlations)', fontsize=16, fontweight='bold')

# Get top correlations (excluding age itself)
top_vars = age_corr_filtered.head(4).index

for idx, var in enumerate(top_vars):
    row = idx // 2
    col = idx % 2
    axes[row, col].scatter(titanic[var], titanic['age'], alpha=0.5, edgecolors='black')
    axes[row, col].set_xlabel(var, fontsize=11)
    axes[row, col].set_ylabel('Age', fontsize=11)
    axes[row, col].set_title(f'Age vs {var} (r={age_correlations[var]:.3f})', fontsize=12)
    axes[row, col].grid(alpha=0.3)

plt.tight_layout()
scatter_path = os.path.join(save_dir, 'age_scatter_plots.png')
plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {scatter_path}")
plt.close()

# 5. Missing values visualization
plt.figure(figsize=(10, 6))
missing_data = titanic.isnull().sum().sort_values(ascending=False)
missing_data = missing_data[missing_data > 0]
if len(missing_data) > 0:
    missing_data.plot(kind='bar', color='coral', edgecolor='black')
    plt.xlabel('Columns', fontsize=12)
    plt.ylabel('Number of Missing Values', fontsize=12)
    plt.title('Missing Values by Column (After Age Imputation)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    missing_path = os.path.join(save_dir, 'missing_values.png')
    plt.savefig(missing_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {missing_path}")
    plt.close()

print("\n" + "=" * 60)
print(f"All plots saved to '{save_dir}/' directory")
print("=" * 60)

# RANDOM FOREST VISUALIZATIONS
print("\n" + "=" * 60)
print("GENERATING RANDOM FOREST VISUALIZATIONS")
print("=" * 60)

# 1. Feature Importance Plot
plt.figure(figsize=(10, 6))
colors_importance = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
bars = plt.barh(feature_importance['Feature'], feature_importance['Importance'],
                color=colors_importance, edgecolor='black', linewidth=1.5)
plt.xlabel('Importance Score', fontsize=13, fontweight='bold')
plt.ylabel('Features', fontsize=13, fontweight='bold')
plt.title('Random Forest Feature Importance\n(Most Influential Factors for Survival)',
          fontsize=15, fontweight='bold', pad=20)
plt.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, feature_importance['Importance'])):
    plt.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
             f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
importance_path = os.path.join(save_dir, 'rf_feature_importance.png')
plt.savefig(importance_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {importance_path}")
plt.close()

# 2. Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(rf_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
            cbar_kws={'label': 'Count'}, linewidths=2, linecolor='black',
            xticklabels=['Died', 'Survived'], yticklabels=['Died', 'Survived'],
            annot_kws={'size': 16, 'fontweight': 'bold'})
plt.xlabel('Predicted Class', fontsize=13, fontweight='bold')
plt.ylabel('Actual Class', fontsize=13, fontweight='bold')
plt.title(
    f'Confusion Matrix - Random Forest\nAccuracy: {rf_results["test_accuracy"]:.2%} | MAE: {rf_results["test_mae"]:.4f}',
    fontsize=15, fontweight='bold', pad=20)
plt.tight_layout()
cm_path = os.path.join(save_dir, 'rf_confusion_matrix.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {cm_path}")
plt.close()

# 3. Prediction Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Random Forest Predictions vs Actual Outcomes', fontsize=16, fontweight='bold')

# Actual distribution
actual_counts = y_test.value_counts().sort_index()
axes[0].bar(['Died', 'Survived'], actual_counts.values,
            color=['#e74c3c', '#2ecc71'], edgecolor='black', linewidth=2, alpha=0.8)
axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
axes[0].set_title('Actual Survival Distribution\n(Test Set)', fontsize=13, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(actual_counts.values):
    axes[0].text(i, v + 2, str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')

# Predicted distribution
pred_counts = pd.Series(y_pred_test).value_counts().sort_index()
axes[1].bar(['Died', 'Survived'], pred_counts.values,
            color=['#e74c3c', '#2ecc71'], edgecolor='black', linewidth=2, alpha=0.8)
axes[1].set_ylabel('Count', fontsize=12, fontweight='bold')
axes[1].set_title('Predicted Survival Distribution\n(Test Set)', fontsize=13, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(pred_counts.values):
    axes[1].text(i, v + 2, str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
dist_path = os.path.join(save_dir, 'rf_prediction_distribution.png')
plt.savefig(dist_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {dist_path}")
plt.close()

# 4. Performance Metrics Summary Plot
fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['Accuracy', 'Precision\n(Survived)', 'Recall\n(Survived)', 'F1-Score\n(Survived)']
from sklearn.metrics import precision_score, recall_score, f1_score

values = [
    rf_results['test_accuracy'],
    precision_score(y_test, y_pred_test, pos_label=1),
    recall_score(y_test, y_pred_test, pos_label=1),
    f1_score(y_test, y_pred_test, pos_label=1)
]

bars = ax.bar(metrics, values, color=['#3498db', '#e74c3c', '#f39c12', '#9b59b6'],
              edgecolor='black', linewidth=2, alpha=0.8)
ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_ylim([0, 1])
ax.set_title('Random Forest Performance Metrics Summary', fontsize=15, fontweight='bold', pad=20)
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='50% Baseline')
ax.grid(axis='y', alpha=0.3)
ax.legend()

# Add value labels
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
            f'{val:.3f}\n({val:.1%})', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

plt.tight_layout()
metrics_path = os.path.join(save_dir, 'rf_metrics_summary.png')
plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {metrics_path}")
plt.close()

print("\n" + "=" * 60)
print(f"All Random Forest plots saved to '{save_dir}/' directory")
print("=" * 60)

# FINAL SUMMARY
print("\n" + "🎯 " * 30)
print("=" * 60)
print("              KEY TAKEAWAYS & SUMMARY")
print("=" * 60)
print("🎯 " * 30)

print("\n1️⃣  DATA IMPUTATION:")
print(f"   • Original dataset: {age_missing} missing ages ({age_missing_percent:.1f}%)")
print(f"   • KNN Imputation MAE: {mae:.2f} years")
print(f"   • Average age changed from {age_before_imputation:.2f} to {age_after_imputation:.2f} years")

print("\n2️⃣  RANDOM FOREST MODEL:")
print(f"   • Test Accuracy: {rf_results['test_accuracy']:.2%}")
print(f"   • Mean Absolute Error: {rf_results['test_mae']:.4f}")
print(
    f"   • Most important feature: {feature_importance.iloc[0]['Feature']} ({feature_importance.iloc[0]['Importance']:.4f})")

print("\n3️⃣  VISUALIZATIONS CREATED:")
plot_files = [
    'imputation_comparison.png',
    'imputation_boxplot.png',
    'knn_validation.png',
    'residual_distribution.png',
    'correlation_heatmap.png',
    'age_correlations.png',
    'age_distribution.png',
    'age_scatter_plots.png',
    'missing_values.png',
    'rf_feature_importance.png',
    'rf_confusion_matrix.png',
    'rf_prediction_distribution.png',
    'rf_metrics_summary.png'
]
print(f"   • Total plots saved: {len(plot_files)}")
print(f"   • Location: {save_dir}/")

print("\n" + "=" * 60)
print("           ✨ ANALYSIS COMPLETE ✨")
print("=" * 60)
print("\n")

WEEK 4 

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Create the DataFrame
molecular_weight = [180, 250, 80, 300, 150, 400, 90, 200, 130, 275, 135, 220]
hydrogen_bond_donors = [5, 2, 1, 1, 4, 3, 0, 2, 3, 1, 1, 3]
hydrogen_bond_acceptors = [6, 3, 2, 2, 5, 4, 1, 3, 4, 2, 3, 2]
water_solubility = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]

molecule_names = [f'Molecule {i+1}' for i in range(len(molecular_weight))]

df = pd.DataFrame({
    'Molecule': molecule_names,
    'Molecular Weight': molecular_weight,
    'Hydrogen Bond Donors': hydrogen_bond_donors,
    'Hydrogen Bond Acceptors': hydrogen_bond_acceptors,
    'Water Solubility': water_solubility
})

# Split into features (X) and target (y)
X = df[['Molecular Weight', 'Hydrogen Bond Donors', 'Hydrogen Bond Acceptors']]
y = df['Water Solubility']

# Create and train the decision tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Create the visualization
plt.figure(figsize=(20, 10))
plot_tree(model,
          feature_names=['Molecular Weight', 'Hydrogen Bond Donors', 'Hydrogen Bond Acceptors'],
          class_names=['Not Soluble', 'Soluble'],
          filled=True,
          rounded=True,
          fontsize=12)

plt.title('Decision Tree for Water Solubility Prediction', fontsize=16, fontweight='bold', pad=20)

# Save the figure
plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Decision tree visualization saved as 'decision_tree_visualization.png'")

# Display the plot
plt.show()

