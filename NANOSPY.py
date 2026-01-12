import sys
import os
import ctypes
import time
from datetime import datetime
import math
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure

from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Optional ML
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# -----------------------------
# === USER CONFIGURATION ===
# -----------------------------
# Edit these defaults if needed
DEFAULT_DLL_PATH = r"C:\Users\hammo\OneDrive\Desktop\Nanos+\Nanos+\Software Paket\PR0351InitTool\PR0351DLL.dll"
DEFAULT_COM_PORT = 10  # adjust to your COM number
# Approximate wavelengths for the 16 spectral channels in nm:
DEFAULT_WAVELENGTHS = np.array([410, 430, 450, 470, 490, 510, 530, 550,
                                570, 590, 610, 630, 650, 670, 690, 710])

SAMPLE_CSV_PATH = "nanos_samples.csv"  # saved labeled samples

# -----------------------------
# === DLL wrapper utilities ===
# -----------------------------
class PR0351Wrapper:
    """
    Minimal wrapper around PR0351 DLL via ctypes.
    Exposes:
     - create/connect/disconnect/delete
     - start measurement / is ready / get results
     - set CTA per pixel (if DLL supports)
    """
    def __init__(self, dll_path: str, com_port: int):
        self.dll_path = dll_path
        self.com_port = com_port
        self.dll = None
        self.instance = ctypes.c_ulong()
        self.loaded = False
        self._setup_structs_and_funcs = False

    def load(self):
        if not os.path.exists(self.dll_path):
            raise FileNotFoundError(f"PR0351 DLL not found at: {self.dll_path}")
        try:
            self.dll = ctypes.WinDLL(self.dll_path)
            self._define_funcs()
            self.loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load DLL: {e}")

    def _define_funcs(self):
        # Basic function prototypes based on header we have used earlier
        # If your DLL uses different names or signatures, adjust here.
        try:
            self.dll.PR0351_CreateInstance.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_ulong)]
            self.dll.PR0351_CreateInstance.restype = ctypes.c_int

            self.dll.PR0351_Connect.argtypes = [ctypes.c_ulong]
            self.dll.PR0351_Connect.restype = ctypes.c_int

            self.dll.PR0351_Disconnect.argtypes = [ctypes.c_ulong]
            self.dll.PR0351_Disconnect.restype = ctypes.c_int

            self.dll.PR0351_DeleteInstance.argtypes = [ctypes.c_ulong]
            self.dll.PR0351_DeleteInstance.restype = ctypes.c_int

            self.dll.PR0351_StartMeasurement.argtypes = [ctypes.c_ulong, ctypes.c_int]
            self.dll.PR0351_StartMeasurement.restype = ctypes.c_int

            self.dll.PR0351_IsMeasurementReady.argtypes = [ctypes.c_ulong, ctypes.POINTER(ctypes.c_int)]
            self.dll.PR0351_IsMeasurementReady.restype = ctypes.c_int

            # Full sensor struct expected (we only read first metapixel later)
            # define minimal struct types for reading
            # constants
            PIXELS_PER_METAPIXEL = 16
            METAPIXELS_PER_ROW = 10
            ROWS_PER_SENSOR = 10

            class METAPIXEL_DATA(ctypes.Structure):
                _fields_ = [("rawMetapixelPixel", ctypes.c_float * PIXELS_PER_METAPIXEL)]
            class METAPIXEL_ROW_DATA(ctypes.Structure):
                _fields_ = [("metaPixel", METAPIXEL_DATA * METAPIXELS_PER_ROW)]
            class FULL_SENSOR_DATA(ctypes.Structure):
                _fields_ = [("measureRow", METAPIXEL_ROW_DATA * ROWS_PER_SENSOR)]

            self._METAPIXEL_DATA = METAPIXEL_DATA
            self._FULL_SENSOR_DATA = FULL_SENSOR_DATA

            self.dll.PR0351_GetRawMeasurementResults.argtypes = [ctypes.c_ulong, ctypes.POINTER(FULL_SENSOR_DATA)]
            self.dll.PR0351_GetRawMeasurementResults.restype = ctypes.c_int

            # CTA setter (may not exist)
            try:
                self.dll.PR0351_SetMeasurementParameterGainCTAForPixel.argtypes = [ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p)]
                self.dll.PR0351_SetMeasurementParameterGainCTAForPixel.restype = ctypes.c_int
            except Exception:
                # ignore if not present
                pass

            self._setup_structs_and_funcs = True
        except Exception as e:
            raise RuntimeError(f"Failed to set up DLL function prototypes: {e}")

    def create_and_connect(self):
        if not self._setup_structs_and_funcs:
            raise RuntimeError("DLL not loaded or prototypes not set")
        # Create instance
        res = self.dll.PR0351_CreateInstance(self.com_port, ctypes.byref(self.instance))
        if res != 0:
            raise RuntimeError(f"PR0351_CreateInstance failed with code {res}")
        # Connect
        res = self.dll.PR0351_Connect(self.instance)
        if res != 0:
            raise RuntimeError(f"PR0351_Connect failed with code {res}")
        return True

    def disconnect_and_delete(self):
        if not self._setup_structs_and_funcs:
            return
        try:
            self.dll.PR0351_Disconnect(self.instance)
        except Exception:
            pass
        try:
            self.dll.PR0351_DeleteInstance(self.instance)
        except Exception:
            pass

    def take_full_measurement(self, timeout_s=5.0):
        """
        Start measurement and return a numpy array of length 16 containing first metapixel.
        Returns None on failure.
        """
        if not self._setup_structs_and_funcs:
            raise RuntimeError("DLL functions not ready")
        MEAS_TYPE_FULL = 1
        res = self.dll.PR0351_StartMeasurement(self.instance, MEAS_TYPE_FULL)
        if res != 0:
            # non-zero means failed to start
            return None, f"StartMeasurement failed ({res})"
        # wait until ready
        ready = ctypes.c_int(0)
        t0 = time.time()
        while True:
            self.dll.PR0351_IsMeasurementReady(self.instance, ctypes.byref(ready))
            if ready.value:
                break
            if time.time() - t0 > timeout_s:
                return None, "Measurement timeout"
            time.sleep(0.02)
        data = self._FULL_SENSOR_DATA()
        res = self.dll.PR0351_GetRawMeasurementResults(self.instance, ctypes.byref(data))
        if res != 0:
            return None, f"GetRawMeasurementResults failed ({res})"
        # extract first metapixel first row (16 channels)
        raw = [float(x) for x in data.measureRow[0].metaPixel[0].rawMetapixelPixel]
        return np.array(raw, dtype=float), None

# -----------------------------
# === Spectral / Indices utils ===
# -----------------------------
def wavelength_to_rgb(wavelength):
    """Convert a wavelength (nm) to an approximate RGB tuple (0..1)."""
    gamma = 0.8
    R = G = B = 0.0
    if 380 <= wavelength < 440:
        R = -(wavelength - 440) / (440 - 380)
        B = 1.0
    elif 440 <= wavelength < 490:
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif 490 <= wavelength < 510:
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength < 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
    elif 580 <= wavelength < 645:
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
    elif 645 <= wavelength <= 780:
        R = 1.0
    # intensity factor
    if 380 <= wavelength < 420:
        factor = 0.3 + 0.7*(wavelength - 380)/(420 - 380)
    elif 420 <= wavelength < 645:
        factor = 1.0
    elif 645 <= wavelength <= 780:
        factor = 0.3 + 0.7*(780 - wavelength)/(780 - 645)
    else:
        factor = 0.0
    R = ((R * factor) ** gamma) if R > 0 else 0.0
    G = ((G * factor) ** gamma) if G > 0 else 0.0
    B = ((B * factor) ** gamma) if B > 0 else 0.0
    return (R, G, B)

def compute_indices(reflectances, wavelengths=DEFAULT_WAVELENGTHS):
    """Compute selected vegetation indices from reflectance vector (length 16)."""
    # pick approximate channel indices by nearest wavelength
    def pick_idx(target_nm):
        idx = int(np.argmin(np.abs(wavelengths - target_nm)))
        return idx

    # approximate picks
    blue = reflectances[pick_idx(440)]
    green = reflectances[pick_idx(510)]
    red = reflectances[pick_idx(630)]
    rededge = reflectances[pick_idx(705)]  # nearest to 700-710
    nir = reflectances[pick_idx(800)] if wavelengths.max() >= 800 else None

    # safe arithmetic helpers
    eps = 1e-9
    def safe_div(a,b): return a/(b+eps)

    results = {}
    # GLI
    results['GLI'] = safe_div((2*green - red - blue), (2*green + red + blue))
    # NGRDI = (G-R)/(G+R)
    results['NGRDI'] = safe_div((green - red), (green + red))
    # VARI = (G - B) / (G + R - B)
    results['VARI'] = safe_div((green - blue), (green + red - blue))
    # pseudo NDVI using rededge if NIR not present
    if nir is not None:
        results['NDVI'] = safe_div((nir - red), (nir + red))
    else:
        results['pseudo_NDVI'] = safe_div((rededge - red), (rededge + red))
    # PSRI ~ (R - B)/RedEdge if rededge available
    if rededge is not None and abs(rededge) > eps:
        results['PSRI'] = safe_div((red - blue), rededge)
    else:
        results['PSRI'] = None
    # additionally return the channel picks
    results['idx_blue'] = pick_idx(440)
    results['idx_green'] = pick_idx(510)
    results['idx_red'] = pick_idx(630)
    results['idx_rededge'] = pick_idx(705)
    return results

# -----------------------------
# === Simple rule based classifier ===
# -----------------------------
def rule_based_classify(indices):
    """
    Smart multi-index rule-based plant health assessment
    Returns:
      - Healthy
      - Stressed
      - Possible disease / pest
      - Uncertain
    """

    gli = indices.get('GLI', 0)
    vari = indices.get('VARI', 0)
    ngrdi = indices.get('NGRDI', 0)
    pndvi = indices.get('pseudo_NDVI', 0)
    psri = indices.get('PSRI', 0)

    score = 0

    # --- GLI ---
    if gli > 0.15:
        score += 2
    elif gli > 0.05:
        score += 1
    else:
        score -= 1

    # --- VARI ---
    if vari > 0.10:
        score += 2
    elif vari > 0:
        score += 1
    else:
        score -= 1

    # --- NGRDI ---
    if ngrdi > 0.10:
        score += 1
    elif ngrdi < 0:
        score -= 1

    # --- pseudo NDVI ---
    if pndvi > 0.20:
        score += 2
    elif pndvi > 0.10:
        score += 1
    else:
        score -= 1

    # --- PSRI (inverse logic) ---
    if psri > 0.4:
        score -= 3
    elif psri > 0.2:
        score -= 1

    # --- Final decision ---
    if score >= 5:
        return "Healthy"
    elif score >= 2:
        return "Mild stress"
    elif score >= -1:
        return "Stressed"
    elif score < -1:
        return "Possible disease / pest"
    else:
        return "Uncertain"


# -----------------------------
# === GUI Application ===
# -----------------------------
class NanosApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NANOS+ EVA - Plant Health Tool")
        self.setMinimumSize(1100, 700)
        # state
        self.dll_path = DEFAULT_DLL_PATH
        self.com_port = DEFAULT_COM_PORT
        self.wavelengths = DEFAULT_WAVELENGTHS.copy()
        self.wrapper = None
        self.connected = False
        self.dark = None
        self.white = None
        self.latest_raw = None
        self.latest_reflectance = None
        self.indices = None
        self.samples_file = Path(SAMPLE_CSV_PATH)
        # ML model
        self.model = None
        # setup UI
        self._build_ui()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Left: controls
        left = QtWidgets.QFrame()
        left.setMaximumWidth(360)
        left_layout = QtWidgets.QVBoxLayout(left)

        # DLL path and COM
        dll_label = QtWidgets.QLabel("PR0351 DLL Path:")
        self.dll_edit = QtWidgets.QLineEdit(self.dll_path)
        com_label = QtWidgets.QLabel("COM Port (number):")
        self.com_edit = QtWidgets.QSpinBox()
        self.com_edit.setRange(1, 256)
        self.com_edit.setValue(self.com_port)

        load_button = QtWidgets.QPushButton("Load DLL & Connect")
        load_button.clicked.connect(self.on_connect_clicked)

        disconnect_button = QtWidgets.QPushButton("Disconnect")
        disconnect_button.clicked.connect(self.on_disconnect_clicked)

        left_layout.addWidget(dll_label)
        left_layout.addWidget(self.dll_edit)
        left_layout.addWidget(com_label)
        left_layout.addWidget(self.com_edit)
        left_layout.addWidget(load_button)
        left_layout.addWidget(disconnect_button)
        left_layout.addSpacing(10)

        # Calibration
        calib_label = QtWidgets.QLabel("<b>Calibration</b>")
        left_layout.addWidget(calib_label)
        dark_btn = QtWidgets.QPushButton("Measure DARK (cover sensor)")
        dark_btn.clicked.connect(self.on_dark_clicked)
        white_btn = QtWidgets.QPushButton("Measure WHITE (white ref)")
        white_btn.clicked.connect(self.on_white_clicked)
        left_layout.addWidget(dark_btn)
        left_layout.addWidget(white_btn)
        left_layout.addSpacing(10)

        # Measurement controls
        meas_label = QtWidgets.QLabel("<b>Measurement</b>")
        left_layout.addWidget(meas_label)
        measure_btn = QtWidgets.QPushButton("Take Measurement")
        measure_btn.clicked.connect(self.on_measure_clicked)
        save_btn = QtWidgets.QPushButton("Save Latest Sample (CSV)")
        save_btn.clicked.connect(self.on_save_sample)
        left_layout.addWidget(measure_btn)
        left_layout.addWidget(save_btn)

        # Labeling
        left_layout.addSpacing(10)
        left_layout.addWidget(QtWidgets.QLabel("<b>Label current sample</b>"))
        self.label_combo = QtWidgets.QComboBox()
        self.label_combo.addItems(["Healthy", "Stressed", "Pest", "Nutrient deficiency", "Other"])
        left_layout.addWidget(self.label_combo)
        save_label_btn = QtWidgets.QPushButton("Save & Label Sample")
        save_label_btn.clicked.connect(self.on_save_labeled_sample)
        left_layout.addWidget(save_label_btn)

        # Classifier
        left_layout.addSpacing(10)
        left_layout.addWidget(QtWidgets.QLabel("<b>Classifier</b>"))
        self.train_btn = QtWidgets.QPushButton("Train Classifier (sklearn required)")
        self.train_btn.clicked.connect(self.on_train_model)
        self.predict_btn = QtWidgets.QPushButton("Predict Current Sample")
        self.predict_btn.clicked.connect(self.on_predict_sample)
        left_layout.addWidget(self.train_btn)
        left_layout.addWidget(self.predict_btn)

        # Export / Load CSV
        left_layout.addSpacing(10)
        load_samples_btn = QtWidgets.QPushButton("Open Samples CSV")
        load_samples_btn.clicked.connect(self.on_open_samples)
        left_layout.addWidget(load_samples_btn)

        # Mode: live DLL or file
        left_layout.addSpacing(10)
        self.mode_label = QtWidgets.QLabel("Mode:")
        left_layout.addWidget(self.mode_label)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["DLL (live)", "File mode (CSV)"])
        left_layout.addWidget(self.mode_combo)

        # Status area
        left_layout.addStretch()
        self.status_text = QtWidgets.QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(160)
        left_layout.addWidget(self.status_text)

        layout.addWidget(left)

        # Right side: plots and info
        right = QtWidgets.QFrame()
        right_layout = QtWidgets.QVBoxLayout(right)

        # Matplotlib figure for spectrum (bar)
        self.fig = Figure(figsize=(6,4))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

        # Info labels and indices table
        info_layout = QtWidgets.QHBoxLayout()
        self.indices_table = QtWidgets.QTableWidget(0,2)
        self.indices_table.setHorizontalHeaderLabels(["Index","Value"])
        self.indices_table.horizontalHeader().setStretchLastSection(True)
        info_layout.addWidget(self.indices_table)

        # Right-hand status
        self.result_label = QtWidgets.QLabel("<b>Prediction: N/A</b>")
        info_layout.addWidget(self.result_label)
        right_layout.addLayout(info_layout)

        layout.addWidget(right, 1)

        # finalize UI
        self.show()
        self.log("Application started. Choose DLL path and COM port, then 'Load DLL & Connect' (or switch to File mode).")

    # --- Logging ---
    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.status_text.append(f"[{ts}] {msg}")

    # --- connect/disconnect ---
    def on_connect_clicked(self):
        self.dll_path = self.dll_edit.text().strip()
        self.com_port = int(self.com_edit.value())
        self.mode_combo.setCurrentIndex(0)  # set to DLL mode
        self.log(f"Loading DLL: {self.dll_path}, COM {self.com_port} ...")
        try:
            self.wrapper = PR0351Wrapper(self.dll_path, self.com_port)
            self.wrapper.load()
            self.wrapper.create_and_connect()
            self.connected = True
            self.log("✅ Connected to device via DLL.")
        except Exception as e:
            self.log(f"❌ Connect failed: {e}")
            self.wrapper = None
            self.connected = False
            QtWidgets.QMessageBox.critical(self, "Connect error", str(e))

    def on_disconnect_clicked(self):
        if self.wrapper:
            try:
                self.wrapper.disconnect_and_delete()
                self.log("Disconnected from device.")
            except Exception as e:
                self.log(f"Error during disconnect: {e}")
        self.connected = False
        self.wrapper = None

    # --- calibration ---
    def on_dark_clicked(self):
        if self.mode_combo.currentText().startswith("File"):
            QtWidgets.QMessageBox.information(self, "Info", "In File mode, dark is read from file or you can set dark manually.")
            return
        if not self.connected:
            QtWidgets.QMessageBox.warning(self, "Warning", "Not connected to device.")
            return
        self.log("Measuring DARK (cover sensor)...")
        raw, err = self.wrapper.take_full_measurement()
        if err:
            self.log(f"Dark measurement failed: {err}")
            QtWidgets.QMessageBox.critical(self, "Error", err)
            return
        self.dark = raw
        self.log("Dark stored.")

    def on_white_clicked(self):
        if self.mode_combo.currentText().startswith("File"):
            QtWidgets.QMessageBox.information(self, "Info", "In File mode, white is read from file or you can set white manually.")
            return
        if not self.connected:
            QtWidgets.QMessageBox.warning(self, "Warning", "Not connected to device.")
            return
        self.log("Measuring WHITE reference (point to white panel)...")
        raw, err = self.wrapper.take_full_measurement()
        if err:
            self.log(f"White measurement failed: {err}")
            QtWidgets.QMessageBox.critical(self, "Error", err)
            return
        self.white = raw
        self.log("White reference stored.")

    # --- measurement ---
    def on_measure_clicked(self):
        mode = self.mode_combo.currentText()
        if mode.startswith("File"):
            # open CSV or ask user to pick samples file to display last row
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open sample CSV", ".", "CSV Files (*.csv);;All Files (*)")
            if not fname:
                return
            samples = self._load_samples_csv(fname)
            if len(samples) == 0:
                QtWidgets.QMessageBox.information(self, "Info", "No samples found in file.")
                return
            last = samples[-1]
            raw = np.array(last['raw'])
            self._process_raw_and_plot(raw)
            self.log(f"Loaded last sample from {fname}.")
            return

        if not self.connected or not self.wrapper:
            QtWidgets.QMessageBox.warning(self, "Warning", "Not connected to device.")
            return
        raw, err = self.wrapper.take_full_measurement()
        if err:
            self.log(f"Measurement error: {err}")
            QtWidgets.QMessageBox.critical(self, "Measurement error", err)
            return
        self._process_raw_and_plot(raw)

    def _process_raw_and_plot(self, raw):
        """
        raw: np.array length 16 (float)
        compute reflectance if dark/white present, compute indices, and update plot/table.
        """
        self.latest_raw = raw
        reflectance = None
        if self.dark is not None and self.white is not None:
            # avoid division by zero
            denom = (self.white - self.dark).astype(np.float64)
            denom[denom == 0] = 1.0
            reflectance = (raw - self.dark) / denom
            reflectance = np.clip(reflectance, 0.0, 2.0)
            self.latest_reflectance = reflectance
            self.log("Reflectance computed from dark & white.")
        else:
            # raw as relative "reflectance"
            reflectance = raw / (np.max(raw) + 1e-9)
            self.latest_reflectance = reflectance
            self.log("No dark/white calibration: using normalized raw as proxy reflectance.")

        # compute indices
        indices = compute_indices(self.latest_reflectance, wavelengths=self.wavelengths)
        self.indices = indices

        # update table
        self._update_indices_table(indices)
        # update plot
        self._update_spectrum_plot(self.latest_reflectance)
        self.log("Measurement displayed.")

    def _update_indices_table(self, indices):
        keys = [k for k in indices.keys() if not k.startswith('idx_')]
        self.indices_table.setRowCount(len(keys))
        for i,k in enumerate(keys):
            val = indices[k]
            item_k = QtWidgets.QTableWidgetItem(k)
            item_v = QtWidgets.QTableWidgetItem(str(np.round(val,4)) if val is not None else "N/A")
            self.indices_table.setItem(i,0,item_k)
            self.indices_table.setItem(i,1,item_v)

    def _update_spectrum_plot(self, reflectance):
        self.ax.clear()
        x = self.wavelengths
        # colors
        colors = [wavelength_to_rgb(wl) for wl in x]
        bars = self.ax.bar(x, reflectance, color=colors, width=(x[1]-x[0])*0.9)
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Reflectance (rel.)")
        self.ax.set_title("Spectrum (16 channels)")
        self.ax.set_xlim(self.wavelengths.min()-10, self.wavelengths.max()+10)
        self.ax.set_ylim(0, max(1.0, float(np.max(reflectance))*1.2))
        self.canvas.draw()

    # --- saving sample ---
    def on_save_sample(self):
        if self.latest_raw is None:
            QtWidgets.QMessageBox.information(self, "Info", "No sample to save. Take measurement first.")
            return
        label = "Unlabeled"
        self._append_sample_csv(self.latest_raw, self.latest_reflectance, self.indices, label)
        self.log(f"Saved sample (unlabeled) to {self.samples_file}")

    def on_save_labeled_sample(self):
        if self.latest_raw is None:
            QtWidgets.QMessageBox.information(self, "Info", "No sample to save. Take measurement first.")
            return
        label = self.label_combo.currentText()
        self._append_sample_csv(self.latest_raw, self.latest_reflectance, self.indices, label)
        self.log(f"Saved labeled sample '{label}' to {self.samples_file}")

    def _append_sample_csv(self, raw, reflectance, indices, label):
        # CSV columns: timestamp, label, raw0..raw15, refl0..refl15, idx_GLI, idx_NGRDI, ...
        hdr_exists = self.samples_file.exists()
        with open(self.samples_file, "a", newline='') as f:
            writer = csv.writer(f)
            if not hdr_exists:
                header = ["timestamp","label"] + [f"raw_{i}" for i in range(len(raw))] + \
                         [f"refl_{i}" for i in range(len(reflectance))] + list(indices.keys())
                writer.writerow(header)
            row = [datetime.now().isoformat(), label] + list(raw) + list(reflectance) + [indices[k] for k in indices.keys()]
            writer.writerow(row)

    def _load_samples_csv(self, fname):
        samples = []
        with open(fname, newline='') as f:
            r = csv.reader(f)
            header = next(r)
            for row in r:
                rec = {}
                rec['timestamp'] = row[0]
                rec['label'] = row[1]
                # raw 16
                raws = [float(x) for x in row[2:2+16]]
                rec['raw'] = raws
                # reflectances next 16
                refl = [float(x) for x in row[2+16:2+32]]
                rec['refl'] = refl
                # remaining indices
                rest = row[2+32:]
                # we won't parse them by name here
                samples.append(rec)
        return samples

    def on_open_samples(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Samples CSV", ".", "CSV files (*.csv);;All Files (*)")
        if not fname:
            return
        samples = self._load_samples_csv(fname)
        QtWidgets.QMessageBox.information(self, "Samples loaded", f"Loaded {len(samples)} samples from {fname}")

    # --- basic ML training/predicting ---
    def on_train_model(self):
        if not SKLEARN_AVAILABLE:
            QtWidgets.QMessageBox.warning(self, "sklearn missing", "scikit-learn is not installed in this environment. Install it to enable model training.")
            return
        if not self.samples_file.exists():
            QtWidgets.QMessageBox.information(self, "Info", "No samples CSV found. Save labeled samples first.")
            return
        # load CSV and assemble features/labels
        X = []
        y = []
        with open(self.samples_file, newline='') as f:
            r = csv.reader(f)
            header = next(r)
            for row in r:
                label = row[1]
                rawvals = [float(x) for x in row[2:2+16]]
                refl = [float(x) for x in row[18:18+16]]
                # indices start at column 34 -> parse a few named indices present if any
                # For simplicity take GLI & pseudo_NDVI if present by name
                # we'll compute features ourselves from reflectance
                feats = []
                feats.extend(refl)  # 16 refl channels as features
                X.append(feats)
                y.append(label)
        if len(X) < 5:
            QtWidgets.QMessageBox.information(self, "Info", f"Need at least 5 labeled samples to train; found {len(X)}.")
            return
        X = np.array(X)
        y = np.array(y)
        # split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        self.model = clf
        QtWidgets.QMessageBox.information(self, "Training completed", f"Trained RandomForest (test accuracy ~ {acc:.2f}).")
        self.log(f"Model trained. Test acc ~ {acc:.3f}")

    def on_predict_sample(self):
        if self.latest_reflectance is None:
            QtWidgets.QMessageBox.information(self, "Info", "No sample to predict. Take measurement first.")
            return
        # prefer ML model if available
        if self.model is not None:
            feats = np.array(self.latest_reflectance).reshape(1, -1)
            pred = self.model.predict(feats)[0]
            self.result_label.setText(f"<b>Prediction (ML): {pred}</b>")
            self.log(f"ML prediction: {pred}")
            return
        # fallback to rule-based
        label = rule_based_classify(self.indices)
        self.result_label.setText(f"<b>Prediction (rule): {label}</b>")
        self.log(f"Rule-based prediction: {label}")

# -----------------------------
# === main entrypoint ===
# -----------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = NanosApp()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
