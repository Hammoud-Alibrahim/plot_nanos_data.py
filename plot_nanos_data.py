[plot_nanos_data.py](https://github.com/user-attachments/files/24387958/plot_nanos_data.py)
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 10:22:48 2025

@author: thorben.wellbrock
"""

"""
Sensor-Daten Visualisierung
Liest die neueste *_Rawdata.csv Datei aus dem Logging-Ordner
und visualisiert die 1600 Kanäle als 40x40 2D-Plot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def find_latest_rawdata_file(folder='Logging'):
    """Findet die neueste *_Rawdata.csv Datei im angegebenen Ordner"""
    search_pattern = os.path.join(folder, '*_Rawdata.csv')
    files = glob.glob(search_pattern)
    
    if not files:
        raise FileNotFoundError(f"Keine *_Rawdata.csv Dateien im Ordner '{folder}' gefunden!")
    
    # Neueste Datei nach Änderungsdatum
    latest_file = max(files, key=os.path.getmtime)
    print(latest_file)
    return latest_file

def load_sensor_data(filepath):
    """Lädt die Sensordaten aus der CSV-Datei"""
    # Deutsche Einstellungen: Semikolon als Trennzeichen, Komma als Dezimaltrennzeichen
    df = pd.read_csv(filepath, sep=';', decimal=',', encoding='utf-8', index_col=False)
    
    # Ab Spalte 13 (Index 13 = Spalte N in Excel) beginnen die Messdaten
    # Spalten 0-12 (A-M) werden ignoriert
    sensor_columns = df.columns[13:]
    
    print(f"Datei geladen: {filepath}")
    print(f"Anzahl Zeitpunkte: {len(df)}")
    print(f"Anzahl Sensorkanäle: {len(sensor_columns)}")
    
    return df, sensor_columns

def reshape_sensor_data_to_2d(data_row, sensor_columns):
    """
    Formt die 1600 Kanäle in ein 40x40 Array um
    
    Struktur:
    - 10 Reihen (R1-R10)
    - 10 Metapixel pro Reihe (M1-M10)
    - 16 Kanäle pro Metapixel (P1-P16) in 4x4 Anordnung
    
    Ergebnis: 40x40 Array (10*4 x 10*4)
    """
    # 40x40 Array initialisieren
    sensor_array = np.zeros((40, 40))
    
    # Durch alle Sensorkanäle iterieren
    for col_name in sensor_columns:
        # Spaltennamen parsen: z.B. "R1_M1_P1"
        parts = col_name.split('_')
        if len(parts) != 3:
            continue
            
        row_num = int(parts[0][1:])      # R1 -> 1
        metapixel_num = int(parts[1][1:])  # M1 -> 1
        pixel_num = int(parts[2][1:])     # P1 -> 1
        
        # Position im 40x40 Array berechnen
        # Jede Reihe hat 4 Zeilen (wegen 4x4 Pixel pro Metapixel)
        # Jedes Metapixel hat 4 Spalten
        
        # Pixel innerhalb des Metapixels (4x4 Anordnung)
        # P1-P4: Zeile 0, P5-P8: Zeile 1, P9-P12: Zeile 2, P13-P16: Zeile 3
        pixel_row = (pixel_num - 1) // 4
        pixel_col = (pixel_num - 1) % 4
        
        # Globale Position im 40x40 Array
        global_row = (row_num - 1) * 4 + pixel_row
        global_col = (metapixel_num - 1) * 4 + pixel_col
        
        # Wert eintragen
        sensor_array[global_row, global_col] = data_row[col_name]
    
    return sensor_array

def plot_sensor_data(sensor_array, timepoint=0, title_suffix=""):
    """Erstellt einen 2D-Plot der Sensordaten"""
    plt.figure(figsize=(12, 10))
    
    # Heatmap erstellen
    im = plt.imshow(sensor_array, cmap='viridis', aspect='auto', interpolation='nearest')
    
    # Colorbar hinzufügen
    cbar = plt.colorbar(im, label='Sensorwert')
    
    # Achsenbeschriftung
    plt.xlabel('Spalte (Metapixel 1-10, je 4 Pixel)')
    plt.ylabel('Zeile (Reihe 1-10, je 4 Pixel)')
    plt.title(f'Sensor 40x40 Visualisierung - Zeitpunkt {timepoint}{title_suffix}')
    
    # Gitterlinien für Metapixel (alle 4 Pixel)
    for i in range(0, 41, 4):
        plt.axhline(y=i-0.5, color='white', linewidth=0.5, alpha=0.3)
        plt.axvline(x=i-0.5, color='white', linewidth=0.5, alpha=0.3)
    
    # Achsenticks für Metapixel
    metapixel_ticks = np.arange(2, 40, 4)  # Mitte jedes Metapixels
    metapixel_labels = [f'M{i}' for i in range(1, 11)]
    
    plt.xticks(metapixel_ticks, metapixel_labels)
    plt.yticks(metapixel_ticks, [f'R{i}' for i in range(1, 11)])
    
    plt.tight_layout()
    plt.show()


"""Hauptfunktion"""
# Neueste Datei finden
filepath = find_latest_rawdata_file('Logging')


# Daten laden
df, sensor_columns = load_sensor_data(filepath)

# Ersten Zeitpunkt visualisieren (kann angepasst werden)
timepoint = 0
data_row = df.iloc[timepoint]

# In 40x40 Array umformen
sensor_array = reshape_sensor_data_to_2d(data_row, sensor_columns)

# Plotten
plot_sensor_data(sensor_array, timepoint, f" (Datei: {os.path.basename(filepath)})")

print(f"\nVisualisierung abgeschlossen!")
print(f"Min-Wert: {sensor_array.min():.2f}")
print(f"Max-Wert: {sensor_array.max():.2f}")
print(f"Mittelwert: {sensor_array.mean():.2f}")

