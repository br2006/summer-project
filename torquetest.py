'''
CODE TO DETERMINE MAXIMUM TORQUE PROVIDED BY OUR MOTOR AT DIFFERENT VOLTAGES

GenAI was used to help import data files and facilitate writing the analysis code
chat link : https://claude.ai/share/e04cd129-ebb1-4245-920a-84a725e2c4a8
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit 
from scipy.signal import savgol_filter, find_peaks
 


#FUNCTIONS TO LOAD/SORT CSV FILES


import csv
import glob
import os




def load_folder(folder):
    """Load every CSV in `folder` into a dict of named arrays.
 
    Each key is the filename without extension (e.g. '12V_1A', '8V_1A'), so the
    name tells you the voltage and current. Each value is an (N, 3) array with
    columns: [time, velocity, angle]. Time is converted from the HH:MM:SS.mmm
    timestamp into seconds (float), zeroed to the start of that run.
    """
    data = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]  # '12V_1A'
 
        rows = []
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if len(row) < 3:
                    continue  # skip blank/short lines
                ts, vel, ang = row[0], row[1], row[2]
                try:
                    h, m, s = ts.split(":")
                    t = int(h) * 3600 + int(m) * 60 + float(s)
                    rows.append((t, float(vel), float(ang)))
                except ValueError:
                    continue  # skip a header row or any unparseable line
 
        if not rows:
            continue  # no usable data in this file
 
        arr = np.array(rows)
        arr[:, 0] -= arr[0, 0]  # make time start at 0 for this run
        data[name] = arr
 
    return data
#COMPUTE MOTOR ACCELERATION FUNCTION
def add_angular_acceleration(data):
    """Return a new dict where each run gains a 4th column: angular acceleration.
 
    Angular acceleration is d(velocity)/d(time), computed with numpy.gradient.
    Passing the time array as the coordinate means the (non-uniform) spacing
    between samples is handled correctly. Columns of each array become:
        [time, velocity, angle, angular_acceleration]
    Units are (velocity units) per second.
    """
    result = {}
    for name, arr in data.items():
        time = arr[:, 0]
        velocity = arr[:, 1]
        accel = np.gradient(velocity, time)
        result[name] = np.column_stack([arr, accel])
    return result       
 

#COMPUTE MOTOR ACCELERATION - NORMAL + SMOOTHED
def add_accelerations(data, window=8, polyorder=3):
    """Add raw AND smoothed angular acceleration columns, for comparison.
 
    Columns of each run become:
        [time, velocity, angle, accel_raw, accel_smooth]
 
    accel_raw    : np.gradient of the raw velocity (noisy).
    accel_smooth : velocity is first smoothed with a Savitzky-Golay filter,
                   then differentiated with np.gradient. Savitzky-Golay's own
                   derivative assumes uniform spacing, so we only use it to
                   smooth, then differentiate against the real (non-uniform)
                   time array.
 
    window   : number of samples in the smoothing window (must be odd; it is
               auto-reduced for short runs). Bigger = smoother but blurs sharp
               transitions. polyorder : order of the local polynomial fit.
    """
    result = {}
    for name, arr in data.items():
        time, velocity = arr[:, 0], arr[:, 1]
 
        accel_raw = np.gradient(velocity, time)
 
        # window must be odd, greater than polyorder, and <= sample count
        w = min(window, len(velocity))
        if w % 2 == 0:
            w -= 1
        if w <= polyorder:
            velocity_smooth = velocity.copy()  # too few points to smooth
        else:
            velocity_smooth = savgol_filter(velocity, w, polyorder)
        accel_smooth = np.gradient(velocity_smooth, time)
 
        # arr[:, :3] keeps just the original 3 columns, so this is safe whether
        # or not you already added a raw-acceleration column earlier.
        result[name] = np.column_stack([arr[:, :3], accel_raw, accel_smooth, velocity_smooth])
    return result

def find_acceleration_peaks(data, accel_col=4, prominence_frac=0.4, height_frac=0.6):
    """Find peaks in each run's angular acceleration and return them separately.
 
    Detection is done on the magnitude |acceleration|, so large spikes in BOTH
    directions are caught (the motor speeding up and braking/reversing). The
    value stored is the absolute (magnitude) acceleration at each peak.
 
    Both thresholds are relative to that run's own maximum |acceleration|, so
    they adapt automatically as spike height changes with voltage:
 
    accel_col      : which column to read. 4 = smoothed (default), 3 = raw.
    prominence_frac: a peak must stand out by at least this fraction of the
                     run's max to count (0.4 -> 40% of max).
    height_frac    : a peak must also reach at least this fraction of the run's
                     max (0.6 -> only points >= 60% of the max are peaks).
 
    Returns a dict keyed by run name. Each value is an array of detected peaks
    with columns: [time, abs_acceleration].
    """
    peaks = {}
    for name, arr in data.items():
        time = arr[:, 0]
        accel = np.abs(arr[:, accel_col])  # work with magnitude
        peak_max = accel.max()
        idx, _ = find_peaks(
            accel,
            prominence=prominence_frac * peak_max,
            height=height_frac * peak_max,
        )
        peaks[name] = np.column_stack([time[idx], accel[idx]])
    return peaks
       
#LOAD DATA

data = load_folder('torque_test')
data = add_accelerations(data)

'''
for i in range(2, 20, 2):
    if i <= 8:
        run = data[f"{i}V_1A"]
        plt.plot(run[:, 0], run[:, 1])
        plt.title(f'{i}V_1A - motor vel')
        plt.show()
        plt.plot(run[:,0], run[:,5])
        plt.title(f'{i}V_1A - motor vel-smoothed')
        plt.show()
        plt.plot(run[:,0], run[:,3])
        plt.title(f'{i}V_1A - motor accel')
        plt.show()
        plt.plot(run[:,0], run[:,4])
        plt.title(f'{i}V_1A - motor accel smoothed')
        plt.show()
        
        
    else:
        run1 = data[f"{i}V_1A"]
        plt.plot(run1[:, 0], run1[:, 1])
        plt.title(f'{i}V_1A - motor vel')
        plt.show()
        plt.plot(run1[:,0], run1[:,5])
        plt.title(f'{i}V_1A - motor vel-smoothed')
        plt.show()
        plt.plot(run1[:,0], run1[:,3])
        plt.title(f'{i}V_1A - motor accel')
        plt.show()
        plt.plot(run1[:,0], run1[:,4])
        plt.title(f'{i}V_1A - motor accel smoothed')
        plt.show()
        
        run2 = data[f"{i}V_2A"]
        plt.plot(run2[:, 0], run2[:, 1])
        plt.title(f'{i}V_2A - motor vel')
        plt.show()
        plt.plot(run2[:,0], run2[:,5])
        plt.title(f'{i}V_2A - motor vel-smoothed')
        plt.show()
        plt.plot(run2[:,0], run2[:,3])
        plt.title(f'{i}V_2A - motor accel')
        plt.show()
        plt.plot(run2[:,0], run2[:,4])
        plt.title(f'{i}V_2A - motor accel smoothed')
        plt.show()
''' 


peaks = find_acceleration_peaks(data, accel_col=4, prominence_frac=0.4, height_frac=0.5)

#p = peaks["6V_1A"]
#peak_accels = p[:, 1]   # all absolute values now

 
    
 # 1A DATA ANALYSIS
data_1A = []
for i in range(2, 20, 2):
    p_i = peaks[f'{i}V_1A']
    peak_accels = p_i[:, 1]

    av_peak_i  = np.mean(peak_accels)
    peak_std_i = np.std(peak_accels) / np.sqrt(len(peak_accels))

    data_1A.append([i, av_peak_i, peak_std_i])

data_1A = np.array(data_1A)   # shape (N, 3): columns are V, mean, std
    

 # 2A DATA ANALYSIS
data_2A = []
for i in range(10, 20, 2):
    p_i = peaks[f'{i}V_2A']
    peak_accels = p_i[:, 1]

    av_peak_i  = np.mean(peak_accels)
    peak_std_i = np.std(peak_accels) / np.sqrt(len(peak_accels))

    data_2A.append([i, av_peak_i, peak_std_i])

data_2A = np.array(data_2A)  

I_flywheel = 0.000276


'''
plt.errorbar(data_1A[:,0], data_1A[:,1]*I_flywheel, yerr = data_1A[:,2]*I_flywheel)
plt.errorbar(data_2A[:,0], data_2A[:,1]*I_flywheel, yerr = data_2A[:,2]*I_flywheel)
plt.show()
'''
plt.figure(figsize=(8, 6))

plt.errorbar(data_1A[:, 0], data_1A[:, 1] * I_flywheel, yerr=data_1A[:, 2] * I_flywheel,
             fmt='o-', capsize=4, label='1 A')
plt.errorbar(data_2A[:, 0], data_2A[:, 1] * I_flywheel, yerr=data_2A[:, 2] * I_flywheel,
             fmt='s-', capsize=4, label='2 A')

plt.xlabel('Voltage (V)')
plt.ylabel('Peak torque (N·m)')
plt.title('Peak torque vs supply voltage')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig('peak_torque.png', dpi=300, bbox_inches='tight')   # save first
plt.show()

