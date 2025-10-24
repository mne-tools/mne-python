# Your EEG Analysis Quick Start Guide

This guide is customized for your specific setup!

## Your Files

**Your EDF folder:** `C:\Users\yildi\OneDrive\Desktop\braindecode-master\EDFS`

**Files mentioned:**
- AS EC QEEG.edf (Eyes Closed QEEG)
- And other EDF files in your EDFS folder

## 🚀 Three Ways to Run (Pick One!)

### ⭐ Option 1: One-Click (Easiest!)

1. **Download these 2 files to your Desktop:**
   - `RUN_ANALYSIS.bat`
   - `batch_process_windows.py`

2. **Double-click `RUN_ANALYSIS.bat`**

That's it! The batch file will:
- Check if Python is installed
- Install required packages automatically
- Analyze all EDF files in your EDFS folder
- Save reports to `EDFS\reports\`

### Option 2: Command Line (More Control)

1. **Open Command Prompt**

2. **Navigate to where you saved the script:**
   ```cmd
   cd C:\Users\yildi\Desktop
   ```

3. **Run the analysis:**
   ```cmd
   python batch_process_windows.py --input "C:\Users\yildi\OneDrive\Desktop\braindecode-master\EDFS"
   ```

### Option 3: Single File Analysis

To analyze just one file (e.g., your AS EC QEEG.edf):

```cmd
python client_eeg_analysis.py --input "C:\Users\yildi\OneDrive\Desktop\AS EC QEEG.edf" --output "AS_EC_Report.html"
```

## 📦 Installation (Only Once)

Before first run, install Python packages:

```cmd
pip install mne matplotlib numpy scipy edfio
```

## 📊 What You'll Get

After running, check your EDFS folder:

```
braindecode-master\EDFS\
├── AS EC QEEG.edf (your original files)
├── [other .edf files]
│
├── reports\  ← NEW FOLDER with HTML reports
│   ├── AS_EC_QEEG_report.html  👈 Double-click to open!
│   ├── file2_report.html
│   └── ...
│
└── figures\  ← NEW FOLDER with all images
    ├── AS_EC_QEEG_topomap_t1.png  👈 Brain activity maps!
    ├── AS_EC_QEEG_topomap_t2.png
    ├── AS_EC_QEEG_topomap_t3.png
    ├── AS_EC_QEEG_topomap_t4.png
    ├── AS_EC_QEEG_topomap_t5.png
    ├── AS_EC_QEEG_psd.png  👈 Frequency spectrum
    └── ...
```

## 🧠 Understanding Your Results

### Topographic Maps (Brain Activity Maps)
- Show electrical activity distribution across the scalp
- **Red/warm colors** = Higher amplitude (more activity)
- **Blue/cool colors** = Lower amplitude (less activity)
- Multiple time points show how activity changes

### Power Spectral Density (PSD) - Frequency Analysis
Shows the dominant brain wave frequencies:
- **Delta (0.5-4 Hz):** Deep sleep
- **Theta (4-8 Hz):** Drowsiness, meditation
- **Alpha (8-13 Hz):** Relaxed, eyes closed (important for your EC QEEG!)
- **Beta (13-30 Hz):** Active thinking, concentration
- **Gamma (30-100 Hz):** High-level cognitive processing

**For "EC" (Eyes Closed) recordings:** You should see prominent alpha waves around 8-13 Hz!

### HTML Report
- Opens in any web browser
- Contains all visualizations
- Shows recording information
- Professional format for clients

## 💡 Tips

### Your "AS EC QEEG.edf" File
- **EC = Eyes Closed** recording
- Expect strong alpha wave activity (8-13 Hz)
- Good for analyzing resting state brain activity

### Batch Processing Multiple Clients
If you have multiple client folders, create a simple batch script:

```cmd
python batch_process_windows.py --input "C:\Clients\Client1\EDFS"
python batch_process_windows.py --input "C:\Clients\Client2\EDFS"
python batch_process_windows.py --input "C:\Clients\Client3\EDFS"
```

### Custom Settings

**For US recordings (60 Hz power line):**
```cmd
python batch_process_windows.py --input "C:\...\EDFS" --notch 60
```

**Custom output location:**
```cmd
python batch_process_windows.py --input "C:\...\EDFS" --output "C:\Reports\Client_Name"
```

## 🆘 Troubleshooting

### Error: "Python not found"
- Install Python: https://www.python.org/downloads/
- **Important:** Check "Add Python to PATH" during installation!

### Error: "No module named 'mne'"
```cmd
pip install mne matplotlib numpy scipy edfio
```

### Error: "File not found"
- Check that your path is correct
- Use quotes if path has spaces: `"C:\path with spaces\file.edf"`
- Make sure you're in the right directory

### Reports look empty or incomplete
- Check that EDF files are not corrupted
- Try opening one EDF file first to test
- Check console output for specific errors

### No topographic maps generated
- Some EDF files don't have electrode position info
- The script will try to use standard 10-20 positions automatically
- PSD and time series plots will still work!

## 📞 Need Help?

1. Check the detailed guides:
   - `WINDOWS_INSTRUCTIONS.md` - Complete Windows guide
   - `QUICK_START_CLIENT_ANALYSIS.md` - Full documentation

2. Run with a test file first to make sure everything works

3. Check the console output for error messages

## ✅ Quick Checklist

- [ ] Python installed
- [ ] Packages installed: `pip install mne matplotlib numpy scipy edfio`
- [ ] Scripts downloaded (`batch_process_windows.py` or `RUN_ANALYSIS.bat`)
- [ ] Path to EDFS folder is correct
- [ ] Ready to run!

---

**You're all set!** Your EEG analysis toolkit is ready to use. Good luck with your client analyses! 🎉
