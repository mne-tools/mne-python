# Quick Commands - Copy & Paste Ready!

**Your EDFS Folder:** `C:\Users\yildi\OneDrive\Desktop\braindecode-master\EDFS`

---

## 🚀 Three Ways to Run (Pick Your Favorite!)

### ⭐ Method 1: One-Click (EASIEST!)

**Just double-click:**
```
ANALYZE_MY_EDFS.bat
```

That's it! Pre-configured for your exact folder.

---

### Method 2: Python Script

**Open Command Prompt and run:**
```cmd
python analyze_my_data.py
```

Also pre-configured for your EDFS folder!

---

### Method 3: Command Line (Most Control)

**Copy and paste this entire command:**

```cmd
python batch_process_windows.py --input "C:\Users\yildi\OneDrive\Desktop\braindecode-master\EDFS" --notch 50
```

For US recordings (60 Hz power), use:
```cmd
python batch_process_windows.py --input "C:\Users\yildi\OneDrive\Desktop\braindecode-master\EDFS" --notch 60
```

---

## 📦 First Time Setup (Do This Once!)

**Install required packages:**

```cmd
pip install mne matplotlib numpy scipy edfio
```

**That's all the setup needed!**

---

## 📂 Where to Find Your Results

After running, your reports will be at:

```
C:\Users\yildi\OneDrive\Desktop\braindecode-master\EDFS\reports\
```

**Files you'll get:**
- `AS_EC_QEEG_report.html` ← Main report (open in browser)
- `figures\AS_EC_QEEG_topomap_t1.png` through t5.png ← Brain maps
- `figures\AS_EC_QEEG_psd.png` ← Frequency analysis

---

## 🎯 Single File Analysis

**To analyze just your AS EC QEEG file:**

```cmd
python client_eeg_analysis.py --input "C:\Users\yildi\OneDrive\Desktop\AS EC QEEG.edf" --output "AS_EC_Report.html"
```

---

## ⚡ Quick Verification

**Test if everything is installed:**

```cmd
python --version
python -c "import mne; print('MNE OK')"
```

If both work, you're ready to go!

---

## 🆘 Troubleshooting One-Liners

**Check if Python is installed:**
```cmd
python --version
```

**Check if packages are installed:**
```cmd
python -c "import mne, matplotlib, numpy, scipy, edfio; print('All packages OK!')"
```

**Check if your EDFS folder exists:**
```cmd
dir "C:\Users\yildi\OneDrive\Desktop\braindecode-master\EDFS\*.edf"
```

**Install packages if missing:**
```cmd
pip install mne matplotlib numpy scipy edfio
```

---

## 📊 Expected Output

**For each EDF file, you'll get:**

```
reports\
├── AS_EC_QEEG_report.html      (3-4 MB) ← Open this!
├── patient2_report.html
└── figures\
    ├── AS_EC_QEEG_topomap_t1.png (150 KB)
    ├── AS_EC_QEEG_topomap_t2.png
    ├── AS_EC_QEEG_topomap_t3.png
    ├── AS_EC_QEEG_topomap_t4.png
    ├── AS_EC_QEEG_topomap_t5.png
    └── AS_EC_QEEG_psd.png (120 KB)
```

---

## ⏱️ How Long Does It Take?

- **1 file (60s recording):** ~30-60 seconds
- **10 files:** ~7-10 minutes
- **AS EC QEEG.edf:** Depends on length, probably 30-90 seconds

---

## 💡 Pro Tips

### Process Multiple Client Folders

```cmd
python batch_process_windows.py --input "C:\Clients\Client1\EDFS"
python batch_process_windows.py --input "C:\Clients\Client2\EDFS"
python batch_process_windows.py --input "C:\Clients\Client3\EDFS"
```

### Custom Output Location

```cmd
python batch_process_windows.py --input "C:\Users\yildi\OneDrive\Desktop\braindecode-master\EDFS" --output "C:\My_Reports"
```

### Create Demo Data for Testing

```cmd
python create_demo_edf.py
python client_eeg_analysis.py --input demo_eeg_data.edf
```

---

## 📋 Copy-Paste Checklist

**Before your first run, copy and paste these commands:**

```cmd
REM 1. Check Python
python --version

REM 2. Install packages
pip install mne matplotlib numpy scipy edfio

REM 3. Verify packages
python -c "import mne; print('Ready!')"

REM 4. Check your files exist
dir "C:\Users\yildi\OneDrive\Desktop\braindecode-master\EDFS\*.edf"

REM 5. Run analysis
python analyze_my_data.py

REM Done! Open the reports folder to see your results.
```

---

## 🎯 Your Exact Setup Summary

**Your folder:** `C:\Users\yildi\OneDrive\Desktop\braindecode-master\EDFS`

**Your key file:** `AS EC QEEG.edf`

**Easiest command:** Just double-click `ANALYZE_MY_EDFS.bat`

**Alternative:** `python analyze_my_data.py`

**Reports location:** `EDFS\reports\`

**Power frequency:** 50 Hz (Europe/Asia) - Change to 60 if US data

---

## ✅ Success Indicators

**You'll know it worked when:**
- ✅ No error messages in console
- ✅ "ANALYSIS COMPLETE!" message appears
- ✅ `reports` folder created in EDFS
- ✅ HTML files appear in reports folder
- ✅ Can open HTML files in browser
- ✅ See brain maps and frequency plots

---

## 🎉 Ready to Run!

**Simplest possible workflow:**

1. Open Command Prompt
2. Type: `cd C:\Users\yildi\Desktop` (or wherever you saved files)
3. Type: `python analyze_my_data.py`
4. Wait ~1-2 minutes per file
5. Open `EDFS\reports\` folder
6. Double-click any `.html` file
7. Done!

---

**All commands above are copy-paste ready!** Just open Command Prompt and paste! 🚀
