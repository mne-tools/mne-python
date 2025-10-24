# How to Run on Windows

## Step 1: Install Python Requirements

Open **Command Prompt** or **PowerShell** and run:

```bash
pip install mne matplotlib numpy scipy edfio
```

## Step 2: Download the Scripts

Download these files to your Desktop or any folder:
- `batch_process_windows.py`
- `client_eeg_analysis.py`

## Step 3: Run the Batch Processor

### For Your EDFS Folder:

Open **Command Prompt** and navigate to where you saved the scripts:

```bash
cd C:\Users\yildi\Desktop
```

Then run:

```bash
python batch_process_windows.py --input "C:\Users\yildi\OneDrive\Desktop\braindecode-master\EDFS"
```

This will:
- ✅ Process ALL EDF files in that folder
- ✅ Create a `reports` folder with HTML reports for each file
- ✅ Generate topographic maps for all recordings
- ✅ Create a `figures` folder with all visualizations

### For a Single File:

```bash
python client_eeg_analysis.py --input "C:\Users\yildi\OneDrive\Desktop\AS EC QEEG.edf" --output "AS_EC_report.html"
```

## What You'll Get

After running, you'll find:

```
braindecode-master\EDFS\
├── reports\
│   ├── file1_report.html
│   ├── file2_report.html
│   └── ...
└── figures\
    ├── file1_topomap_t1.png
    ├── file1_topomap_t2.png
    ├── file1_psd.png
    └── ...
```

## Open the Reports

Simply **double-click** any `.html` file to open it in your browser!

## Troubleshooting

### Error: "pip not found"
- Install Python from: https://www.python.org/downloads/
- Make sure to check "Add Python to PATH" during installation

### Error: "No module named 'mne'" or "No module named 'edfio'"
- Run: `pip install mne matplotlib numpy scipy edfio`

### Error: "No EDF files found"
- Check the path is correct
- Make sure path is in quotes if it has spaces

## Advanced Options

### Change Filter Settings:

```bash
python batch_process_windows.py --input "C:\path\to\EDFS" --notch 60
```

Options:
- `--notch 50`: For Europe/Asia (50 Hz power line)
- `--notch 60`: For US/Canada (60 Hz power line)

### Custom Output Location:

```bash
python batch_process_windows.py --input "C:\path\to\EDFS" --output "C:\path\to\my_reports"
```

## Need Help?

If you get errors:
1. Make sure Python is installed
2. Make sure all packages are installed: `pip install mne matplotlib numpy scipy`
3. Check that your EDF file paths are correct
4. Make sure paths with spaces are in quotes

---

**That's it! Your EEG analysis is ready to go on Windows!** 🚀
