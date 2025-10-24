# Quick Start Guide: EEG Client Data Analysis

This guide will help you analyze your client's EDF files and generate professional reports with topographic maps using MNE-Python.

## Installation

```bash
# Install required packages
pip install mne matplotlib numpy scipy
```

## Usage

### Basic Usage (Simplest!)

```bash
python client_eeg_analysis.py --input patient_data.edf
```

This will:
- Load the EDF file
- Apply preprocessing (filtering, denoising)
- Generate topographic maps
- Create power spectral density plots
- Generate a comprehensive HTML report: `eeg_analysis_report.html`

### Advanced Usage

```bash
python client_eeg_analysis.py \
    --input patient_data.edf \
    --output client_report_2025.html \
    --lowpass 0.5 \
    --highpass 40.0 \
    --notch 60.0
```

### Options

- `--input` or `-i`: Path to your EDF file (REQUIRED)
- `--output` or `-o`: Name for the HTML report (default: eeg_analysis_report.html)
- `--lowpass` or `-l`: Low-pass filter frequency in Hz (default: 0.5)
- `--highpass` or `-h`: High-pass filter frequency in Hz (default: 40.0)
- `--notch`: Notch filter for line noise - use 50 Hz for Europe/Asia, 60 Hz for US (default: 50)

## What You Get

After running the script, you'll have:

1. **HTML Report** (`eeg_analysis_report.html`)
   - Interactive, professional-looking report
   - All visualizations in one place
   - Patient/recording information
   - Open in any web browser

2. **Figures Folder** (`eeg_figures/`)
   - Topographic maps (showing brain activity distribution)
   - Power spectral density plots
   - Time series plots
   - All saved as high-quality PNG files

## Example Workflow for Multiple Clients

### Process Multiple Files:

```bash
# Client 1
python client_eeg_analysis.py --input client1.edf --output client1_report.html

# Client 2
python client_eeg_analysis.py --input client2.edf --output client2_report.html

# Client 3
python client_eeg_analysis.py --input client3.edf --output client3_report.html
```

### Batch Processing Script:

Create a file called `batch_process.sh`:

```bash
#!/bin/bash

# Process all EDF files in a directory
for edf_file in *.edf; do
    output_name="${edf_file%.edf}_report.html"
    echo "Processing $edf_file..."
    python client_eeg_analysis.py --input "$edf_file" --output "$output_name"
done

echo "All files processed!"
```

Then run:
```bash
chmod +x batch_process.sh
./batch_process.sh
```

## Understanding the Output

### Topographic Maps
- Show spatial distribution of electrical activity across the scalp
- Warmer colors (red) = higher amplitude
- Cooler colors (blue) = lower amplitude
- Multiple time points show how activity changes over time

### Power Spectral Density (PSD)
- Shows frequency content of the EEG signals
- Identifies dominant frequencies
- Useful for detecting abnormal rhythms
- Common bands:
  - Delta (0.5-4 Hz): Deep sleep
  - Theta (4-8 Hz): Drowsiness, meditation
  - Alpha (8-13 Hz): Relaxed, eyes closed
  - Beta (13-30 Hz): Alert, active thinking
  - Gamma (30-100 Hz): High-level processing

### Time Series
- Raw signal traces for each channel
- Shows amplitude changes over time
- Useful for identifying artifacts or abnormal events

## Troubleshooting

### Error: "File not found"
- Make sure you provide the full path to your EDF file
- Example: `--input /home/user/data/patient1.edf`

### Error: "No montage found"
- Some EDF files don't have channel location information
- The script will still work but topomaps may not display
- You may need to add channel locations manually

### Report looks empty
- Check that your EDF file has valid EEG data
- Make sure the file isn't corrupted
- Try opening the EDF file in a viewer first

## Python API (For Custom Analysis)

If you want to customize the analysis, you can import the functions:

```python
import mne
from client_eeg_analysis import load_edf_file, preprocess_data, create_topographic_maps

# Load your data
raw = load_edf_file('patient.edf')

# Preprocess
raw = preprocess_data(raw, l_freq=0.5, h_freq=40.0)

# Create custom visualizations
raw.plot_sensors(show_names=True)  # Show sensor locations
raw.plot(duration=20)               # Plot 20 seconds of data

# Extract specific frequency bands
raw_alpha = raw.copy().filter(8, 13)  # Alpha band
raw_beta = raw.copy().filter(13, 30)  # Beta band

# Compute evoked responses (if you have events)
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.2, tmax=0.5)
evoked = epochs.average()
evoked.plot_topomap(times=[0, 0.1, 0.2, 0.3])
```

## Comparison with MATLAB

| Feature | MNE-Python (This Tool) | MATLAB/EEGLAB |
|---------|------------------------|---------------|
| Cost | FREE | $2,000+ license |
| Setup Time | 5 minutes | 1-2 hours |
| Automation | Easy (Python scripts) | Harder |
| Reports | Auto-generated HTML | Manual |
| Topomaps | ✓ | ✓ |
| EDF Support | ✓ | ✓ |
| Learning Curve | Moderate | Moderate |

## Need Help?

- MNE-Python Documentation: https://mne.tools/
- Tutorials: https://mne.tools/stable/auto_tutorials/index.html
- Forum: https://mne.discourse.group/

## Next Steps

1. Try the script with a sample EDF file
2. Open the generated HTML report in your browser
3. Customize the parameters for your specific needs
4. Create batch scripts for multiple clients
5. Explore the MNE-Python documentation for advanced analyses

Happy analyzing!
