# Understanding Your EEG Analysis Results

This guide helps you interpret the outputs from your EEG analysis.

## 📁 Output Files Overview

After analysis, you'll get:

```
EDFS\
├── reports\
│   └── AS_EC_QEEG_report.html  ← Main report (open in browser)
└── figures\
    ├── AS_EC_QEEG_topomap_t1.png  ← Topographic maps (5 time points)
    ├── AS_EC_QEEG_topomap_t2.png
    ├── AS_EC_QEEG_topomap_t3.png
    ├── AS_EC_QEEG_topomap_t4.png
    ├── AS_EC_QEEG_topomap_t5.png
    └── AS_EC_QEEG_psd.png  ← Frequency analysis
```

## 🧠 1. Topographic Maps (Brain Activity Distribution)

### What They Show
Topographic maps display the **spatial distribution** of electrical activity across the scalp at specific time points.

### How to Read Them

**Color Scale:**
- 🔴 **Red/Warm colors** = High amplitude (strong activity)
- 🔵 **Blue/Cool colors** = Low amplitude (weak activity)
- ⚪ **White/Middle** = Average activity

**Head View:**
- **Top of circle** = Front of head (forehead)
- **Bottom of circle** = Back of head (occipital)
- **Left side** = Left hemisphere
- **Right side** = Right hemisphere

### What to Look For

**For Eyes Closed (EC) recordings:**
- Strong activity in **posterior (back) regions** = Normal alpha waves
- Should see **symmetric patterns** between left and right
- Activity concentrated in **occipital area** (back of head)

**Normal Patterns:**
- Symmetric left-right distribution
- Smooth gradients (no sudden spikes)
- Consistent patterns across time points

**Potential Issues to Note:**
- Extreme asymmetry (one side much stronger)
- Very localized "hot spots"
- Unusual frontal dominance in EC recordings

## 📊 2. Power Spectral Density (PSD) - Frequency Analysis

### What It Shows
PSD shows **how much power** (energy) is present at each frequency in the EEG signal.

### Brain Wave Bands

| Band | Frequency | State | What to Expect |
|------|-----------|-------|----------------|
| **Delta (δ)** | 0.5-4 Hz | Deep sleep, unconscious | Low in EC recordings |
| **Theta (θ)** | 4-8 Hz | Drowsiness, meditation | Moderate levels |
| **Alpha (α)** | 8-13 Hz | Relaxed, eyes closed | 🔥 **HIGHEST in EC!** |
| **Beta (β)** | 13-30 Hz | Alert, active thinking | Lower in EC |
| **Gamma (γ)** | 30-100 Hz | High cognition | Low amplitude |

### For Your "AS EC QEEG" (Eyes Closed) Recording

**Expected Pattern:**
- 📈 **Peak around 10 Hz** (alpha range) - This is GOOD!
- Higher posterior alpha (back channels)
- Lower beta activity (since not actively thinking)
- Minimal delta (unless drowsy)

**Alpha Peak Characteristics:**
- **Frequency:** Usually 8-13 Hz (commonly 9-11 Hz)
- **Amplitude:** Should be the dominant peak
- **Location:** Strongest in occipital (O1, O2, Oz) electrodes

### How to Read the PSD Plot

**Y-axis:** Power/Amplitude (in dB or μV²)
- Higher = More activity at that frequency

**X-axis:** Frequency (Hz)
- Left = Slower waves (0-4 Hz = Delta)
- Middle = Alpha range (8-13 Hz)
- Right = Faster waves (13+ Hz = Beta, Gamma)

**Multiple Lines:**
- Each line = One channel/electrode
- Compare patterns across channels
- Look for dominant frequency peaks

## 📈 3. Time Series (Raw Signal View)

### What It Shows
Shows the actual EEG waveforms over time for each channel.

### How to Read

**Y-axis:** Each horizontal line = One channel
- Top channels = Frontal electrodes (Fp1, Fp2, F3, F4...)
- Bottom channels = Posterior electrodes (P3, P4, O1, O2...)

**X-axis:** Time in seconds

**Amplitude:** Vertical deflections
- Positive deflections (upward)
- Negative deflections (downward)

### What to Look For

**Normal Alpha Rhythm (EC):**
- Regular, rhythmic waves at 8-13 Hz
- Amplitude: 20-100 μV (microvolts)
- Most prominent in posterior channels
- Waxes and wanes (varies in amplitude)

**Good Quality Signals:**
- Clear, regular waveforms
- No extreme jumps or flat lines
- Consistent amplitude across similar channels

**Artifacts (Things to Watch Out For):**
- 👁️ **Eye blinks:** Large spikes in frontal channels (Fp1, Fp2)
- 💪 **Muscle activity:** High-frequency noise (looks "fuzzy")
- 📱 **Electrical interference:** Regular, sharp spikes
- 🔌 **Line noise:** 50 Hz or 60 Hz oscillations

## 🎯 Interpreting Your "AS EC QEEG" Specifically

### What "EC" Means
- **EC = Eyes Closed**
- Used to assess resting state brain activity
- Alpha waves should be dominant

### Normal EC QEEG Expectations

1. **Topographic Maps:**
   - Posterior dominance (red/warm at back of head)
   - Symmetric left-right patterns
   - Smooth gradients

2. **PSD Plot:**
   - Clear alpha peak at 8-13 Hz
   - Peak should be 2-5x higher than adjacent frequencies
   - Lower frequencies in frontal vs. posterior

3. **Clinical Significance:**
   - **Strong alpha:** Good relaxation, healthy resting state
   - **Weak alpha:** May indicate anxiety, hyperarousal, or eyes weren't fully closed
   - **Asymmetric alpha:** May warrant further investigation

## 📋 Quick Reference: What's Normal vs. What's Not

### ✅ Normal Patterns (EC Recording)

- [ ] Strong alpha peak (8-13 Hz) in PSD
- [ ] Posterior dominance in topomaps
- [ ] Left-right symmetry
- [ ] Regular rhythmic waves in time series
- [ ] Clear posterior channels (O1, O2, Oz)

### ⚠️ Patterns to Note (For Further Review)

- [ ] Asymmetric activity (>50% difference L-R)
- [ ] Absent or very weak alpha
- [ ] Dominant delta/theta in awake EC
- [ ] Excessive high-frequency activity
- [ ] Very localized activity patterns

### 🚨 Technical Issues (Not Brain Activity)

- [ ] Flat lines (electrode disconnection)
- [ ] Extreme spikes (artifact)
- [ ] Regular 50/60 Hz noise (power line)
- [ ] All channels showing identical patterns (reference issue)

## 💡 Tips for Client Reports

### What to Highlight

1. **Dominant Frequency:**
   - "Peak alpha frequency at 10.5 Hz"
   - "Strong posterior alpha rhythm"

2. **Distribution:**
   - "Symmetric bilateral distribution"
   - "Appropriate posterior dominance"

3. **Amplitude:**
   - "Alpha amplitude within normal range"
   - "Good signal-to-noise ratio"

### Professional Interpretation Examples

**Good EC Recording:**
> "The recording demonstrates a well-defined posterior alpha rhythm at 10 Hz with appropriate reactivity. Topographic maps show expected occipital dominance with symmetric bilateral distribution. Power spectral analysis confirms prominent alpha activity (8-13 Hz) consistent with a relaxed, eyes-closed state."

**Recording with Findings:**
> "The recording shows reduced alpha amplitude in posterior regions compared to age-matched norms. Power spectral density reveals a shift toward theta dominance (6-8 Hz), suggesting possible drowsiness or hyperarousal. Asymmetry noted with reduced left posterior activity (15% lower than right)."

## 🔬 Advanced: Comparing Multiple Recordings

If analyzing multiple files:

1. **Baseline vs. Task:**
   - Compare EC (eyes closed) with EO (eyes open)
   - Alpha should decrease significantly when eyes open

2. **Before/After Treatment:**
   - Look for changes in peak frequency
   - Monitor amplitude changes
   - Track symmetry improvements

3. **Multi-Client Comparison:**
   - Note age differences (alpha slows with age)
   - Consider medication effects
   - Compare to normative databases

## 📚 Further Learning

### Key Concepts to Understand

1. **Alpha Blocking:** Alpha disappears when eyes open (normal)
2. **Alpha Asymmetry:** Linked to emotional processing
3. **Peak Alpha Frequency (PAF):** Individual marker, changes with cognitive state
4. **Alpha Power:** Amplitude of alpha activity, varies by person

### Frequency Bands Quick Reference

```
0.5 ─────── 4 ────── 8 ──────── 13 ────── 30 ────── 100 Hz
    Delta      Theta    Alpha       Beta        Gamma
    Sleep      Drowsy   Relaxed     Alert       Focus
```

## ✅ Quality Checklist for Each Report

Before sharing with clients:

- [ ] Topographic maps show reasonable patterns
- [ ] PSD has identifiable peaks
- [ ] Time series looks clean (minimal artifacts)
- [ ] Recording information is accurate
- [ ] Appropriate frequency bands are prominent for recording type
- [ ] No obvious technical errors

---

## 🎓 Summary

**For EC (Eyes Closed) QEEG:**
- ✅ Look for strong alpha (8-13 Hz)
- ✅ Check posterior dominance in topomaps
- ✅ Verify left-right symmetry
- ✅ Clean signals without excessive artifacts

**The HTML report combines all these visualizations in one professional document!**

Happy analyzing! 🧠✨
