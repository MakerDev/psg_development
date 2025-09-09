# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an integrated sleep analysis system that combines three main components for comprehensive sleep monitoring: **Sleep Stage Classification**, **Arousal Detection**, and **Micro Event Detection** (spindles and K-complexes). The system processes EEG signals from EDF files and provides XML-formatted predictions for clinical use.

## Core Architecture

### Main Components

1. **Sleep Stage Classification** (`sleep_stage/`)
   - CNN encoders (ResNet, RegNet, Swin Transformer, ConvNext) for 5-class sleep stage prediction
   - Input: 9-channel EEG data (30-second epochs at 50Hz)
   - Output: Wake, REM, N1, N2, N3 classifications

2. **Arousal Detection** (`arousal/`)
   - DeepSleep models for arousal event detection
   - Supports both time-domain and spectrogram-based approaches
   - Multi-channel ensemble predictions

3. **Micro Event Detection** (`micro_event/`)
   - REDv2Time models for sleep spindle and K-complex detection
   - Channel-specific event detection with temporal post-processing
   - Event duration and morphology analysis

4. **Integrated Pipeline** (`int_sleep_score.py`)
   - Unified processing combining all three components
   - Cross-validation between different detection systems
   - Correction algorithms using micro-events to refine sleep staging

### Key Data Flow

1. **EDF Loading**: Multi-channel EEG data from clinical recordings
2. **Channel Mapping**: 9-channel standardization (F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1, LOC, ROC, EMG)
3. **Preprocessing**: Filtering, resampling to 50Hz, epoch segmentation
4. **Parallel Processing**: Sleep stages, arousals, and micro-events detected simultaneously
5. **Post-processing**: Temporal smoothing, stage transition validation, micro-event correction
6. **XML Output**: Clinical-format annotations with timestamps and confidence scores

### Signal Configuration

- **Channels**: 9-channel EEG setup with EOG and EMG
- **Sampling Rate**: 50Hz (supports 100Hz input with resampling)
- **Epoch Length**: 30 seconds (1500 samples at 50Hz)
- **Sleep Stages**: 5 classes (Wake=0, REM=1, N1=2, N2=3, N3=4)
- **Event Types**: Arousals, Sleep Spindles, K-complexes

## Common Development Commands

### Integrated Processing (Recommended)

```bash
# Full integrated analysis (all three components)
python int_sleep_score.py --edf /path/to/file.edf --dest /output/dir --gpu 0

# With specific parameters
python int_sleep_score.py --edf file.edf --dest /output --gpu 0 --event_type spindle --th_mul 1.3
```

### Individual Component Analysis

#### Sleep Stage Classification
```bash
# Single file inference
python sleep_stage/integrated_demo_cnn.py --edf /path/to/file.edf --dest /output/dir --gpu 0

# Batch processing
./sleep_stage/runVerify.sh /path/to/edf/dir /output/dir 0

# Multi-channel analysis
python sleep_stage/integrated_demo_cnn_mc.py --edf file.edf --dest /output --gpu 0
```

#### Arousal Detection
```bash
# Single file arousal detection
python arousal/integrated_demo_deepsleep.py --edf /path/to/file.edf --dest /output/dir --gpu 0

# Batch processing
./arousal/runVerify.sh /path/to/edf/dir /output/dir 0

# Spectrogram-based approach
python arousal/demo_spec.py --edf file.edf --gpu 0 --type spec
```

#### Micro Event Detection
```bash
# Spindle detection
python micro_event/me_score.py --edf /path/to/file.edf --event_type spindle --gpu 0

# K-complex detection
python micro_event/me_score.py --edf /path/to/file.edf --event_type kcomplex --gpu 0
```

### Training Models

#### Sleep Stage Models
```bash
# Train CNN model
python sleep_stage/train_cnn.py --model resnet18 --gpu 0 --seed 5

# Train StageNet
python sleep_stage/train_stagenet.py --gpu 0 --epochs 100
```

#### Arousal Models
```bash
# Train DeepSleep model
python arousal/train_deepsleep.py --model DeepSleepSota --gpu 0 --epochs 50
```

#### Micro Event Models
```bash
# Train micro event detector
python micro_event/train_hn.py --event_type spindle --gpu 0 --epochs 100
```

### Testing

```bash
# Run all tests
python -m pytest sleep_stage/tests/ arousal/tests/ -v

# Test integrated pipeline
python -m pytest sleep_stage/tests/test_integrated_pipeline.py -v

# Test specific component
python -m pytest arousal/tests/test_arousal_final.py -v
```

### Data Preprocessing

```bash
# Sleep stage preprocessing
python sleep_stage/prep_window_wise.py --input_dir /edf/dir --output_dir /pickle/dir --fs 50

# Arousal spectrogram preprocessing
python arousal/prep_spectrogram.py --input_dir /edf/dir --output_dir /spec/dir

# Micro event preprocessing (automatic in training)
```

## Model Files and Configuration

### Pretrained Models

#### Sleep Stage Models (`sleep_stage/saved_models/`)
- `pretrained_asam_ver3.pt` - Main production model
- `pretrained_asam_ver2.pt` - Alternative model
- `pretrained_miss1.pt` through `pretrained_miss8.pt` - Missing channel variants

#### Arousal Models (`arousal/saved_models/`)
- `deepsleep_spec_0.923.pt` - Spectrogram-based model
- `deepsleep_time_*.pt` - Time-domain models with various configurations
- `deepsleep_tight_*.pt` and `deepsleep_loose_*.pt` - Different sensitivity variants

#### Micro Event Models (`micro_event/saved_models/`)
- `HN_spindle_ep006_f10.5243_newall_th0.2657.pth` - Spindle detection
- `HN_kcomplex_ep012_f10.4473_newall_th0.2433.pth` - K-complex detection

### Model Architecture Support
- **CNN Models**: ResNet (18, 50), RegNet (16, 128), Swin Transformer, ConvNext
- **DeepSleep**: Various architectures for arousal detection
- **REDv2Time**: Temporal models for micro-event detection

## Configuration and Data Formats

### Channel Mapping
- Standard 9-channel setup with automatic mapping from various EEG system naming conventions
- Configurable handling of missing channels (raise error, interpolate, or skip)
- Channel names mapped through configuration files in each component

### Input/Output Formats
- **Input**: EDF files with optional XML annotations
- **Output**: XML files with predictions, timestamps, and confidence scores
- **Training Data**: Preprocessed pickle files for each component

### Key Parameters
- `--gpu`: GPU device number (default: 0)
- `--fs`: Sampling frequency (default: 50Hz)
- `--seed`: Random seed for reproducibility
- `--start_time`: Custom analysis start time
- `--th_mul`: Threshold multiplier for micro-event sensitivity

## Development Notes

### Error Handling
- EDF files must contain required channels or analysis will fail with descriptive errors
- Missing channels can be configured to interpolate or raise errors
- GPU/CPU fallback is automatic based on availability

### Performance Considerations
- Integrated processing recommended for clinical workflows
- Individual components can be run separately for development/debugging
- Memory usage scales with file duration and number of channels
- Batch processing scripts available for multiple files

### Cross-Component Integration
- Micro-events used to validate and correct sleep stage predictions
- Arousal detections influence sleep stage transitions
- Temporal consistency enforced across all predictions
- Confidence scores used for decision weighting in integrated analysis