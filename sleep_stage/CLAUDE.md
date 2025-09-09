# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a sleep stage classification system that processes EEG signals and predicts sleep stages using deep learning models. The system can handle both single-file inference and batch processing of EDF files with XML annotations.

## Core Architecture

### Main Components

- **Models**: CNN encoders (ResNet, RegNet, Swin Transformer, ConvNext) for sleep stage classification located in `models/`
- **Data Processing**: EDF file loading, preprocessing, and XML handling in `modules/`
- **Inference Pipeline**: Multiple demo scripts for different use cases
- **Training**: CNN and StageNet training scripts with XGBoost support

### Key Data Flow

1. **EDF Loading**: Raw EEG data loaded from EDF files using `modules/iofiles/edf.py`
2. **Preprocessing**: Signal filtering, normalization, and epoch extraction via `modules/preprocessing.py`
3. **Model Inference**: CNN models process 30-second epochs (1500 samples at 50Hz)
4. **Post-processing**: Temporal smoothing and stage transitions in `utils/post_process.py`
5. **Output**: Sleep stage predictions saved as XML files

### Signal Configuration

- **Channels**: 9-channel EEG setup (F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1, LOC, ROC, EMG)
- **Sampling Rate**: 50Hz (configurable, supports 100Hz)
- **Epoch Length**: 30 seconds (1500 samples at 50Hz)
- **Sleep Stages**: 5 classes (Wake=0, REM=1, N1=2, N2=3, N3=4)

## Common Development Commands

### Running Inference

```bash
# Single file inference with CNN model
python integrated_demo_cnn.py --edf /path/to/file.edf --dest /output/dir --gpu 0

# Batch processing multiple EDF files
./runVerify.sh /path/to/edf/dir /output/dir 0

# With specific model architecture
python integrated_demo_cnn.py --edf file.edf --dest /output --model resnet50 --gpu 0
```

### Training Models

```bash
# Train CNN model
python train_cnn.py --model resnet18 --gpu 0 --seed 5

# Train StageNet model  
python train_stagenet.py --gpu 0 --epochs 100

# XGBoost training (Jupyter notebook)
jupyter notebook train_xgboost.ipynb
```

### Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_integrated_pipeline.py -v

# Test with pretrained model (requires model file)
python -m pytest tests/test_integrated_pipeline.py::TestIntegratedSleepPipeline::test_with_pretrained_model -v
```

### Data Preprocessing

```bash
# Window-wise preprocessing for CNN training
python prep_window_wise.py --input_dir /edf/dir --output_dir /pickle/dir --fs 50

# Spectrogram-based preprocessing
python prep_spectrogram.py --input_dir /edf/dir --output_dir /spec/dir

# XGBoost feature extraction
python prep_xgboost.py --input_dir /edf/dir --output_dir /features/dir
```

## Model Files

### Pretrained Models
- Located in `saved_models/` directory
- Main models: `pretrained_asam_ver3.pt`, `pretrained_asam_ver2.pt`
- Models expect 9-channel input with 1500 time points per epoch

### Model Architecture Support
- **ResNet**: `resnet18`, `resnet50` 
- **RegNet**: `regnet16`, `regnet128`
- **Vision Transformers**: `swin`, `convnext`
- **StageNet**: Custom architecture in `models/stagenet.py`

## Configuration

### Channel Mapping
- EEG channel names are mapped through `utils/config.py`
- Supports various naming conventions from different EEG systems
- Missing channels can be handled with interpolation or raising errors

### Data Formats
- **Input**: EDF files with corresponding XML annotations (optional)
- **Output**: XML files with sleep stage predictions and confidence scores
- **Training Data**: Pickled numpy arrays with preprocessed epochs

## Development Notes

### Error Handling
- Missing channels can be configured to raise errors or interpolate
- EDF files without proper channel mapping will fail with descriptive errors
- GPU/CPU fallback is automatic based on availability

### Performance Considerations
- Models expect exactly 1500 samples per epoch (30 seconds at 50Hz)
- Batch processing recommended for multiple files
- GPU memory usage scales with batch size and model complexity

### Post-processing
- Temporal smoothing applied with configurable window sizes (default: 6 epochs)
- Stage transition constraints can be applied
- Confidence scores preserved in XML output