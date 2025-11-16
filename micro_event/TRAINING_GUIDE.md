# K-Complex Training Guide - STRICT Clinical Standards

This guide explains how to train the improved K-complex detector with strict postprocessing.

## Quick Start

```bash
# Basic training with default STRICT settings
python micro_event/train_kcomplex_improved.py --gpu 0 --epochs 50 --save True

# Training with custom amplitude thresholds
python micro_event/train_kcomplex_improved.py \
    --gpu 0 \
    --min_amplitude 75 \
    --max_amplitude 300 \
    --epochs 50
```

## Key Features

### 1. **Dual Evaluation System**

The training script evaluates each epoch TWICE:

- **BEFORE Postprocessing**: Raw model predictions (baseline performance)
- **AFTER Postprocessing**: Strict validation with N3 filtering enabled

**Model selection is based on POSTPROCESSED F1 score** for better real-world performance.

### 2. **STRICT Clinical Standards**

All parameters are set to clinical standards by default:

- **Amplitude**: 75-300 µV (AASM guidelines)
- **Shape Quality**: Minimum 0.6 (prominence, symmetry, sharpness)
- **SNR**: Minimum 2.5 (signal-to-noise ratio)
- **N3 Filtering**: Enabled (temporal isolation + periodicity detection)

### 3. **N3 Slow Wave Filtering**

The postprocessing applies TWO critical checks:

1. **Temporal Isolation**: Checks 3-second windows before/after event
   - Rejects if similar events nearby (correlation > 0.7)
   - Filters continuous N3 slow waves

2. **Periodicity Detection**: Analyzes 10-second window
   - Autocorrelation analysis for rhythmic patterns
   - Rejects if periodicity_strength > 0.6
   - Filters regular N3 slow waves

## Command-Line Arguments

### Training Parameters

```bash
--gpu 0                    # GPU device number
--lr 1e-4                  # Learning rate
--batch_size 16            # Batch size
--epochs 50                # Number of epochs
--seed 42                  # Random seed for reproducibility
--save True                # Save best model
--tag "experiment_name"    # Experiment tag for model filename
```

### Loss Weights (Shape-Aware)

```bash
--weight_detection 1.0     # Primary K-complex detection
--weight_peak_align 0.4    # Peak alignment (increased from 0.3)
--weight_peak_order 0.3    # Peak ordering (pos before neg)
--weight_zerocross 0.2     # Zero-crossing boundaries
--weight_shape 0.5         # Shape consistency (CRITICAL - was 0.1)
```

### Amplitude Constraints (Clinical Standards)

```bash
--min_amplitude 75         # Minimum amplitude in µV (AASM standard)
--max_amplitude 300        # Maximum amplitude in µV
```

### Postprocessing Parameters

```bash
--use_postprocessing True  # Enable strict postprocessing during validation
--min_shape_quality 0.6    # Minimum shape quality score (0-1)
--min_snr 2.5              # Minimum signal-to-noise ratio
```

### Multi-Task Learning

```bash
--use_auxiliary True       # Enable auxiliary tasks (peak + zero-crossing)
```

## Usage Examples

### Example 1: Default STRICT Training

```bash
python micro_event/train_kcomplex_improved.py \
    --gpu 0 \
    --epochs 50 \
    --save True \
    --tag "strict_v1"
```

**Output:**
```
================================================================================
K-Complex Detection Training - Improved Architecture
================================================================================
Device: cuda:0
Use auxiliary tasks: True
Batch size: 16
Learning rate: 0.0001
...

Loss function configuration:
  Weights:
    Detection: 1.0
    Peak alignment: 0.4
    Peak ordering: 0.3
    Zero-crossing: 0.2
    Shape consistency: 0.5 (CRITICAL)
  Amplitude constraints:
    Minimum: 75 µV (clinical standard)
    Maximum: 300 µV

Postprocessing configuration:
  Enabled: True
  Min shape quality: 0.6
  Min SNR: 2.5
  N3 filtering: Enabled (temporal isolation + periodicity detection)

================================================================================
Epoch 001/050
--------------------------------------------------------------------------------
Training: 100%|████████████████████| 120/120 [01:24<00:00,  1.42it/s, loss=0.5234]
Train Loss: 0.5234
  Detection: 0.4123 | Peak align: 0.0512 | Peak order: 0.0234 | Zero-cross: 0.0156 | Shape: 0.0209

Val Loss: 0.4567 | Threshold: 0.3245
BEFORE Postprocessing:
  AUPRC: 0.6234 | P=0.5123 R=0.7234 F1=0.6012
AFTER Strict Postprocessing (N3 filtering enabled):
  P=0.7345 R=0.6123 F1=0.6678
  Rejection rate: 35.67% (candidates: 1234, validated: 794)
✓ Model saved: KC_strict_strict_v1_ep001_f10.6678_th0.3245.pth
```

### Example 2: Custom Loss Weights

Emphasize shape consistency even more:

```bash
python micro_event/train_kcomplex_improved.py \
    --gpu 0 \
    --weight_shape 0.8 \
    --weight_peak_align 0.5 \
    --weight_peak_order 0.4 \
    --epochs 50 \
    --tag "high_shape_weight"
```

### Example 3: Relaxed Amplitude (Not Recommended)

For research/comparison purposes only:

```bash
python micro_event/train_kcomplex_improved.py \
    --gpu 0 \
    --min_amplitude 50 \
    --max_amplitude 400 \
    --tag "relaxed_amplitude"
```

### Example 4: Disable Postprocessing

To see raw model performance without postprocessing:

```bash
python micro_event/train_kcomplex_improved.py \
    --gpu 0 \
    --use_postprocessing False \
    --tag "no_postprocessing"
```

**Note**: Model will still be trained with shape-aware loss, but validation won't apply strict postprocessing.

### Example 5: Baseline Comparison

Train without auxiliary tasks (baseline):

```bash
python micro_event/train_kcomplex_improved.py \
    --gpu 0 \
    --use_auxiliary False \
    --tag "baseline_no_auxiliary"
```

## Understanding the Output

### Training Epoch Output

```
Epoch 001/050
--------------------------------------------------------------------------------
Training: 100%|████████████████████| 120/120 [01:24<00:00,  1.42it/s, loss=0.5234]
Train Loss: 0.5234
  Detection: 0.4123 | Peak align: 0.0512 | Peak order: 0.0234 | Zero-cross: 0.0156 | Shape: 0.0209
```

- **Total Loss**: Weighted sum of all components
- **Detection**: Primary K-complex detection loss
- **Peak align**: Peak alignment within events
- **Peak order**: Positive before negative peak constraint
- **Zero-cross**: Zero-crossing boundary loss
- **Shape**: Shape consistency loss (CRITICAL)

### Validation Output

```
Val Loss: 0.4567 | Threshold: 0.3245
BEFORE Postprocessing:
  AUPRC: 0.6234 | P=0.5123 R=0.7234 F1=0.6012
AFTER Strict Postprocessing (N3 filtering enabled):
  P=0.7345 R=0.6123 F1=0.6678
  Rejection rate: 35.67% (candidates: 1234, validated: 794)
```

- **BEFORE**: Raw model predictions (all events > threshold)
- **AFTER**: Validated events after strict postprocessing
- **Rejection rate**: % of candidates rejected by postprocessing
- **Candidates**: Number of raw detections
- **Validated**: Number passing all validation criteria

**Key Insight**: High rejection rate (30-40%) is EXPECTED and GOOD. It means:
- Model is sensitive (high recall)
- Postprocessing filters false positives effectively
- N3 slow waves are being removed correctly

## Model Selection Strategy

The training script uses **POSTPROCESSED F1 score** for model selection:

```python
# Save best model based on POSTPROCESSED F1 score
f1_score = f1_after if after_metrics else f1_before
```

**Why?**
- Raw model F1 may be inflated by false positives
- Postprocessed F1 reflects real-world performance
- Better generalization to clinical data

## File Naming Convention

Saved models follow this format:

```
KC_strict_{tag}_ep{epoch:03d}_f1{f1:.4f}_th{threshold:.4f}.pth
```

Example:
```
KC_strict_strict_v1_ep023_f10.7234_th0.3456.pth
```

- `KC_strict`: K-complex with strict standards
- `{tag}`: Your experiment tag
- `ep023`: Epoch 23
- `f10.7234`: F1 score 0.7234
- `th0.3456`: Optimal threshold 0.3456

## Best Practices

### 1. **Monitor Both Metrics**

Always check both before/after postprocessing:

- **High BEFORE F1, Low AFTER F1**: Model generating too many false positives
- **Similar BEFORE/AFTER F1**: Model learning correct patterns (GOOD!)
- **Low BEFORE F1, High AFTER F1**: Unlikely, check for bugs

### 2. **Rejection Rate**

Typical rejection rates:

- **20-30%**: Good, moderate filtering
- **30-40%**: Normal, strict filtering (especially with N3 filtering)
- **>50%**: Model may be too sensitive, consider increasing detection threshold
- **<10%**: Model may be too conservative OR not enough N3 data

### 3. **Loss Component Balance**

Check loss components during training:

```
Detection: 0.41 | Peak align: 0.05 | Peak order: 0.02 | Zero-cross: 0.02 | Shape: 0.02
```

- **Detection should dominate** (0.3-0.5)
- **Auxiliary losses** should be smaller (0.01-0.10)
- **Shape loss** should be moderate (0.01-0.05)

If shape loss is very high (>0.2), model may be struggling to learn valid K-complex patterns.

### 4. **Hyperparameter Tuning**

Start with defaults, then tune:

1. **First**: Train with defaults
2. **If precision low**: Increase `--weight_shape` or `--min_shape_quality`
3. **If recall low**: Decrease `--min_amplitude` or `--min_shape_quality`
4. **If N3 false positives**: Already handled by isolation/periodicity checks

## Integration with Inference

After training, use the model with strict postprocessing:

```python
import torch
from models.kcomplex_detector import KComplexDetector
from postprocess.kcomplex_postprocessor_strict import postprocess_kcomplex_predictions_strict

# Load model
model = KComplexDetector(in_channels=1)
model.load_state_dict(torch.load('KC_strict_v1_ep023_f10.7234.pth'))
model.eval()

# Inference
with torch.no_grad():
    outputs = model(eeg_signal, return_auxiliary=True)
    logits = outputs['logits']
    probs = torch.softmax(logits, dim=-1)[:, :, 1]  # P(K-complex)

# Apply STRICT postprocessing
refined_preds, events_info = postprocess_kcomplex_predictions_strict(
    predictions=probs.cpu().numpy(),
    raw_signal=raw_eeg.cpu().numpy(),
    fs=200,
    threshold=0.3456,  # Use threshold from training
    min_amplitude=75,
    min_shape_quality=0.6,
    min_snr=2.5,
    check_context=True,  # N3 filtering
    refine_boundaries=True
)

# Results
for event in events_info:
    print(f"K-complex: {event['start_time']:.2f}s - {event['end_time']:.2f}s")
    print(f"  Amplitude: {event['amplitude']:.1f} µV")
    print(f"  Shape quality: {event['shape_quality']:.3f}")
```

## Troubleshooting

### Issue 1: Low F1 Score After Postprocessing

**Symptoms**: High F1 before postprocessing (0.7+), low after (0.3-)

**Causes**:
- Model not learning valid K-complex patterns
- Too many false positives in N3 regions

**Solutions**:
1. Increase `--weight_shape` to 0.8 or 1.0
2. Ensure training data includes N2 pages (default is correct)
3. Check if ground truth labels are accurate

### Issue 2: High Rejection Rate (>60%)

**Symptoms**: Most candidates rejected by postprocessing

**Causes**:
- Model threshold too low
- Model generating artifacts

**Solutions**:
1. The optimal threshold is learned during validation, this should auto-correct
2. Increase `--weight_peak_order` to enforce peak ordering
3. Increase `--min_amplitude` to filter small artifacts

### Issue 3: OOM (Out of Memory)

**Symptoms**: CUDA out of memory error

**Solutions**:
1. Reduce `--batch_size` to 8 or 4
2. Disable auxiliary tasks: `--use_auxiliary False`
3. Use gradient accumulation (requires code modification)

## Related Documentation

- `KCOMPLEX_STRICT_STANDARDS.md`: Complete validation criteria
- `KCOMPLEX_IMPROVEMENTS.md`: Architecture and implementation details
- `kcomplex_postprocessor_strict.py`: Postprocessing implementation
- `losses_kcomplex.py`: Loss function implementation

## Citation

If you use this code, please reference:

```
K-Complex Detection with Strict Clinical Standards
- Multi-task learning with shape-aware loss
- N3 slow wave filtering via temporal isolation and periodicity detection
- Clinical amplitude standards (75-300 µV, AASM guidelines)
```
