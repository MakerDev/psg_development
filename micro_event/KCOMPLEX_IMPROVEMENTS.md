# K-Complex Detection - Improved Architecture

이 문서는 K-complex 탐지 성능을 개선하기 위한 새로운 아키텍처와 방법론을 설명합니다.

## 문제 분석

### 기존 구현의 한계

1. **모델 아키텍처**
   - 단순 binary classification만 수행
   - K-complex의 형태적 특성(positive peak → negative peak)을 학습하지 않음
   - Zero-crossing 정보를 활용하지 않음

2. **Loss 함수**
   - 일반적인 focal loss만 사용
   - K-complex의 특성(peak 위치, amplitude, duration)을 반영하지 않음

3. **후처리**
   - `validate_kcomplex()`에서 올바른 검증 로직이 있지만, 학습 시점에는 반영되지 않음
   - 모델이 후처리에서 걸러질 False Positive를 계속 생성

## K-Complex의 특성

K-complex는 다음과 같은 명확한 특성을 가집니다:

1. **Peak 패턴**: Positive peak → Negative peak (이 순서가 중요!)
2. **Zero-crossing**: 이벤트 시작과 끝은 zero-crossing point에 위치
3. **Duration**: 0.08~0.7초 (positive peak에서 negative peak까지)
4. **Amplitude**: 15~250 µV (너무 크거나 작으면 artifact)
5. **단일 Peak**: 하나의 확실한 positive peak와 하나의 negative peak

## 개선 솔루션

### 1. Multi-Task Learning 모델 (`KComplexDetector`)

**위치**: `micro_event/models/kcomplex_detector.py`

#### 주요 특징:

```python
# Multi-scale input processing
입력: [raw_signal, abs(signal), derivative(signal)]
```

- **Raw signal**: 원본 EEG
- **Absolute value**: Amplitude 정보 강조
- **Derivative**: Zero-crossing 탐지에 유리 (부호 변화 감지)

#### Multi-Task Outputs:

1. **Primary Task**: K-complex detection (binary classification)
2. **Auxiliary Task 1**: Peak location prediction (positive/negative peak 위치)
3. **Auxiliary Task 2**: Zero-crossing detection (이벤트 경계)

```python
outputs = model(x, return_auxiliary=True)
# returns:
# {
#   'logits': (batch, time, 2),        # K-complex detection
#   'peaks': (batch, time, 2),         # [pos_peak, neg_peak] probabilities
#   'zerocross': (batch, time, 1)      # zero-crossing probabilities
# }
```

#### 구조적 개선:

- **Multi-dilated convolutions**: 다양한 시간 스케일 포착
- **Bidirectional LSTM**: 시간적 문맥 파악
- **Separate heads**: 각 task별 전문화된 출력 헤드

### 2. Shape-Aware Loss 함수 (`KComplexLoss`)

**위치**: `micro_event/losses_kcomplex.py`

#### 구성 요소:

##### a) Detection Loss (Focal Loss)
```python
# 기본 K-complex detection
masked_focal_loss(logits, targets, mask)
```

##### b) Peak Alignment Loss
```python
# K-complex 이벤트 내에서 peak 확률을 최대화
# 이벤트 외부에서는 최소화
peak_alignment_loss(peak_probs, targets, mask)
```

##### c) Peak Ordering Loss
```python
# Positive peak이 negative peak보다 먼저 나타나도록 강제
# 시간 차이가 0.08~0.7초 사이에 있도록 제약
peak_ordering_loss(peak_probs, targets, mask)
```

##### d) Zero-Crossing Boundary Loss
```python
# 이벤트 경계에서 zero-crossing 확률을 최대화
zerocrossing_boundary_loss(zerocross_probs, targets, mask)
```

##### e) Shape Consistency Loss
```python
# 원본 신호에서 실제로 올바른 K-complex 패턴을 검증
# - Positive peak → negative peak 순서
# - 적절한 amplitude (15-250 µV)
# - 적절한 duration (0.08-0.7초)
shape_consistency_loss(raw_signal, predictions, targets, mask)
```

#### 사용 예시:

```python
criterion = KComplexLoss(
    weight_detection=1.0,      # Primary task
    weight_peak_align=0.3,     # Auxiliary: peak detection
    weight_peak_order=0.2,     # Auxiliary: peak ordering
    weight_zerocross=0.2,      # Auxiliary: boundary detection
    weight_shape=0.1,          # Shape validation
    fs=200
)

loss, loss_dict = criterion(outputs, targets, mask, raw_signal)
```

### 3. Zero-Crossing 기반 후처리

**위치**: `micro_event/postprocess/kcomplex_postprocessor.py`

#### 주요 기능:

##### a) Zero-Crossing Boundary Refinement
```python
refined_start, refined_end = refine_event_boundaries_with_zerocrossing(
    signal, event_start, event_end, fs=200
)
```
- 이벤트 경계를 가장 가까운 zero-crossing point로 조정
- 최대 300ms까지만 확장 (과도한 확장 방지)

##### b) Peak Detection and Validation
```python
peak_info = detect_kcomplex_peaks(signal, fs=200)
# Returns:
# {
#   'has_valid_pattern': bool,
#   'pos_peak_idx': int,
#   'neg_peak_idx': int,
#   'pos_amplitude': float,
#   'neg_amplitude': float,
#   'peak_to_peak': float,
#   'duration': float
# }
```

검증 항목:
- Positive peak이 negative peak보다 먼저 나타나는가?
- 시간 차이가 0.08~0.7초 사이인가?
- Multiple similar peaks가 없는가? (artifact 제거)
- Amplitude가 적절한가? (15-250 µV)

##### c) Comprehensive Event Validation
```python
is_valid, info = validate_kcomplex_event(
    raw_signal, event_start, event_end, fs=200,
    min_amplitude=15, max_amplitude=250,
    min_duration=0.15, max_duration=1.5
)
```

검증 항목:
1. SNR (Signal-to-Noise Ratio) ≥ 1.5
2. Valid peak pattern (pos → neg)
3. Amplitude constraints
4. Duration constraints
5. Artifact detection (급격한 변화 제거)

##### d) End-to-End Post-processing
```python
refined_preds, events_info = postprocess_kcomplex_predictions(
    predictions, raw_signal, fs=200,
    min_amplitude=15, max_amplitude=250,
    min_duration=0.15, max_duration=1.5,
    threshold=0.5,
    refine_boundaries=True
)
```

### 4. 개선된 Training Script

**위치**: `micro_event/train_kcomplex_improved.py`

#### 사용 방법:

```bash
# 기본 학습 (모든 auxiliary tasks 활성화)
python micro_event/train_kcomplex_improved.py \
    --gpu 0 \
    --lr 1e-4 \
    --batch_size 16 \
    --epochs 50 \
    --use_auxiliary True \
    --save True \
    --tag "v1"

# Loss weights 조정
python micro_event/train_kcomplex_improved.py \
    --gpu 0 \
    --weight_detection 1.0 \
    --weight_peak_align 0.3 \
    --weight_peak_order 0.2 \
    --weight_zerocross 0.2 \
    --weight_shape 0.1 \
    --tag "custom_weights"

# Auxiliary tasks 없이 (baseline 비교용)
python micro_event/train_kcomplex_improved.py \
    --gpu 0 \
    --use_auxiliary False \
    --tag "baseline"
```

#### 주요 기능:

1. **Multi-task training**: Detection + Peak + Zero-crossing
2. **Loss monitoring**: 각 loss component별 추적
3. **Best model saving**: Validation F1 기준 자동 저장
4. **Reproducibility**: Random seed 고정

## 성능 개선 포인트

### 1. 학습 시점 개선
- **Before**: 모델이 K-complex 특성을 모르고 학습 → 후처리에서 많이 걸러짐
- **After**: 학습 시점에 K-complex 특성을 반영 → 더 정확한 detection

### 2. Zero-Crossing 활용
- **Before**: 후처리에서만 사용
- **After**: 모델이 zero-crossing을 학습 + 후처리에서 boundary 정제

### 3. Peak Pattern 학습
- **Before**: 단순 binary classification
- **After**: Positive/negative peak 위치를 명시적으로 학습

### 4. Amplitude & Duration 제약
- **Before**: 후처리에서만 검증
- **After**: Loss 함수에 반영 → 학습 중에도 제약 적용

## 예상 개선 효과

1. **False Positive 감소**
   - Peak pattern이 올바르지 않은 이벤트를 학습 중에 제거
   - Amplitude/duration이 부적절한 이벤트 필터링

2. **Boundary 정확도 향상**
   - Zero-crossing 학습으로 더 정확한 경계 예측
   - 후처리의 boundary refinement로 최종 정제

3. **Robust Detection**
   - Multi-scale input (raw, abs, derivative)으로 다양한 특성 포착
   - Shape consistency loss로 artifact 제거

4. **Interpretability**
   - Peak location prediction으로 모델이 "어디를" 보는지 알 수 있음
   - 각 loss component를 추적하여 학습 과정 분석 가능

## 사용 워크플로우

### 1. Training

```bash
# 1단계: 기본 학습
python micro_event/train_kcomplex_improved.py \
    --gpu 0 \
    --epochs 50 \
    --save True

# 2단계: Hyperparameter 튜닝 (필요시)
python micro_event/train_kcomplex_improved.py \
    --gpu 0 \
    --weight_peak_align 0.5 \
    --weight_shape 0.2 \
    --tag "tuned"
```

### 2. Inference (예시)

```python
import torch
from models.kcomplex_detector import KComplexDetector
from postprocess.kcomplex_postprocessor import postprocess_kcomplex_predictions

# Load model
model = KComplexDetector(in_channels=1)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Inference
with torch.no_grad():
    outputs = model(eeg_signal, return_auxiliary=True)
    logits = outputs['logits']
    probs = torch.softmax(logits, dim=-1)[:, :, 1]  # P(K-complex)

# Post-processing
refined_preds, events_info = postprocess_kcomplex_predictions(
    probs.cpu().numpy(),
    raw_eeg.cpu().numpy(),
    fs=200,
    threshold=0.4
)

# Results
for event in events_info:
    print(f"K-complex: {event['start_time']:.2f}s - {event['end_time']:.2f}s")
    print(f"  Amplitude: {event['amplitude']:.1f} µV")
    print(f"  Duration: {event['duration']:.3f}s")
```

## 파일 구조

```
micro_event/
├── models/
│   └── kcomplex_detector.py          # 개선된 모델 아키텍처
├── losses_kcomplex.py                 # K-complex 전용 loss 함수
├── postprocess/
│   └── kcomplex_postprocessor.py     # Zero-crossing 기반 후처리
├── train_kcomplex_improved.py        # 학습 스크립트
└── KCOMPLEX_IMPROVEMENTS.md          # 이 문서
```

## 기존 코드와의 호환성

- 기존 dataset 클래스 (`SleepEventDatasetEBX`) 그대로 사용 가능
- 기존 후처리 함수들과 병행 사용 가능
- 점진적 migration 가능:
  1. 먼저 새 모델만 테스트
  2. 그 다음 새 loss 함수 적용
  3. 마지막으로 후처리 개선

## 추가 개선 가능 항목

1. **Data Augmentation**
   - Time stretching (duration 변화)
   - Amplitude scaling
   - Noise injection

2. **Advanced Architecture**
   - Transformer-based attention
   - Multi-resolution analysis
   - Ensemble methods

3. **Active Learning**
   - False positive 케이스를 선별적으로 재학습
   - Hard negative mining

4. **Transfer Learning**
   - Spindle detector에서 학습한 feature 활용
   - Cross-subject adaptation

## 문의

이 개선 사항에 대한 질문이나 피드백이 있으시면 이슈를 등록해주세요.
