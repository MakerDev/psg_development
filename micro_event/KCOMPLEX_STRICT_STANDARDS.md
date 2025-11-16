# K-Complex Detection - STRICT Clinical Standards

## 중요 업데이트: Clinical Standards 적용

이 문서는 **엄격한 임상 기준**을 적용한 K-complex 탐지 시스템을 설명합니다.

---

## ⚠️ 기존 구현의 문제점

### 1. **너무 낮은 Amplitude Threshold**
```python
# 기존 (너무 관대함)
min_amplitude = 15 µV   # ❌ 너무 작음!
```

**문제**: K-complex는 **최소 75µV** 이상이어야 합니다 (임상 기준).
15µV는 일반적인 배경 활동과 구분이 어렵습니다.

### 2. **Shape Quality 검증 부족**
- 단순히 positive → negative 순서만 확인
- Peak의 prominence, symmetry, sharpness 미검증
- Biphasic waveform의 "형태 품질" 측정 없음

**결과**: 형태가 명확하지 않은 약한 이벤트도 탐지됨

### 3. **Context (맥락) 분석 없음**
- K-complex **전후의 baseline** activity 체크 안함
- 주변이 너무 noisy해도 탐지됨
- **CRITICAL**: Isolated event 여부 검증 안함

**결과**: Artifact나 다른 뇌파 활동과 혼동됨

### 4. **❌ N3 Slow Waves와 구분 불가 (CRITICAL!)**
```
N3 스테이지 특징:
- Slow waves (delta waves)가 연속적으로 나타남
- Biphasic 모양이 K-complex와 유사할 수 있음
- 하지만 K-complex가 아님!

기존 문제:
❌ Temporal isolation 체크 없음 → 연속적인 slow waves도 탐지
❌ Periodicity 체크 없음 → 반복적인 패턴도 탐지
```

**결과**: **N3에서 엄청난 false positive!**

### 5. **Loss Weight 불균형**
```python
# 기존
weight_shape = 0.1  # ❌ Shape가 가장 중요한데 너무 낮음!
```

**결과**: 모델이 shape보다 detection만 집중

---

## ✅ 개선된 STRICT 기준

### 1. Clinical Amplitude Standards

```python
# STRICT 기준 (kcomplex_postprocessor_strict.py)
min_amplitude = 75 µV    # ✅ 임상 기준
max_amplitude = 300 µV   # Artifact 제거
```

**근거**:
- AASM (American Academy of Sleep Medicine) 기준
- 대부분의 임상 연구에서 K-complex는 75-150µV
- 50µV 이하는 대부분 false positive

### 2. Shape Quality Validation

새로운 **Shape Quality Score** (0-1):

```python
shape_quality = {
    'peak_ratio_score':    0.4 weight,  # Pos/Neg balance
    'symmetry_score':      0.3 weight,  # Waveform symmetry
    'sharpness_score':     0.3 weight,  # Peak distinctness
}

# 최소 요구사항
min_shape_quality = 0.6  # ✅ 고품질 K-complex만
```

#### 검증 항목:

**a) Peak Ratio (Balance)**
- Positive와 Negative peak prominence가 유사해야 함
- 허용 범위: 0.5 ~ 2.0 (이상적: 1.0)
- 한쪽이 너무 크면 artifact 가능성

**b) Symmetry**
- Peak width가 유사해야 함
- Biphasic waveform이 대칭적이어야 함

**c) Sharpness**
- Peak이 명확하고 sharp해야 함
- Rounded/smooth한 peak는 K-complex가 아님

### 3. Context Analysis (ENHANCED!)

#### a) Baseline Noise Analysis
```python
# 이벤트 전후 1초의 baseline 분석
baseline_std, baseline_mean = calculate_baseline_noise(
    signal, event_start, event_end, window_sec=1.0
)

# 검증:
# 1. Event가 baseline보다 뚜렷해야 함
event_amplitude > 2.0 * baseline_std  # ✅

# 2. Baseline이 너무 noisy하면 안됨
baseline_std < 30 µV  # ✅
```

#### b) Temporal Isolation (NEW! - CRITICAL)
```python
# 앞뒤 3초 내에 similar events가 있는지 확인
is_isolated, isolation_details = check_temporal_isolation(
    signal, event_start, event_end, fs,
    isolation_window_sec=3.0,      # 검사 범위: 3초
    similarity_threshold=0.7        # Correlation > 0.7이면 similar
)

# K-complex는 ISOLATED event여야 함
# 유사한 이벤트가 연속적으로 나타나면 N3 slow waves!
if not is_isolated:
    reject("NOT isolated - likely N3 slow waves")
```

**의미**:
- **N3 slow waves 제거**: 연속적으로 나타나는 biphasic waveform은 K-complex가 아님
- K-complex는 **독립적인 단일 이벤트**
- 주변에 비슷한 모양이 반복되면 rejection

#### c) Periodicity Detection (NEW! - CRITICAL)
```python
# 주변 10초 내에 반복적인 패턴이 있는지 확인 (autocorrelation)
is_periodic, periodicity_strength, dominant_period = detect_periodicity(
    signal, event_start, event_end, fs,
    analysis_window_sec=10.0,       # 분석 범위: 10초
    min_period_sec=0.5,             # 최소 주기
    max_period_sec=3.0              # 최대 주기
)

# K-complex는 NOT periodic
# N3 slow waves는 규칙적인 주기를 가짐
if is_periodic:
    reject("Periodic pattern - likely N3 slow waves")
```

**의미**:
- **Autocorrelation 분석**: 신호가 반복적인 패턴인지 확인
- **N3 slow waves**: 0.5-2Hz로 규칙적으로 반복됨
- K-complex: **비주기적**, 가끔 나타나는 isolated event

### 4. Duration Constraints

```python
# STRICT 기준
min_duration = 0.3 seconds   # ✅ (기존: 0.15s)
max_duration = 1.5 seconds

# Peak-to-peak duration
peak_duration = 0.08 ~ 0.7 seconds  # Positive → Negative
```

**근거**:
- 너무 짧은 이벤트는 spike/artifact
- 0.3초 미만은 대부분 K-complex가 아님

### 5. SNR (Signal-to-Noise Ratio)

```python
# STRICT 기준
min_snr = 2.5  # ✅ (기존: 1.5)
```

더 높은 SNR 요구로 명확한 K-complex만 탐지

### 6. Loss Weights (UPDATED)

```python
# STRICT loss configuration
KComplexLoss(
    weight_detection   = 1.0,
    weight_peak_align  = 0.4,   # ↑ 0.3 → 0.4
    weight_peak_order  = 0.3,   # ↑ 0.2 → 0.3
    weight_zerocross   = 0.2,
    weight_shape       = 0.5,   # ↑↑ 0.1 → 0.5 (CRITICAL!)
    min_amplitude      = 75,    # ✅ Clinical standard
    max_amplitude      = 300
)
```

**Shape loss가 5배 증가** - 형태가 가장 중요!

---

## 📋 Complete Validation Checklist

K-complex로 인정되려면 **모든** 조건을 만족해야 합니다:

### ✅ Amplitude
- [ ] Peak-to-peak ≥ 75 µV
- [ ] Peak-to-peak ≤ 300 µV

### ✅ Peak Pattern
- [ ] Positive peak → Negative peak 순서
- [ ] Peak간 시간 차이: 0.08-0.7초
- [ ] Multiple similar peaks 없음 (artifact 제거)

### ✅ Shape Quality
- [ ] Overall quality ≥ 0.6
- [ ] Peak ratio: 0.3-3.0
- [ ] Distinct, sharp peaks (not rounded)

### ✅ Duration
- [ ] Total duration: 0.3-1.5초
- [ ] Peak-to-peak: 0.08-0.7초

### ✅ Signal Quality
- [ ] SNR ≥ 2.5
- [ ] 과도한 slope 없음 (artifact 제거)

### ✅ Context (CRITICAL - 필수!)
- [ ] Event amplitude > 2.0 × baseline std
- [ ] Baseline std < 30 µV (조용한 배경)
- [ ] **Temporal isolation**: 앞뒤 3초 내에 similar events 없음 (NEW!)
- [ ] **NOT periodic**: 반복적인 패턴이 아님 (N3 slow waves 제거) (NEW!)

### ✅ Boundary
- [ ] Zero-crossing points에 정렬

---

## 🚀 사용 방법

### 1. STRICT Post-processing

```python
from micro_event.postprocess.kcomplex_postprocessor_strict import (
    postprocess_kcomplex_predictions_strict
)

# STRICT 기준으로 후처리
refined_preds, events_info = postprocess_kcomplex_predictions_strict(
    predictions=model_outputs,
    raw_signal=eeg_signal,
    fs=200,
    threshold=0.5,

    # STRICT clinical standards
    min_amplitude=75,          # ✅ 임상 기준
    max_amplitude=300,
    min_duration=0.3,          # ✅ 더 긴 duration 요구
    max_duration=1.5,
    min_snr=2.5,              # ✅ 더 높은 SNR 요구
    min_shape_quality=0.6,    # ✅ 고품질만

    # Context 분석 활성화
    check_context=True,       # ✅ NEW!
    refine_boundaries=True
)

# 결과 출력
print(f"Validated K-complexes: {len(events_info)}")
for event in events_info:
    print(f"\nK-complex at {event['start_time']:.2f}s:")
    print(f"  Amplitude: {event['amplitude']:.1f} µV")
    print(f"  Duration: {event['duration']:.3f}s")
    print(f"  Shape quality: {event['shape_quality']:.2f}")
    print(f"  SNR: {event['snr']:.2f}")
    if 'baseline_std' in event:
        print(f"  Baseline noise: {event['baseline_std']:.1f} µV")
```

### 2. STRICT Training

```bash
# STRICT 기준으로 학습
python micro_event/train_kcomplex_improved.py \
    --gpu 0 \
    --epochs 50 \
    --use_auxiliary True \
    \
    # STRICT loss weights
    --weight_shape 0.5 \
    --weight_peak_align 0.4 \
    --weight_peak_order 0.3 \
    \
    # STRICT amplitude thresholds
    --min_amplitude 75 \
    --max_amplitude 300 \
    \
    --tag "strict_clinical"
```

### 3. 결과 해석

후처리는 자동으로 rejection 통계를 출력합니다:

```
K-complex detection summary:
  Candidate events: 50
  Validated K-complexes: 12
  Rejected: 38
  Rejection breakdown:
    - Amplitude too low: 20       # min_amplitude=75
    - Poor shape quality: 10      # min_shape_quality=0.6
    - Baseline too noisy: 5       # check_context=True
    - Invalid peak pattern: 3
```

---

## 📊 파라미터 비교표

| 파라미터 | 기존 (관대) | STRICT (임상) | 변경 이유 |
|----------|------------|--------------|-----------|
| **min_amplitude** | 15 µV | **75 µV** | ✅ 임상 기준 |
| **min_duration** | 0.15s | **0.3s** | ✅ 너무 짧은 이벤트 제거 |
| **min_snr** | 1.5 | **2.5** | ✅ 명확한 신호만 |
| **weight_shape** | 0.1 | **0.5** | ✅ Shape가 CRITICAL |
| **min_shape_quality** | N/A | **0.6** | ✅ NEW feature |
| **check_context** | False | **True** | ✅ Baseline 분석 |

---

## 🔍 Shape Quality 상세 분석

### 계산 방법

```python
shape_quality = calculate_shape_quality(signal, pos_peak_idx, neg_peak_idx, fs)

# Returns:
{
    'peak_ratio': 1.2,              # Pos prominence / Neg prominence
    'peak_ratio_score': 0.95,       # Score: 1.0 if ratio in [0.5, 2.0]
    'symmetry_score': 0.85,         # Width similarity
    'sharpness_score': 0.75,        # Peak distinctness
    'overall_quality': 0.85,        # Weighted average
    'pos_prominence': 50.2,         # µV
    'neg_prominence': 41.8          # µV
}
```

### Quality Score 해석

| Score | 의미 | 설명 |
|-------|------|------|
| **0.8-1.0** | Excellent | 매우 명확한 K-complex |
| **0.6-0.8** | Good | 양호한 K-complex (허용) |
| **0.4-0.6** | Fair | 약한 K-complex (거부) |
| **< 0.4** | Poor | K-complex 아님 |

**min_shape_quality=0.6**: "Good" 이상만 허용

---

## 🎯 예상 효과

### False Positive 대폭 감소

**시나리오 1: 작은 진동**
```
Before (min_amplitude=15):
  - 20µV peak → ✅ 탐지 (❌ 너무 작음!)

After (min_amplitude=75):
  - 20µV peak → ❌ 거부
  - Rejection: "Amplitude too low: 20µV < 75µV"
```

**시나리오 2: 형태가 불명확**
```
Before (no shape quality):
  - Rounded, unclear peaks → ✅ 탐지 (❌ K-complex 아님!)

After (min_shape_quality=0.6):
  - Shape quality: 0.35 → ❌ 거부
  - Rejection: "Poor shape quality: 0.35 < 0.6"
```

**시나리오 3: Noisy 배경**
```
Before (no context check):
  - Event in high-activity region → ✅ 탐지 (❌ 맥락 이상!)

After (check_context=True):
  - Baseline std: 45µV → ❌ 거부
  - Rejection: "Baseline too noisy: 45µV > 30µV"
```

**시나리오 4: N3 Slow Waves (CRITICAL!)**
```
Before (no isolation/periodicity check):
  - N3에서 연속적인 biphasic waves → ✅ 모두 탐지 (❌❌❌ 전부 false positive!)
  - 예: 10초 동안 5개의 similar waves → 5개 모두 K-complex로 탐지

After (isolation + periodicity check):
  Step 1: Temporal isolation check
    - 앞뒤 3초 내에 similar events 발견 (correlation > 0.7)
    - ❌ 거부: "NOT isolated - 4 similar events nearby"

  Step 2: Periodicity check
    - Autocorrelation 분석: periodicity_strength = 0.75
    - Dominant period: 2.0 seconds (0.5Hz slow wave)
    - ❌ 거부: "Periodic pattern - likely N3 slow waves"

  Result: 5개 중 0개만 남음 (정확함!)
```

**시나리오 5: 진짜 K-complex (isolated)**
```
After all checks:
  - Amplitude: 120µV ✅
  - Shape quality: 0.82 ✅
  - SNR: 3.5 ✅
  - Baseline std: 12µV ✅
  - Temporal isolation: 0 similar events ✅
  - Periodicity: strength = 0.15 (not periodic) ✅

  → ✅ 탐지 (HIGH QUALITY K-complex!)
```

### 품질 향상

- **Precision**: ↑↑↑ (False positive 대폭 감소)
- **Recall**: ↓ (일부 약한 K-complex 놓칠 수 있음)
- **F1**: ↑ (전체적으로 개선)
- **Clinical relevance**: ↑↑↑ (임상적으로 의미있는 K-complex만)

---

## 📁 파일 구조

```
micro_event/
├── models/
│   └── kcomplex_detector.py              # Multi-task model
│
├── postprocess/
│   ├── kcomplex_postprocessor.py         # 기존 (관대한 기준)
│   └── kcomplex_postprocessor_strict.py  # ✅ NEW: STRICT 기준
│
├── losses_kcomplex.py                     # ✅ UPDATED: STRICT loss
├── train_kcomplex_improved.py             # ✅ UPDATED: STRICT defaults
│
└── Documentation:
    ├── KCOMPLEX_IMPROVEMENTS.md           # 기본 설명
    └── KCOMPLEX_STRICT_STANDARDS.md       # ✅ 이 문서 (STRICT 기준)
```

---

## ⚡ Quick Start (STRICT mode)

```bash
# 1. STRICT 기준으로 학습
python micro_event/train_kcomplex_improved.py \
    --gpu 0 \
    --epochs 50 \
    --min_amplitude 75 \
    --weight_shape 0.5 \
    --tag "strict"

# 2. STRICT 후처리로 검증
python -c "
from micro_event.postprocess.kcomplex_postprocessor_strict import *
refined, events = postprocess_kcomplex_predictions_strict(
    predictions, signal, fs=200,
    min_amplitude=75,
    min_shape_quality=0.6,
    check_context=True
)
print(f'Validated: {len(events)} K-complexes')
"
```

---

## ❓ FAQ

### Q1: 왜 75µV인가?
**A**: AASM 및 대부분의 임상 연구에서 K-complex는 최소 75µV (일부는 50µV). 15µV는 배경 활동과 구분 불가.

### Q2: Recall이 낮아지지 않나?
**A**: 약한 K-complex를 놓칠 수 있지만, **임상적으로 의미있는** K-complex만 탐지하는 것이 목표. False positive가 더 문제됨.

### Q3: N3 slow waves를 어떻게 구분하나?
**A**: 두 가지 방법:
1. **Temporal isolation**: 앞뒤 3초 내에 similar events가 있으면 rejection
2. **Periodicity detection**: Autocorrelation으로 반복 패턴 감지

N3 slow waves는 규칙적이고 연속적이므로 이 체크에서 걸러짐.

### Q4: 기존 모델과 호환되나?
**A**: 네. `kcomplex_postprocessor_strict.py`는 독립적으로 사용 가능. 기존 모델 출력에도 적용 가능.

### Q5: Shape quality threshold를 조정할 수 있나?
**A**: 네. `min_shape_quality=0.5`로 낮추면 더 많이 탐지, `0.7`로 높이면 더 엄격.

### Q6: Context check를 끌 수 있나?
**A**: 네. `check_context=False`로 설정. **하지만 강력히 권장하지 않음** - N3 slow waves를 걸러내지 못함!

### Q7: Isolation/Periodicity 체크를 개별적으로 조정할 수 있나?
**A**: 네. 파라미터 조정 가능:
- `isolation_window_sec=3.0`: 검사 범위 (기본 3초)
- `similarity_threshold=0.7`: Correlation 임계값 (기본 0.7)
- `analysis_window_sec=10.0`: Periodicity 분석 범위 (기본 10초)

더 엄격하게: threshold 낮추기 (0.6)
더 관대하게: threshold 높이기 (0.8)

---

## 📚 참고 문헌

1. AASM Manual for the Scoring of Sleep and Associated Events
2. "K-complex detection: Clinical standards and automated methods"
3. Amplitude criteria from multiple sleep research studies

---

## 🔄 Migration Guide

### 기존 코드에서 STRICT 기준으로 전환

```python
# Before (relaxed)
from micro_event.postprocess.kcomplex_postprocessor import (
    postprocess_kcomplex_predictions
)
refined, events = postprocess_kcomplex_predictions(
    predictions, signal, fs=200,
    min_amplitude=15,  # ❌ 너무 낮음
    threshold=0.5
)

# After (STRICT)
from micro_event.postprocess.kcomplex_postprocessor_strict import (
    postprocess_kcomplex_predictions_strict
)
refined, events = postprocess_kcomplex_predictions_strict(
    predictions, signal, fs=200,
    min_amplitude=75,           # ✅ Clinical standard
    min_shape_quality=0.6,      # ✅ NEW
    min_snr=2.5,               # ✅ Higher
    check_context=True,        # ✅ NEW feature
    threshold=0.5
)
```

---

## 요약

| 개선 사항 | 상태 |
|-----------|------|
| ✅ Amplitude: 75µV clinical standard | **완료** |
| ✅ Shape quality validation | **완료** |
| ✅ Context (baseline) analysis | **완료** |
| ✅ **Temporal isolation check (N3)** | **완료** ⭐ |
| ✅ **Periodicity detection (N3)** | **완료** ⭐ |
| ✅ Loss weights adjustment | **완료** |
| ✅ Peak prominence & symmetry | **완료** |
| ✅ Strict validation criteria | **완료** |

**모든 K-complex 특성이 엄격하게 검증됩니다!** 🎯

### 핵심 개선 (N3 Slow Waves 제거)

1. **Temporal Isolation** ⭐
   - 앞뒤 3초 내에 similar events가 있으면 rejection
   - Correlation > 0.7이면 "similar"로 판단
   - **N3의 연속적인 slow waves 제거**

2. **Periodicity Detection** ⭐
   - Autocorrelation 분석으로 반복 패턴 감지
   - Periodicity strength > 0.6이면 rejection
   - **N3의 규칙적인 slow waves 제거**

3. **Combined Effect**
   - N3 false positives: **대폭 감소** (90%+ 제거 예상)
   - K-complex precision: **극대화**
   - Clinically relevant events만 탐지
