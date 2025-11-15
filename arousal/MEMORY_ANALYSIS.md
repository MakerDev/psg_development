# Memory Usage Analysis - Multimodal Arousal Detection

## Overview
이 문서는 multimodal arousal detection 시스템의 메모리 사용량을 상세히 분석합니다.

---

## 1. 전처리 단계 (prep_spectrogram_combined.py)

### 입력 데이터
- **EDF 파일**: 8시간 수면 기록
- **채널 수**: 9 (F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1, LOC, ROC, EMG)
- **샘플링 레이트**: 50 Hz
- **총 샘플 수**: 8 hours × 3600 sec × 50 Hz = 1,440,000 samples

### 메모리 계산

#### 원본 데이터 로딩
```python
data = raw.get_data()  # (channels, time)
```
- Shape: `(9, 1,440,000)`
- dtype: `float64` (8 bytes)
- **메모리**: 9 × 1,440,000 × 8 = **103.68 MB**

#### 정규화된 데이터
```python
data_norm = robust_scale(data, fs=50)
```
- Shape: `(9, 1,440,000)`
- dtype: `float32` (4 bytes)
- **메모리**: 9 × 1,440,000 × 4 = **51.84 MB**

#### 60초 청크 하나당 메모리

**청크 크기**: 60 sec × 50 Hz = 3,000 samples

##### 1) Time Domain Features

**a) x_time_raw**
```python
x_time_raw = data_chunk.T  # (time, channels)
```
- Shape: `(3000, 9)`
- dtype: `float32`
- **메모리**: 3000 × 9 × 4 = **108 KB**

**b) envelope (Hilbert transform)**
```python
envelope = extract_amplitude_envelope(data_chunk)
```
- Shape: `(9, 3000)`
- dtype: `float32`
- **메모리**: 9 × 3000 × 4 = **108 KB**

**c) first_deriv, second_deriv**
```python
first_deriv = np.gradient(envelope, axis=1) * fs
second_deriv = np.gradient(first_deriv, axis=1) * fs
```
- 각각 Shape: `(9, 3000)`
- dtype: `float32`
- **메모리 (각)**: 9 × 3000 × 4 = **108 KB**
- **총**: 108 KB × 2 = **216 KB**

**d) x_time_combined**
```python
x_time_combined = np.stack([data_chunk, envelope, first_deriv, second_deriv], axis=1)
```
- Shape: `(9, 4, 3000)`
- dtype: `float32`
- **메모리**: 9 × 4 × 3000 × 4 = **432 KB**

##### 2) Frequency Domain Features

**Spectrogram 파라미터**:
- nperseg = 100 (2초)
- noverlap = 50 (1초)
- 60초 데이터 → 약 119 time bins
- freq bins = nperseg/2 + 1 = 51

**x_spec**
```python
spec, freqs, times = make_spectrogram(x_time_raw, fs=50, nperseg=100, noverlap=50)
```
- Shape: `(9, 51, 119)`
- dtype: `float32`
- **메모리**: 9 × 51 × 119 × 4 = **219 KB**

##### 3) Statistical Features

**파라미터**:
- window_sec = 2초 → window_size = 100 samples
- stride = 50 samples (50% overlap)
- 60초 → 약 59 windows
- n_features = 6 (mean, std, min, max, skewness, kurtosis)

**stat_features**
```python
stat_features = extract_statistical_features(data_chunk, fs=50, window_sec=2)
```
- Shape: `(9, 6, 59)`
- dtype: `float32`
- **메모리**: 9 × 6 × 59 × 4 = **12.7 KB**

##### 4) Labels

**y_time**
- Shape: `(3000,)`
- dtype: `float32`
- **메모리**: 3000 × 4 = **12 KB**

**y_spec**
- Shape: `(119,)`
- dtype: `float32`
- **메모리**: 119 × 4 = **0.5 KB**

##### 5) Metadata
```python
freqs, times, artifact_mask, meas_date, chunk_idx, etc.
```
- **메모리**: 약 **5 KB**

### 청크 하나당 총 메모리

| 항목 | 크기 |
|------|------|
| x_time_raw | 108 KB |
| x_time_combined | 432 KB |
| envelope | 108 KB |
| x_spec | 219 KB |
| stat_features | 12.7 KB |
| y_time | 12 KB |
| y_spec | 0.5 KB |
| metadata | 5 KB |
| **총계** | **~897 KB ≈ 0.88 MB** |

### 전체 파일 처리 시 메모리

**8시간 파일**:
- 총 청크 수: 8 × 60 = 480 chunks
- 저장되는 총 데이터: 480 × 0.88 MB = **422 MB**

**처리 중 피크 메모리**:
- 원본 데이터: 103.68 MB
- 정규화 데이터: 51.84 MB
- 현재 청크 처리: ~10 MB (임시 배열들)
- **피크 메모리**: **~165 MB**

---

## 2. 학습 단계 (train_deepsleep.py)

### 모델 파라미터 메모리

#### DeepSleepMultimodal 구조
```python
model = DeepSleepMultimodal(n_channels=9, base_ch=32, use_attention=True)
```

**파라미터 수 분석**:

##### Time Branch
- feature_conv: 9×32×4 = 1,152 params
- temporal_conv: 32×64 + 64×64 = 6,144 params
- down1: 64×128 conv = 122,880 params
- down2: 128×256 conv = 114,688 params
- **소계**: ~245K params

##### Frequency Branch
- conv1: 9×32 conv2d = 2,880 params
- down1: 32×64 conv2d = 18,432 params
- down2: 64×128 conv2d = 73,728 params
- down3: 128×256 conv2d = 294,912 params
- **소계**: ~390K params

##### Amplitude Branch
- feature_conv: 9×32×6 = 1,728 params
- temporal_conv: 32×64 + 64×128 = 10,240 params
- **소계**: ~12K params

##### Attention Fusion
- channel_attention: (256+256+128) → 640/4 → 640 = ~410K params
- gates: 3 params
- **소계**: ~410K params

##### Final Layers
- final_conv: 640→256→128→64 = ~738K params
- upsample: 64→32→16 = ~10K params
- out_conv: 16→1 = 16 params
- **소계**: ~748K params

**총 파라미터 수**: ~2,458,763 params

**모델 파라미터 메모리**:
- float32: 2,458,763 × 4 = **9.4 MB**
- 그래디언트 (float32): 2,458,763 × 4 = **9.4 MB**
- 옵티마이저 상태 (Adam, 2개 momentum): 2,458,763 × 4 × 2 = **18.8 MB**
- **총 모델 관련 메모리**: **37.6 MB**

### 배치 처리 메모리

#### Batch Size = 4일 때

**입력 데이터**:

1. **x_time_combined**: `(4, 9, 4, 3000)`
   - 메모리: 4 × 9 × 4 × 3000 × 4 = **1.73 MB**

2. **x_spec**: `(4, 9, 51, 119)`
   - 메모리: 4 × 9 × 51 × 119 × 4 = **0.88 MB**

3. **x_stat**: `(4, 9, 6, 59)`
   - 메모리: 4 × 9 × 6 × 59 × 4 = **0.05 MB**

4. **y**: `(4, 3000)`
   - 메모리: 4 × 3000 × 4 = **0.05 MB**

**입력 총계**: **2.71 MB**

#### Forward Pass 중간 활성화 (Activations)

**Time Branch**:
- x0: (4, 32, 3000) = 4 × 32 × 3000 × 4 = **1.54 MB**
- x1: (4, 64, 1500) = 4 × 64 × 1500 × 4 = **1.54 MB**
- x2: (4, 128, 750) = 4 × 128 × 750 × 4 = **1.54 MB**
- x3: (4, 256, 375) = 4 × 256 × 375 × 4 = **1.54 MB**
- **소계**: ~6.16 MB

**Frequency Branch**:
- x0: (4, 32, 51, 119) = 4 × 32 × 51 × 119 × 4 = **3.12 MB**
- x1: (4, 64, 26, 60) = 4 × 64 × 26 × 60 × 4 = **1.60 MB**
- x2: (4, 128, 13, 30) = 4 × 128 × 13 × 30 × 4 = **0.80 MB**
- x3: (4, 256, 7, 15) = 4 × 256 × 7 × 15 × 4 = **0.43 MB**
- **소계**: ~5.95 MB

**Amplitude Branch**:
- x0: (4, 32, 59) = 4 × 32 × 59 × 4 = **0.03 MB**
- x1: (4, 64, 59) = 4 × 64 × 59 × 4 = **0.06 MB**
- x2: (4, 128, 59) = 4 × 128 × 59 × 4 = **0.12 MB**
- **소계**: ~0.21 MB

**Fusion + Final**:
- fused: (4, 640, 375) = 4 × 640 × 375 × 4 = **3.84 MB**
- final_conv: (4, 64, 375) = 4 × 64 × 375 × 4 = **0.38 MB**
- upsample: (4, 16, 1500) = 4 × 16 × 1500 × 4 = **0.38 MB**
- output: (4, 1, 3000) = 4 × 1 × 3000 × 4 = **0.05 MB**
- **소계**: ~4.65 MB

**총 중간 활성화**: ~17 MB

#### Backward Pass (Gradients)

역전파 시에는 활성화와 동일한 크기의 그래디언트가 필요합니다.
- **그래디언트 메모리**: ~17 MB

### GPU 메모리 총계 (Batch Size = 4)

| 항목 | 메모리 |
|------|--------|
| 모델 파라미터 | 9.4 MB |
| 모델 그래디언트 | 9.4 MB |
| 옵티마이저 상태 | 18.8 MB |
| 입력 데이터 | 2.71 MB |
| Forward 활성화 | 17 MB |
| Backward 그래디언트 | 17 MB |
| CUDA 오버헤드 | ~500 MB |
| **총계** | **~574 MB** |

실제로는 PyTorch의 메모리 캐싱과 연산 버퍼로 인해 약간 더 사용:
- **실제 예상 GPU 메모리**: **~800 MB - 1.2 GB**

### 다양한 Batch Size별 메모리 사용량

| Batch Size | 입력 데이터 | 활성화 | 그래디언트 | 총 GPU 메모리 (예상) |
|-----------|------------|--------|-----------|-------------------|
| 1 | 0.68 MB | 4.25 MB | 4.25 MB | **600 MB - 800 MB** |
| 2 | 1.36 MB | 8.5 MB | 8.5 MB | **700 MB - 900 MB** |
| 4 | 2.71 MB | 17 MB | 17 MB | **800 MB - 1.2 GB** |
| 8 | 5.42 MB | 34 MB | 34 MB | **1.2 GB - 1.8 GB** |
| 16 | 10.84 MB | 68 MB | 68 MB | **2.0 GB - 2.8 GB** |

### DataLoader 메모리 (num_workers=4)

각 워커는 배치를 prefetch합니다:
- 워커당 메모리: ~10 MB (청크 로딩 + 변환)
- 4개 워커: 4 × 10 MB = **40 MB**
- 메인 프로세스: 배치 큐 = 2 × 2.71 MB = **5.4 MB**
- **총 DataLoader 메모리**: **~50 MB**

---

## 3. 전체 학습 프로세스 메모리 (시스템 RAM)

### 데이터셋 로딩

파일 경로만 메모리에 유지:
- 480 chunks × 100 bytes/path = **48 KB**

데이터는 on-the-fly로 로딩되므로 전체를 메모리에 유지하지 않음.

### 학습 중 RAM 사용량

| 항목 | 메모리 |
|------|--------|
| Python 인터프리터 | ~100 MB |
| PyTorch 라이브러리 | ~500 MB |
| 데이터 경로 | 0.05 MB |
| DataLoader workers | 50 MB |
| 기타 버퍼 | ~100 MB |
| **총 RAM** | **~750 MB - 1 GB** |

---

## 4. 추론 단계 메모리

### 단일 60초 청크 추론

**GPU 메모리**:
- 모델 파라미터: 9.4 MB
- 입력 데이터 (batch=1): 0.68 MB
- Forward 활성화: 4.25 MB
- CUDA 오버헤드: ~300 MB
- **총**: **~315 MB - 400 MB**

### 8시간 파일 전체 추론

청크별로 순차 처리하므로:
- **GPU 메모리**: **~315 MB - 400 MB** (일정)
- **처리 시간**: 480 chunks × 10 ms = **~5초**

---

## 5. 메모리 최적화 방안

### 현재 최적화

✅ **60초 청크**:
- 전체 파일(103 MB)을 한 번에 로딩하지 않음
- 청크별 처리로 메모리 사용 감소

✅ **On-the-fly 로딩**:
- 전체 데이터셋을 메모리에 유지하지 않음
- 필요할 때만 pickle 파일 로딩

✅ **Float32 사용**:
- Float64 대신 Float32 사용으로 **50% 메모리 절감**

### 추가 최적화 가능 방안

#### 1. Mixed Precision Training (AMP)
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Forward
with autocast():
    y_pred = model(x_time, x_spec, x_stat)
    loss = criterion(y_pred, y)

# Backward
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
**예상 절감**: 30-50% GPU 메모리 (FP16 사용)

#### 2. Gradient Checkpointing
```python
from torch.utils.checkpoint import checkpoint

# 중간 활성화를 저장하지 않고 재계산
out = checkpoint(self.time_branch, x_time)
```
**예상 절감**: 40-60% 활성화 메모리
**트레이드오프**: 학습 속도 ~20% 감소

#### 3. 더 작은 Base Channel
```python
model = DeepSleepMultimodal(n_channels=9, base_ch=16)  # 32 → 16
```
**예상 절감**: 파라미터 ~75% 감소, 메모리 ~60% 감소

#### 4. Smaller Batch Size
```bash
python train_deepsleep.py --batch_size 2  # 4 → 2
```
**예상 절감**: 활성화 메모리 50% 감소

---

## 6. 실제 측정 방법

### GPU 메모리 모니터링
```bash
# 학습 중 실시간 모니터링
watch -n 0.5 nvidia-smi
```

### 코드 내 메모리 측정
```python
import torch

# GPU 메모리
print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
print(f"Max allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")

# CPU 메모리
import psutil
process = psutil.Process()
print(f"RAM: {process.memory_info().rss / 1024**2:.2f} MB")
```

---

## 7. 요약 및 권장사항

### 메모리 사용량 요약

| 단계 | RAM | GPU (batch_size=4) |
|------|-----|-------------------|
| 전처리 (1개 파일) | ~165 MB | N/A |
| 학습 | ~750 MB - 1 GB | ~800 MB - 1.2 GB |
| 추론 (60초 청크) | ~100 MB | ~315 MB - 400 MB |

### 하드웨어 권장사항

#### 최소 사양
- **RAM**: 8 GB
- **GPU**: 2 GB VRAM (GTX 1050 Ti급)
- **Batch size**: 2
- **학습 가능**: ✅

#### 권장 사양
- **RAM**: 16 GB
- **GPU**: 6 GB VRAM (RTX 2060급)
- **Batch size**: 4-8
- **학습 속도**: 빠름

#### 최적 사양
- **RAM**: 32 GB
- **GPU**: 12 GB VRAM (RTX 3080급)
- **Batch size**: 16
- **Mixed precision**: 가능
- **학습 속도**: 매우 빠름

### 메모리 부족 시 대처법

**GPU 메모리 부족**:
```bash
# 1. Batch size 감소
python train_deepsleep.py --batch_size 2

# 2. Base channels 감소
python train_deepsleep.py --base_ch 16

# 3. 둘 다 적용
python train_deepsleep.py --batch_size 2 --base_ch 16
```

**RAM 부족** (거의 발생하지 않음):
```bash
# num_workers 감소
# train_deepsleep.py 수정: num_workers=4 → num_workers=2
```

---

## 결론

이 multimodal arousal detection 시스템은 **매우 효율적인 메모리 사용**을 보입니다:

✅ **전처리**: 165 MB (8시간 파일 처리 시)
✅ **학습**: GPU 1.2 GB, RAM 1 GB (batch_size=4)
✅ **추론**: GPU 400 MB

**60초 청크 방식**으로 인해 메모리 사용량이 매우 낮으며, **GTX 1050 Ti (4GB)** 수준의 GPU에서도 학습이 가능합니다.

대부분의 현대 워크스테이션에서 **문제없이 실행 가능**합니다.
