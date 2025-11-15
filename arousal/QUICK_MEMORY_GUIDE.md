# 메모리 사용량 빠른 가이드

## 📊 요약

### 메모리 사용량 한눈에 보기

| 단계 | RAM | GPU (batch_size=4) |
|------|-----|-------------------|
| **전처리** | 165 MB | - |
| **학습** | 1 GB | 1.2 GB |
| **추론** | 100 MB | 400 MB |

### 최소 하드웨어 요구사항

- **RAM**: 8 GB
- **GPU**: GTX 1050 Ti (4 GB) 이상
- **저장공간**: 1 GB (8시간 파일 480 청크 기준)

---

## 🔍 실제 메모리 측정하기

### 1. 메모리 측정 스크립트 실행

```bash
cd /home/user/psg_development/arousal

# 기본 설정 (batch_size=4, base_ch=32)
python measure_memory.py

# 다양한 설정으로 테스트
python measure_memory.py --batch_size 2 --base_ch 16
python measure_memory.py --batch_size 8 --base_ch 32
```

### 2. 출력 예시

```
==============================================================
🧠 Multimodal Arousal Detection - Memory Measurement
==============================================================

System Information:
  Python: 3.9.0
  PyTorch: 2.0.1
  CUDA Available: True
  GPU: NVIDIA RTX 3080
  GPU Memory: 10.00 GB
  Total RAM: 32.00 GB

🔍 Estimating Preprocessing Memory (60-second chunk)...
  x_time_raw:      108.00 KB
  x_time_combined: 432.00 KB
  x_spec:          219.22 KB
  x_stat:          12.73 KB
  Total:           896.85 KB

For 8-hour file (480 chunks):
  Total storage:   430.49 MB

🔍 Measuring Model Initialization Memory...
✅ Model created on GPU: 9.36 MB

Model Statistics:
  Total parameters:     2,458,763
  Trainable parameters: 2,458,763
  Parameter size:       9.36 MB

==============================================================
Memory Usage: After Forward Pass (batch_size=4)
==============================================================
GPU Memory:
  Allocated:     137.25 MB
  Reserved:      184.00 MB
  Max Allocated: 142.18 MB

==============================================================
Memory Usage: Peak During Training Step (batch_size=4)
==============================================================
GPU Memory:
  Allocated:     198.47 MB
  Reserved:      256.00 MB
  Max Allocated: 1,156.32 MB

📊 Summary
Total GPU Memory Used: 198.47 MB
Peak GPU Memory: 1,156.32 MB

💡 Recommendations:
  ✅ Memory usage is low (1156 MB)
  ✅ Can increase batch_size or base_ch for better performance
```

---

## 💾 상세 메모리 분석

### 60초 청크 하나당

```
x_time_raw:      108 KB   (raw signals)
envelope:        108 KB   (amplitude envelope)
derivatives:     216 KB   (1st + 2nd derivative)
x_time_combined: 432 KB   (combined time features)
x_spec:          219 KB   (spectrogram)
x_stat:          13 KB    (statistical features)
y_time:          12 KB    (time labels)
y_spec:          0.5 KB   (spec labels)
──────────────────────────
Total:           ~900 KB  (~0.88 MB)
```

### 8시간 파일 전체

```
청크 수: 480 (8시간 × 60분)
총 저장공간: 480 × 0.88 MB = 422 MB
```

### 학습 시 GPU 메모리 (batch_size=4)

```
모델 파라미터:        9.4 MB
모델 그래디언트:      9.4 MB
옵티마이저 상태:     18.8 MB  (Adam momentum × 2)
입력 데이터:          2.7 MB
Forward 활성화:      17.0 MB
Backward 그래디언트: 17.0 MB
CUDA 오버헤드:      500 MB
────────────────────────────
총계:               ~574 MB
실제 사용량:     800 MB - 1.2 GB
```

---

## ⚙️ Batch Size별 메모리 사용량

| Batch Size | 입력 | 활성화 | GPU 총계 | 추천 GPU |
|-----------|------|--------|---------|----------|
| 1 | 0.7 MB | 4 MB | 600-800 MB | GTX 1050 (2GB) |
| 2 | 1.4 MB | 9 MB | 700-900 MB | GTX 1050 Ti (4GB) |
| 4 | 2.7 MB | 17 MB | 800 MB-1.2 GB | GTX 1060 (6GB) |
| 8 | 5.4 MB | 34 MB | 1.2-1.8 GB | RTX 2060 (6GB) |
| 16 | 11 MB | 68 MB | 2.0-2.8 GB | RTX 3060 (12GB) |

---

## 🔧 메모리 최적화 방법

### 1. Batch Size 줄이기 (가장 효과적)

```bash
# 메모리 50% 감소
python train_deepsleep.py --batch_size 2

# 메모리 75% 감소
python train_deepsleep.py --batch_size 1
```

**효과**: GPU 메모리 50-75% 감소
**단점**: 학습 속도 느려짐, gradient noise 증가

### 2. Base Channels 줄이기

```bash
# 메모리 ~60% 감소
python train_deepsleep.py --base_ch 16

# 메모리 ~40% 감소
python train_deepsleep.py --base_ch 24
```

**효과**: 모델 크기와 메모리 감소
**단점**: 모델 표현력 감소 가능

### 3. 둘 다 적용 (극한 상황)

```bash
# 메모리 ~80% 감소
python train_deepsleep.py --batch_size 1 --base_ch 16
```

**효과**: GTX 1050 (2GB)에서도 실행 가능
**단점**: 학습 시간 증가, 성능 소폭 감소 가능

### 4. Mixed Precision Training (고급)

```python
# train_deepsleep.py 수정
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Training loop
with autocast():
    output = model(x_time, x_spec, x_stat)
    loss = criterion(output, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**효과**: GPU 메모리 30-50% 감소
**단점**: 코드 수정 필요, 약간의 정밀도 손실

---

## 🎯 상황별 추천 설정

### Case 1: 메모리 여유로움 (GPU 8GB+)

```bash
python train_deepsleep.py \
    --batch_size 8 \
    --base_ch 32 \
    --epochs 50
```

**예상 메모리**: 1.5-2.0 GB
**학습 시간**: 빠름
**성능**: 최고

### Case 2: 일반적인 상황 (GPU 6GB)

```bash
python train_deepsleep.py \
    --batch_size 4 \
    --base_ch 32 \
    --epochs 50
```

**예상 메모리**: 1.0-1.2 GB
**학습 시간**: 보통
**성능**: 좋음

### Case 3: 메모리 부족 (GPU 4GB)

```bash
python train_deepsleep.py \
    --batch_size 2 \
    --base_ch 24 \
    --epochs 50
```

**예상 메모리**: 600-800 MB
**학습 시간**: 느림
**성능**: 양호

### Case 4: 극한 절약 (GPU 2GB)

```bash
python train_deepsleep.py \
    --batch_size 1 \
    --base_ch 16 \
    --epochs 50
```

**예상 메모리**: 400-600 MB
**학습 시간**: 매우 느림
**성능**: 허용 가능

---

## 📈 실시간 메모리 모니터링

### GPU 메모리 모니터링

```bash
# 실시간 모니터링 (0.5초마다 업데이트)
watch -n 0.5 nvidia-smi

# 또는 gpustat 사용 (설치 필요)
pip install gpustat
watch -n 0.5 gpustat -cp
```

### 학습 중 메모리 로깅

```python
# train_deepsleep.py에 추가
if (i + 1) % 10 == 0:
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

---

## ❗ 메모리 오류 해결

### CUDA Out of Memory 에러

```
RuntimeError: CUDA out of memory. Tried to allocate XX.XX MiB
```

**해결 방법**:

1. **Batch size 줄이기** (가장 효과적)
   ```bash
   python train_deepsleep.py --batch_size 2
   ```

2. **캐시 정리**
   ```python
   torch.cuda.empty_cache()
   ```

3. **모델 크기 줄이기**
   ```bash
   python train_deepsleep.py --base_ch 16
   ```

4. **다른 프로세스 종료**
   ```bash
   # GPU 사용 중인 프로세스 확인
   nvidia-smi

   # 프로세스 종료
   kill -9 [PID]
   ```

### CPU Memory 부족 (드물음)

**해결 방법**:

1. **DataLoader workers 줄이기**
   ```python
   # train_deepsleep.py 수정
   num_workers=2  # 4 → 2로 변경
   ```

2. **스왑 메모리 증가** (Linux)
   ```bash
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

---

## 🚀 성능 vs 메모리 트레이드오프

| 설정 | GPU 메모리 | 학습 속도 | 예상 성능 |
|------|-----------|---------|----------|
| batch=16, ch=32 | 2.5 GB | 매우 빠름 | 최고 |
| batch=8, ch=32 | 1.5 GB | 빠름 | 최고 |
| batch=4, ch=32 | 1.0 GB | 보통 | 좋음 |
| batch=2, ch=32 | 700 MB | 느림 | 좋음 |
| batch=4, ch=24 | 750 MB | 보통 | 양호 |
| batch=2, ch=24 | 600 MB | 느림 | 양호 |
| batch=1, ch=16 | 500 MB | 매우 느림 | 허용 |

**추천**: **batch=4, ch=32** (최적의 균형)

---

## 📝 체크리스트

학습 시작 전 확인:

- [ ] GPU 메모리 확인: `nvidia-smi`
- [ ] 메모리 측정 실행: `python measure_memory.py`
- [ ] 적절한 batch_size 선택
- [ ] 저장 공간 확인 (최소 1GB)
- [ ] 전처리 완료 여부 확인

---

## 🎓 결론

이 multimodal arousal detection 시스템은:

✅ **매우 효율적**: 1.2 GB GPU 메모리 (batch=4)
✅ **확장 가능**: batch_size, base_ch로 조절 가능
✅ **접근성 높음**: GTX 1050 Ti에서도 실행 가능
✅ **청크 방식**: 메모리 문제 거의 없음

**대부분의 GPU에서 문제없이 학습 가능합니다!** 🎉

---

## 📞 추가 도움말

메모리 문제 발생 시:
1. `measure_memory.py` 실행해서 정확한 사용량 확인
2. `--batch_size 2` 시도
3. 그래도 안되면 `--base_ch 16` 추가
4. 여전히 문제면 GPU 사양 확인

자세한 분석은 `MEMORY_ANALYSIS.md` 참조
