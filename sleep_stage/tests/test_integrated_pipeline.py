import pytest
import torch
import numpy as np
import os
from datetime import datetime

from models.cnn_encoders import resnet18
from utils.transforms import build_transforms
from utils.post_process import run_postprocess


class TestIntegratedSleepPipeline:
    """Sleep Stage 전체 파이프라인 통합 테스트"""
    
    @pytest.fixture
    def sample_eeg_data(self):
        """테스트용 EEG 데이터 생성"""
        # 10분 데이터 (20 epochs)
        duration_sec = 600
        fs = 50
        num_channels = 9
        num_samples = duration_sec * fs
        
        # 실제 EEG와 유사한 패턴 생성
        t = np.linspace(0, duration_sec, num_samples)
        data = np.zeros((num_samples, num_channels))
        
        for ch in range(num_channels):
            # 다양한 주파수 성분 추가
            data[:, ch] = (
                0.5 * np.sin(2 * np.pi * 0.5 * t) +  # 0.5Hz (slow wave)
                0.3 * np.sin(2 * np.pi * 8 * t) +    # 8Hz (alpha)
                0.2 * np.sin(2 * np.pi * 15 * t) +   # 15Hz (beta)
                0.1 * np.random.randn(num_samples)    # noise
            )
        
        return data, fs
    
    @pytest.fixture
    def sleep_stage_model(self):
        """Sleep stage 모델 생성"""
        model = resnet18(num_channels=9, num_classes=5, pretrained=False)
        model.eval()
        return model
    
    def test_full_pipeline(self, sample_eeg_data, sleep_stage_model):
        """전체 파이프라인 테스트"""
        data, fs = sample_eeg_data
        
        # 1. 데이터를 30초 에포크로 분할
        epoch_length = 30 * fs
        num_epochs = len(data) // epoch_length
        epochs = []
        
        for i in range(num_epochs):
            start = i * epoch_length
            end = (i + 1) * epoch_length
            epoch = data[start:end, :]
            epochs.append(epoch)
        
        epochs = np.array(epochs)  # (num_epochs, 1500, 9)
        
        # 2. Transform 적용
        transforms = build_transforms(["NormaliseOnly"], n_channels=9)
        
        # 3. 배치 처리를 위한 준비
        batch_size = 4
        all_preds = []
        
        for i in range(0, num_epochs, batch_size):
            batch_epochs = epochs[i:i+batch_size]
            batch_tensor = []
            
            for epoch in batch_epochs:
                # Transform 적용
                epoch_t = epoch.T  # (9, 1500)
                epoch_transformed, _ = transforms(epoch_t, 0)
                batch_tensor.append(epoch_transformed)
            
            # 배치 텐서 생성
            batch_tensor = torch.stack(batch_tensor)
            batch_tensor = batch_tensor.unsqueeze(2)  # (batch, 9, 1, 1500)
            
            # 4. 모델 추론
            with torch.no_grad():
                output = sleep_stage_model(batch_tensor)
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
        
        # 5. 후처리 적용
        processed_preds = run_postprocess(all_preds, window_size=6)
        
        # 검증
        assert len(processed_preds) == num_epochs
        assert all(0 <= p <= 4 for p in processed_preds)
        assert isinstance(processed_preds, list)
    
    @pytest.mark.skipif(not os.path.exists('/home/honeynaps/data/shared/sleep_stage/saved_models/pretrained_asam_ver3.pt'),
                       reason="Pretrained model not found")
    def test_with_pretrained_model(self, sample_eeg_data):
        """사전 학습 모델을 사용한 테스트"""
        data, fs = sample_eeg_data
        
        # 모델 로드
        model = resnet18(num_channels=9, num_classes=5, pretrained=False)
        pretrained_path = '/home/honeynaps/data/shared/sleep_stage/saved_models/pretrained_asam_ver3.pt'
        model.load_state_dict(torch.load(pretrained_path, map_location='cpu', weights_only=True))
        model.eval()
        
        # 30초 에포크 생성
        epoch_length = 30 * fs
        num_epochs = len(data) // epoch_length
        
        transforms = build_transforms(["NormaliseOnly"], n_channels=9)
        all_preds = []
        all_probs = []
        
        for i in range(num_epochs):
            start = i * epoch_length
            end = (i + 1) * epoch_length
            epoch = data[start:end, :].T  # (9, 1500)
            
            # Transform 적용
            epoch_transformed, _ = transforms(epoch, 0)
            epoch_tensor = epoch_transformed.unsqueeze(0).unsqueeze(2)  # (1, 9, 1, 1500)
            
            # 추론
            with torch.no_grad():
                output = model(epoch_tensor)
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_preds.append(pred.item())
                all_probs.append(probs.squeeze().cpu().numpy())
        
        # 후처리
        processed_preds = run_postprocess(all_preds, window_size=6)
        
        # 검증
        assert len(processed_preds) == num_epochs
        assert len(all_probs) == num_epochs
        assert all(0 <= p <= 4 for p in processed_preds)
        
        # 확률 분포 확인
        all_probs = np.array(all_probs)
        assert all_probs.shape == (num_epochs, 5)
        assert np.allclose(all_probs.sum(axis=1), 1.0, atol=1e-6)
    
    def test_batch_consistency(self, sample_eeg_data, sleep_stage_model):
        """배치 처리와 개별 처리의 일관성 테스트"""
        data, fs = sample_eeg_data
        
        # 4개 에포크만 사용
        epoch_length = 30 * fs
        epochs = []
        for i in range(4):
            start = i * epoch_length
            end = (i + 1) * epoch_length
            epoch = data[start:end, :].T  # (9, 1500)
            epochs.append(epoch)
        
        transforms = build_transforms(["NormaliseOnly"], n_channels=9)
        
        # 개별 처리
        individual_outputs = []
        for epoch in epochs:
            epoch_transformed, _ = transforms(epoch, 0)
            epoch_tensor = epoch_transformed.unsqueeze(0).unsqueeze(2)
            with torch.no_grad():
                output = sleep_stage_model(epoch_tensor)
                individual_outputs.append(output)
        
        individual_outputs = torch.cat(individual_outputs, dim=0)
        
        # 배치 처리
        batch_tensor = []
        for epoch in epochs:
            epoch_transformed, _ = transforms(epoch, 0)
            batch_tensor.append(epoch_transformed)
        
        batch_tensor = torch.stack(batch_tensor).unsqueeze(2)
        with torch.no_grad():
            batch_output = sleep_stage_model(batch_tensor)
        
        # 결과 비교
        assert torch.allclose(individual_outputs, batch_output, atol=1e-6)