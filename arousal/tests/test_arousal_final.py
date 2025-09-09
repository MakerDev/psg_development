import pytest
import torch
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ArousalFinal import ArousalFinal
from ProgNoti import ProgNoti


class TestArousalFinal:
    """ArousalFinal 클래스의 unit test - time과 spec 타입만 테스트"""
    
    @pytest.fixture
    def mock_signals(self):
        """테스트용 신호 데이터 생성"""
        signal_length = 100000
        return {
            'F3_2': np.random.randn(signal_length),
            'F4_1': np.random.randn(signal_length),
            'C3_2': np.random.randn(signal_length),
            'C4_1': np.random.randn(signal_length),
            'O1_2': np.random.randn(signal_length),
            'O2_1': np.random.randn(signal_length),
            'LOC': np.random.randn(signal_length),
            'ROC': np.random.randn(signal_length),
            'CHIN': np.random.randn(signal_length)
        }
    
    @pytest.fixture
    def arousal_detector_time(self, mock_signals):
        """ArousalFinal 인스턴스 생성 - time 타입"""
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        return ArousalFinal(
            sigs=mock_signals,
            base_time=base_time,
            min_secs=3,
            gpu=0,
            seed=42,
            num_channels=9,
            fs=50,
            type='time'
        )
    
    @pytest.fixture
    def arousal_detector_spec(self, mock_signals):
        """ArousalFinal 인스턴스 생성 - spec 타입"""
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        return ArousalFinal(
            sigs=mock_signals,
            base_time=base_time,
            min_secs=3,
            gpu=0,
            seed=42,
            num_channels=9,
            fs=50,
            type='spec'
        )
    
    def test_initialization_time(self, arousal_detector_time):
        """time 타입 초기화 테스트"""
        assert arousal_detector_time.fs == 50
        assert arousal_detector_time.num_channels == 9
        assert arousal_detector_time.min_secs == 3
        assert arousal_detector_time.type == 'time'
        assert arousal_detector_time.seed == 42
        assert arousal_detector_time.is_ensemble == False
    
    def test_initialization_spec(self, arousal_detector_spec):
        """spec 타입 초기화 테스트"""
        assert arousal_detector_spec.fs == 50
        assert arousal_detector_spec.num_channels == 9
        assert arousal_detector_spec.min_secs == 3
        assert arousal_detector_spec.type == 'spec'
        assert arousal_detector_spec.seed == 42
        assert arousal_detector_spec.is_ensemble == False
    
    def test_invalid_type(self, mock_signals):
        """잘못된 type 파라미터 테스트"""
        with pytest.raises(ValueError, match="Invalid type"):
            ArousalFinal(
                sigs=mock_signals,
                base_time=datetime.now(),
                type='invalid_type'
            )
    
    def test_map_spec_pred_to_time(self, arousal_detector_spec):
        """스펙트로그램 예측을 시간 도메인으로 매핑하는 함수 테스트"""
        pred_1d = np.array([1, 0, 1, 0, 1])
        times = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        total_samples = 150  # 3초 * 50Hz
        
        result = arousal_detector_spec.map_spec_pred_to_time(
            pred_1d, times, total_samples, fs=50, nperseg=50, mode='average'
        )
        
        assert len(result) == total_samples
        assert result.dtype == np.float32
        assert np.all(result >= 0) and np.all(result <= 1)
        
        # max mode 테스트
        result_max = arousal_detector_spec.map_spec_pred_to_time(
            pred_1d, times, total_samples, fs=50, nperseg=50, mode='max'
        )
        assert len(result_max) == total_samples
    
    def test_postprocess_arousal_preds(self, arousal_detector_time):
        """Arousal 예측 후처리 테스트"""
        # 짧은 이벤트와 긴 이벤트를 포함한 예측
        fs = 50
        min_len_sec = 5
        min_len_samples = min_len_sec * fs  # 250 샘플
        
        # 100 샘플 (2초) 이벤트와 300 샘플 (6초) 이벤트
        preds = np.zeros(1000)
        preds[100:200] = 1  # 2초 이벤트 - 제거될 것
        preds[400:700] = 1  # 6초 이벤트 - 유지될 것
        
        result = arousal_detector_time.postprocess_arousal_preds(
            preds, min_len=min_len_sec, fs=fs
        )
        
        assert len(result) == len(preds)
        assert result.dtype == int
        assert np.all(np.isin(result, [0, 1]))
        # 짧은 이벤트는 제거되고 긴 이벤트만 남아야 함
        assert np.sum(result[100:200]) == 0  # 짧은 이벤트 제거됨
        assert np.sum(result[400:700]) == 300  # 긴 이벤트 유지됨
    
    def test_fill_missing_channels(self, arousal_detector_time):
        """누락된 채널 채우기 테스트"""
        # 일부 채널이 누락된 데이터
        recording = np.random.randn(50000, 7)  # 9개 중 7개 채널만 있음
        missing_channels = [2, 5]  # C3_2, O2_1 누락
        
        result = arousal_detector_time.fill_missing_channels(recording, missing_channels)
        
        assert result.shape == (50000, 9)
    
    @pytest.mark.skipif(not os.path.exists('/home/honeynaps/data/shared/arousal/saved_models'),
                       reason="Pretrained models directory not found")
    def test_build_model_time_type(self, arousal_detector_time):
        """시간 도메인 모델 빌드 테스트 - 실제 모델 로드"""
        device = torch.device('cpu')
        missing_channels = []
        pretrained_dir = '/home/honeynaps/data/shared/arousal/saved_models'
        
        model, threshold, _, _, _, _ = arousal_detector_time._build_model(
            missing_channels, pretrained_dir, device
        )
        
        assert model is not None
        assert isinstance(threshold, float)
        assert threshold > 0 and threshold < 1
    
    @pytest.mark.skipif(not os.path.exists('/home/honeynaps/data/shared/arousal/saved_models'),
                       reason="Pretrained models directory not found")
    def test_build_model_spec_type(self, arousal_detector_spec):
        """스펙트로그램 도메인 모델 빌드 테스트 - 실제 모델 로드"""
        device = torch.device('cpu')
        missing_channels = []
        pretrained_dir = '/home/honeynaps/data/shared/arousal/saved_models'

        model, threshold, _, _, _, _ = arousal_detector_spec._build_model(
            missing_channels, pretrained_dir, device
        )
        
        assert model is not None
        assert isinstance(threshold, float)
        assert threshold > 0 and threshold < 1
        

    @pytest.mark.skipif(not os.path.exists('/home/honeynaps/data/shared/arousal/saved_models'),
                       reason="Pretrained models directory not found")
    def test_evaluate_time_model_real(self, arousal_detector_time):
        """시간 모델 평가 테스트 - 실제 모델 사용"""
        from arousal.models.DeepSleepSota import DeepSleepNetSota
        
        # 실제 모델 생성 및 로드
        model = DeepSleepNetSota(n_channels=9)
        pretrained_path = '/home/honeynaps/data/shared/arousal/saved_models/deepsleep_tight_asam_0.6587.pt'
        
        if os.path.exists(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path, map_location='cpu', weights_only=True))
            model.eval()
            
            # 배치 차원을 포함한 입력 데이터
            batch_size = 2
            x = torch.randn(batch_size, 9, 50000)
            label = torch.ones(batch_size, 50000) * -1  # padding
            label[:, :25000] = 0  # 실제 라벨
            threshold = 0.5
            
            result = arousal_detector_time.evaluate_time_model(
                model, x, label, threshold
            )
            
            assert len(result) == 25000 * batch_size
            assert result.dtype == int
    
    @pytest.mark.skipif(not os.path.exists('/home/honeynaps/data/shared/arousal/saved_models'),
                       reason="Pretrained models directory not found")
    def test_evaluate_spec_model_real(self, arousal_detector_spec):
        """스펙트로그램 모델 평가 테스트 - 실제 모델 사용"""
        from arousal.models.DeepSleepAttn2D import DeepSleepAttn2D
        
        # 실제 모델 생성 및 로드
        model = DeepSleepAttn2D(in_channels=9, base_ch=16, num_layers=4,
                               transformer_layers=2, nhead=4, dropout=0.25)
        pretrained_path = '/home/honeynaps/data/shared/arousal/saved_models/deepsleep_spec_attn_0.5931.pt'
        
        if os.path.exists(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path, map_location='cpu', weights_only=True))
            model.eval()
            
            # 배치 차원을 포함한 입력 데이터
            batch_size = 1
            freq_bins = 26
            time_bins = 3000
            x = torch.randn(batch_size, 9, freq_bins, time_bins)
            threshold = 0.5
            times = np.linspace(0, 300, time_bins)
            total_samples = 6000
            
            result = arousal_detector_spec.evaluate_spec_model(
                model, x, threshold, times, total_samples
            )
            
            assert len(result) == total_samples
            assert result.dtype == int
    

class TestDataSet:
    """ArousalFinal._DataSet 클래스 테스트"""
    
    def test_dataset_initialization(self):
        """데이터셋 초기화 테스트"""
        x = np.random.randn(100, 50)
        y = np.random.randint(0, 2, 100)
        
        dataset = ArousalFinal._DataSet('test', x, y)
        
        assert len(dataset) == 1
        assert dataset.name == 'test'
    
    def test_dataset_getitem(self):
        """데이터셋 아이템 접근 테스트"""
        x = np.random.randn(100, 50)
        y = np.random.randint(0, 2, 100)
        
        dataset = ArousalFinal._DataSet('test', x, y)
        name, data, label = dataset[0]
        
        assert name == 'test'
        assert np.array_equal(data, x)
        assert np.array_equal(label, y)
    
    def test_dataset_with_transforms(self):
        """Transform이 있는 데이터셋 테스트"""
        x = np.random.randn(100, 50)
        y = np.random.randint(0, 2, 100)
        
        # 간단한 transform mock
        def mock_transform(x, y):
            return x * 2, y
        
        dataset = ArousalFinal._DataSet('test', x, y, transforms=mock_transform)
        name, data, label = dataset[0]
        
        assert name == 'test'
        assert np.array_equal(data, x * 2)
        assert np.array_equal(label, y)