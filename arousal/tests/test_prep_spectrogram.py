import pytest
import numpy as np
from unittest.mock import patch, Mock, MagicMock
import datetime
import pickle
import os
import sys
sys.path.append('/home/honeynaps/data/shared')

from arousal.prep_spectrogram_tech import (
    moving_window_mean_rms_norm,
    create_arousal_labels_extended,
    robust_scale,
    make_spectrogram,
    map_label_to_spec_time,
    expand_label_freq,
    process_edf_arousal_spec
)


class TestPrepSpectrogramTech:
    """Spectrogram 전처리 함수들의 unit test"""
    
    def test_moving_window_mean_rms_norm(self):
        """Moving window 정규화 테스트"""
        # 테스트 데이터 생성
        x = np.random.randn(9, 50000)  # 9채널, 1000초 데이터
        fs = 50
        window_min = 1  # 1분 window
        
        result = moving_window_mean_rms_norm(x, fs=fs, window_min=window_min)
        
        assert result.shape == x.shape
        assert result.dtype == np.float32
        # 정규화 후 평균은 0에 가깝고 표준편차는 1에 가까워야 함
        assert np.abs(np.mean(result)) < 0.1
    
    def test_robust_scale(self):
        """Robust scaling 테스트"""
        x = np.random.randn(9, 1000)
        
        result = robust_scale(x)
        
        assert result.shape == x.shape
        # Median이 0에 가까워야 함
        median = np.median(result, axis=1)
        assert np.allclose(median, 0, atol=1e-6)
    
    def test_create_arousal_labels_extended(self):
        """Arousal 라벨 생성 테스트"""
        # Mock 이벤트 생성
        meas_date = datetime.datetime(2024, 1, 1)
        events = [
            {
                "onset": meas_date + datetime.timedelta(seconds=10),
                "duration": 5.0
            },
            {
                "onset": meas_date + datetime.timedelta(seconds=30),
                "duration": 10.0
            }
        ]
        total_samples = 2500  # 50초 * 50Hz
        sfreq = 50
        
        result = create_arousal_labels_extended(
            events, meas_date, total_samples, sfreq
        )
        
        assert len(result) == total_samples
        assert result.dtype == np.float32
        # 라벨이 설정된 구간 확인
        assert np.sum(result[500:750]) == 250  # 10-15초
        assert np.sum(result[1500:2000]) == 500  # 30-40초
    
    def test_make_spectrogram(self):
        """스펙트로그램 생성 테스트"""
        # 테스트 신호 생성
        fs = 50
        duration = 10  # 10초
        n_channels = 9
        data = np.random.randn(duration * fs, n_channels)
        
        spec, freqs, times = make_spectrogram(
            data, fs=fs, nperseg=100, noverlap=50
        )
        
        assert spec.shape[0] == n_channels
        assert len(freqs) == spec.shape[1]
        assert len(times) == spec.shape[2]
        assert np.all(np.isfinite(spec))
    
    def test_map_label_to_spec_time(self):
        """시간 도메인 라벨을 스펙트로그램 시간으로 매핑 테스트"""
        fs = 50
        y_1d = np.zeros(500)  # 10초
        y_1d[100:200] = 1  # 2-4초에 arousal
        
        t_array = np.linspace(0, 10, 20)  # 20개 시간 빈
        
        result = map_label_to_spec_time(y_1d, t_array, fs=fs, nperseg=100)
        
        assert len(result) == len(t_array)
        assert result.dtype == np.float32
        assert np.sum(result) > 0  # 일부 구간에 1이 있어야 함
    
    def test_expand_label_freq(self):
        """라벨을 주파수 차원으로 확장 테스트"""
        label_1d = np.array([0, 1, 0, 1, 1])
        freq_bins = 26
        
        result = expand_label_freq(label_1d, freq_bins)
        
        assert result.shape == (freq_bins, len(label_1d))
        # 각 주파수 빈에서 동일한 라벨
        assert np.all(result[:, 1] == 1)
        assert np.all(result[:, 0] == 0)
    
    @patch('arousal.prep_spectrogram_tech.load_edf_file')
    @patch('arousal.prep_spectrogram_tech.load_arousal_xml')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('pickle.dump')
    def test_process_edf_arousal_spec(self, mock_dump, mock_open, 
                                     mock_load_xml, mock_load_edf):
        """EDF 파일 처리 및 스펙트로그램 저장 테스트"""
        # Mock 설정
        mock_raw = Mock()
        mock_raw.info = {'meas_date': datetime.datetime(2024, 1, 1)}
        mock_raw.get_data.return_value = np.random.randn(9, 50000)
        mock_load_edf.return_value = mock_raw
        
        mock_events = [
            {
                "onset": datetime.datetime(2024, 1, 1, 0, 0, 10),
                "duration": 5.0
            }
        ]
        mock_load_xml.return_value = mock_events
        
        # 함수 실행
        edf_path = "test.edf"
        xml_path = "test.xml"
        save_dir = "/tmp"
        
        spec, label = process_edf_arousal_spec(
            edf_path, xml_path, save_dir, fs=50
        )
        
        assert spec is not None
        assert label is not None
        assert mock_dump.called