import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pickle

from prep_window_wise import (
    epoching_with_events, epoching_from_time,
    load_edf_for_demo, load_only_edf
)


class TestPrepWindowWise:
    """Window-wise 전처리 함수들의 unit test"""
    
    def test_epoching_with_events(self):
        """이벤트 기반 에포킹 테스트"""
        # 테스트 데이터 생성
        sfreq = 50
        duration = 300  # 5분
        x = np.random.randn(duration * sfreq, 9)  # 9채널
        
        # 이벤트 생성
        events = [
            {'s_sec': 0, 'e_sec': 30, 'annotation': 'SLEEP-W'},
            {'s_sec': 30, 'e_sec': 60, 'annotation': 'SLEEP-1'},
            {'s_sec': 60, 'e_sec': 90, 'annotation': 'SLEEP-2'},
            {'s_sec': 90, 'e_sec': 120, 'annotation': 'SLEEP-3'},
            {'s_sec': 120, 'e_sec': 150, 'annotation': 'SLEEP-R'},
            # 마지막 2개는 무시됨
            {'s_sec': 150, 'e_sec': 180, 'annotation': 'SLEEP-W'},
            {'s_sec': 180, 'e_sec': 210, 'annotation': 'SLEEP-W'}
        ]
        
        X, Y = epoching_with_events(x, events, sfreq=sfreq)
        
        # 마지막 2개를 제외한 에포크만 처리됨
        assert X.shape == (5, 30 * sfreq, 9)
        assert Y.shape == (5,)
        assert list(Y) == [0, 2, 3, 4, 1]  # 라벨 매핑 확인
    
    def test_epoching_from_time(self):
        """시간 기반 에포킹 테스트"""
        sfreq = 50
        window_size = 30
        duration = 150  # 5분
        x = np.random.randn(duration * sfreq, 9)
        
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        start_time = base_time + timedelta(seconds=30)
        
        X = epoching_from_time(x, base_time, start_time, sfreq=sfreq, window_size=window_size)
        
        # 30초부터 시작해서 30초 윈도우로 나눔
        expected_epochs = (duration - 30) // window_size
        assert X.shape == (expected_epochs, window_size * sfreq, 9)
    
    @patch('prep_window_wise.SCH')
    def test_load_edf_for_demo(self, mock_sch_class):
        """EDF 파일 로드 및 전처리 테스트"""
        # Mock 설정
        mock_dataset = Mock()
        mock_sch_class.return_value = mock_dataset
        
        # Mock EDF
        mock_raw = Mock()
        mock_raw.info = {'meas_date': datetime(2024, 1, 1)}
        mock_raw.get_data.return_value = np.random.randn(9, 15000)  # 9채널, 300초
        mock_dataset.load_edf.return_value = mock_raw
        
        # Mock XML events
        mock_events = [
            {'s_sec': 0, 'e_sec': 30, 'annotation': 'SLEEP-W'},
            {'s_sec': 30, 'e_sec': 60, 'annotation': 'SLEEP-1'},
            {'s_sec': 60, 'e_sec': 90, 'annotation': 'SLEEP-2'},
            {'s_sec': 270, 'e_sec': 300, 'annotation': 'SLEEP-2'},  # 마지막
            {'s_sec': 300, 'e_sec': 330, 'annotation': 'SLEEP-2'}   # 마지막
        ]
        mock_dataset.load_events.return_value = mock_events
        
        # 함수 실행
        edf_path = "test.edf"
        xml_path = "test.xml"
        
        with patch('prep_window_wise.prep_psg_signal') as mock_prep:
            mock_prep.return_value = np.random.randn(15000, 9)
            
            result = load_edf_for_demo(edf_path, xml_path, sfreq=50)
        
        assert isinstance(result, dict)
        assert 'x' in result and 'y' in result
        assert result['x'].shape[0] == 3  # 마지막 2개 제외
        assert result['y'].shape[0] == 3
    
    @patch('prep_window_wise.edf_io.load')
    def test_load_only_edf(self, mock_load):
        """XML 없이 EDF만 로드하는 테스트"""
        # Mock 설정
        mock_raw = Mock()
        mock_raw.info = {'meas_date': datetime(2024, 1, 1)}
        mock_raw.get_data.return_value = np.random.randn(9, 15000)
        mock_load.return_value = (mock_raw, 0)
        
        with patch('prep_window_wise.prep_psg_signal') as mock_prep:
            mock_prep.return_value = np.random.randn(15000, 9)
            
            X = load_only_edf("test.edf", None, sfreq=50)
        
        assert isinstance(X, np.ndarray)
        assert X.shape[1] == 1500  # 30초 * 50Hz
        assert X.shape[2] == 9      # 9채널