import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

from utils.tools import (
    load_edf_file, save_sleepstage_xml, str2bool
)
from utils.transforms import build_transforms, NormaliseOnly
from utils.losses import ASLSingleLabel


class TestSleepUtils:
    @patch('mne.io.read_raw_edf')
    def test_load_edf_file_basic(self, mock_read_raw):
        # 함수 실행
        raw = load_edf_file("/home/honeynaps/data/GOLDEN/EDF2/SCH-230114R3_M-60-OV-SE.edf", preload=True,
                            preset="STAGENET", exclude=False)

    
    def test_save_sleepstage_xml(self):
        """Sleep stage XML 저장 테스트"""
        meas_date = datetime(2024, 1, 1, 0, 0, 0)
        y = [0, 1, 2, 3, 4, 0, 1]  # 7 epochs
        probs = [
            [0.9, 0.05, 0.03, 0.01, 0.01],
            [0.1, 0.8, 0.05, 0.03, 0.02],
            [0.05, 0.1, 0.7, 0.1, 0.05],
            [0.02, 0.03, 0.15, 0.7, 0.1],
            [0.01, 0.02, 0.07, 0.2, 0.7],
            [0.85, 0.1, 0.03, 0.01, 0.01],
            [0.15, 0.75, 0.05, 0.03, 0.02]
        ]
        
        with patch('xml.etree.ElementTree.ElementTree.write') as mock_write:
            save_sleepstage_xml(meas_date, y, "test.xml", probs=probs)
            mock_write.assert_called_once()
            
            # 생성된 XML 구조 확인
            args = mock_write.call_args
            assert args[0][0] == "test.xml"
    
    def test_asl_single_label_loss(self):
        """ASLSingleLabel 손실 함수 테스트"""
        loss_fn = ASLSingleLabel(gamma_pos=0, gamma_neg=4, eps=0.1)
        
        # 테스트 입력
        batch_size = 4
        num_classes = 5
        inputs = torch.randn(batch_size, num_classes)
        targets = torch.tensor([0, 1, 2, 3])
        
        loss = loss_fn(inputs, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # 스칼라
        assert loss.item() > 0  # 양수
        assert not torch.isnan(loss)
    
    def test_build_transforms_sleep(self):
        transforms = build_transforms(["NormaliseOnly"], n_channels=9)
        
        # 테스트 데이터
        recording = np.random.randn(9, 1500)
        label = 0
        
        rec_tensor, label_tensor = transforms(recording, label)
        
        assert isinstance(rec_tensor, torch.Tensor)
        assert rec_tensor.shape == (9, 1500)
        # 정규화 확인
        means = torch.mean(rec_tensor, dim=1)
        assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)