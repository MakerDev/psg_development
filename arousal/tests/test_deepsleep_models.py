import pytest
import torch
import torch.nn as nn
import numpy as np
import os

from models.DeepSleepSota import DeepSleepNetSota
from models.DeepSleepAttn2D import (
    DeepSleepAttn2D, DoubleConv2D, Down2D, Up2D, SEBlock,
    center_crop_or_pad
)


class TestDeepSleepSota:
    """DeepSleepNetSota 모델의 unit test"""
    
    def test_initialization(self):
        """모델 초기화 테스트"""
        model = DeepSleepNetSota(n_channels=9)
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'up1')
        assert hasattr(model, 'out_conv')
        assert hasattr(model, 'down1')
    
    def test_forward_pass(self):
        """Forward pass 테스트 - 배치 차원 포함"""
        model = DeepSleepNetSota(n_channels=9)
        model.eval()
        
        # 배치 차원을 포함한 입력 데이터
        batch_size = 2
        n_channels = 9
        seq_len = 1500 * 50  # 30초 * 50Hz
        x = torch.randn(batch_size, n_channels, seq_len)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 1, seq_len)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_training_mode(self):
        """학습 모드에서의 forward pass 테스트"""
        model = DeepSleepNetSota(n_channels=9)
        model.train()
        
        batch_size = 2
        x = torch.randn(batch_size, 9, 75000)
        
        # 학습 모드에서도 정상 작동해야 함
        output = model(x)
        assert output.shape == (batch_size, 1, 75000)
        assert output.requires_grad
    
    @pytest.mark.skipif(not torch.cuda.is_available(), 
                       reason="CUDA not available")
    
    def test_cuda_inference(self):
        """CUDA에서의 추론 테스트"""
        model = DeepSleepNetSota(n_channels=9).cuda()
        model.eval()
        
        batch_size = 2
        x = torch.randn(batch_size, 9, 75000).cuda()
        
        with torch.no_grad():
            output = model(x)
        
        assert output.is_cuda
        assert output.shape == (batch_size, 1, 75000)
    
    @pytest.mark.skipif(not os.path.exists('/home/honeynaps/data/shared/arousal/saved_models/deepsleep_tight_asam_0.6587.pt'),
                       reason="Pretrained model not found")
    def test_load_pretrained(self):
        """사전 학습 모델 로드 테스트"""
        model = DeepSleepNetSota(n_channels=9)
        pretrained_path = '/home/honeynaps/data/shared/arousal/saved_models/deepsleep_tight_asam_0.6587.pt'
        
        # 모델 로드
        model.load_state_dict(torch.load(pretrained_path, map_location='cpu',weights_only=True))
        model.eval()
        
        # 추론 테스트
        batch_size = 2
        x = torch.randn(batch_size, 9, 75000)

        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 1, 75000)
        # 출력이 sigmoid 범위 내에 있어야 함
        assert torch.all(output >= 0) and torch.all(output <= 1)


class TestDeepSleepAttn2D:
    """DeepSleepAttn2D 모델의 unit test"""
    
    def test_initialization(self):
        """모델 초기화 테스트"""
        model = DeepSleepAttn2D(
            in_channels=9,
            base_ch=16,
            num_layers=4,
            transformer_layers=2,
            nhead=4,
            dropout=0.1
        )
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'inc')
        assert hasattr(model, 'bot')
        assert len(model.downs) == 4
        assert len(model.ups) == 4
        assert len(model.heads) == 4
    
    def test_forward_pass(self):
        """Forward pass 테스트 - 배치 차원 포함"""
        model = DeepSleepAttn2D(in_channels=9)
        model.eval()
        
        batch_size = 2
        channels = 9
        freq_bins = 51
        time_bins = 100
        x = torch.randn(batch_size, channels, freq_bins, time_bins)
        
        with torch.no_grad():
            # 일반 forward
            output = model(x, train=False)
            assert output.shape == (batch_size, 1, freq_bins, time_bins)
            assert not torch.isnan(output).any()
            
            # Training mode forward (deep supervision)
            outputs = model(x, train=True)
            assert isinstance(outputs, list)
            assert len(outputs) == 4  # main output + 4 deep supervision outputs
            for out in outputs:
                assert out.shape == (batch_size, 1, freq_bins, time_bins)
    
    def test_forward_without_activation(self):
        """활성화 함수 없이 forward pass 테스트"""
        model = DeepSleepAttn2D(in_channels=9)
        model.eval()
        
        batch_size = 2
        x = torch.randn(batch_size, 9, 51, 100)
        
        with torch.no_grad():
            # act=False로 테스트
            output = model(x, train=False, act=False)
            
        assert output.shape == (batch_size, 1, 51, 100)
    
    def test_freq_conv_mode(self):
        """주파수 축 convolution 모드 테스트"""
        model = DeepSleepAttn2D(in_channels=9)
        model.eval()
        
        batch_size = 2
        x = torch.randn(batch_size, 9, 26, 100)
        
        with torch.no_grad():
            output = model(x, train=False, freq=True)
            
        # freq_conv가 적용되면 주파수 차원이 1로 축소됨
        assert output.shape == (batch_size, 1, 100)
    
    def test_different_input_sizes(self):
        """다양한 입력 크기 테스트"""
        model = DeepSleepAttn2D(in_channels=9)
        model.eval()
        
        batch_size = 2
        input_sizes = [(26, 50), (51, 100), (51, 200)]
        
        for freq_bins, time_bins in input_sizes:
            x = torch.randn(batch_size, 9, freq_bins, time_bins)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (batch_size, 1, freq_bins, time_bins)
            assert not torch.isnan(output).any()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), 
                       reason="CUDA not available")
    def test_cuda_inference(self):
        """CUDA에서의 추론 테스트"""
        model = DeepSleepAttn2D(in_channels=9).cuda()
        model.eval()
        
        batch_size = 2
        x = torch.randn(batch_size, 9, 51, 100).cuda()
        
        with torch.no_grad():
            output = model(x)
        
        assert output.is_cuda
        assert output.shape == (batch_size, 1, 51, 100)
    
    @pytest.mark.skipif(not os.path.exists('/home/honeynaps/data/shared/arousal/saved_models/deepsleep_spec_attn_0.5931.pt'),
                       reason="Pretrained model not found")
    def test_load_pretrained(self):
        """사전 학습 모델 로드 테스트"""
        model = DeepSleepAttn2D(in_channels=9, base_ch=16, num_layers=4,
                               transformer_layers=2, nhead=4, dropout=0.25)
        pretrained_path = '/home/honeynaps/data/shared/arousal/saved_models/deepsleep_spec_attn_0.5931.pt'
        
        # 모델 로드
        model.load_state_dict(torch.load(pretrained_path, map_location='cpu', weights_only=True))
        model.eval()
        
        # 추론 테스트
        batch_size = 2
        x = torch.randn(batch_size, 9, 51, 100)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 1, 51, 100)
        # 출력이 sigmoid 범위 내에 있어야 함
        assert torch.all(output >= 0) and torch.all(output <= 1)


class TestDeepSleepAttn2DComponents:
    """DeepSleepAttn2D 컴포넌트 테스트"""
    
    def test_center_crop_or_pad(self):
        """center_crop_or_pad 함수 테스트"""
        batch_size = 2
        channels = 16
        
        # Crop 테스트
        x = torch.randn(batch_size, channels, 64, 64)
        result = center_crop_or_pad(x, 32, 32)
        assert result.shape == (batch_size, channels, 32, 32)
        
        # Pad 테스트
        x = torch.randn(batch_size, channels, 16, 16)
        result = center_crop_or_pad(x, 32, 32)
        assert result.shape == (batch_size, channels, 32, 32)
        
        # 혼합 테스트 (한 축은 crop, 다른 축은 pad)
        x = torch.randn(batch_size, channels, 64, 16)
        result = center_crop_or_pad(x, 32, 32)
        assert result.shape == (batch_size, channels, 32, 32)
    
    def test_double_conv2d(self):
        """DoubleConv2D 블록 테스트"""
        batch_size = 2
        block = DoubleConv2D(in_ch=16, out_ch=32)
        x = torch.randn(batch_size, 16, 64, 64)
        
        output = block(x)
        assert output.shape == (batch_size, 32, 64, 64)
        assert not torch.isnan(output).any()
    
    def test_down2d(self):
        """Down2D 블록 테스트"""
        batch_size = 2
        block = Down2D(in_ch=16, out_ch=32)
        x = torch.randn(batch_size, 16, 64, 64)
        
        output = block(x)
        # MaxPool2d with ceil_mode=True
        assert output.shape == (batch_size, 32, 32, 32)
        assert not torch.isnan(output).any()
    
    def test_up2d(self):
        """Up2D 블록 테스트"""
        batch_size = 2
        block = Up2D(in_ch=64, out_ch=32)
        x = torch.randn(batch_size, 64, 16, 16)
        skip = torch.randn(batch_size, 32, 32, 32)
        
        output = block(x, skip)
        assert output.shape == skip.shape
        assert not torch.isnan(output).any()
    
    def test_se_block(self):
        """SEBlock 테스트"""
        batch_size = 2
        channels = 64
        block = SEBlock(channels=channels, reduction=16)
        x = torch.randn(batch_size, channels, 32, 32)
        
        output = block(x)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()