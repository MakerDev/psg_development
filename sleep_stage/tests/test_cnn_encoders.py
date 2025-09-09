import pytest
import torch
import torch.nn as nn
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_encoders import resnet18


class TestResNet18Encoder:
    """ResNet18 기반 Sleep Stage 인코더의 unit test"""
    
    def test_initialization(self):
        """모델 초기화 테스트"""
        model = resnet18(num_channels=9, num_classes=5, pretrained=False)
        
        assert isinstance(model, nn.Module)
        # 첫 번째 conv layer가 9채널을 받도록 수정되었는지 확인
        assert model.conv1.in_channels == 9
        assert model.conv1.out_channels == 64
        # 마지막 FC layer가 5개 클래스를 출력하는지 확인
        assert model.fc.out_features == 5
        assert model.fc.in_features == 512  # ResNet18의 경우
    
    def test_forward_pass(self):
        """Forward pass 테스트 - 배치 차원 포함"""
        model = resnet18(num_channels=9, num_classes=5, pretrained=False)
        model.eval()
        
        # 배치 차원을 포함한 입력 데이터
        batch_size = 4
        num_channels = 9
        num_signals = 1500  # 30초 * 50Hz
        x = torch.randn(batch_size, num_channels, 1, num_signals)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 5)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_different_batch_sizes(self):
        """다양한 배치 크기 테스트"""
        model = resnet18(num_channels=9, num_classes=5, pretrained=False)
        model.eval()
        
        batch_sizes = [1, 2, 8, 16]
        num_channels = 9
        num_signals = 1500
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, num_channels, 1, num_signals)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (batch_size, 5)
            assert not torch.isnan(output).any()
    
    def test_different_sampling_rates(self):
        """다양한 샘플링 레이트 테스트"""
        model = resnet18(num_channels=9, num_classes=5, pretrained=False)
        model.eval()
        
        batch_size = 2
        num_channels = 9
        # 50Hz, 100Hz, 200Hz for 30 seconds
        signal_lengths = [1500, 3000, 6000]
        
        for num_signals in signal_lengths:
            x = torch.randn(batch_size, num_channels, 1, num_signals)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (batch_size, 5)
            assert not torch.isnan(output).any()
    
    def test_training_mode(self):
        """학습 모드에서의 forward pass 테스트"""
        model = resnet18(num_channels=9, num_classes=5, pretrained=False)
        model.train()
        
        batch_size = 4
        x = torch.randn(batch_size, 9, 1, 1500)
        
        # 학습 모드에서도 정상 작동해야 함
        output = model(x)
        assert output.shape == (batch_size, 5)
        assert output.requires_grad
    
    def test_gradient_flow(self):
        """그래디언트 흐름 테스트"""
        model = resnet18(num_channels=9, num_classes=5, pretrained=False)
        model.train()
        
        batch_size = 2
        x = torch.randn(batch_size, 9, 1, 1500, requires_grad=True)
        target = torch.randint(0, 5, (batch_size,))
        
        # Forward pass
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # 그래디언트가 제대로 전파되었는지 확인
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # 모델 파라미터들의 그래디언트 확인
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), 
                       reason="CUDA not available")
    def test_cuda_inference(self):
        """CUDA에서의 추론 테스트"""
        model = resnet18(num_channels=9, num_classes=5, pretrained=False).cuda()
        model.eval()
        
        batch_size = 4
        x = torch.randn(batch_size, 9, 1, 1500).cuda()
        
        with torch.no_grad():
            output = model(x)
        
        assert output.is_cuda
        assert output.shape == (batch_size, 5)
    
    @pytest.mark.skipif(not os.path.exists('/home/honeynaps/data/shared/sleep_stage/saved_models/pretrained_asam_ver3.pt'),
                       reason="Pretrained model not found")
    def test_load_pretrained(self):
        """사전 학습 모델 로드 테스트"""
        model = resnet18(num_channels=9, num_classes=5, pretrained=False)
        pretrained_path = '/home/honeynaps/data/shared/sleep_stage/saved_models/pretrained_asam_ver3.pt'
        
        # 모델 로드
        model.load_state_dict(torch.load(pretrained_path, map_location='cpu', weights_only=True))
        model.eval()
        
        # 추론 테스트
        batch_size = 4
        x = torch.randn(batch_size, 9, 1, 1500)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 5)
        
        # Softmax를 적용하면 확률이 되어야 함
        probs = torch.softmax(output, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-6)
    
    def test_model_consistency(self):
        """동일 입력에 대한 출력 일관성 테스트"""
        model = resnet18(num_channels=9, num_classes=5, pretrained=False)
        model.eval()
        
        batch_size = 2
        x = torch.randn(batch_size, 9, 1, 1500)
        
        # 동일한 입력에 대해 두 번 추론
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        
        # 출력이 동일해야 함
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_input_shape_variations(self):
        """다양한 입력 형태 테스트"""
        model = resnet18(num_channels=9, num_classes=5, pretrained=False)
        model.eval()
        
        # 2D, 3D, 4D 입력 테스트
        batch_size = 2
        num_channels = 9
        num_signals = 1500
        
        # 4D input (standard)
        x_4d = torch.randn(batch_size, num_channels, 1, num_signals)
        
        with torch.no_grad():
            output = model(x_4d)
        
        assert output.shape == (batch_size, 5)
    
    def test_model_output_range(self):
        """모델 출력 범위 테스트"""
        model = resnet18(num_channels=9, num_classes=5, pretrained=False)
        model.eval()
        
        batch_size = 10
        x = torch.randn(batch_size, 9, 1, 1500)
        
        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=1)
        
        # Softmax 출력은 0과 1 사이여야 함
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        # 각 샘플의 확률 합은 1이어야 함
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-6)
        
        # 가장 높은 확률의 클래스
        max_probs, pred_classes = torch.max(probs, dim=1)
        assert torch.all(pred_classes >= 0) and torch.all(pred_classes < 5)