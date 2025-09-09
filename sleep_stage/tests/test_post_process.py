import pytest
import numpy as np
import copy

from utils.post_process import (
    build_pred_pattern_dict, evaluate_rule_effect,
    collect_rule_candidates, build_rules_greedy,
    run_postprocess, run_advanced_pattern_postprocess
)
from utils.post_process_preset import RULES


class TestPostProcess:
    """후처리 함수들의 unit test"""
    
    def test_build_pred_pattern_dict(self):
        """예측 패턴 사전 생성 테스트"""
        preds = [1, 2, 1, 1, 1, 2, 1, 1, 1]
        labels = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        window_size = 3
        
        pattern_dict = build_pred_pattern_dict(preds, labels, window_size)
        
        # "121" 패턴이 "111"로 나타나야 함
        assert "121" in pattern_dict
        assert "111" in pattern_dict["121"]
        assert pattern_dict["121"]["111"] == 2

    def test_evaluate_rule_effect(self):
        """규칙 적용 효과 평가 테스트"""
        preds = [1, 2, 1, 1, 1, 2, 2, 2]
        labels = [1, 1, 1, 1, 1, 2, 2, 2]
        rule = ("121", "111")
        window_size = 3
        
        delta_correct, new_preds = evaluate_rule_effect(
            preds, labels, rule, window_size
        )
        
        # 규칙 적용 후 개선된 정확도
        assert delta_correct == 1  # 1개 더 맞춤
        assert new_preds[0:3] == [1, 1, 1]  # 121 -> 111로 변경됨
    
    def test_collect_rule_candidates(self):
        """규칙 후보 수집 테스트"""
        pattern_dict = {
            "121": {"111": 5, "122": 1},
            "232": {"222": 3, "233": 2},
            "343": {"333": 1}  # min_count 미만
        }
        
        candidates = collect_rule_candidates(
            pattern_dict, top_k=2, min_count=2
        )
        
        # min_count 이상인 규칙만 포함
        assert ("121", "111") in candidates
        assert ("232", "222") in candidates
        assert ("232", "233") in candidates
        assert ("343", "333") not in candidates  # count가 1이므로 제외
    
    def test_build_rules_greedy(self):
        """Greedy 규칙 생성 테스트"""
        preds = [1, 2, 1, 1, 1, 2, 3, 3, 3, 2]
        labels = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        
        candidates = {("121", "111"), ("233", "222"), ("332", "222")}
        
        rules, final_preds = build_rules_greedy(
            preds, labels, candidates, window_size=3, max_rules=10
        )
        
        # 개선이 있는 규칙들이 선택되어야 함
        assert len(rules) > 0
        # 최종 예측이 원본보다 나아야 함
        original_correct = sum(p == l for p, l in zip(preds, labels))
        final_correct = sum(p == l for p, l in zip(final_preds, labels))
        assert final_correct >= original_correct
    
    def test_run_postprocess_preset(self):
        """사전 정의된 규칙으로 후처리 테스트"""
        # 실제 RULES에서 하나 선택
        if 5 in RULES and len(RULES[5]) > 0:
            test_rule = RULES[5][0]  # 예: ('12111', '11111')
            pattern, replacement = test_rule
            
            # 해당 패턴을 포함하는 예측 생성
            preds = list(map(int, pattern)) + [0, 0, 0]
            
            result = run_postprocess(preds, window_size=5)
            
            # 패턴이 교체되었는지 확인
            expected = list(map(int, replacement)) + [0, 0, 0]
            assert result[:5] == expected[:5]
    
    def test_run_advanced_pattern_postprocess(self):
        # 간단한 테스트 케이스
        preds = [1, 2, 1, 1, 1, 2, 3, 3, 3, 2, 2, 2]
        labels = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
        
        rules, final_preds = run_advanced_pattern_postprocess(
            preds, labels, window_size=3, top_k=3, min_count=1, max_rules=5
        )
        
        assert isinstance(rules, list)
        assert isinstance(final_preds, list)
        assert len(final_preds) == len(preds)
        
        # 정확도가 개선되거나 동일해야 함
        original_acc = sum(p == l for p, l in zip(preds, labels)) / len(preds)
        final_acc = sum(p == l for p, l in zip(final_preds, labels)) / len(preds)
        assert final_acc >= original_acc