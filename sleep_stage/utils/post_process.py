import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import natsort
import argparse
import random
import datetime
import copy
import torch.utils.tensorboard as tb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from ..models.cnn_encoders import *
from .transforms import *
from .tools import str2bool
from collections import defaultdict
from .post_process_preset import RULES


def build_pred_pattern_dict(preds, labels, window_size=3):
    pred_pattern_dict = defaultdict(lambda: defaultdict(int))

    for i in range(len(preds) - window_size + 1):
        window_pred = preds[i:i+window_size]
        window_true = labels[i:i+window_size]

        str_pred = "".join(map(str, window_pred))
        str_true = "".join(map(str, window_true))

        pred_pattern_dict[str_pred][str_true] += 1

    return pred_pattern_dict


def evaluate_rule_effect(preds, labels, rule, window_size=3):
    pred_pattern, replace_pattern = rule
    preds_copy = copy.deepcopy(preds)
    L = len(preds)

    old_correct = sum(p == t for p, t in zip(preds_copy, labels))

    # 룰 적용
    for i in range(L - window_size + 1):
        window_pred = preds_copy[i:i+window_size]
        str_pred = "".join(map(str, window_pred))
        if str_pred == pred_pattern:
            new_labels = [int(x) for x in replace_pattern]
            preds_copy[i:i+window_size] = new_labels

    new_correct = sum(p == t for p, t in zip(preds_copy, labels))
    delta_correct = new_correct - old_correct

    return delta_correct, preds_copy


def collect_rule_candidates(pred_pattern_dict, top_k=2, min_count=2):
    rule_candidates = set()
    for pred_pat, true_dict in pred_pattern_dict.items():
        sorted_trues = sorted(true_dict.items(), key=lambda x: x[1], reverse=True)
        for i in range(min(top_k, len(sorted_trues))):
            true_pat, count = sorted_trues[i]
            if count >= min_count:
                rule_candidates.add((pred_pat, true_pat))
    return rule_candidates


def build_rules_greedy(preds, labels, rule_candidates, window_size=3, max_rules=20):
    selected_rules = []
    current_preds = copy.deepcopy(preds)
    used_candidates = set()

    for step in range(max_rules):
        best_rule = None
        best_improvement = 0
        best_new_preds = None

        # 모든 후보를 탐색하며, 가장 delta_correct 높은 룰 찾기
        for rule in rule_candidates:
            if rule in used_candidates:
                continue

            delta_correct, new_preds = evaluate_rule_effect(
                current_preds, labels, rule, window_size=window_size
            )
            if delta_correct > best_improvement:
                best_improvement = delta_correct
                best_rule = rule
                best_new_preds = new_preds

        # 개선이 없으면 종료
        if not best_rule or best_improvement <= 0:
            print(f"[build_rules_greedy] No more improvement at step={step}. Stop.")
            break

        selected_rules.append(best_rule)
        used_candidates.add(best_rule)
        current_preds = best_new_preds

        print(f"[build_rules_greedy] step={step}, best_rule={best_rule}, improvement={best_improvement}")

    return selected_rules, current_preds

def run_postprocess(preds, window_size):
    if window_size not in RULES:
        raise ValueError(f"No preset for: {window_size}")

    selected_rules = RULES[window_size]
    final_preds = copy.deepcopy(preds)

    for rule in selected_rules:
        _, final_preds = evaluate_rule_effect(final_preds, final_preds, rule, window_size=window_size)
    return final_preds

def run_advanced_pattern_postprocess(preds, labels, window_size=3, top_k=3, min_count=2, max_rules=150):
    pred_pattern_dict = build_pred_pattern_dict(preds, labels, window_size=window_size)
    rule_candidates = collect_rule_candidates(pred_pattern_dict, top_k=top_k, min_count=min_count)
    selected_rules, final_preds = build_rules_greedy(
        preds, labels, rule_candidates, window_size=window_size, max_rules=max_rules
    )

    return selected_rules, final_preds

