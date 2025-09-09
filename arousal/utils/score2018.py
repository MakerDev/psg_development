#!/usr/bin/env python3
# Official 2018 Physionet scoring class available at:
# https://physionet.org/content/challenge-2018/1.0.0/score2018.py

import sys
import os
import numpy
import h5py
import argparse

class Challenge2018Score:
    """Class used to compute scores for the 2018 PhysioNet/CinC Challenge.

    A Challenge2018Score object aggregates the outputs of a proposed
    classification algorithm, and calculates the area under the
    precision-recall curve, as well as the area under the receiver
    operating characteristic curve.

    After creating an instance of this class, call score_record() for
    each record being tested.  To calculate scores for a particular
    record, call record_auprc() and record_auroc().  After scoring all
    records, call gross_auprc() and gross_auroc() to obtain the scores
    for the database as a whole.
    """

    def __init__(self, input_digits=None):
        """Initialize a new scoring buffer.

        If 'input_digits' is given, it is the number of decimal digits
        of precision used in input probability values.
        """
        if input_digits is None:
            input_digits = 3
        self._scale = 10**input_digits
        self._pos_values = numpy.zeros(self._scale + 1, dtype=numpy.int64)
        self._neg_values = numpy.zeros(self._scale + 1, dtype=numpy.int64)
        self._record_auc = {}

    def score_record(self, truth, predictions, record_name=None, threshold=0.5):
        """Add results for a given record to the buffer.

        'truth' is a vector of arousal values: zero for non-arousal
        regions, positive for target arousal regions, and negative for
        unscored regions.

        'predictions' is a vector of probabilities produced by the
        classification algorithm being tested.  This vector must be
        the same length as 'truth', and each value must be between 0
        and 1.

        If 'record_name' is specified, it can be used to obtain
        per-record scores afterwards, by calling record_auroc() and
        record_auprc().
        """
        # Check if length is correct
        if len(predictions) != len(truth):
            raise ValueError("length of 'predictions' does not match 'truth'")

        # Compute the histogram of all input probabilities
        b = self._scale + 1
        r = (-threshold / self._scale, 1.0 + threshold / self._scale)
        all_values = numpy.histogram(predictions, bins=b, range=r)[0]

        # Check if input contains any out-of-bounds or NaN values
        # (which are ignored by numpy.histogram)
        if numpy.sum(all_values) != len(predictions):
            raise ValueError("invalid values in 'predictions'")

        # Compute the histogram of probabilities within arousal regions
        pred_pos = predictions[truth > 0]
        pos_values = numpy.histogram(pred_pos, bins=b, range=r)[0]

        # Compute the histogram of probabilities within unscored regions
        pred_ign = predictions[truth < 0]
        ign_values = numpy.histogram(pred_ign, bins=b, range=r)[0]

        # Compute the histogram of probabilities in non-arousal regions,
        # given the above
        neg_values = all_values - pos_values - ign_values

        self._pos_values += pos_values
        self._neg_values += neg_values

        if record_name is not None:
            self._record_auc[record_name] = self._auc(pos_values, neg_values)

    def _auc(self, pos_values, neg_values):
        # Calculate areas under the ROC and PR curves by iterating
        # over the possible threshold values.

        # At the minimum threshold value, all samples are classified as
        # positive, and thus TPR = 1 and FPR = 1 (since all negatives are misclassified as well).
        tp = numpy.sum(pos_values)
        fp = numpy.sum(neg_values)
        tn = 0
        fn = 0
        
        # 계산 초기값 설정
        # TPR = TP/(TP+FN), 여기서는 초기에는 모든 것을 positive로 예측하므로 TPR=1
        # FPR = FP/(FP+TN), 초기에는 모든 음성을 양성으로 분류하므로 FPR=1
        tpr = 1.0
        fpr = 1.0
        # PPV = TP/(TP+FP)
        ppv = float(tp) / (tp + fp) if (tp+fp)>0 else 0.0

        # If either class is empty, scores are undefined.
        if tp == 0 or fp == 0:
            return (float('nan'), float('nan'))

        # 사다리꼴 적분을 위해 이전 값을 저장
        tpr_prev = tpr
        fpr_prev = fpr
        ppv_prev = ppv

        auroc = 0.0
        auprc = 0.0

        # Threshold를 높여감에 따라 TP->FN, FP->TN으로 이동
        for (n_pos, n_neg) in zip(pos_values, neg_values):
            # 이전 값 저장
            tpr_prev, fpr_prev, ppv_prev = tpr, fpr, ppv

            # threshold 이동에 따른 값 업데이트
            tp -= n_pos
            fn += n_pos
            fp -= n_neg
            tn += n_neg

            if (tp+fn) > 0:
                tpr = float(tp) / (tp + fn)
            else:
                tpr = 0.0

            if (fp+tn) > 0:
                fpr = float(fp) / (fp + tn)
            else:
                fpr = 0.0

            if (tp+fp) > 0:
                ppv = float(tp) / (tp + fp)
            else:
                ppv = ppv_prev

            # AUROC: TPR vs FPR에 대한 사다리꼴 적분
            # 면적 = (fpr_prev - fpr) * (tpr_prev + tpr) / 2
            # 여기서 fpr_prev > fpr 인 상태이므로 음수가 되지 않도록 주의.
            # 현재 fpr은 감소하는 형태로 가므로 fpr_prev - fpr는 양수가 될 것.
            auroc += (fpr_prev - fpr) * (tpr_prev + tpr) * 0.5

            # AUPRC: TPR vs PPV에 대한 유사한 접근
            # 면적 = (tpr_prev - tpr)*ppv_prev (tpr 기준 적분)
            auprc += (tpr_prev - tpr) * ppv_prev

        return (auroc, auprc)

    def gross_auroc(self):
        """Compute the area under the ROC curve.

        The result will be NaN if none of the records processed so far
        contained any target arousals.
        """
        return self._auc(self._pos_values, self._neg_values)[0]

    def gross_auprc(self):
        """Compute the area under the precision-recall curve.

        The result will be NaN if none of the records processed so far
        contained any target arousals.
        """
        return self._auc(self._pos_values, self._neg_values)[1]

    def record_auroc(self, record_name):
        """Compute the area under the ROC curve for a single record.

        The result will be NaN if the record did not contain any
        target arousals.

        The given record must have previously been processed by
        calling score_record().
        """
        return self._record_auc[record_name][0]

    def record_auprc(self, record_name):
        """Compute the area under the PR curve for a single record.

        The result will be NaN if the record did not contain any
        target arousals.

        The given record must have previously been processed by
        calling score_record().
        """
        return self._record_auc[record_name][1]

class Challenge2018ScoreVer2:
    def __init__(self, input_digits=None):
        if input_digits is None:
            input_digits = 3
        self._scale = 10**input_digits
        self._pos_values = numpy.zeros(self._scale + 1, dtype=numpy.int64)
        self._neg_values = numpy.zeros(self._scale + 1, dtype=numpy.int64)

        # _record_auc에 (auroc, auprc, best_thr, best_f1)를 저장
        #   예: self._record_auc[record_name] = (0.9, 0.45, 0.3, 0.55)
        self._record_auc = {}

    def score_record(self, truth, predictions, record_name=None, threshold=0.5):
        if len(predictions) != len(truth):
            raise ValueError("length of 'predictions' does not match 'truth'")

        b = self._scale + 1
        r = (-threshold / self._scale, 1.0 + threshold / self._scale)
        all_values = numpy.histogram(predictions, bins=b, range=r)[0]

        if numpy.sum(all_values) != len(predictions):
            raise ValueError("invalid values in 'predictions'")

        pred_pos = predictions[truth > 0]
        pos_values = numpy.histogram(pred_pos, bins=b, range=r)[0]

        pred_ign = predictions[truth < 0]
        ign_values = numpy.histogram(pred_ign, bins=b, range=r)[0]

        neg_values = all_values - pos_values - ign_values

        self._pos_values += pos_values
        self._neg_values += neg_values

        if record_name is not None:
            # _auc()가 (auroc, auprc, best_thr, best_f1) 리턴하도록 수정
            auroc, auprc, best_thr, best_f1 = self._auc(pos_values, neg_values)
            self._record_auc[record_name] = (auroc, auprc, best_thr, best_f1)

    def _auc(self, pos_values, neg_values):
        """Returns (auroc, auprc, best_thr, best_f1)."""
        # 초기값: threshold = 0 (모두 양성으로 분류)
        tp = numpy.sum(pos_values)
        fp = numpy.sum(neg_values)
        tn = 0
        fn = 0
        
        # 초기 TPR=1, FPR=1, PPV=TP/(TP+FP)
        tpr = 1.0
        fpr = 1.0
        ppv = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0

        # TP=0이거나 FP=0이면 roc/pr 계산이 제대로 정의되지 않을 수 있음
        # 일단 Nan 리턴
        if tp == 0 or fp == 0:
            return (float('nan'), float('nan'), float('nan'), float('nan'))

        # 사다리꼴 적분
        auroc = 0.0
        auprc = 0.0

        # ────────── 추가: best F1 추적용 변수 ──────────
        best_f1 = 0.0
        best_thr_bin = 0  # self._scale 단위로 본 bin index

        # threshold를 1→0 방향(또는 0→1)으로 이동
        for i, (n_pos, n_neg) in enumerate(zip(pos_values, neg_values)):
            # 현재 threshold(i)에서의 F1 계산
            recall = tpr   # recall = TP/(TP+FN)
            precision = ppv
            if precision + recall > 0:
                f1 = 2.0 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            # best_f1 갱신
            if f1 > best_f1:
                best_f1 = f1
                best_thr_bin = i  # i는 histogram bin index

            # 이전 값 저장
            tpr_prev, fpr_prev, ppv_prev = tpr, fpr, ppv

            # threshold를 한 단계 높이는 효과:
            #   n_pos개 양성을 음성으로 이동(TP->FN), n_neg개 음성을 진짜 음성(FP->TN)로 이동
            tp -= n_pos
            fn += n_pos
            fp -= n_neg
            tn += n_neg

            tpr = float(tp)/(tp + fn) if (tp + fn) > 0 else 0.0
            fpr = float(fp)/(fp + tn) if (fp + tn) > 0 else 0.0
            ppv = float(tp)/(tp + fp) if (tp + fp) > 0 else ppv_prev

            # AUROC 적분
            auroc += (fpr_prev - fpr) * (tpr_prev + tpr) * 0.5
            # AUPRC 적분 (tpr vs ppv)
            auprc += (tpr_prev - tpr) * ppv_prev

        # 루프 종료 후, 마지막 점에서 한 번 더 F1 확인
        recall = tpr
        precision = ppv
        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_thr_bin = len(pos_values)  # 마지막 bin index

        # best_thr_bin을 실제 [0,1] 확률값으로 변환
        #   bin i 에 대응하는 threshold ~ i / self._scale
        best_thr = best_thr_bin / float(self._scale)

        return (auroc, auprc, best_thr, best_f1)

    def gross_auroc(self):
        return self._auc(self._pos_values, self._neg_values)[0]

    def gross_auprc(self):
        return self._auc(self._pos_values, self._neg_values)[1]

    def record_auroc(self, record_name):
        return self._record_auc[record_name][0]

    def record_auprc(self, record_name):
        return self._record_auc[record_name][1]

    # ────────── 새로 추가: record의 best F1값을 반환 ──────────
    def record_f1(self, record_name):
        """
        Return the best F1 score found for this record over all thresholds.
        """
        return self._record_auc[record_name][3]

    # 필요하다면 record의 best threshold를 얻는 함수도 추가할 수 있음
    def record_best_threshold(self, record_name):
        """
        Return the threshold at which the best F1 occurred for this record.
        """
        return self._record_auc[record_name][2]



################################################################
# Command line interface
################################################################

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('vecfiles', metavar='RECORD.vec', nargs='+',
                   help='vector of probabilities to score')
    p.add_argument('-r', '--reference-dir', metavar='DIR', default='training',
                   help='location of reference arousal.mat files')
    args = p.parse_args()

    print('Record          AUROC     AUPRC')
    print('_______________________________')
    s = Challenge2018Score()
    failed = 0
    for vec_file in args.vecfiles:
        record = os.path.basename(vec_file)
        if record.endswith('.vec'):
            record = record[:-4]

        arousal_file = os.path.join(args.reference_dir, record,
                                    record + '-arousal.mat')
        try:
            # Load reference annotations from the arousal.mat file
            with h5py.File(arousal_file, 'r') as af:
                truth = numpy.ravel(af['data']['arousals'])

            # Load predictions from the vec file
            predictions = numpy.zeros(len(truth), dtype=numpy.float32)
            with open(vec_file, 'rb') as vf:
                i = -1
                for (i, v) in enumerate(vf):
                    try:
                        predictions[i] = v
                    except IndexError:
                        break
                if i != len(truth) - 1:
                    print('Warning: wrong number of samples in %s'
                          % vec_file)

            # Compute and print scores for this record
            s.score_record(truth, predictions, record)
            auroc = s.record_auroc(record)
            auprc = s.record_auprc(record)
            print('%-11s  %8.6f  %8.6f' % (record, auroc, auprc))
        except Exception as exc:
            print(exc)
            print('%-11s  %8s  %8s' % (record, 'error', 'error'))
            failed = 1

    # Compute and print overall scores
    auroc = s.gross_auroc()
    auprc = s.gross_auprc()
    print('_______________________________')
    print('%-11s  %8.6f  %8.6f' % ('Overall', auroc, auprc))
    sys.exit(failed)
