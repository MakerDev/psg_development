import pandas as pd
import numpy as np

def find_events(sequence):
    events = []
    in_event = False
    start = 0
    length = len(sequence)
    for i in range(length):
        if not in_event:
            if sequence[i] == 1:
                in_event = True
                start = i
        else:
            if sequence[i] == 0:
                end = i - 1
                events.append((start, end))
                in_event = False

    if in_event:
        events.append((start, length - 1))

    return events


def find_events_with_confidence(pred_label, pred_prob):
    if len(pred_label) != len(pred_prob):
        raise ValueError("pred_label and pred_prob must have the same length.")
    
    events = []
    in_event = False
    start = 0
    length = len(pred_label)

    for i in range(length):
        if not in_event:
            if pred_label[i] == 1:
                in_event = True
                start = i
        else:
            if pred_label[i] == 0:
                # 이벤트 종료 시점
                end = i - 1
                # 평균 확률
                avg_p = float(pred_prob[start:end+1].mean())
                events.append((start, end, avg_p))
                in_event = False

    # 마지막까지 1이었다면 처리
    if in_event:
        end = length - 1
        avg_p = float(pred_prob[start:end+1].mean())
        events.append((start, end, avg_p))

    return events


def get_y_sleep_stage(y_sleep, start, end):
    SLEEPSTAGE_TO_LABEL = {
        "SLEEP-U":-1,
        "SLEEP-W":0, 
        "SLEEP-R":1,
        "SLEEP-1":2, 
        "SLEEP-2":3, 
        "SLEEP-3":4,
    }
    LABEL_TO_SLEEPSTAGE = {v:k for k,v in SLEEPSTAGE_TO_LABEL.items()}
    if y_sleep is None:
        return "None"

    if start < 0 or end >= len(y_sleep):
        raise ValueError("start or end index out of bounds")

    sleep_stages = y_sleep[start:end+1] # (0,1,2,3,4)
    major_stage_label = np.bincount(sleep_stages).argmax()
    major_stage = LABEL_TO_SLEEPSTAGE[major_stage_label]

    return major_stage

def compare_events_without_conf(
    gt_events,      # list of (gt_start, gt_end)
    pred_events,    # list of (pred_start, pred_end, avg_prob)
    overlap_ratio_th=0.1,
    sfreq=50
):
    df_gt_columns = [
        "gt_idx", "gt_len", "pred_start", "pred_end", "gt_start", "gt_end",
        "overlap_YN", "overlap_ratio",
        "front_overhang", "back_overhang",
        "front_underhang", "back_underhang",
        "pred_len", "stage_pred", "stage_gt"
    ]
    gt_rows = []

    used_pred_indices = set()

    for i, (gt_s, gt_e) in enumerate(gt_events):
        gt_len = (gt_e - gt_s + 1)

        best_ratio = 0.0
        best_pred_idx = None
        best_pred_event = None  # (ps, pe, pred_len, avg_prob)

        for j, (ps, pe) in enumerate(pred_events):
            pred_len = (pe - ps + 1)
            overlap_s = max(gt_s, ps)
            overlap_e = min(gt_e, pe)
            overlap_len = max(0, overlap_e - overlap_s + 1)
            ratio = overlap_len / gt_len

            if ratio > best_ratio:
                best_ratio = ratio
                best_pred_idx = j
                best_pred_event = (ps, pe, pred_len)

        if best_pred_event is not None and best_ratio >= overlap_ratio_th:
            (ps, pe, pred_len) = best_pred_event
            used_pred_indices.add(best_pred_idx)

            front_overhang = max(0, gt_s - ps)  
            back_overhang  = max(0, pe - gt_e)  

            front_underhang = max(0, ps - gt_s)
            back_underhang  = max(0, gt_e - pe)

            row = {
                "gt_idx": i,
                "gt_len": gt_len / sfreq,
                "pred_start": ps,
                "pred_end": pe,
                "gt_start": gt_s,
                "gt_end": gt_e,
                "overlap_YN": "Y",
                "overlap_ratio": best_ratio,
                "front_overhang": front_overhang / sfreq,
                "back_overhang": back_overhang / sfreq,
                "front_underhang": front_underhang / sfreq,
                "back_underhang": back_underhang / sfreq,
                "pred_len": pred_len / sfreq,
                "stage_pred": get_y_sleep_stage(None, ps, pe),
                "stage_gt": get_y_sleep_stage(None, gt_s, gt_e)
            }
        else:
            row = {
                "gt_idx": i,
                "gt_len": gt_len / sfreq,
                "pred_start": 0,
                "pred_end": 0,
                "gt_start": gt_s,
                "gt_end": gt_e,
                "overlap_YN": "N",
                "overlap_ratio": 0.0,
                "front_overhang": 0,
                "back_overhang": 0,
                "front_underhang": 0,
                "back_underhang": 0,
                "pred_len": 0,
                "stage_pred": 0,
                "stage_gt": get_y_sleep_stage(None, gt_s, gt_e)
            }
        gt_rows.append(row)

    df_gt = pd.DataFrame(gt_rows, columns=df_gt_columns)

    unmatched = []
    for j, (ps, pe) in enumerate(pred_events):
        if j not in used_pred_indices:
            pred_len = (pe - ps + 1)
            unmatched.append({
                "pred_start": ps,
                "pred_end": pe,
                "pred_idx": j,
                "gt_start": 0,
                "gt_end": 0,
                "pred_len": pred_len / sfreq,
                "stage": get_y_sleep_stage(None, ps, pe)
            })
    df_unmatched = pd.DataFrame(unmatched, columns=["pred_start", "pred_end", "pred_idx", "gt_start", "gt_end", "pred_len", "stage"])

    return df_gt, df_unmatched



def compare_events_with_conf(
    gt_events,      # list of (gt_start, gt_end)
    pred_events,    # list of (pred_start, pred_end, avg_prob)
    y_sleep=None,
    overlap_ratio_th=0.1,
    sfreq=50
):
    df_gt_columns = [
        "gt_idx", "gt_len", 
        "overlap_YN", "overlap_ratio",
        "front_overhang", "back_overhang",
        "front_underhang", "back_underhang",
        "pred_len", "pred_conf", "stage_pred", "stage_gt"
    ]
    gt_rows = []

    used_pred_indices = set()

    for i, (gt_s, gt_e) in enumerate(gt_events):
        gt_len = (gt_e - gt_s + 1)

        best_ratio = 0.0
        best_pred_idx = None
        best_pred_event = None  # (ps, pe, pred_len, avg_prob)

        for j, (ps, pe, pconf) in enumerate(pred_events):
            pred_len = (pe - ps + 1)
            overlap_s = max(gt_s, ps)
            overlap_e = min(gt_e, pe)
            overlap_len = max(0, overlap_e - overlap_s + 1)
            ratio = overlap_len / gt_len

            if ratio > best_ratio:
                best_ratio = ratio
                best_pred_idx = j
                best_pred_event = (ps, pe, pred_len, pconf)
        
        if best_pred_event is not None and best_ratio >= overlap_ratio_th:
            (ps, pe, pred_len, pconf) = best_pred_event
            used_pred_indices.add(best_pred_idx)

            front_overhang = max(0, gt_s - ps) 
            back_overhang  = max(0, pe - gt_e) 

            front_underhang = max(0, ps - gt_s)
            back_underhang  = max(0, gt_e - pe)

            row = {
                "gt_idx": i,
                "gt_len": gt_len / sfreq,
                "overlap_YN": "Y",
                "overlap_ratio": best_ratio,
                "front_overhang": front_overhang / sfreq,
                "back_overhang": back_overhang / sfreq,
                "front_underhang": front_underhang / sfreq,
                "back_underhang": back_underhang / sfreq,
                "pred_len": pred_len / sfreq,
                "pred_conf": f"{pconf:.3f}",
                "stage_pred": get_y_sleep_stage(y_sleep, ps, pe),
                "stage_gt": get_y_sleep_stage(y_sleep, gt_s, gt_e)
            }
        else:
            # overlap N
            row = {
                "gt_idx": i,
                "gt_len": gt_len / sfreq,
                "overlap_YN": "N",
                "overlap_ratio": 0.0,
                "front_overhang": 0,
                "back_overhang": 0,
                "front_underhang": 0,
                "back_underhang": 0,
                "pred_len": 0,
                "pred_conf": "0.000",
                "stage_pred": get_y_sleep_stage(y_sleep, ps, pe),
                "stage_gt": get_y_sleep_stage(y_sleep, gt_s, gt_e)
            }
        gt_rows.append(row)

    df_gt = pd.DataFrame(gt_rows, columns=df_gt_columns)

    unmatched = []
    for j, (ps, pe, pconf) in enumerate(pred_events):
        if j not in used_pred_indices:
            pred_len = (pe - ps + 1)
            unmatched.append({
                "pred_idx": j,
                "pred_len": pred_len / sfreq,
                "pred_conf": f"{pconf:.3f}",
                "stage": get_y_sleep_stage(y_sleep, ps, pe)
            })
    df_unmatched = pd.DataFrame(unmatched, columns=["pred_idx","pred_len","pred_conf", "stage"])

    return df_gt, df_unmatched



def export_two_sheets_to_excel(df_gt, df_unmatched, excel_path="results.xlsx"):
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_gt.to_excel(writer, sheet_name="GT_Overlap", index=False)
        df_unmatched.to_excel(writer, sheet_name="Pred_Unmatched", index=False)


def event_level_analysis(y_pred, y_target, y_prob=None, excel_path=None, sfreq=50, overlap_th=0.1, return_stats=False, y_sleep=None, verbose=False):
    gt_events = find_events(y_target)

    if y_prob is None:
        pred_events = find_events(y_pred)
        df_gt, df_unmatched = compare_events_without_conf(
            gt_events, pred_events, overlap_ratio_th=overlap_th, sfreq=sfreq
        )
    else:
        pred_events = find_events_with_confidence(y_pred, y_prob)

        df_gt, df_unmatched = compare_events_with_conf(
            gt_events, pred_events, overlap_ratio_th=overlap_th, sfreq=sfreq, y_sleep=y_sleep
        )

    if verbose:
        for i, pred_event in enumerate(pred_events):
            # print event length
            print(f"{(pred_event[1] - pred_event[0])//sfreq} sec", end = ' ')
            if i % 10 == 0:
                print()
        print()

    if excel_path is not None:
        export_two_sheets_to_excel(df_gt, df_unmatched, excel_path)
        print(f"Saved 2-sheet Excel to {excel_path}")

    n_events_found = len(df_gt[df_gt["overlap_YN"] == "Y"])
    n_events_missed = len(df_gt[df_gt["overlap_YN"] == "N"])
    n_events_unmatched = len(df_unmatched)  # 어떤 GT와도 매칭 못한 pred

    total_gt = len(gt_events)
    if total_gt == 0:
        detection_ratio = 0.0
    else:
        detection_ratio = n_events_found / total_gt

    df_matched = df_gt[df_gt["overlap_YN"] == "Y"]

    if len(df_matched) > 0:
        avg_front_over = df_matched[df_matched["front_overhang"] > 0]["front_overhang"].mean()
        avg_back_over  = df_matched[df_matched["back_overhang"] > 0]["back_overhang"].mean()
        avg_front_under= df_matched[df_matched["front_underhang"] > 0]["front_underhang"].mean()
        avg_back_under = df_matched[df_matched["back_underhang"] > 0]["back_underhang"].mean()
    else:
        avg_front_over = 0
        avg_back_over = 0
        avg_front_under = 0
        avg_back_under = 0

    fp_events, fn_events, tp_events = [], [], []
    for _, row in df_gt.iterrows():
        if row["overlap_YN"] == "Y":
            tp_events.append(row)          # TP
        else:
            fn_events.append(row)          # FN
    for _, row in df_unmatched.iterrows():
        fp_events.append(row)              # FP

    matched_only = np.zeros_like(y_pred, dtype=y_pred.dtype)
    for row in tp_events:
        ps, pe = int(row["pred_start"]), int(row["pred_end"])
        matched_only[ps:pe+1] = y_pred[ps:pe+1]   # 보통 1로 채워짐

    missed_only = np.zeros_like(y_target, dtype=y_target.dtype)
    for row in fn_events:
        gs, ge = int(row["gt_start"]), int(row["gt_end"])
        missed_only[gs:ge+1] = y_target[gs:ge+1]  # 1 유지  

    wrong_only = np.zeros_like(y_pred, dtype=y_pred.dtype)
    for row in fp_events:
        ps, pe = int(row["pred_start"]), int(row["pred_end"])
        wrong_only[ps:pe+1] = y_pred[ps:pe+1]

    matched_pred_count = len(pred_events) - n_events_unmatched

    if return_stats:
        return {
            "n_events_found": n_events_found,
            "n_events_missed": n_events_missed,
            "n_events_unmatched": n_events_unmatched,
            "detection_ratio": detection_ratio,
            "mean_overlap_ratio": df_gt[df_gt["overlap_YN"] == "Y"]["overlap_ratio"].mean(),
            "avg_front_overhang": avg_front_over,
            "avg_back_overhang": avg_back_over,
            "avg_front_underhang": avg_front_under,
            "avg_back_underhang": avg_back_under,
            "matched_pred_ratio": matched_pred_count / len(pred_events) if len(pred_events) > 0 else 0,
            "matched_only": matched_only,
            "missed_only": missed_only,
            "wrong_only": wrong_only,
        }

    return gt_events, pred_events, n_events_found, n_events_missed, n_events_unmatched

def events_to_1d_vector(events, length):
    """
    events: list of (start_idx, end_idx)
    length: total length of the signal
    return: np.ndarray shape=(length,) in {0,1}
    """
    out = np.zeros(length, dtype=int)

    if len(events[0]) == 2:
        for (s, e) in events:
            out[s:e+1] = 1
    elif len(events[0]) == 3:
        for (s, e, _) in events:
            out[s:e+1] = 1

    return out

def find_overlap_pairs(ev1, ev2):
    pairs = []

    for i1, (s1,e1) in enumerate(ev1):
        for i2, (u1,v1) in enumerate(ev2):
            overlap_s = max(s1, u1)
            overlap_e = min(e1, v1)
            if overlap_e >= overlap_s:
                pairs.append((i1, i2, overlap_s, overlap_e))
    return pairs

def combine_overlapped_events(ev1, ev2, mode='intersection'):
    """
    두 이벤트 리스트(ev1, ev2)에서 오버랩이 있는 쌍만 최종 이벤트로 만든다.
    결합 방식(mode)에 따라 최종 구간 결정.

    mode in ['intersection', 'union', 'model1', 'model2']
    """
    
    if len(ev1[0]) == 3:
        ev1 = [(s,e) for (s,e,_) in ev1]
        ev2 = [(s,e) for (s,e,_) in ev2]

    pairs = find_overlap_pairs(ev1, ev2)
    out = []

    for (i1, i2, overlap_s, overlap_e) in pairs:
        s1,e1 = ev1[i1]
        s2,e2 = ev2[i2]

        if mode == 'intersection':
            # 이미 (overlap_s, overlap_e) 구했음
            # => 최종 이벤트 = overlap 구간 그대로
            final_s = overlap_s
            final_e = overlap_e

        elif mode == 'union':
            final_s = min(s1, s2)
            final_e = max(e1, e2)

        elif mode == 'model1':
            final_s, final_e = s1, e1
        
        elif mode == 'model2':
            final_s, final_e = s2, e2
        
        else:
            raise ValueError(f"Unknown mode: {mode}")

        out.append((final_s, final_e))

    # out 리스트에는 (start, end)가 여러 개 들어 있음.
    # 필요하다면 중복/겹침 병합
    # => sort & merge
    out.sort(key=lambda x: x[0])
    
    merged = []
    for interval in out:
        if not merged:
            merged.append(interval)
        else:
            ps,pe = merged[-1]
            cs,ce = interval
            if cs <= pe+1:
                merged[-1] = (ps, max(pe,ce))
            else:
                merged.append(interval)
    return merged


def combine_two_models_events(
    spec_pred_events,  # Model1 이벤트
    time_pred_events,  # Model2 이벤트
    total_length,      # 시계열 길이
    mode='intersection' # ['intersection', 'union', 'model1', 'model2']
):
    final_events = combine_overlapped_events(spec_pred_events, time_pred_events, mode=mode)
    final_1d = events_to_1d_vector(final_events, total_length)
    return final_1d, final_events
