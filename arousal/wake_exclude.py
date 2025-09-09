import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

def parse_stage_xml(stage_xml_path):
    """
    Sleep stage XML을 파싱하여,
    (start_time, end_time, stage_description, location) 형태 튜플의 리스트를 반환.
    """
    tree = ET.parse(stage_xml_path)
    root = tree.getroot()  # <annotationlist>
    
    intervals = []
    for ann in root.findall('annotation'):
        onset_str = ann.find('onset').text        # 예: 2023-11-28T22:19:30.000000
        duration_str = ann.find('duration').text  # 예: 30.000000 (초 단위)
        description = ann.find('description').text
        # location = ann.find('location').text

        onset_dt = datetime.strptime(onset_str, "%Y-%m-%dT%H:%M:%S.%f")
        duration_sec = float(duration_str)
        end_dt = onset_dt + timedelta(seconds=duration_sec)
        
        intervals.append((onset_dt, end_dt, description, ""))
    return intervals


def parse_arousal_xml(arousal_xml_path):
    """
    Arousal XML을 파싱하여,
    (start_time, end_time, description, location) 형태 튜플의 리스트를 반환.
    """
    tree = ET.parse(arousal_xml_path)
    root = tree.getroot()  # <annotationlist>
    
    arousals = []
    for ann in root.findall('annotation'):
        onset_str = ann.find('onset').text
        duration_str = ann.find('duration').text
        description = ann.find('description').text
        location = ann.find('location').text

        onset_dt = datetime.strptime(onset_str, "%Y-%m-%dT%H:%M:%S.%f")
        duration_sec = float(duration_str)
        end_dt = onset_dt + timedelta(seconds=duration_sec)
        
        arousals.append((onset_dt, end_dt, description, location))
    return arousals


def intervals_overlap(a_start, a_end, b_start, b_end):
    """
    두 구간 (a_start, a_end)와 (b_start, b_end)가
    한 샘플이라도 겹치면 True, 겹치지 않으면 False 반환
    """
    return (a_start < b_end) and (b_start < a_end)


def filter_arousals_by_sleep_w(arousals, stage_intervals):
    """
    SLEEP-W 구간과 겹치는 Arousal들은 모두 제거.
    
    arousals: [(a_start_dt, a_end_dt, 'AROUS', location), ...]
    stage_intervals: [(s_start_dt, s_end_dt, stage_desc, location), ...]
                    이 중 stage_desc가 'SLEEP-W'인 구간만 고려

    리턴: 최종 arousal 리스트
    """
    # 1) SLEEP-W 구간만 추출
    sleep_w_intervals = [
        (st, ed) for (st, ed, desc, loc) in stage_intervals if desc == "SLEEP-W"
    ]
    
    # 2) arousal이 sleep-w와 한 샘플이라도 겹치면 제거
    filtered_arousals = []
    for (a_start, a_end, desc, loc) in arousals:
        overlap_flag = False
        for (w_start, w_end) in sleep_w_intervals:
            if intervals_overlap(a_start, a_end, w_start, w_end):
                # 겹치면 바로 이 arousal은 버림
                overlap_flag = True
                break
        if not overlap_flag:
            filtered_arousals.append((a_start, a_end, desc, loc))
    
    return filtered_arousals


def save_arousal_xml(arousals, save_path):
    """
    arousals 리스트를 XML 파일로 저장
    arousals: [(start_dt, end_dt, 'AROUS', location), ...]
    """
    # XML 구조: <annotationlist> 내부에 <annotation>들이 나열
    # 각 <annotation> 안에 <onset>, <duration>, <description>, <location>
    
    root = ET.Element('annotationlist')
    
    for (start_dt, end_dt, desc, loc) in arousals:
        annotation = ET.SubElement(root, 'annotation')
        
        onset_elem = ET.SubElement(annotation, 'onset')
        # datetime을 문자열(ISO 형식)로 변환
        onset_str = start_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
        onset_elem.text = onset_str
        
        duration_elem = ET.SubElement(annotation, 'duration')
        duration_sec = (end_dt - start_dt).total_seconds()
        duration_elem.text = f"{duration_sec:.6f}"
        
        desc_elem = ET.SubElement(annotation, 'description')
        desc_elem.text = desc
        
        loc_elem = ET.SubElement(annotation, 'location')
        loc_elem.text = loc
    
    # 최종 XML 쓰기
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(save_path, encoding='utf-8', xml_declaration=True)
    print(f"[INFO] Saved final arousal xml to {save_path}")


if __name__ == "__main__":
    """
    예시 사용:
    python filter_arousal.py 
        --stage_xml path/to/stage.xml
        --arousal_xml path/to/arousal.xml
        --output_xml path/to/output.xml
    """
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage_xml', type=str, default="/home/honeynaps/data/GOLDEN/EBX2/SLEEP/SCH_F_20_OB_231128R4_NO_SLEEP.xml")
    parser.add_argument('--arousal_xml', type=str, default="/home/honeynaps/data/shared/arousal/SHR_SCH_F_20_OB_231128R4_NO_AROUS.xml")
    parser.add_argument('--output_xml', type=str,  default="/home/honeynaps/data/shared/arousal/SHR_SCH_F_20_OB_231128R4_NO_AROUS_FILTERED.xml")
    args = parser.parse_args()
    
    # 1) Stage XML 파싱
    stage_intervals = parse_stage_xml(args.stage_xml)
    
    # 2) Arousal XML 파싱
    arousals = parse_arousal_xml(args.arousal_xml)
    
    # 3) SLEEP-W와 겹치는 Arousal 제거
    final_arousals = filter_arousals_by_sleep_w(arousals, stage_intervals)
    
    # 4) 최종 Arousal XML 저장
    save_arousal_xml(final_arousals, args.output_xml)
