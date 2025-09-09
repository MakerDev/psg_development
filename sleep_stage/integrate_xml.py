import xml.etree.ElementTree as ET

def parse_sleep_xml(xml_path):
    """
    XML 파일(수면 단계)에서
    onset -> (duration, description, location, probability) 
    형태로 딕셔너리 반환
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()  # <annotationlist>
    
    data_dict = {}
    for annotation in root.findall('annotation'):
        onset_text = annotation.find('onset').text.strip()  # "2023-07-15T22:13:00.000000"
        duration_text = annotation.find('duration').text.strip()
        descr = annotation.find('description').text.strip()
        
        loc_elem = annotation.find('location')
        location = loc_elem.text.strip() if loc_elem is not None else "UNKNOWN"
        
        prob_elem = annotation.find('probability')
        probability = prob_elem.text.strip() if prob_elem is not None else "1.0"
        
        data_dict[onset_text] = {
            'duration': duration_text,
            'description': descr,
            'location': location,
            'prob': probability
        }
    return data_dict

def short_stage(desc):
    """
    'SLEEP-W' -> 'W'
    'SLEEP-R' -> 'R'
    'SLEEP-1' -> 'N1'
    'SLEEP-2' -> 'N2'
    'SLEEP-3' -> 'N3'
    그 외는 그대로 혹은 원하는 방식으로 처리
    """
    desc = desc.upper()  # 대문자로
    if 'SLEEP-W' in desc:
        return 'W'
    elif 'SLEEP-R' in desc:
        return 'R'
    elif 'SLEEP-1' in desc:
        return 'N1'
    elif 'SLEEP-2' in desc:
        return 'N2'
    elif 'SLEEP-3' in desc:
        return 'N3'
    else:
        # 필요시 확장
        return desc.replace('SLEEP-', '')

def merge_sleep_xml(truth_xml_path, pred_xml_path, output_xml_path):
    truth_dict = parse_sleep_xml(truth_xml_path)
    pred_dict = parse_sleep_xml(pred_xml_path)

    # 결과를 기록할 onset 리스트 (truth, pred의 onset 합집합)
    all_onsets = sorted(set(list(truth_dict.keys()) + list(pred_dict.keys())))
    
    # XML 구조 생성
    root = ET.Element('annotationlist')

    for onset in all_onsets:
        # 정답 정보
        truth_info = truth_dict.get(onset, None)
        # 예측 정보
        pred_info = pred_dict.get(onset, None)

        if truth_info is None:
            truth_desc = "NONE"
        else:
            truth_desc = truth_info['description']
        if pred_info is None:
            pred_desc = "NONE"
        else:
            pred_desc = pred_info['description']
        
        t_stage = short_stage(truth_desc)
        p_stage = short_stage(pred_desc)

        # 규칙
        if t_stage == p_stage:
            new_desc = f"CORRECT-{t_stage}"
        else:
            new_desc = f"WRONG-{t_stage}:{p_stage}"

        # duration, location, probability => pred 기준(또는 truth 기준)
        # 여기서는 pred 기준
        duration = pred_info['duration'] if (pred_info is not None) else \
                   (truth_info['duration'] if truth_info is not None else "30.000000")
        location = pred_info['location'] if (pred_info is not None) else "UNKNOWN"
        probability = pred_info['prob'] if (pred_info is not None) else "0.0"

        # annotation 엘리먼트 생성
        ann = ET.Element('annotation')
        
        onset_elem = ET.SubElement(ann, 'onset')
        onset_elem.text = onset
        
        dur_elem = ET.SubElement(ann, 'duration')
        dur_elem.text = duration
        
        desc_elem = ET.SubElement(ann, 'description')
        desc_elem.text = new_desc
        
        loc_elem = ET.SubElement(ann, 'location')
        loc_elem.text = location
        

        root.append(ann)

    # 트리 저장
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)
    print(f"[Saved merged XML] {output_xml_path}")

if __name__ == "__main__":
    truth_xml = "/home/honeynaps/data/GOLDEN/EBX2/SLEEP/SCH_F_40_NW_231130R4_MO_SLEEP.xml"
    pred_xml = "/home/honeynaps/data/shared/SCH_F_40_NW_231130R4_MO_SLEEP_PRED.xml"
    out_xml = "/home/honeynaps/data/shared/INT_SCH_F_40_NW_231130R4_MO_SLEEP_PRED.xml"

    merge_sleep_xml(truth_xml, pred_xml, out_xml)
    