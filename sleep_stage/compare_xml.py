import os
import xml.etree.ElementTree as ET

def count_diff_descriptions(pred, true, type='1'):
    tree1 = ET.parse(pred)
    tree2 = ET.parse(true)
    root1 = tree1.getroot()
    root2 = tree2.getroot()

    annotations1 = root1.findall('annotation')
    annotations2 = root2.findall('annotation')

    annotations1 = annotations1[:len(annotations2)]

    min_len = min(len(annotations1), len(annotations2))
    
    total_len = 0
    diff_count = 0
    for i in range(min_len):
        desc1 = annotations1[i].find('description').text
        desc2 = annotations2[i].find('description').text

        if "-U" in desc1 or "-U" in desc2: # Unknown은 제외
            continue

        if desc1 != desc2:
            diff_count += 1

        total_len += 1

    accuracy = 1 - diff_count / total_len
    return accuracy

if __name__ == "__main__":
    preds_dir = "/home/honeynaps/data/shared/sleep_stage/preds"
    label_dir = "/home/honeynaps/data/GOLDEN/EBX2/SLEEP"

    preds = [f for f in os.listdir(preds_dir) if f.endswith('.xml')]
    labels = [f for f in os.listdir(label_dir) if f.endswith('.xml')]

    overall_acc = 0
    for i in range(len(preds)):
        pred = os.path.join(preds_dir, preds[i])
        label = os.path.join(label_dir, labels[i])

        accuracy = count_diff_descriptions(pred, label)
        overall_acc += accuracy
        print(f"{preds[i]}'s Accuracy: {accuracy}")

    overall_acc /= len(preds)
    print(f"Overall Accuracy: {overall_acc}")

    # pred = "/home/honeynaps/data/shared/SCH_F_20_OV_230715R3_MO_SLEEP.xml"
    # label = "/home/honeynaps/data/GOLDEN/EBX2/SLEEP/SCH_F_20_OV_230715R3_MO_SLEEP.xml"

    # result = count_diff_descriptions(pred, label)
    # print(f"description이 다른 line의 개수: {result}")