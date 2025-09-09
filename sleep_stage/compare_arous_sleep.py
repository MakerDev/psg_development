import os
import xml.etree.ElementTree as ET

def compare_first_onset(arousal, sleep_stage):
    tree1 = ET.parse(arousal)
    tree2 = ET.parse(sleep_stage)
    root1 = tree1.getroot()
    root2 = tree2.getroot()

    annotations1 = root1.findall('annotation')
    annotations2 = root2.findall('annotation')


    onset1 = annotations1[0].find('onset').text
    onset2 = annotations2[0].find('onset').text

    # if onset1 is earlier than onset2, print("onset1 is earlier than onset2")
    is_onset1_earlier = onset1 < onset2
    if is_onset1_earlier:
        print(f"onset1: {onset1}, onset2: {onset2}")
        return True

    return False


def compare_last_onset(arousal, sleep_stage):
    tree1 = ET.parse(arousal)
    tree2 = ET.parse(sleep_stage)
    root1 = tree1.getroot()
    root2 = tree2.getroot()

    annotations1 = root1.findall('annotation')
    annotations2 = root2.findall('annotation')

    # Count how many events in arousal exist outside of the last sleep stage event
    last_sleep_onset = annotations2[-1].find('onset').text
    count = 0
    for annotation in annotations1:
        onset = annotation.find('onset').text
        if onset > last_sleep_onset:
            count += 1

    return count  


if __name__ == "__main__":
    arous_dir = "/home/honeynaps/data/dataset2/EBX/AROUS"
    sleep_dir = "/home/honeynaps/data/dataset2/EBX/SLEEP"
    arous_dir = "/home/honeynaps/data/GOLDEN/EBX2/AROUS"
    sleep_dir = "/home/honeynaps/data/GOLDEN/EBX2/SLEEP"

    arousals = [f for f in os.listdir(arous_dir) if f.endswith('.xml')]

    for i in range(len(arousals)):
        arousal = os.path.join(arous_dir, arousals[i])
        sleep_filename =  arousals[i].replace("_AROUS.xml", "_SLEEP.xml")

        if sleep_filename not in os.listdir(sleep_dir):
            print(f"{sleep_filename} is not in {sleep_dir}")
            continue

        sleep = os.path.join(sleep_dir, sleep_filename)
        if compare_first_onset(arousal, sleep):
            print(f"{arousals[i]} is earlier than {sleep_filename}")

        c = compare_last_onset(arousal, sleep)
        if c != 0:
            print(f"{arousals[i]} has {c} events outside of {sleep_filename}")
