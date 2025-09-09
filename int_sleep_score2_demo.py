#!/usr/bin/env python3
"""
Demo version of int_sleep_score2.py that works with simulated data
to demonstrate the post-processing analysis functionality
"""

import os
import numpy as np
from datetime import datetime, timedelta
from tools.post_process_enhanced import correct_sleep_stages_with_tracking, PostProcessInfo


def save_enhanced_sleepstage_xml(meas_date, original_stages, corrected_stages, 
                                post_process_info, ground_truth, xml_save_path, 
                                location="EEG-F4"):
    """
    Save sleep stage predictions with detailed post-processing analysis to XML.
    
    Format: {STAGE}_{TRUE/FALSE}_{POST_PROCESS_WRONG}_{POST_PROCESS_REASON}
    """
    import xml.etree.ElementTree as ET
    
    label_to_stage = {
        0: "SLEEP-W",
        1: "SLEEP-R", 
        2: "SLEEP-1",
        3: "SLEEP-2",
        4: "SLEEP-3"
    }

    root = ET.Element("annotationlist")

    for i, (original_stage, corrected_stage, gt_stage) in enumerate(zip(original_stages, corrected_stages, ground_truth)):
        start_sec = i * 30
        onset_time = meas_date + timedelta(seconds=start_sec)
        duration = 30.0
        
        # Get post-processing info for this epoch
        pp_info = post_process_info[i] if i < len(post_process_info) else None
        
        # Build description string
        stage_name = label_to_stage.get(corrected_stage, "SLEEP-U")
        
        # Check if prediction is correct
        is_correct = "TRUE" if corrected_stage == gt_stage else "FALSE"
        
        # Check if post-processing caused error
        post_process_wrong = "FALSE"
        post_process_reason = "NONE"
        
        if pp_info and pp_info.was_changed:
            # Original was correct but corrected is wrong
            if original_stage == gt_stage and corrected_stage != gt_stage:
                post_process_wrong = "TRUE"
            post_process_reason = pp_info.change_reason
        
        # Format: {STAGE}_{TRUE/FALSE}_{POST_PROCESS_WRONG}_{POST_PROCESS_REASON}
        description = f"{stage_name}_{is_correct}_{post_process_wrong}_{post_process_reason}"

        annotation = ET.SubElement(root, "annotation")

        # onset
        onset_elem = ET.SubElement(annotation, "onset")
        onset_elem.text = onset_time.strftime("%Y-%m-%dT%H:%M:%S.%f")

        # duration
        duration_elem = ET.SubElement(annotation, "duration")
        duration_elem.text = f"{duration:.6f}"

        # description
        desc_elem = ET.SubElement(annotation, "description")
        desc_elem.text = description

        # location
        location_elem = ET.SubElement(annotation, "location")
        location_elem.text = location

    # XML tree 작성
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)

    # XML 파일로 저장
    tree.write(xml_save_path, encoding="UTF-8", xml_declaration=True)


def main():
    """
    Demo with simulated data to show post-processing analysis
    """
    
    print("=== Int Sleep Score 2 Demo ===")
    print("Demonstrating post-processing analysis with simulated data\n")
    
    # Simulate realistic sleep data
    n_epochs = 50
    np.random.seed(42)
    
    # Create realistic sleep pattern
    sleep_preds = []
    ground_truth = []
    
    # Simulate a sleep pattern: Wake -> N1 -> N2 -> N3 -> N2 -> REM -> Wake
    for i in range(n_epochs):
        if i < 5:  # Initial wake
            gt = pred = 0  # Wake
        elif i < 10:  # Sleep onset
            gt = pred = 2  # N1
        elif i < 20:  # Light sleep
            gt = 3  # N2
            pred = 2 if np.random.random() < 0.3 else 3  # Some misclassifications
        elif i < 30:  # Deep sleep
            gt = pred = 4  # N3
        elif i < 40:  # Light sleep again
            gt = 3  # N2
            pred = 2 if np.random.random() < 0.2 else 3
        else:  # REM sleep
            gt = pred = 1  # REM
        
        sleep_preds.append(pred)
        ground_truth.append(gt)
    
    original_sleep_preds = sleep_preds.copy()
    
    # Simulate arousal events (start_time_sec, duration_sec)
    arousal_preds = [
        (8.0 * 30 + 10, 2.0),   # Arousal in epoch 8
        (15.0 * 30 + 5, 1.5),   # Arousal in epoch 15
        (25.0 * 30 + 20, 3.0),  # Arousal in epoch 25
    ]
    
    # Simulate micro events (K-complexes) - 25Hz data
    n_samples_25hz = n_epochs * 30 * 25  # 30 sec per epoch, 25Hz
    micro_events = {
        'C3': np.zeros(n_samples_25hz, dtype=int),
        'C4': np.zeros(n_samples_25hz, dtype=int),
    }
    
    # Add some K-complexes in the first half of N2 epochs
    for epoch_idx in range(n_epochs):
        if ground_truth[epoch_idx] == 3:  # N2 epochs
            if np.random.random() < 0.6:  # 60% chance of K-complex
                # Add K-complex in first half of epoch
                start_sample = epoch_idx * 30 * 25 + np.random.randint(0, 10 * 25)  # First 10 seconds
                duration = np.random.randint(25, 75)  # 1-3 seconds
                micro_events['C3'][start_sample:start_sample + duration] = 1
                if np.random.random() < 0.7:  # 70% chance in both channels
                    micro_events['C4'][start_sample:start_sample + duration] = 1
    
    print("Original sleep stage accuracy:")
    accuracy_orig = sum(1 for i in range(n_epochs) if sleep_preds[i] == ground_truth[i]) / n_epochs
    print(f"Accuracy: {accuracy_orig:.2%} ({sum(1 for i in range(n_epochs) if sleep_preds[i] == ground_truth[i])}/{n_epochs})")
    
    # Apply post-processing with tracking
    print("\nApplying post-processing correction...")
    corrected_sleep_stages, post_process_info = correct_sleep_stages_with_tracking(
        arousal_preds, 
        sleep_preds, 
        micro_events
    )
    
    print("Post-processed sleep stage accuracy:")
    accuracy_corr = sum(1 for i in range(n_epochs) if corrected_sleep_stages[i] == ground_truth[i]) / n_epochs
    print(f"Accuracy: {accuracy_corr:.2%} ({sum(1 for i in range(n_epochs) if corrected_sleep_stages[i] == ground_truth[i])}/{n_epochs})")
    
    # Analyze changes
    changes_count = sum(1 for info in post_process_info if info.was_changed)
    correct_to_wrong = sum(1 for i, info in enumerate(post_process_info) 
                          if info.was_changed and sleep_preds[i] == ground_truth[i] and corrected_sleep_stages[i] != ground_truth[i])
    wrong_to_correct = sum(1 for i, info in enumerate(post_process_info) 
                          if info.was_changed and sleep_preds[i] != ground_truth[i] and corrected_sleep_stages[i] == ground_truth[i])
    
    print(f"\nPost-processing Analysis:")
    print(f"Total epochs changed: {changes_count}")
    print(f"Correct -> Wrong: {correct_to_wrong}")
    print(f"Wrong -> Correct: {wrong_to_correct}")
    print(f"Net improvement: {wrong_to_correct - correct_to_wrong}")
    print(f"Accuracy improvement: {accuracy_corr - accuracy_orig:.2%}")
    
    # Show detailed changes
    print("\nDetailed changes:")
    for i, info in enumerate(post_process_info):
        if info.was_changed:
            orig_correct = "✓" if sleep_preds[i] == ground_truth[i] else "✗"
            corr_correct = "✓" if corrected_sleep_stages[i] == ground_truth[i] else "✗"
            print(f"Epoch {i:2d}: {sleep_preds[i]} -> {corrected_sleep_stages[i]} (GT: {ground_truth[i]}) [{orig_correct}->{corr_correct}] {info.change_reason}")
    
    # Save enhanced XML
    output_file = "/tmp/demo_sleep_analysis.xml"
    base_time = datetime.now()
    
    print(f"\nSaving enhanced XML to: {output_file}")
    save_enhanced_sleepstage_xml(
        meas_date=base_time,
        original_stages=sleep_preds,
        corrected_stages=corrected_sleep_stages,
        post_process_info=post_process_info,
        ground_truth=ground_truth,
        xml_save_path=output_file
    )
    
    print("Demo completed!")
    print(f"\nYou can examine the XML file with detailed post-processing analysis at: {output_file}")
    
    # Show a few example XML entries
    print("\nSample XML entries:")
    with open(output_file, 'r') as f:
        lines = f.readlines()
        in_annotation = False
        count = 0
        for line in lines:
            if '<annotation>' in line:
                in_annotation = True
                count += 1
            if in_annotation:
                print(line.rstrip())
            if '</annotation>' in line:
                in_annotation = False
                if count >= 3:  # Show first 3 annotations
                    break


if __name__ == "__main__":
    main()