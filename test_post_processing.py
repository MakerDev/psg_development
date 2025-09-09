#!/usr/bin/env python3
"""
Test script for post-processing functionality
"""

import numpy as np
from datetime import datetime
from tools.post_process_enhanced import correct_sleep_stages_with_tracking, PostProcessInfo

def test_enhanced_save_xml():
    """Test the enhanced XML save function with sample data"""
    
    # Create sample data
    n_epochs = 10
    original_stages = [2, 3, 2, 1, 0, 2, 3, 2, 1, 0]  # Sample sleep stages
    corrected_stages = [2, 3, 3, 1, 0, 3, 3, 2, 1, 0]  # Some stages corrected
    ground_truth = [2, 3, 2, 1, 0, 3, 2, 2, 1, 0]     # Ground truth labels
    
    # Create sample post-processing info
    post_process_info = []
    for i in range(n_epochs):
        was_changed = original_stages[i] != corrected_stages[i]
        change_reason = "KCOMPLEX_TO_N2" if was_changed else "NO_CHANGE"
        
        post_process_info.append(PostProcessInfo(
            epoch_idx=i,
            original_stage=original_stages[i],
            corrected_stage=corrected_stages[i],
            was_changed=was_changed,
            change_reason=change_reason
        ))
    
    # Import the save function
    from int_sleep_score2 import save_enhanced_sleepstage_xml
    
    # Test save function
    meas_date = datetime.now()
    xml_path = "/tmp/test_sleep_analysis.xml"
    
    print("Testing enhanced XML save function...")
    save_enhanced_sleepstage_xml(
        meas_date=meas_date,
        original_stages=original_stages,
        corrected_stages=corrected_stages,
        post_process_info=post_process_info,
        ground_truth=ground_truth,
        xml_save_path=xml_path
    )
    
    print(f"XML saved to: {xml_path}")
    
    # Read and display the XML content
    with open(xml_path, 'r') as f:
        content = f.read()
        print("\nGenerated XML content:")
        print(content[:1000] + "..." if len(content) > 1000 else content)
    
    # Analyze the results
    print("\nAnalysis:")
    for i, info in enumerate(post_process_info):
        if info.was_changed:
            orig_correct = original_stages[i] == ground_truth[i]
            corr_correct = corrected_stages[i] == ground_truth[i]
            
            print(f"Epoch {i}: {original_stages[i]} -> {corrected_stages[i]} (GT: {ground_truth[i]})")
            print(f"  Original correct: {orig_correct}, Corrected correct: {corr_correct}")
            print(f"  Change reason: {info.change_reason}")
            
            if orig_correct and not corr_correct:
                print("  ⚠️  Post-processing made correct prediction wrong!")
            elif not orig_correct and corr_correct:
                print("  ✅ Post-processing fixed wrong prediction!")

def test_post_processing_tracking():
    """Test the post-processing tracking functionality"""
    
    print("\nTesting post-processing tracking...")
    
    # Create sample data
    arousal_preds = [(5.0, 2.0), (15.5, 1.5)]  # Some arousal events
    sleep_preds = [2, 3, 2, 1, 0, 2, 3, 2]    # Sleep stages
    
    # Create sample micro event data (25Hz)
    n_samples_25hz = len(sleep_preds) * 30 * 25  # 30 sec per epoch, 25Hz
    micro_events = {
        'C3': np.random.choice([0, 1], size=n_samples_25hz, p=[0.9, 0.1]),
        'C4': np.random.choice([0, 1], size=n_samples_25hz, p=[0.9, 0.1])
    }
    
    # Add some specific K-complexes in first half of epochs
    # Epoch 1 (samples 750-1500 at 25Hz) -> first half is 0-375
    start_idx = 1 * 30 * 25 + 100  # 10 seconds into epoch 1
    micro_events['C3'][start_idx:start_idx+50] = 1  # K-complex in first half
    
    corrected_stages, post_info = correct_sleep_stages_with_tracking(
        arousal_preds, sleep_preds, micro_events
    )
    
    print(f"Original stages:  {sleep_preds}")
    print(f"Corrected stages: {corrected_stages}")
    
    changes = [info for info in post_info if info.was_changed]
    print(f"\nChanges made: {len(changes)}")
    
    for info in changes:
        print(f"Epoch {info.epoch_idx}: {info.original_stage} -> {info.corrected_stage} ({info.change_reason})")

if __name__ == "__main__":
    test_enhanced_save_xml()
    test_post_processing_tracking()
    print("\nAll tests completed!")