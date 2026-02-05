"""
This script tests the temporal_nms_merge function from the TimelineGenerator class.
"""

import sys
import os

# Mocking VisionModelWrapper to avoid heavy imports
class MockVisionModelWrapper:
    def __init__(self, *args, **kwargs):
        pass

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import numpy as np
from timeline_generator import TimelineGenerator

# Patch VisionModelWrapper
import timeline_generator
timeline_generator.VisionModelWrapper = MockVisionModelWrapper

def test_temporal_nms_merge():
    generator = TimelineGenerator(model_type='mock')
    
    # Synthetic candidates
    # Sorted by score
    candidates = [
        {'peak_second': 10, 'start_second': 8, 'end_second': 12, 'score': 0.9},  # Best
        {'peak_second': 15, 'start_second': 11, 'end_second': 18, 'score': 0.8}, # Overlaps with 1st (11-12)
        {'peak_second': 25, 'start_second': 22, 'end_second': 28, 'score': 0.7}, # No overlap with 1st/2nd
        {'peak_second': 17, 'start_second': 16, 'end_second': 20, 'score': 0.6}, # Overlaps with 2nd (already merged into 1st)
        {'peak_second': 30, 'start_second': 27, 'end_second': 33, 'score': 0.5}, # Overlaps with 3rd
    ]
    
    # Test k=2
    # Expectation:
    # 1. Start with 1st [8, 12].
    # 2. Find overlaps:
    #    - 2nd [11, 18] overlaps with [8, 12]. Merge -> [8, 18].
    #    - 4th [16, 20] overlaps with [8, 18]. Merge -> [8, 20].
    #    - Result 1: [8, 20], confidence 0.9
    # 3. Next best remaining: 3rd [22, 28].
    # 4. Find overlaps with [22, 28]:
    #    - 5th [27, 33] overlaps. Merge -> [22, 33].
    #    - Result 2: [22, 33], confidence 0.7
    # Total 2 results.
    
    merged = generator.temporal_nms_merge(candidates, k=2)
    
    print("Merged results:")
    for i, m in enumerate(merged):
        print(f"Result {i+1}: Start={m['start_second']}, End={m['end_second']}, Peak={m['peak_second']}, Conf={m['score']:.2f}")
    
    assert len(merged) == 2
    assert merged[0]['start_second'] == 8
    assert merged[0]['end_second'] == 20
    assert merged[0]['score'] == 0.9
    
    assert merged[1]['start_second'] == 22
    assert merged[1]['end_second'] == 33
    assert merged[1]['score'] == 0.7
    
    print("\nSUCCESS: Merging logic is correct!")

if __name__ == '__main__':
    test_temporal_nms_merge()
