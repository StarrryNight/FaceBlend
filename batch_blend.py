#!/usr/bin/env python3
"""
Batch Face Blender
Reads pairs from Faces.txt and generates blended faces for each pair.
"""

import os
import sys
from pathlib import Path

# Import the face blender module
from face_blender import process_images

def main():
    script_dir = Path(__file__).parent.resolve()
    
    # Paths
    faces_file = script_dir / "Faces.txt"
    faces_folder = script_dir / "Faces"
    results_folder = script_dir / "Results"
    
    # Check Faces.txt exists
    if not faces_file.exists():
        print(f"ERROR: {faces_file} not found")
        sys.exit(1)
    
    # Check Faces folder exists
    if not faces_folder.exists():
        print(f"ERROR: {faces_folder} folder not found")
        sys.exit(1)
    
    # Create Results folder if it doesn't exist
    results_folder.mkdir(exist_ok=True)
    
    # Read pairs from Faces.txt
    with open(faces_file, 'r') as f:
        lines = f.readlines()
    
    pairs = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split(', ')
        if len(parts) == 2:
            name1 = parts[0].strip()
            name2 = parts[1].strip()
            pairs.append((name1, name2))
        else:
            print(f"WARNING: Skipping invalid line: {line}")
    
    if not pairs:
        print("No valid pairs found in Faces.txt")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"BATCH FACE BLENDER")
    print(f"{'='*60}")
    print(f"\nFound {len(pairs)} pairs to process\n")
    
    success_count = 0
    fail_count = 0
    
    for i, (name1, name2) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] Processing: {name1} + {name2}")
        print("-" * 40)
        
        # Build file paths
        img1_path = faces_folder / f"{name1}.png"
        img2_path = faces_folder / f"{name2}.png"
        output_path = results_folder / f"{name1}+{name2}.png"
        
        # Check input files exist
        if not img1_path.exists():
            print(f"  ERROR: {img1_path} not found")
            fail_count += 1
            continue
        
        if not img2_path.exists():
            print(f"  ERROR: {img2_path} not found")
            fail_count += 1
            continue
        
        # Process the pair
        try:
            success = process_images(
                str(img1_path),
                str(img2_path),
                str(output_path),
                output_size=(600, 600),
                blend_ratio=0.5
            )
            
            if success:
                print(f"  SUCCESS: Saved to {output_path.name}")
                success_count += 1
            else:
                print(f"  FAILED: Could not process")
                fail_count += 1
                
        except Exception as e:
            print(f"  ERROR: {e}")
            fail_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"  Success: {success_count}/{len(pairs)}")
    print(f"  Failed:  {fail_count}/{len(pairs)}")
    print(f"  Results saved to: {results_folder}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

