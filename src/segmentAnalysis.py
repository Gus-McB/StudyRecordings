import pandas as pd

def time_to_seconds(time_str):
    """Convert time string to seconds."""
    parts = time_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

def find_overlapping_segments(ai_start, ai_end, corrected_df):
    """Find all corrected segments that overlap with an AI segment."""
    ai_start_sec = time_to_seconds(ai_start)
    ai_end_sec = time_to_seconds(ai_end)
    
    overlaps = []
    for idx, row in corrected_df.iterrows():
        corr_start_sec = time_to_seconds(row['start_time'])
        corr_end_sec = time_to_seconds(row['end_time'])
        
        # Check if time ranges overlap
        if not (ai_end_sec <= corr_start_sec or ai_start_sec >= corr_end_sec):
            overlap_start = max(ai_start_sec, corr_start_sec)
            overlap_end = min(ai_end_sec, corr_end_sec)
            overlap_duration = overlap_end - overlap_start
            
            overlaps.append({
                'overlap_seconds': overlap_duration,
                'corrected_idx': idx,
                'start_time': row['start_time'],
                'end_time': row['end_time'],
                'speaker': row['speaker'],
                'text': row['text']
            })
    
    return sorted(overlaps, key=lambda x: x['overlap_seconds'], reverse=True)

# Load data
ai_unmatched = pd.read_csv('./transcript_unmatched_segments.csv')
original_corrected = pd.read_csv('./200604_0046_transcript_correction.csv')

print("\n" + "="*100)
print("SEGMENT REORGANIZATION ANALYSIS")
print("Investigating where unmatched AI segments went in the corrected version")
print("="*100 + "\n")

# Categorize unmatched segments
merged_segments = []
split_segments = []
removed_segments = []

for ai_idx, ai_row in ai_unmatched.iterrows():
    overlaps = find_overlapping_segments(ai_row['ai_start'], ai_row['ai_end'], original_corrected)
    
    if len(overlaps) == 0:
        # No overlap - segment was removed or significantly moved
        removed_segments.append({
            'ai_row': int(ai_row['ai_row']),
            'ai_start': ai_row['ai_start'],
            'ai_end': ai_row['ai_end'],
            'ai_speaker': ai_row['ai_speaker'],
            'ai_text': ai_row['ai_text']
        })
    elif len(overlaps) == 1:
        # Single overlap - could be merged or just repositioned
        overlap = overlaps[0]
        merged_segments.append({
            'ai_row': int(ai_row['ai_row']),
            'ai_start': ai_row['ai_start'],
            'ai_end': ai_row['ai_end'],
            'ai_speaker': ai_row['ai_speaker'],
            'ai_text': ai_row['ai_text'],
            'corr_start': overlap['start_time'],
            'corr_end': overlap['end_time'],
            'corr_speaker': overlap['speaker'],
            'corr_text': overlap['text'],
            'overlap_seconds': overlap['overlap_seconds']
        })
    else:
        # Multiple overlaps - segment was split
        split_segments.append({
            'ai_row': int(ai_row['ai_row']),
            'ai_start': ai_row['ai_start'],
            'ai_end': ai_row['ai_end'],
            'ai_speaker': ai_row['ai_speaker'],
            'ai_text': ai_row['ai_text'],
            'num_overlaps': len(overlaps),
            'overlaps': overlaps
        })

print(f"Total unmatched AI segments: {len(ai_unmatched)}")
print(f"  → Merged/repositioned: {len(merged_segments)}")
print(f"  → Split into multiple: {len(split_segments)}")
print(f"  → Removed/not found: {len(removed_segments)}\n")

# Export results
merged_df = pd.DataFrame(merged_segments)
if len(merged_df) > 0:
    merged_df.to_csv('./segment_analysis_merged.csv', index=False)
    print(f"✓ Merged segments saved to: segment_analysis_merged.csv")

split_df = pd.DataFrame(split_segments)
if len(split_df) > 0:
    # Only keep key info for split segments
    split_export = split_df[['ai_row', 'ai_start', 'ai_end', 'ai_speaker', 'ai_text', 'num_overlaps']].copy()
    split_export.to_csv('./segment_analysis_split.csv', index=False)
    print(f"✓ Split segments saved to: segment_analysis_split.csv")

removed_df = pd.DataFrame(removed_segments)
if len(removed_df) > 0:
    removed_df.to_csv('./segment_analysis_removed.csv', index=False)
    print(f"✓ Removed segments saved to: segment_analysis_removed.csv")

print("\n" + "="*100)
print("DETAILED BREAKDOWN")
print("="*100 + "\n")

if len(merged_df) > 0:
    print(f"MERGED/REPOSITIONED SEGMENTS ({len(merged_df)}):")
    print("These segments appear in the corrected version with the same or overlapping content:\n")
    for _, row in merged_df.head(15).iterrows():
        print(f"AI Row {int(row['ai_row'])}: [{row['ai_start']}-{row['ai_end']}] {row['ai_speaker']}")
        print(f"  AI Text: {row['ai_text']}")
        print(f"  Found in Corrected: [{row['corr_start']}-{row['corr_end']}] {row['corr_speaker']}")
        print(f"  Corrected Text: {row['corr_text']}")
        print(f"  Overlap duration: {row['overlap_seconds']} seconds\n")

if len(split_df) > 0:
    print(f"\nSPLIT SEGMENTS ({len(split_df)}):")
    print("These segments overlap with multiple segments in the corrected version:\n")
    for _, row in split_df.head(10).iterrows():
        print(f"AI Row {int(row['ai_row'])}: [{row['ai_start']}-{row['ai_end']}] {row['ai_speaker']}")
        print(f"  AI Text: {row['ai_text']}")
        print(f"  Split across {row['num_overlaps']} corrected segments:")
        for overlap in row['overlaps'][:3]:  # Show top 3
            print(f"    - [{overlap['start_time']}-{overlap['end_time']}] {overlap['speaker']}: {overlap['text'][:60]}...")
        print()

if len(removed_df) > 0:
    print(f"\nREMOVED/NOT FOUND SEGMENTS ({len(removed_df)}):")
    print("These segments have no time overlap with any corrected segment:\n")
    for _, row in removed_df.head(20).iterrows():
        print(f"AI Row {int(row['ai_row'])}: [{row['ai_start']}-{row['ai_end']}] {row['ai_speaker']}")
        print(f"  Text: {row['ai_text']}\n")

print("\n" + "="*100)
print("CONCLUSION")
print("="*100)
print(f"\nOut of {len(ai_unmatched)} unmatched AI segments:")
print(f"  • {len(merged_segments)} can be found in corrected version (likely segment merging)")
print(f"  • {len(split_segments)} are split across multiple corrected segments")
print(f"  • {len(removed_segments)} appear to be removed or significantly changed")
print("\nThe corrected version appears to have reorganized segments for better coherence.")
print("="*100 + "\n")
