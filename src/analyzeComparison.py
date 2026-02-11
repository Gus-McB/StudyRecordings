import pandas as pd

# Read the comparison report
comparison_df = pd.read_csv('./transcript_comparison_report.csv')

# Show summary by match status
print("\n" + "="*100)
print("MATCH STATUS BREAKDOWN")
print("="*100 + "\n")

match_summary = comparison_df['match_status'].value_counts()
print(match_summary)
print()

# Show where the problems are (find time ranges with most unmatched segments)
unmatched_df = comparison_df[comparison_df['match_status'] == 'UNMATCHED']

if len(unmatched_df) > 0:
    print("\n" + "="*100)
    print(f"UNMATCHED AI SEGMENTS BY TIME RANGE ({len(unmatched_df)} total)")
    print("="*100 + "\n")
    
    # Group by start times to find clusters
    for idx, row in unmatched_df.head(20).iterrows():
        print(f"AI Row {int(row['ai_row'])}: [{row['ai_start']}] {row['ai_speaker']}")
        print(f"  Text: {row['ai_text']}")

# Show the matched ones with issues
print("\n" + "="*100)
print("MATCHED SEGMENTS WITH TEXT DIFFERENCES")
print("="*100 + "\n")

text_diff_df = comparison_df[(comparison_df['text_match'] == 'NO') & (comparison_df['match_status'] == 'MATCHED')]
print(f"Found {len(text_diff_df)} matched segments with text differences:\n")

for idx, row in text_diff_df.iterrows():
    print(f"AI Row {int(row['ai_row'])}: [{row['ai_start']}-{row['ai_end']}] {row['ai_speaker']}")
    print(f"  AI:        {row['ai_text']}")
    print(f"  Corrected: {row['corrected_text']}")
    if pd.notna(row['word_differences']) and row['word_differences'] != '':
        print(f"  Changes:   {row['word_differences']}")
    print()

# Export just the problematic matched pairs to CSV 
print("\n" + "="*100)
print("EXPORTING RESULTS")
print("="*100 + "\n")

text_diff_df.to_csv('./transcript_text_differences.csv', index=False)
print(f"✓ Text differences saved to: transcript_text_differences.csv")

unmatched_df.to_csv('./transcript_unmatched_segments.csv', index=False)
print(f"✓ Unmatched segments saved to: transcript_unmatched_segments.csv")

print("\nTo understand the unmatched segments, check if they were:")
print("  1. Merged into longer segments in the corrected version")
print("  2. Split into multiple segments in the corrected version")
print("  3. Completely removed or added")
