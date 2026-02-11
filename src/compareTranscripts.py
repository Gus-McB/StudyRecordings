import pandas as pd
import os
from difflib import SequenceMatcher

def load_transcripts(ai_path, corrected_path):
    """Load both transcript files."""
    ai_df = pd.read_csv(ai_path)
    corrected_df = pd.read_csv(corrected_path)
    return ai_df, corrected_df

def time_to_seconds(time_str):
    """Convert time string (H:MM:SS) to seconds."""
    parts = time_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

def segments_overlap(start1, end1, start2, end2):
    """Check if two time segments overlap."""
    s1, e1 = time_to_seconds(start1), time_to_seconds(end1)
    s2, e2 = time_to_seconds(start2), time_to_seconds(end2)
    return not (e1 <= s2 or e2 <= s1)

def find_matching_segment(ai_row, corrected_df, matched_indices):
    """Find the best matching segment in corrected transcript by time overlap + speaker."""
    best_match_idx = None
    best_overlap = 0
    
    ai_start = time_to_seconds(ai_row['start_time'])
    ai_end = time_to_seconds(ai_row['end_time'])
    ai_speaker = ai_row['speaker']
    
    for idx, corr_row in corrected_df.iterrows():
        if idx in matched_indices:  # Already matched
            continue
        
        corr_start = time_to_seconds(corr_row['start_time'])
        corr_end = time_to_seconds(corr_row['end_time'])
        
        # Check if same speaker and time overlaps
        if ai_speaker == corr_row['speaker']:
            # Calculate overlap
            overlap_start = max(ai_start, corr_start)
            overlap_end = min(ai_end, corr_end)
            if overlap_end > overlap_start:
                overlap = overlap_end - overlap_start
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match_idx = idx
    
    return best_match_idx

def compare_transcripts(ai_df, corrected_df, output_file):
    """Compare AI and corrected transcripts by matching segments."""
    
    comparison_results = []
    matched_corrected_indices = set()
    
    print(f"\n{'='*100}")
    print(f"SMART TRANSCRIPT COMPARISON ANALYSIS")
    print(f"Matching segments by time overlap and speaker (not row-by-row)")
    print(f"{'='*100}\n")
    
    print(f"AI Transcript segments: {len(ai_df)}")
    print(f"Corrected Transcript segments: {len(corrected_df)}")
    print(f"Difference: {abs(len(ai_df) - len(corrected_df))} segments\n")
    
    speaker_mismatches = 0
    text_differences = 0
    perfect_matches = 0
    unmatched_ai = 0
    
    # Match each AI segment to corrected segment
    for ai_idx, ai_row in ai_df.iterrows():
        result = {
            'ai_row': ai_idx + 1,
            'ai_start': ai_row['start_time'],
            'ai_end': ai_row['end_time'],
            'ai_speaker': ai_row['speaker'],
            'ai_text': ai_row['text'],
            'corrected_start': '',
            'corrected_end': '',
            'corrected_speaker': '',
            'corrected_text': '',
            'speaker_match': '',
            'text_match': '',
            'word_differences': '',
            'match_status': ''
        }
        
        # Find matching segment in corrected transcript
        best_match_idx = find_matching_segment(ai_row, corrected_df, matched_corrected_indices)
        
        if best_match_idx is not None:
            matched_corrected_indices.add(best_match_idx)
            corr_row = corrected_df.iloc[best_match_idx]
            
            result['corrected_start'] = corr_row['start_time']
            result['corrected_end'] = corr_row['end_time']
            result['corrected_speaker'] = corr_row['speaker']
            result['corrected_text'] = corr_row['text']
            result['match_status'] = 'MATCHED'
            
            # Compare speakers
            speaker_match = ai_row['speaker'] == corr_row['speaker']
            result['speaker_match'] = 'YES' if speaker_match else 'NO'
            if not speaker_match:
                speaker_mismatches += 1
            
            # Compare text
            text_match = ai_row['text'] == corr_row['text']
            result['text_match'] = 'YES' if text_match else 'NO'
            
            if text_match and speaker_match:
                perfect_matches += 1
            else:
                text_differences += 1
            
            # Find word-level differences
            if not text_match:
                ai_words = str(ai_row['text']).split()
                corrected_words = str(corr_row['text']).split()
                diff_words = []
                
                for j, (ai_w, corr_w) in enumerate(zip(ai_words, corrected_words)):
                    if ai_w != corr_w:
                        diff_words.append(f"'{ai_w}'→'{corr_w}'")
                
                if len(corrected_words) != len(ai_words):
                    diff_words.append(f"WordCnt:{len(ai_words)}→{len(corrected_words)}")
                
                result['word_differences'] = ' | '.join(diff_words[:5])  # Show first 5 diffs
        else:
            result['match_status'] = 'UNMATCHED'
            unmatched_ai += 1
        
        comparison_results.append(result)
    
    # Save to CSV
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(output_file, index=False)
    print(f"Detailed comparison saved to: {output_file}\n")
    
    # Print summary statistics
    print(f"{'='*100}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*100}\n")
    
    print(f"Matched segments: {len(ai_df) - unmatched_ai}")
    print(f"Perfect matches (speaker & text correct): {perfect_matches}")
    print(f"Speaker mismatches: {speaker_mismatches}")
    print(f"Text differences: {text_differences}")
    print(f"Unmatched segments (no equivalent in corrected): {unmatched_ai}")
    print(f"Unmatched segments in corrected (extra): {len(corrected_df) - len(matched_corrected_indices)}\n")
    
    if len(ai_df) > 0:
        accuracy_percent = (perfect_matches / len(ai_df)) * 100
        print(f"Overall accuracy (AI vs Corrected): {accuracy_percent:.2f}%\n")
    
    # Detailed issues
    print(f"{'='*100}")
    print(f"DETAILED ISSUES")
    print(f"{'='*100}\n")
    
    unmatched_segments = comparison_df[comparison_df['match_status'] == 'UNMATCHED']
    if len(unmatched_segments) > 0:
        print(f"UNMATCHED AI SEGMENTS ({len(unmatched_segments)} found):")
        for _, row in unmatched_segments.iterrows():
            print(f"  AI Row {int(row['ai_row'])}: [{row['ai_start']}-{row['ai_end']}] {row['ai_speaker']}")
            print(f"    Text: {row['ai_text']}\n")
    
    speaker_issues = comparison_df[(comparison_df['speaker_match'] == 'NO') & (comparison_df['match_status'] == 'MATCHED')]
    if len(speaker_issues) > 0:
        print(f"\nSPEAKER MISMATCHES ({len(speaker_issues)} found):")
        for _, row in speaker_issues.iterrows():
            print(f"  AI Row {int(row['ai_row'])} → Corrected [{row['corrected_start']}-{row['corrected_end']}]")
            print(f"    AI Speaker: {row['ai_speaker']} → Corrected Speaker: {row['corrected_speaker']}")
            print(f"    AI Text: {row['ai_text']}")
            print(f"    Corrected Text: {row['corrected_text']}\n")
    
    text_issues = comparison_df[(comparison_df['text_match'] == 'NO') & (comparison_df['match_status'] == 'MATCHED')]
    if len(text_issues) > 0:
        print(f"\nTEXT DIFFERENCES ({len(text_issues)} found):")
        for _, row in text_issues.iterrows():
            if pd.notna(row['word_differences']) and row['word_differences'] != '':
                print(f"  AI Row {int(row['ai_row'])} [{row['ai_start']}-{row['ai_end']}] - {row['ai_speaker']}")
                print(f"    Changes: {row['word_differences']}")
                print(f"    AI: {row['ai_text']}")
                print(f"    Corrected: {row['corrected_text']}\n")
    
    # Show unmatched corrected segments
    unmatched_corrected = corrected_df[~corrected_df.index.isin(matched_corrected_indices)]
    if len(unmatched_corrected) > 0:
        print(f"\nEXTRA SEGMENTS IN CORRECTED ({len(unmatched_corrected)} found):")
        for idx, row in unmatched_corrected.iterrows():
            print(f"  Corrected Row {idx+1}: [{row['start_time']}-{row['end_time']}] {row['speaker']}")
            print(f"    Text: {row['text']}\n")
    
    print(f"{'='*100}\n")

if __name__ == "__main__":
    # Define file paths
    ai_transcript = './output/transcripts/200604_0046_transcript.csv'
    corrected_transcript = './200604_0046_transcript_correction.csv'
    output_comparison = './transcript_comparison_report.csv'
    
    # Load and compare
    ai_df, corrected_df = load_transcripts(ai_transcript, corrected_transcript)
    compare_transcripts(ai_df, corrected_df, output_comparison)
