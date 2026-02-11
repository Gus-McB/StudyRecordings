import pandas as pd
from difflib import SequenceMatcher
import re

def similarity_ratio(text1, text2):
    """Calculate text similarity ratio using SequenceMatcher."""
    if not text1 or not text2:
        return 0
    return SequenceMatcher(None, str(text1).lower(), str(text2).lower()).ratio()

def extract_words(text):
    """Extract words from text."""
    return set(str(text).lower().split())

def word_overlap(text1, text2):
    """Calculate word overlap percentage."""
    words1 = extract_words(text1)
    words2 = extract_words(text2)
    
    if not words1 or not words2:
        return 0
    
    overlap = len(words1.intersection(words2))
    total = len(words1.union(words2))
    return overlap / total if total > 0 else 0

def time_to_seconds(time_str):
    """Convert time string to seconds."""
    parts = time_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

def find_best_text_match(ai_segment, corrected_unmatched_reset_df, matched_pairs, min_similarity=0.5):
    """Find best matching segment in corrected list based on text similarity."""
    ai_text = ai_segment['ai_text']
    ai_speaker = ai_segment['ai_speaker']
    
    best_match = None
    best_score = 0
    
    for pos, corr_row in corrected_unmatched_reset_df.iterrows():
        # Skip if already matched
        if pos in matched_pairs:
            continue
        
        corr_text = corr_row['text']
        corr_speaker = corr_row['speaker']
        
        # Prefer same speaker
        speaker_bonus = 0.2 if ai_speaker == corr_speaker else 0
        
        # Calculate similarity
        text_sim = similarity_ratio(ai_text, corr_text)
        word_overlap_score = word_overlap(ai_text, corr_text)
        
        # Combined score: 60% text similarity, 40% word overlap
        combined_score = (text_sim * 0.6 + word_overlap_score * 0.4) + speaker_bonus
        
        if combined_score > best_score and combined_score >= min_similarity:
            best_score = combined_score
            best_match = (pos, combined_score)
    
    return best_match

# Load data
print("\n" + "="*100)
print("ADVANCED MATCHING: Unmatched Segments by Text Similarity")
print("="*100 + "\n")

ai_unmatched = pd.read_csv('./transcript_unmatched_segments.csv')
corrected_df = pd.read_csv('./output/transcripts/200604_0046_transcript.csv')
original_corrected = pd.read_csv('./200604_0046_transcript_correction.csv')

# Find which corrected segments are unmatched
comparison_df = pd.read_csv('./transcript_comparison_report.csv')
matched_corrected_indices = set(comparison_df[comparison_df['corrected_start'] != ''].index)

corrected_unmatched = original_corrected[~original_corrected.index.isin(matched_corrected_indices)]

print(f"AI segments without matches: {len(ai_unmatched)}")
print(f"Corrected segments without matches: {len(corrected_unmatched)}")
print(f"\nAttempting to match by text similarity...\n")

# Store all new matches
new_matches = []
matched_corrected_set = set()

# Reset corrected_unmatched index for consistent positioning
corrected_unmatched_reset = corrected_unmatched.reset_index(drop=True)

for ai_idx, ai_row in ai_unmatched.iterrows():
    best_match = find_best_text_match(ai_row, corrected_unmatched_reset, matched_corrected_set, min_similarity=0.45)
    
    if best_match:
        corr_pos, score = best_match
        matched_corrected_set.add(corr_pos)
        corr_row = corrected_unmatched_reset.iloc[corr_pos]
        
        new_matches.append({
            'ai_row': int(ai_row['ai_row']),
            'ai_start': ai_row['ai_start'],
            'ai_end': ai_row['ai_end'],
            'ai_speaker': ai_row['ai_speaker'],
            'ai_text': ai_row['ai_text'],
            'corrected_start': corr_row['start_time'],
            'corrected_end': corr_row['end_time'],
            'corrected_speaker': corr_row['speaker'],
            'corrected_text': corr_row['text'],
            'match_score': round(score, 3),
            'speaker_match': 'YES' if ai_row['ai_speaker'] == corr_row['speaker'] else 'NO',
            'text_similarity': round(similarity_ratio(ai_row['ai_text'], corr_row['text']), 3)
        })

new_matches_df = pd.DataFrame(new_matches)

if len(new_matches_df) > 0:
    # Sort by match score
    new_matches_df = new_matches_df.sort_values('match_score', ascending=False)
    
    print(f"Found {len(new_matches_df)} additional matches!\n")
    
    # Save results
    new_matches_df.to_csv('./transcript_advanced_matches.csv', index=False)
    print(f"✓ Advanced matches saved to: transcript_advanced_matches.csv\n")
    
    # Print top matches
    print("="*100)
    print("TOP MATCHES BY CONFIDENCE SCORE")
    print("="*100 + "\n")
    
    for idx, row in new_matches_df.head(30).iterrows():
        print(f"Match Score: {row['match_score']} | Speaker: {row['speaker_match']} | Similarity: {row['text_similarity']}")
        print(f"  AI Row {int(row['ai_row'])}: [{row['ai_start']}-{row['ai_end']}] {row['ai_speaker']}")
        print(f"    AI:        {row['ai_text']}")
        print(f"  [{row['corrected_start']}-{row['corrected_end']}] {row['corrected_speaker']}")
        print(f"    Corrected: {row['corrected_text']}\n")
    
    # Summary
    print("="*100)
    print("UPDATED SUMMARY")
    print("="*100 + "\n")
    
    total_matched = 257 + len(new_matches_df)  # Perfect matches + new matches
    total_ai = 403
    
    print(f"Originally matched perfectly: 257 segments")
    print(f"New matches found by text similarity: {len(new_matches_df)}")
    print(f"Total matched segments (with reasonable confidence): {total_matched}")
    print(f"Still unmatched in AI: {total_ai - total_matched}")
    print(f"Still unmatched in Corrected: {len(corrected_unmatched) - len(matched_corrected_set)}")
    print(f"\nImproved accuracy: {(total_matched / total_ai * 100):.2f}%\n")
    
    # Analyze match quality
    high_confidence = len(new_matches_df[new_matches_df['match_score'] >= 0.75])
    medium_confidence = len(new_matches_df[(new_matches_df['match_score'] >= 0.6) & (new_matches_df['match_score'] < 0.75)])
    low_confidence = len(new_matches_df[new_matches_df['match_score'] < 0.6])
    
    print("Match Quality Distribution:")
    print(f"  High confidence (≥0.75): {high_confidence}")
    print(f"  Medium confidence (0.60-0.74): {medium_confidence}")
    print(f"  Low confidence (<0.60): {low_confidence}\n")
    
else:
    print("No matches found with text similarity matching.")
