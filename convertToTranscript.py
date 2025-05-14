import os
import pandas as pd
from datetime import timedelta
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import whisper
import ffmpeg
import tempfile

# --- CONSTANTS ---
MIN_SPEAKERS = 2
MAX_SPEAKERS = 6

# --- CONVERT WMA TO WAV ---
def convert_to_wav(input_file, output_dir, output_filename=None):
    if not output_filename:
        output_filename = os.path.splitext(os.path.basename(input_file))[0] + ".wav"
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)
    os.system(f"ffmpeg -y -i \"{input_file}\" -ar 16000 -ac 1 \"{output_path}\"")
    return output_path

# --- EXTRACT AUDIO SEGMENTS FOR EMBEDDING ---
def extract_segment(wav_path, start_sec, end_sec):
    out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    ffmpeg.input(wav_path, ss=start_sec, to=end_sec).output(out_path, ar=16000, ac=1).run(quiet=True, overwrite_output=True)
    return out_path

# --- MAIN FUNCTION ---
def transcribe_and_diarize(input_file, base_output_dir):
    # Prepare output paths
    wav_output_dir = os.path.join(base_output_dir, "convertedRecordings")
    transcript_output_dir = os.path.join(base_output_dir, "transcripts")
    os.makedirs(wav_output_dir, exist_ok=True)
    os.makedirs(transcript_output_dir, exist_ok=True)

    # Convert and prepare file names
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    wav_path = convert_to_wav(input_file, wav_output_dir, base_name + ".wav")
    transcript_path = os.path.join(transcript_output_dir, base_name + "_transcript.csv")

    # Load Whisper model and transcribe
    model = whisper.load_model("medium")
    result = model.transcribe(wav_path, language="en")
    segments = result["segments"]

    # Embed segments
    encoder = VoiceEncoder()
    embeddings = []
    valid_segments = []

    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        if end - start < 0.5:
            continue

        segment_audio_path = extract_segment(wav_path, start, end)
        try:
            wav = preprocess_wav(segment_audio_path)
            embed = encoder.embed_utterance(wav)
            embeddings.append(embed)
            valid_segments.append(segment)
        except Exception as e:
            print(f"Error embedding segment ({start}-{end}): {e}")
        finally:
            os.remove(segment_audio_path)

    if len(embeddings) < MIN_SPEAKERS:
        print("Not enough segments for speaker estimation.")
        return

    # Estimate speakers
    print("Estimating number of speakers...")
    best_n = MIN_SPEAKERS
    best_score = -1
    best_labels = []

    for n_clusters in range(MIN_SPEAKERS, min(MAX_SPEAKERS + 1, len(embeddings))):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
        labels = kmeans.labels_
        if len(set(labels)) == 1:
            continue
        score = silhouette_score(embeddings, labels)
        print(f"n_clusters={n_clusters}, silhouette={score:.3f}")
        if score > best_score:
            best_score = score
            best_n = n_clusters
            best_labels = labels

    print(f"Using estimated number of speakers: {best_n}")

    # Save transcript with speaker labels
    rows = []
    for segment, label in zip(valid_segments, best_labels):
        rows.append({
            "start_time": str(timedelta(seconds=int(segment["start"]))),
            "end_time": str(timedelta(seconds=int(segment["end"]))),
            "speaker": f"Person {label + 1}",
            "text": segment["text"].strip()
        })

    df = pd.DataFrame(rows)
    df.to_csv(transcript_path, index=False)
    print(f"Transcript saved to: {transcript_path}")
    return transcript_path, wav_path
