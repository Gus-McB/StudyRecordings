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
SUPPORTED_AUDIO_EXTENSIONS = {'.amr', '.wma', '.mp3', '.m4a', '.wav'}

# --- Convert any audio to WAV (Mono, 16kHz) ---
def convert_to_wav(input_file, output_dir, output_filename=None):
    if not output_filename:
        output_filename = os.path.splitext(os.path.basename(input_file))[0] + ".wav"
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)
    os.system(f"ffmpeg -y -i \"{input_file}\" -ar 16000 -ac 1 \"{output_path}\"")
    return output_path

# --- Extract audio segment for speaker embedding ---
def extract_segment(wav_path, start_sec, end_sec):
    out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    ffmpeg.input(wav_path, ss=start_sec, to=end_sec).output(out_path, ar=16000, ac=1).run(quiet=True, overwrite_output=True)
    return out_path

# --- Transcribe + Diarize ---
def transcribe_and_diarize(input_file, wav_output_dir, transcript_output_dir):
    # Prepare paths
    os.makedirs(wav_output_dir, exist_ok=True)
    os.makedirs(transcript_output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # Convert input audio to WAV
    wav_file = convert_to_wav(input_file, wav_output_dir, base_name + ".wav")

    # Transcribe using Whisper
    model = whisper.load_model("medium")
    result = model.transcribe(wav_file, language="en")

    segments = result["segments"]
    embeddings = []
    valid_segments = []

    encoder = VoiceEncoder()

    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        if end - start < 0.5:
            continue

        try:
            segment_audio_path = extract_segment(wav_file, start, end)
            wav = preprocess_wav(segment_audio_path)
            embed = encoder.embed_utterance(wav)
            embeddings.append(embed)
            valid_segments.append(segment)
        except Exception as e:
            print(f"Error embedding segment ({start}-{end}): {e}")
        finally:
            if os.path.exists(segment_audio_path):
                os.remove(segment_audio_path)

    if len(embeddings) < 2:
        print("Not enough segments for speaker estimation.")
        return

    # Cluster speaker embeddings
    best_n = MIN_SPEAKERS
    best_score = -1
    best_labels = []

    for n_clusters in range(MIN_SPEAKERS, min(MAX_SPEAKERS + 1, len(embeddings))):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
        labels = kmeans.labels_
        if len(set(labels)) == 1:
            continue
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score = score
            best_n = n_clusters
            best_labels = labels

    # Save diarized transcript
    rows = []
    for segment, label in zip(valid_segments, best_labels):
        start = str(timedelta(seconds=int(segment["start"])))
        end = str(timedelta(seconds=int(segment["end"])))
        text = segment["text"].strip()
        speaker = f"Person {label + 1}"
        rows.append({
            "start_time": start,
            "end_time": end,
            "speaker": speaker,
            "text": text
        })

    df = pd.DataFrame(rows)
    transcript_file = os.path.join(transcript_output_dir, base_name + "_transcript.csv")
    df.to_csv(transcript_file, index=False)
