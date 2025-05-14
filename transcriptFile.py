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

# --- SETTINGS ---
AUDIO_FILE = "/home/Angus/Desktop/StudyRecordings/Test 1/Test 1/Local 1/160119_0030.WMA"
CSV_OUTPUT = "transcript.csv"
MIN_SPEAKERS = 2
MAX_SPEAKERS = 6

# --- CONVERT WMA TO WAV ---
def convert_to_wav(input_file, output_file="converted.wav"):
    os.system(f"ffmpeg -y -i \"{input_file}\" -ar 16000 -ac 1 \"{output_file}\"")
    return output_file

# --- EXTRACT AUDIO SEGMENTS FOR EMBEDDING ---
def extract_segment(wav_path, start_sec, end_sec):
    out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    ffmpeg.input(wav_path, ss=start_sec, to=end_sec).output(out_path, ar=16000, ac=1).run(quiet=True, overwrite_output=True)
    return out_path

# --- TRANSCRIBE AND DIARIZE ---
def transcribe_with_auto_speaker_estimation(audio_file):
    wav_file = convert_to_wav(audio_file)
    model = whisper.load_model("medium")
    result = model.transcribe(wav_file, language="en")


    segments = result["segments"]
    embeddings = []
    valid_segments = []

    encoder = VoiceEncoder()

    # Step 1: Embed each Whisper segment
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        if end - start < 0.5:
            continue

        segment_audio_path = extract_segment(wav_file, start, end)
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

    # Step 2: Estimate number of speakers using silhouette score
    best_n = MIN_SPEAKERS
    best_score = -1
    best_labels = []

    print("Estimating number of speakers...")
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

    # Step 3: Output transcript with speaker labels
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
    df.to_csv(CSV_OUTPUT, index=False)
    print(f"Transcript with speaker labels saved to {CSV_OUTPUT}")

# --- RUN ---
transcribe_with_auto_speaker_estimation(AUDIO_FILE)
