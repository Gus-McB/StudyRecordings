import whisper
import pandas as pd
import os
from datetime import timedelta

# --- SETTINGS ---
AUDIO_FILE = "/home/Angus/Desktop/StudyRecordings/Test 1/Test 1/Local 1/160119_0030.WMA"
CSV_OUTPUT = "transcript.csv"

# --- CONVERT WMA TO WAV ---
def convert_to_wav(input_file, output_file="converted.wav"):
    os.system(f"ffmpeg -y -i \"{input_file}\" -ar 16000 -ac 1 \"{output_file}\"")
    return output_file

# --- LOAD WHISPER MODEL ---
model = whisper.load_model("large-v2")

# --- TRANSCRIBE WITH SPEAKER CHANGE DETECTION ---
def transcribe(audio_file):
    wav_file = convert_to_wav(audio_file)

    result = model.transcribe(wav_file, verbose=False)

    rows = []
    current_speaker = 1

    for i, segment in enumerate(result['segments']):
        start = str(timedelta(seconds=int(segment['start'])))
        end = str(timedelta(seconds=int(segment['end'])))
        text = segment['text'].strip()

        # Simulate speaker change detection
        # If this segment is far from the last one or starts with something like "Speaker" or a long pause
        if i > 0:
            prev_end = result['segments'][i - 1]['end']
            if segment['start'] - prev_end > 2:  # simulate speaker change if there's a pause >2s
                current_speaker += 1

        rows.append({
            "start_time": start,
            "end_time": end,
            "speaker": f"Person {current_speaker}",
            "text": text
        })

    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUTPUT, index=False)
    print(f"Transcript saved to {CSV_OUTPUT}")

# --- RUN ---
transcribe(AUDIO_FILE)