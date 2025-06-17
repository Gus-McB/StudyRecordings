import os
import pandas as pd
from datetime import timedelta
import assemblyai as aai
from dotenv import load_dotenv

# --- CONSTANTS ---
load_dotenv()
SUPPORTED_AUDIO_EXTENSIONS = {'.amr', '.wma', '.mp3', '.m4a', '.wav'}
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# --- Convert to WAV (Mono, 16kHz) ---
def convert_to_wav(input_file, output_dir, output_filename=None):
    if not output_filename:
        output_filename = os.path.splitext(os.path.basename(input_file))[0] + ".wav"
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)
    os.system(f"ffmpeg -y -i \"{input_file}\" -ar 16000 -ac 1 \"{output_path}\"")
    return output_path

# --- Transcribe + Diarize using AssemblyAI ---
def transcribe_and_diarize(input_file, wav_output_dir, transcript_output_dir):
    os.makedirs(wav_output_dir, exist_ok=True)
    os.makedirs(transcript_output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # Convert to WAV
    wav_file = convert_to_wav(input_file, wav_output_dir, base_name + ".wav")

    # Transcribe with speaker diarization
    aai.settings.api_key = ASSEMBLYAI_API_KEY
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(wav_file, config=aai.TranscriptionConfig(speaker_labels=True))

    if transcript.status != 'completed':
        print(f"Transcription failed: {transcript.status}")
        return

    # --- Diarized transcript output ---
    rows = []
    utterances = transcript.utterances
    speaker_times = {}
    full_timeline = []

    for utterance in utterances:
        start_sec = utterance.start / 1000
        end_sec = utterance.end / 1000
        speaker_id = f"Person {utterance.speaker}"
        duration = end_sec - start_sec

        # Accumulate speaking time
        speaker_times[speaker_id] = speaker_times.get(speaker_id, 0) + duration

        # Save timeline for overlap/silence calc
        full_timeline.append((start_sec, end_sec, speaker_id))

        # Save to transcript
        rows.append({
            "start_time": str(timedelta(seconds=int(start_sec))),
            "end_time": str(timedelta(seconds=int(end_sec))),
            "speaker": speaker_id,
            "text": utterance.text.strip()
        })

    df = pd.DataFrame(rows)
    transcript_file = os.path.join(transcript_output_dir, base_name + "_transcript.csv")
    df.to_csv(transcript_file, index=False)

    # --- Overlap and Silence Analysis ---
    timeline = sorted(full_timeline, key=lambda x: x[0])
    current_speakers = []
    previous_end = 0
    overlap_time = 0
    silence_time = 0

    events = []
    for start, end, speaker in timeline:
        events.append((start, 'start', speaker))
        events.append((end, 'end', speaker))

    events.sort()
    active_speakers = set()
    last_timestamp = 0

    for timestamp, event_type, speaker in events:
        if timestamp > last_timestamp:
            delta = timestamp - last_timestamp
            if len(active_speakers) == 0:
                silence_time += delta
            elif len(active_speakers) > 1:
                overlap_time += delta
        if event_type == 'start':
            active_speakers.add(speaker)
        else:
            active_speakers.discard(speaker)
        last_timestamp = timestamp

    # --- Save summary ---
    summary_rows = [{"metric": "Total Speaking Time", "value": ""}]
    for speaker, seconds in speaker_times.items():
        summary_rows.append({
            "metric": f"{speaker} speaking time",
            "value": str(timedelta(seconds=int(seconds)))
        })

    summary_rows += [
        {"metric": "Overlapping speech time", "value": str(timedelta(seconds=int(overlap_time)))},
        {"metric": "Silence (no one speaking)", "value": str(timedelta(seconds=int(silence_time)))}
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_file = os.path.join(transcript_output_dir, base_name + "_summary.csv")
    summary_df.to_csv(summary_file, index=False)

    # Also print summary
    print("\n=== SPEAKING TIME SUMMARY ===")
    for row in summary_rows:
        print(f"{row['metric']}: {row['value']}")
