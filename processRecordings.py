import os
import convertToTranscript as ctt

SUPPORTED_EXTENSIONS = {'.wma', '.wav', '.mp3', '.m4a', '.amr'}

def is_audio_file(filename):
    return os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS

def process_directory(root_dir, output_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if is_audio_file(file):
                input_path = os.path.join(dirpath, file)

                # Compute relative path (e.g. 'folderA/audio1.wma')
                rel_path = os.path.relpath(input_path, root_dir)
                rel_folder = os.path.dirname(rel_path)
                base_name = os.path.splitext(os.path.basename(input_path))[0]

                # Output paths, preserving folder structure
                transcript_subdir = os.path.join(output_dir, "transcripts", rel_folder)
                wav_subdir = os.path.join(output_dir, "convertedRecordings", rel_folder)

                transcript_path = os.path.join(transcript_subdir, base_name + "_transcript.csv")

                # Skip if already processed
                if os.path.exists(transcript_path):
                    print(f"Skipping (already processed): {input_path}")
                    continue

                print(f"\nProcessing: {input_path}")
                try:
                    os.makedirs(transcript_subdir, exist_ok=True)
                    os.makedirs(wav_subdir, exist_ok=True)

                    ctt.transcribe_and_diarize(
                        input_file=input_path,
                        wav_output_dir=wav_subdir,
                        transcript_output_dir=transcript_subdir
                    )
                except Exception as e:
                    print(f"Failed to process {input_path}: {e}")

if __name__ == "__main__":
    input_dir = r"C:\Users\angus\Desktop\PersonalVSCode\Github\StudyRecordings\Test 1"
    output_dir = r"C:\Users\angus\Desktop\PersonalVSCode\Github\StudyRecordings"
    process_directory(input_dir, output_dir)
