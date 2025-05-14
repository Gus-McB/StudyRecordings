import os
from processRecordings import transcribe_and_diarize

def process_directory(root_dir, output_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            input_path = os.path.join(dirpath, file)
            print(f"\nProcessing: {input_path}")
            try:
                transcribe_and_diarize(input_path, output_dir)
            except Exception as e:
                print(f"Failed to process {input_path}: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch transcribe and diarize audio files in a folder.")
    parser.add_argument("input_dir", help="Root directory containing audio files")
    parser.add_argument("output_dir", help="Directory to store converted recordings and transcripts")

    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)
