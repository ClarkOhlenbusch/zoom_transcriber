import os
import sys
import argparse
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcription.transcription_manager import TranscriptionManager

def main():
    """
    Command-line interface for the transcription module
    """
    parser = argparse.ArgumentParser(description="Transcribe Zoom recordings with speaker identification")
    parser.add_argument("recording_path", help="Path to the Zoom recording (MP4 or M4A)")
    parser.add_argument("--output-dir", "-o", help="Directory to save output files")
    parser.add_argument("--language", "-l", help="Language code for transcription (e.g., 'en')")
    parser.add_argument("--model-size", "-m", default="base", 
                        choices=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"],
                        help="Size of the Whisper model to use")
    parser.add_argument("--max-speakers", "-s", type=int, default=2,
                        help="Maximum number of speakers to identify")
    
    args = parser.parse_args()
    
    # Create transcription manager
    manager = TranscriptionManager(
        whisper_model_size=args.model_size,
        max_speakers=args.max_speakers
    )
    
    # Process recording
    try:
        print(f"Processing recording: {args.recording_path}")
        result = manager.process_recording(
            args.recording_path,
            output_dir=args.output_dir,
            language=args.language
        )
        
        # Print summary
        print("\nTranscription completed successfully!")
        print(f"Number of speakers identified: {result['num_speakers']}")
        print(f"Language: {result['language']} (confidence: {result['language_probability']:.2f})")
        print(f"Number of segments: {len(result['segments'])}")
        
        # Print output file paths
        output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.recording_path))
        recording_path = Path(args.recording_path)
        json_path = os.path.join(output_dir, f"{recording_path.stem}_transcript.json")
        text_path = os.path.join(output_dir, f"{recording_path.stem}_transcript.txt")
        
        print(f"\nOutput files:")
        print(f"  - JSON: {json_path}")
        print(f"  - Text: {text_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
