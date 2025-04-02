import os
import sys
import argparse
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """
    Command-line interface for the Zoom Transcriber and Summarizer
    """
    parser = argparse.ArgumentParser(
        description="Zoom Transcriber and Summarizer - Transcribe and summarize Zoom recordings"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Transcribe command
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe a Zoom recording")
    transcribe_parser.add_argument("recording_path", help="Path to the Zoom recording (MP4 or M4A)")
    transcribe_parser.add_argument("--output-dir", "-o", help="Directory to save output files")
    transcribe_parser.add_argument("--language", "-l", help="Language code for transcription (e.g., 'en')")
    transcribe_parser.add_argument("--model-size", "-m", default="base", 
                        choices=["tiny", "base", "small", "medium"],
                        help="Size of the Whisper model to use")
    transcribe_parser.add_argument("--max-speakers", "-s", type=int, default=2,
                        help="Maximum number of speakers to identify")
    
    # Summarize command (API version)
    summarize_parser = subparsers.add_parser("summarize", help="Summarize a transcript (requires HuggingFace API token)")
    summarize_parser.add_argument("transcript_path", help="Path to the transcript JSON file")
    summarize_parser.add_argument("--output-dir", "-o", help="Directory to save output files")
    summarize_parser.add_argument("--model", "-m", default="google/flan-t5-base", 
                        help="HuggingFace model to use for summarization")
    
    # Local summarize command
    local_summarize_parser = subparsers.add_parser("local-summarize", help="Summarize a transcript using local processing (no API required)")
    local_summarize_parser.add_argument("transcript_path", help="Path to the transcript JSON file")
    local_summarize_parser.add_argument("--output-dir", "-o", help="Directory to save output files")
    
    # UI command (API version)
    ui_parser = subparsers.add_parser("ui", help="Start the web interface (requires HuggingFace API token)")
    ui_parser.add_argument("--port", "-p", type=int, default=8501, help="Port to run the web interface on")
    
    # Local UI command
    local_ui_parser = subparsers.add_parser("local-ui", help="Start the web interface with local summarization (no API required)")
    local_ui_parser.add_argument("--port", "-p", type=int, default=8501, help="Port to run the web interface on")
    
    args = parser.parse_args()
    
    if args.command == "transcribe":
        from src.transcription.cli import main as transcribe_main
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        return transcribe_main()
    
    elif args.command == "summarize":
        from src.summarization.cli import main as summarize_main
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        return summarize_main()
    
    elif args.command == "local-summarize":
        from src.summarization.local_cli import main as local_summarize_main
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        return local_summarize_main()
    
    elif args.command == "ui":
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "ui", "app.py"), "--server.port", str(args.port)]
        stcli.main()
        return 0
    
    elif args.command == "local-ui":
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "ui", "local_app.py"), "--server.port", str(args.port)]
        stcli.main()
        return 0
    
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
