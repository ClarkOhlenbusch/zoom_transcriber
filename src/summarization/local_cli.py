import os
import sys
import json
import argparse
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from summarization.local_summarizer import LocalTranscriptSummarizer

def main():
    """
    Command-line interface for the local summarization module
    """
    parser = argparse.ArgumentParser(description="Summarize meeting transcripts using local processing (no API required)")
    parser.add_argument("transcript_path", help="Path to the transcript JSON file")
    parser.add_argument("--output-dir", "-o", help="Directory to save output files")
    
    args = parser.parse_args()
    
    # Load transcript data
    try:
        with open(args.transcript_path, 'r') as f:
            transcript_data = json.load(f)
    except Exception as e:
        print(f"Error loading transcript file: {e}")
        return 1
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.transcript_path))
    
    # Add filename to transcript data for reference
    transcript_data["filename"] = args.transcript_path
    
    # Create summarizer
    summarizer = LocalTranscriptSummarizer()
    
    # Generate summary
    try:
        print(f"Summarizing transcript: {args.transcript_path}")
        summary = summarizer.summarize(transcript_data, output_dir=args.output_dir)
        
        # Print summary sections
        print("\nSummary generated successfully!")
        print("\n=== Meeting Summary ===")
        print(summary["summary"])
        print("\n=== Action Items ===")
        print(summary["action_items"])
        print("\n=== Decisions ===")
        print(summary["decisions"])
        print("\n=== Timeline ===")
        print(summary["timeline"])
        
        # Print output file paths
        transcript_path = Path(args.transcript_path)
        base_name = transcript_path.stem
        json_path = os.path.join(args.output_dir, f"{base_name}_summary.json")
        text_path = os.path.join(args.output_dir, f"{base_name}_summary.txt")
        
        print(f"\nOutput files:")
        print(f"  - JSON: {json_path}")
        print(f"  - Text: {text_path}")
        
    except Exception as e:
        print(f"Error generating summary: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
