# Zoom Transcriber and Summarizer

A simple, user-friendly application that transcribes Zoom calls with speaker identification and automatically generates comprehensive meeting summaries.

## Features

- Transcribes Zoom recordings (MP4, M4A, WAV) with speaker identification
- Generates concise meeting summaries
- Extracts action items and key decisions
- Creates a timeline of discussion topics
- Provides a user-friendly web interface
- Outputs in both text and JSON formats

## Installation

### Prerequisites

- Python 3.10+
- FFmpeg (for audio extraction)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/zoom-transcriber.git
cd zoom-transcriber
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Interface

1. Start the Streamlit web application:
```bash
cd src
streamlit run ui/app.py
```

2. Open your browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Upload a Zoom recording file (MP4 or M4A)

4. Configure transcription and summarization settings if needed

5. Click "Transcribe and Summarize" and wait for processing to complete

6. View and download the results

### Command Line Interface

#### Transcription Only

```bash
python -m transcription.cli recording.mp4 --output-dir ./output --model-size base --max-speakers 2
```

#### Summarization Only (for existing transcripts)

```bash
python -m summarization.cli transcript.json --output-dir ./output --model google/flan-t5-large
```

## How It Works

1. **Audio Extraction**: Extracts audio from video files using FFmpeg
2. **Transcription**: Uses faster-whisper to transcribe the audio
3. **Speaker Identification**: Identifies different speakers using a lightweight diarization algorithm
4. **Summarization**: Processes the transcript using LangChain and HuggingFace models to generate summaries
5. **Output Generation**: Creates formatted output files in both text and JSON formats

## Configuration Options

### Transcription

- **Whisper Model Size**: Choose from tiny, base, small, or medium (larger models are more accurate but require more resources)
- **Maximum Speakers**: Set the maximum number of speakers to identify
- **Language**: Optionally specify the language code (e.g., 'en' for English)

### Summarization

- **Summarization Model**: Choose from different HuggingFace models for summarization

## Output Files

- **Transcript JSON**: Detailed transcript with timestamps and speaker information
- **Transcript Text**: Human-readable transcript with speaker labels
- **Summary JSON**: Structured summary with sections for summary, action items, decisions, and timeline
- **Summary Text**: Formatted text summary with clear headings

## Limitations

- Speaker identification accuracy may vary depending on audio quality
- Processing large files may take significant time
- Summarization quality depends on the transcript quality and chosen model

## License

This project is licensed under the MIT License - see the LICENSE file for details.
