# Zoom Transcriber and Summarizer - Program Architecture

## Overview

This document outlines the architecture for the Zoom Transcriber and Summarizer application. The program will provide an end-to-end solution for transcribing Zoom recordings with speaker identification and generating comprehensive meeting summaries.

## System Components

The application will consist of the following main components:

1. **User Interface (UI)** - A simple, intuitive interface for users to upload Zoom recordings and view results
2. **Transcription Module** - Handles audio processing and speech-to-text with speaker identification
3. **Summarization Module** - Processes transcripts to generate meeting summaries, action items, and key points
4. **File Management** - Handles file I/O operations and format conversions
5. **Configuration Manager** - Manages API keys and user preferences

## Architecture Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  User Interface │────▶│   File Manager  │────▶│  Transcription  │
│                 │     │                 │     │     Module      │
│                 │◀────│                 │◀────│                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Output Format  │◀────│  Summarization  │◀────│   Transcript    │
│    Generator    │     │     Module      │     │    Processor    │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Component Details

### 1. User Interface (UI)

**Implementation:** Web-based interface using Flask or Streamlit

**Features:**
- File upload for Zoom recordings (MP4 or M4A formats)
- Progress indicators for transcription and summarization
- Display and download options for results
- Configuration settings for API keys and preferences

### 2. Transcription Module

**Implementation:** Based on research findings, we'll use a hybrid approach:

**Primary Option:** OpenAI Whisper + Pyannote-Audio
- Whisper for high-quality transcription
- Pyannote-Audio for speaker diarization
- Local processing to avoid API costs

**Alternative Option:** AssemblyAI API
- Simpler implementation with built-in speaker diarization
- Limited by free tier credits ($50)

**Features:**
- Audio extraction from video files
- Speech-to-text conversion
- Speaker identification and labeling
- Timestamp generation

### 3. Summarization Module

**Implementation:** Flexible approach with multiple options:

**Primary Option:** LangChain with open-source models
- Use LangChain's map-reduce approach for handling long transcripts
- Integrate with Hugging Face models (BART, T5, or Pegasus)
- Completely free, open-source solution

**Alternative Option:** Minimal API usage with commercial providers
- Use OpenAI or Claude API only for the final summarization step
- Minimize token usage to reduce costs
- Higher quality summaries

**Features:**
- Generate concise meeting summaries
- Extract action items and assignments
- Identify key decisions
- Create timeline of discussion topics

### 4. File Management

**Implementation:** Python utilities for file handling

**Features:**
- Audio extraction from video files
- Format conversions
- Temporary file management
- Output file generation (PDF, TXT)

### 5. Configuration Manager

**Implementation:** Simple configuration file or database

**Features:**
- Store API keys securely
- User preferences
- Model selection options

## Data Flow

1. User uploads Zoom recording through the UI
2. File Manager processes the file and extracts audio if needed
3. Transcription Module processes audio to generate transcript with speaker identification
4. Transcript Processor formats and prepares the transcript for summarization
5. Summarization Module generates meeting summary, action items, and key points
6. Output Format Generator creates the final document in the desired format
7. User receives notification and can view/download the results

## Technical Stack

- **Programming Language:** Python 3.x
- **Web Framework:** Flask or Streamlit for UI
- **Transcription:** Whisper + Pyannote-Audio or AssemblyAI API
- **Summarization:** LangChain with Hugging Face models or commercial API
- **Audio Processing:** FFmpeg for media handling
- **Output Generation:** ReportLab or WeasyPrint for PDF generation

## Implementation Considerations

### Transcription Approach

We'll implement the Whisper + Pyannote-Audio approach as the primary solution since it:
- Is completely free and open-source
- Can run locally without API costs
- Provides high-quality transcription with speaker identification

We'll include AssemblyAI as an alternative option for users who prefer a simpler setup and don't mind using the free tier credits.

### Summarization Approach

We'll implement a flexible approach for summarization:
1. Default: LangChain with open-source models for a completely free solution
2. Optional: Integration with commercial APIs (OpenAI/Claude) for higher quality summaries

This allows users to choose based on their quality requirements and budget constraints.

### Scalability and Performance

- For long recordings, we'll implement chunking strategies to handle memory constraints
- Progress indicators will keep users informed during longer processing times
- Temporary files will be used to manage memory usage efficiently

## Next Steps

1. Set up the project structure and dependencies
2. Implement the transcription module with Whisper and Pyannote-Audio
3. Implement the summarization module with LangChain
4. Create a simple web interface using Streamlit
5. Integrate all components and test with sample recordings
6. Refine and optimize based on testing results
7. Prepare documentation and delivery package
