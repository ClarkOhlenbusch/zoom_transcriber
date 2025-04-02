#!/bin/bash

# Run the Zoom Transcriber and Summarizer application with local summarization (no API required)

# Navigate to the src directory
cd "$(dirname "$0")/src"

# Run the Streamlit application with local summarization
streamlit run ui/local_app.py "$@"
