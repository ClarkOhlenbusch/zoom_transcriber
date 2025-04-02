#!/bin/bash

# Run the Zoom Transcriber and Summarizer application

# Navigate to the src directory
cd "$(dirname "$0")/src"

# Run the Streamlit application
streamlit run ui/app.py "$@"
