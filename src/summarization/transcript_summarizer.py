import os
import json
from typing import Dict, List, Optional, Any
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

class TranscriptSummarizer:
    """
    Enhanced class for summarizing meeting transcripts using LangChain
    with improved prompts for more concise, intelligent summaries
    """
    
    def __init__(self, huggingface_model="google/flan-t5-large"):
        """
        Initialize the TranscriptSummarizer
        
        Args:
            huggingface_model (str): HuggingFace model to use for summarization
        """
        self.huggingface_model = huggingface_model
        self.llm = None
        
    def _initialize_llm(self):
        """Initialize the language model if not already initialized"""
        if self.llm is None:
            # Use HuggingFace models through their API
            self.llm = HuggingFaceHub(
                repo_id=self.huggingface_model,
                model_kwargs={"temperature": 0.3, "max_length": 512}  # Lower temperature for more focused output
            )
    
    def _format_transcript_for_summarization(self, transcript_data: Dict[str, Any]) -> str:
        """
        Format transcript data into a string for summarization
        
        Args:
            transcript_data (dict): Transcript data from TranscriptionManager
        
        Returns:
            str: Formatted transcript text
        """
        formatted_text = "Meeting Transcript:\n\n"
        
        for segment in transcript_data["segments"]:
            speaker = segment.get("speaker", "Unknown Speaker")
            text = segment.get("text", "")
            timestamp = self._format_time(segment.get("start", 0))
            formatted_text += f"[{timestamp}] {speaker}: {text}\n"
        
        return formatted_text
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def _split_transcript(self, transcript_text: str) -> List[str]:
        """
        Split transcript into chunks for processing
        
        Args:
            transcript_text (str): Formatted transcript text
        
        Returns:
            list: List of transcript chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=400,  # Increased overlap for better context preservation
            separators=["\n\n", "\n", " ", ""]
        )
        
        return text_splitter.split_text(transcript_text)
    
    def _extract_meeting_metadata(self, transcript_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata about the meeting to provide context for summarization
        
        Args:
            transcript_data (dict): Transcript data from TranscriptionManager
            
        Returns:
            dict: Meeting metadata
        """
        # Extract filename if available
        filename = ""
        if "filename" in transcript_data:
            filename = os.path.basename(transcript_data["filename"])
            # Clean up filename to extract potential meeting name
            filename = os.path.splitext(filename)[0]
            filename = filename.replace("_", " ").replace("-", " ")
        
        # Get duration
        duration = 0
        if transcript_data.get("segments"):
            last_segment = transcript_data["segments"][-1]
            duration = last_segment.get("end", 0)
        
        # Get number of speakers
        speakers = set()
        for segment in transcript_data.get("segments", []):
            if "speaker" in segment:
                speakers.add(segment["speaker"])
        
        return {
            "title": filename,
            "duration_minutes": int(duration // 60),
            "num_speakers": len(speakers),
            "speakers": list(speakers)
        }
    
    def summarize(self, transcript_data: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Summarize a meeting transcript with improved prompts for more concise, intelligent summaries
        
        Args:
            transcript_data (dict): Transcript data from TranscriptionManager
            output_dir (str, optional): Directory to save output files
        
        Returns:
            dict: Dictionary containing summary, action items, and key points
        """
        # Initialize language model
        self._initialize_llm()
        
        # Extract meeting metadata
        metadata = self._extract_meeting_metadata(transcript_data)
        
        # Format transcript for summarization
        transcript_text = self._format_transcript_for_summarization(transcript_data)
        
        # Split transcript into chunks
        transcript_chunks = self._split_transcript(transcript_text)
        
        # Create improved map prompt for individual chunk summarization
        map_template = """
        You are an expert meeting analyst tasked with creating concise, insightful summaries of meeting transcripts.
        
        Below is a portion of a meeting transcript:
        
        {text}
        
        Create a brief, focused summary of this portion that:
        1. Captures the key points and essential information
        2. Identifies any decisions made
        3. Notes any action items mentioned
        4. Highlights important questions or concerns raised
        
        Focus on extracting meaningful insights rather than simply repeating what was said.
        Be concise and clear, avoiding unnecessary details or filler content.
        """
        
        map_prompt = PromptTemplate(template=map_template, input_variables=["text"])
        
        # Create improved combine prompt for final summary
        combine_template = """
        You are an expert meeting analyst tasked with creating a comprehensive yet concise summary of a meeting.
        
        Below are summaries of different portions of a meeting transcript:
        
        {text}
        
        Based on these summaries, create a clear, concise, and insightful meeting summary that includes:
        
        1. A concise executive summary (2-3 paragraphs) that captures the essence of the meeting
        2. A list of key action items with assigned responsibilities and deadlines (if mentioned)
        3. A list of important decisions made during the meeting
        4. A brief timeline of the main topics discussed
        
        Guidelines:
        - Be concise and focused - the summary should be easy to scan and understand quickly
        - Use clear headings to separate each section
        - Focus on extracting meaningful insights rather than simply repeating what was said
        - Organize information logically and prioritize the most important points
        - Use bullet points for action items, decisions, and timeline points
        - Write in a professional, clear tone
        
        The goal is to create a summary that provides maximum value with minimum reading time.
        """
        
        combine_prompt = PromptTemplate(template=combine_template, input_variables=["text"])
        
        # Create and run the summarization chain
        chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True
        )
        
        # Run the chain
        result = chain.run(transcript_chunks)
        
        # Parse the result into sections
        sections = self._parse_summary_sections(result)
        
        # Save results if output directory is provided
        if output_dir:
            self._save_summary(sections, transcript_data, output_dir)
        
        return sections
    
    def _parse_summary_sections(self, summary_text: str) -> Dict[str, str]:
        """
        Parse the summary text into sections with improved pattern matching
        
        Args:
            summary_text (str): Raw summary text from the LLM
        
        Returns:
            dict: Dictionary with sections (summary, action_items, decisions, timeline)
        """
        sections = {
            "summary": "",
            "action_items": "",
            "decisions": "",
            "timeline": ""
        }
        
        # Improved parsing with better pattern matching
        current_section = "summary"
        lines = summary_text.split("\n")
        
        # Define patterns for section headers
        summary_patterns = [r"executive\s+summary", r"summary", r"overview"]
        action_patterns = [r"action\s+items?", r"next\s+steps", r"to-?dos?", r"tasks?"]
        decision_patterns = [r"decisions?", r"conclusions?", r"outcomes?", r"agreed"]
        timeline_patterns = [r"timeline", r"topics?", r"agenda", r"discussion\s+points"]
        
        for line in lines:
            lower_line = line.lower()
            
            # Check for section headers
            if any(re.search(pattern, lower_line) for pattern in summary_patterns):
                current_section = "summary"
                continue
            elif any(re.search(pattern, lower_line) for pattern in action_patterns):
                current_section = "action_items"
                continue
            elif any(re.search(pattern, lower_line) for pattern in decision_patterns):
                current_section = "decisions"
                continue
            elif any(re.search(pattern, lower_line) for pattern in timeline_patterns):
                current_section = "timeline"
                continue
            
            # Add line to current section
            if sections[current_section]:
                sections[current_section] += "\n" + line
            else:
                sections[current_section] += line
        
        # Clean up sections
        for section in sections:
            sections[section] = sections[section].strip()
            
            # If summary section is empty but there's content before any headers, use that as summary
            if section == "summary" and not sections[section]:
                # Find content before first header
                first_header_idx = float('inf')
                for pattern_list in [action_patterns, decision_patterns, timeline_patterns]:
                    for pattern in pattern_list:
                        matches = [i for i, line in enumerate(lines) if re.search(pattern, line.lower())]
                        if matches:
                            first_header_idx = min(first_header_idx, matches[0])
                
                if first_header_idx < float('inf'):
                    sections["summary"] = "\n".join(lines[:first_header_idx]).strip()
        
        return sections
    
    def _save_summary(self, summary_sections: Dict[str, str], transcript_data: Dict[str, Any], output_dir: str) -> None:
        """
        Save summary to files with improved formatting
        
        Args:
            summary_sections (dict): Dictionary with summary sections
            transcript_data (dict): Original transcript data
            output_dir (str): Directory to save output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine base filename from transcript data
        if "filename" in transcript_data:
            base_name = os.path.splitext(os.path.basename(transcript_data["filename"]))[0]
        else:
            base_name = "meeting"
        
        # Save JSON summary
        json_path = os.path.join(output_dir, f"{base_name}_summary.json")
        with open(json_path, 'w') as f:
            json.dump(summary_sections, f, indent=2)
        
        # Save text summary with improved formatting
        text_path = os.path.join(output_dir, f"{base_name}_summary.txt")
        with open(text_path, 'w') as f:
            f.write("# Meeting Summary\n\n")
            f.write(summary_sections["summary"])
            f.write("\n\n# Action Items\n\n")
            f.write(summary_sections["action_items"])
            f.write("\n\n# Decisions\n\n")
            f.write(summary_sections["decisions"])
            f.write("\n\n# Timeline\n\n")
            f.write(summary_sections["timeline"])
        
        print(f"Summary saved to {json_path} and {text_path}")
