import os
import json
from typing import Dict, List, Optional, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

class TranscriptSummarizer:
    """
    Class for summarizing meeting transcripts using LangChain
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
                model_kwargs={"temperature": 0.5, "max_length": 512}
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
            formatted_text += f"{speaker}: {text}\n"
        
        return formatted_text
    
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
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        return text_splitter.split_text(transcript_text)
    
    def summarize(self, transcript_data: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Summarize a meeting transcript
        
        Args:
            transcript_data (dict): Transcript data from TranscriptionManager
            output_dir (str, optional): Directory to save output files
        
        Returns:
            dict: Dictionary containing summary, action items, and key points
        """
        # Initialize language model
        self._initialize_llm()
        
        # Format transcript for summarization
        transcript_text = self._format_transcript_for_summarization(transcript_data)
        
        # Split transcript into chunks
        transcript_chunks = self._split_transcript(transcript_text)
        
        # Create summary chain
        summary_template = """
        You are an AI assistant tasked with summarizing meeting transcripts.
        
        Below is a portion of a meeting transcript:
        
        {text}
        
        Please provide a concise summary of this portion of the meeting.
        """
        
        summary_prompt = PromptTemplate(template=summary_template, input_variables=["text"])
        
        combine_template = """
        You are an AI assistant tasked with summarizing meeting transcripts.
        
        Below are summaries of different portions of a meeting transcript:
        
        {text}
        
        Based on these summaries, please provide:
        
        1. A concise overall summary of the meeting (2-3 paragraphs)
        2. A list of key action items and who is responsible for them
        3. A list of important decisions made during the meeting
        4. A brief timeline of the main topics discussed
        
        Format your response with clear headings for each section.
        """
        
        combine_prompt = PromptTemplate(template=combine_template, input_variables=["text"])
        
        chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=summary_prompt,
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
        Parse the summary text into sections
        
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
        
        # Simple parsing based on headings
        current_section = "summary"
        lines = summary_text.split("\n")
        
        for line in lines:
            lower_line = line.lower()
            
            if "action item" in lower_line or "action items" in lower_line:
                current_section = "action_items"
                continue
            elif "decision" in lower_line or "decisions" in lower_line:
                current_section = "decisions"
                continue
            elif "timeline" in lower_line or "topics" in lower_line:
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
        
        return sections
    
    def _save_summary(self, summary_sections: Dict[str, str], transcript_data: Dict[str, Any], output_dir: str) -> None:
        """
        Save summary to files
        
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
        
        # Save text summary
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
