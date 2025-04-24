import os
import json
import sys
import torch
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sap_analyzer.extractor import extract_text_from_pdf
from sap_analyzer.judge import AnswerJudge
import argparse
import logging

class SAPAnalyzer:
    def __init__(self, model_name: str = "google/flan-t5-large"):
        self.model_name = model_name
        
        # Check if MPS is available
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        # Initialize pipeline
        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device if self.device != "mps" else -1,
            max_length=2048
        )
        
        # Initialize judge system
        self.judge = AnswerJudge(model_name)
        
        self.extracted_text = None
        self.max_length = 2048
        self.chunk_size = 1024

    def _generate_qa_response(self, question: str, context: str) -> str:
        """Generate a response in Q&A format."""
        if len(self.tokenizer.encode(context)) > self.max_length:
            chunks = [context[i:i+self.chunk_size] for i in range(0, len(context), self.chunk_size)]
            context = " ".join(chunks[:2])
        
        prompt = f"""
        You are an expert in analyzing Statistical Analysis Plans (SAPs). 
        Please answer the following question based on the provided document excerpt.
        Focus specifically on answering the question asked, not providing general information.

        Question: {question}

        Document Excerpt:
        {context}

        Please provide a direct answer to the question, followed by supporting details from the document.
        Structure your response as:

        1. Direct Answer:
        [Provide a clear, concise answer to the specific question asked]

        2. Supporting Evidence:
        [Cite specific details from the document that support your answer]

        3. Additional Context:
        [Provide any relevant additional information that helps understand the answer]

        If the information is not available in the document, please state that clearly.
        """
        
        response = self.pipe(
            prompt,
            max_length=self.max_length,
            do_sample=True,
            temperature=0.3,
            num_return_sequences=1
        )[0]['generated_text']
        
        return response.strip()

    def process_document(self, pdf_path: str):
        """Process the document and extract text."""
        try:
            self.extracted_text = extract_text_from_pdf(pdf_path)
            if not self.extracted_text:
                raise ValueError("No text could be extracted from the PDF")
            return True
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            return False

    def answer_question(self, question: str) -> str:
        """Answer a specific question about the document."""
        if not self.extracted_text:
            raise ValueError("No document has been processed yet. Please run process_document first.")
        
        try:
            # Generate initial answer
            answer = self._generate_qa_response(question, self.extracted_text)
            if not answer:
                return "I couldn't generate a response. Please try rephrasing your question."
            
            # Evaluate the answer
            evaluation = self.judge.evaluate_answer(question, answer, self.extracted_text)
            formatted_evaluation = self.judge.format_evaluation(evaluation)
            
            # Combine answer and evaluation
            final_response = f"{answer}\n{formatted_evaluation}"
            return final_response
        except Exception as e:
            print(f"Error answering question: {str(e)}")
            return None

def setup_logger():
    """Set up a logger for the application."""
    logger = logging.getLogger("SAPAnalyzer")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="SAP Analyzer Tool")
    parser.add_argument("pdf_path", help="Path to the PDF file to analyze")
    parser.add_argument("--model", default="google/flan-t5-large", help="Name of the model to use (default: google/flan-t5-large)")
    return parser.parse_args()

def main():
    logger = setup_logger()
    args = parse_arguments()

    pdf_path = args.pdf_path
    model_name = args.model

    if not os.path.exists(pdf_path):
        logger.error(f"Error: PDF file not found at {pdf_path}")
        return

    if not pdf_path.lower().endswith(".pdf"):
        logger.error("Error: The provided file is not a PDF.")
        return

    # Initialize analyzer
    analyzer = SAPAnalyzer(model_name=model_name)
    
    # Process the document
    if not analyzer.process_document(pdf_path):
        logger.error("Failed to process the document.")
        return

    # Interactive Q&A session
    print("\n=== Question Answering Mode ===")
    print("Type 'exit' to end the session")
    print("Note: All responses are based on the information available in the document. Please verify specific details with the complete SAP document.\n")
    
    try:
        while True:
            question = input("\nYour question: ").strip()
            
            if question.lower() == 'exit':
                print("\nEnding session. Goodbye!")
                break
                
            if question:
                try:
                    answer = analyzer.answer_question(question)
                    if answer:
                        print(f"\n{answer}")
                except Exception as e:
                    logger.error(f"Error answering question: {str(e)}")
    except KeyboardInterrupt:
        print("\nSession interrupted. Goodbye!")

if __name__ == '__main__':
    main()
