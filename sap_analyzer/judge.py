import logging
from typing import Dict, Optional
from transformers import pipeline

class AnswerJudge:
    def __init__(self, model_name: str = "google/flan-t5-large", max_length: int = 2048):
        """Initialize the answer judge system."""
        self.model_name = model_name
        self.max_length = max_length
        self.logger = logging.getLogger(__name__)
        
        # Initialize the pipeline
        try:
            self.pipe = pipeline(
                "text2text-generation",
                model=model_name,
                max_length=max_length
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {str(e)}")
            raise
        
        # Define evaluation criteria
        self.criteria = {
            "relevance": "How well does the answer address the question?",
            "accuracy": "How accurate is the information provided?",
            "completeness": "How comprehensive is the answer?",
            "clarity": "How clear and well-structured is the response?",
            "specificity": "How specific and detailed is the answer?",
            "documentation": "How well is the answer supported by the document?"
        }

    def evaluate_answer(self, question: str, answer: str, context: Optional[str] = None) -> Dict:
        """Evaluate the quality and relevance of an answer."""
        try:
            evaluation_prompt = self._create_evaluation_prompt(question, answer, context)
            evaluation = self.pipe(
                evaluation_prompt,
                max_length=self.max_length,
                do_sample=True,
                temperature=0.3,
                num_return_sequences=1
            )[0]['generated_text']
            return self._parse_evaluation(evaluation)
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            return {
                "scores": {},
                "strengths": [],
                "improvements": [],
                "comments": [],
                "error": str(e)
            }

    def _create_evaluation_prompt(self, question: str, answer: str, context: Optional[str] = None) -> str:
        """Create the evaluation prompt."""
        prompt = f"""
        Evaluate the following Q&A pair for a Statistical Analysis Plan (SAP):

        Question: {question}
        Answer: {answer}
        {f"Context: {context}" if context else ""}

        Please evaluate based on these criteria:
        """
        
        for criterion, description in self.criteria.items():
            prompt += f"\n{criterion.capitalize()} (1-5): {description}"
        
        prompt += """
        
        Provide your evaluation in this format:
        Relevance: [score]/5
        Accuracy: [score]/5
        Completeness: [score]/5
        Clarity: [score]/5
        Specificity: [score]/5
        Documentation: [score]/5
        Overall Score: [average]/5
        Strengths: [list key strengths]
        Areas for Improvement: [list areas that could be improved]
        Comments: [specific feedback]
        """
        
        return prompt

    def _parse_evaluation(self, evaluation: str) -> Dict:
        """Parse the evaluation text into a structured format."""
        scores = {}
        strengths = []
        improvements = []
        comments = []
        
        lines = evaluation.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key in self.criteria:
                    try:
                        score = float(value.split('/')[0])
                        scores[key] = score
                    except (ValueError, IndexError):
                        scores[key] = 0
                elif key == 'strengths':
                    current_section = 'strengths'
                elif key == 'areas for improvement':
                    current_section = 'improvements'
                elif key == 'comments':
                    current_section = 'comments'
                elif key == 'overall score':
                    try:
                        scores['overall'] = float(value.split('/')[0])
                    except (ValueError, IndexError):
                        scores['overall'] = 0
            else:
                if current_section == 'strengths':
                    strengths.append(line)
                elif current_section == 'improvements':
                    improvements.append(line)
                elif current_section == 'comments':
                    comments.append(line)
        
        return {
            'scores': scores,
            'strengths': strengths,
            'improvements': improvements,
            'comments': comments
        }

    def format_evaluation(self, evaluation: Dict) -> str:
        """Format the evaluation into a readable string."""
        output = []
        
        # Add scores
        output.append("\n--- Evaluation ---")
        for criterion in self.criteria:
            if criterion in evaluation['scores']:
                output.append(f"{criterion.capitalize()}: {evaluation['scores'][criterion]}/5")
        
        if 'overall' in evaluation['scores']:
            output.append(f"\nOverall Score: {evaluation['scores']['overall']}/5")
        
        # Add strengths
        if evaluation['strengths']:
            output.append("\nStrengths:")
            for strength in evaluation['strengths']:
                output.append(f"- {strength}")
        
        # Add areas for improvement
        if evaluation['improvements']:
            output.append("\nAreas for Improvement:")
            for improvement in evaluation['improvements']:
                output.append(f"- {improvement}")
        
        # Add comments
        if evaluation['comments']:
            output.append("\nComments:")
            for comment in evaluation['comments']:
                output.append(f"- {comment}")
        
        return '\n'.join(output)