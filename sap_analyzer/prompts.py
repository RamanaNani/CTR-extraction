"""
This module contains the system prompts used for SAP analysis.
"""

# Base template for SAP analysis
system_template = '''
You are a clinical trial SAP analysis assistant.
Use the following SAP document to answer all questions.

------START OF DOCUMENT------
{document}
------END OF DOCUMENT------

If the question is about:
- Primary efficacy outcome
- Safety analysis
- Statistical methodology
- FDA E9 compliance

Respond clearly with structure:
1. Outcome:
2. Methods:
3. Interpretation:
4. FDA E9 Compliance: Yes/No and Why

Additional Guidelines:
- Be precise and specific in your analysis
- Cite relevant sections from the document when possible
- Highlight any potential concerns or limitations
- Ensure compliance with regulatory requirements
'''

# Specialized prompts
safety_analysis_prompt = '''
Analyze the safety aspects of the clinical trial from the SAP document.
Focus on:
1. Safety endpoints
2. Adverse event reporting
3. Safety monitoring plan
4. Statistical methods for safety analysis
'''

statistical_methodology_prompt = '''
Analyze the statistical methodology described in the SAP document.
Focus on:
1. Sample size calculation
2. Statistical tests planned
3. Handling of missing data
4. Interim analysis plan
'''

secondary_efficacy_prompt = '''
Analyze the secondary efficacy outcomes described in the SAP document.
Focus on:
1. Key secondary endpoints
2. Statistical methods for secondary outcomes
3. Interpretation of results
4. Potential limitations or challenges
'''

data_integrity_prompt = '''
Evaluate the data integrity and quality aspects of the SAP document.
Focus on:
1. Data collection methods
2. Data cleaning and validation procedures
3. Handling of outliers and missing data
4. Compliance with Good Clinical Practice (GCP) guidelines
'''

# Utility function for formatting prompts
def format_prompt(template: str, **kwargs) -> str:
    """
    Format a prompt template by replacing placeholders with provided values.

    Args:
        template: The prompt template containing placeholders (e.g., {document}).
        kwargs: Key-value pairs to replace placeholders in the template.

    Returns:
        The formatted prompt with placeholders replaced.

    Raises:
        KeyError: If a required placeholder is missing in kwargs.
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise KeyError(f"Missing placeholder in template: {str(e)}")
