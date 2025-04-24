# SAP Clinical Trial Analysis Tool

This tool analyzes Statistical Analysis Plans (SAP) from clinical trials using advanced natural language processing models to extract and interpret key information.

## Features
- **PDF Text Extraction**: Extracts text from PDF documents, including encrypted files.
- **Automated SAP Analysis**: Analyzes SAP documents for primary efficacy outcomes, safety analysis, statistical methodology, and more.
- **Structured Output**: Provides structured responses with clear sections for outcomes, methods, interpretations, and compliance.
- **FDA E9 Compliance Checking**: Assesses compliance with FDA E9 guidelines.
- **Answer Evaluation**: Evaluates the quality of answers based on relevance, accuracy, completeness, and other criteria.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command-Line Interface
Run the tool using the following command:
```bash
python -m sap_analyzer.main <path_to_pdf>
```

### Interactive Q&A Mode
After processing the document, the tool enters an interactive mode where you can ask questions about the SAP document. Type your question and receive structured answers. Type `exit` to end the session.

### Output
The tool generates a JSON output file in the `outputs` directory with the analysis results.

## Output Format

The output JSON file contains:
- **Primary Efficacy Outcome**: Key findings and methods.
- **Safety Analysis**: Safety endpoints, adverse event reporting, and monitoring plans.
- **Statistical Methodology**: Sample size calculations, statistical tests, and handling of missing data.
- **FDA E9 Compliance**: Assessment of compliance with regulatory guidelines.
- **Answer Evaluation**: Scores and feedback on relevance, accuracy, and other criteria.

## Requirements
- Python 3.8+
- CUDA-capable GPU (recommended for faster processing)
- See `requirements.txt` for package dependencies

## Development

### Key Modules
- **`extractor.py`**: Handles PDF text extraction with support for encrypted files.
- **`judge.py`**: Evaluates the quality of answers based on predefined criteria.
- **`prompts.py`**: Contains system prompts and utilities for formatting prompts.
- **`main.py`**: Entry point for the tool, including document processing and interactive Q&A.

### Testing
Unit tests are located in the `tests` directory. Run tests using:
```bash
pytest
```

## Logging
The tool uses Python's `logging` module for error handling and debugging. Logs are displayed in the console.

## Future Enhancements
- Support for additional regulatory guidelines (e.g., EMA, ICH).
- Localization for non-English SAP documents.
- Enhanced evaluation metrics for answer quality.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
