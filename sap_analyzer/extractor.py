import pdfplumber
import logging
from typing import Optional
from pathlib import Path

def extract_text_from_pdf(pdf_path: str, max_chars: int = 8000, password: Optional[str] = None, logger: Optional[logging.Logger] = None) -> Optional[str]:
    """
    Extract text from a PDF file with error handling and logging.
    
    Args:
        pdf_path: Path to the PDF file
        max_chars: Maximum number of characters to extract (default: 8000)
        password: Password for encrypted PDFs (default: None)
        logger: Custom logger instance (default: None)
        
    Returns:
        Extracted text as string or None if extraction fails
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        if not Path(pdf_path).exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return None
            
        text = ''
        logger.info(f"Extracting text from {pdf_path}")
        
        with pdfplumber.open(pdf_path, password=password) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Found {total_pages} pages in PDF")
            
            for i, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
                    logger.debug(f"Extracted text from page {i}/{total_pages}")
                
                if len(text) >= max_chars:
                    logger.warning(f"Reached maximum character limit ({max_chars}) at page {i}")
                    break
                    
        if not text:
            logger.warning("No text could be extracted from the PDF")
            return None
            
        logger.info(f"Successfully extracted {len(text)} characters from PDF")
        return text[:max_chars]
        
    except pdfplumber.PDFSyntaxError as e:
        logger.error(f"PDF syntax error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return None