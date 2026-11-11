"""
PDF-to-text conversion for TECLA corpus module.

This module performs a *neutral* extraction of textual content from PDFs
using pdfminer.six. No linguistic or structural cleaning is applied.
Cleaning operations must be explicitly requested through the cleaning module.
"""

from pdfminer.high_level import extract_text


def pdf_to_text(pdf_path):
    """
    Convert a PDF file into plain text.

    No cleanup heuristics (e.g., removing headers, page numbers, short lines)
    are applied here. The goal is faithful extraction of the raw textual content.

    Parameters:
        pdf_path (str): Full path to the PDF file.

    Returns:
        str: Extracted raw text from the PDF.
    """
    try:
        raw_text = extract_text(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Could not extract text from {pdf_path}: {e}")

    return raw_text or ""
