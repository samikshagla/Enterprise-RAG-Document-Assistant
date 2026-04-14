import fitz  # PyMuPDF
import re
import argparse

def clean_text(text: str) -> str:
    """
    Cleans extracted text by normalizing whitespace.
    - Replaces multiple spaces or tabs with a single space
    - Replaces excessive newlines with a single newline
    """
    # Replace multiple spaces/tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Replace 3 or more newlines with exactly two newlines (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip leading/trailing whitespace
    return text.strip()

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Reads a complex PDF, extracts its text, and cleans it up.
    """
    try:
        # Open the PDF document
        doc = fitz.open(pdf_path)
        extracted_pages = []

        print(f"Processing '{pdf_path}' ({len(doc)} pages)...")

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract text
            # 'text' parameter attempts to maintain logical reading order and paragraph structure
            text = page.get_text("text")
            
            if text:
                cleaned_page_text = clean_text(text)
                extracted_pages.append(cleaned_page_text)
        
        # Join all extracted pages, separated by a distinct marker or double newline
        final_text = "\n\n--- PAGE BREAK ---\n\n".join(extracted_pages)
        return final_text
    
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Ingestion and Text Extraction")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file to ingest")
    
    args = parser.parse_args()
    
    result = extract_text_from_pdf(args.pdf_path)
    
    if result:
        print("\n--- Extracted Content Preview ---")
        # Print the first 1000 characters as a preview
        print(result[:1000])
        if len(result) > 1000:
            print("\n... [Content Truncated] ...")
        
        print("\n[SUCCESS] Document parsed and text extracted successfully.")
