# Import required libraries
import os
import base64
import requests
from io import BytesIO
from PIL import Image
import PyPDF2
from mistralai import Mistral
from tqdm import tqdm
import time
import json

class MistralOCRProcessor:
    def __init__(self, api_key):
        """Initialize the Mistral OCR processor with API key"""
        self.client = Mistral(api_key=api_key)
        self.api_key = api_key
        self.extracted_pages = []

    def upload_file_to_mistral(self, file_path):
        """Upload file to Mistral and get signed URL"""
        try:
            with open(file_path, 'rb') as file:
                uploaded_file = self.client.files.upload(
                    file={
                        "file_name": os.path.basename(file_path),
                        "content": file,
                    },
                    purpose="ocr"
                )

            # Get signed URL
            signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id)
            return signed_url.url, uploaded_file.id

        except Exception as e:
            print(f"Error uploading file: {str(e)}")
            return None, None

    def process_entire_document(self, document_url):
        """Process entire document using Mistral OCR API"""
        try:
            print("Processing document with OCR...")

            # Process the document with OCR (all pages at once)
            response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": document_url
                }
            )

            return response

        except Exception as e:
            print(f"Error processing document: {str(e)}")
            return None

    def process_pdf_pages(self, pdf_path, max_pages=49):
        """Process PDF and extract text and tables only"""
        try:
            # Upload PDF to Mistral
            print("Uploading PDF to Mistral...")
            document_url, file_id = self.upload_file_to_mistral(pdf_path)

            if not document_url:
                print("Failed to upload PDF")
                return False

            print(f"Starting OCR processing...")

            # Process the entire document
            result = self.process_entire_document(document_url)

            if result and hasattr(result, 'pages') and result.pages:
                print(f"Successfully processed {len(result.pages)} pages")

                # Process each page from the response
                for i, page_data in enumerate(result.pages):
                    if i >= max_pages:  # Limit to max_pages
                        break

                    try:
                        # Extract content from the page
                        if hasattr(page_data, 'markdown') and page_data.markdown:
                            # Filter text and table content only (no images)
                            filtered_content = self.filter_text_and_tables(page_data.markdown)
                        else:
                            filtered_content = ""

                        page_info = {
                            'page_number': i + 1,
                            'content': filtered_content,
                            'success': True
                        }

                        self.extracted_pages.append(page_info)

                        # Print progress
                        print(f"\n--- PAGE {i + 1} CONTENT ---")
                        print(filtered_content[:500] + "..." if len(filtered_content) > 500 else filtered_content)
                        print(f"--- END PAGE {i + 1} ---\n")

                    except Exception as e:
                        print(f"Error processing page {i + 1}: {str(e)}")
                        self.extracted_pages.append({
                            'page_number': i + 1,
                            'content': "",
                            'success': False,
                            'error': str(e)
                        })

                return True
            else:
                print("No pages found in OCR response")
                return False

        except Exception as e:
            print(f"Error in processing PDF: {str(e)}")
            return False

    def filter_text_and_tables(self, markdown_content):
        """Filter content to keep only text and tables, remove image references"""
        if not markdown_content:
            return ""

        lines = markdown_content.split('\n')
        filtered_lines = []

        for line in lines:
            # Skip image references
            if line.strip().startswith('![') and '](' in line and line.strip().endswith(')'):
                continue
            # Skip standalone image tags
            if line.strip().startswith('<img') and line.strip().endswith('>'):
                continue
            # Keep everything else (text, tables, headers, etc.)
            filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def save_to_markdown(self, output_filename="extracted_content.md"):
        """Save all extracted content to a markdown file"""
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write("# OCR Extracted Content\n\n")
                f.write(f"Total pages processed: {len(self.extracted_pages)}\n")
                f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                for page_info in self.extracted_pages:
                    f.write(f"## Page {page_info['page_number']}\n\n")

                    if page_info['success']:
                        f.write(page_info['content'])
                        f.write("\n\n---\n\n")
                    else:
                        f.write("*Failed to extract content from this page*\n")
                        if 'error' in page_info:
                            f.write(f"Error: {page_info['error']}\n")
                        f.write("\n---\n\n")

            print(f"\n✅ Content saved to {output_filename}")
            return output_filename

        except Exception as e:
            print(f"Error saving to markdown: {str(e)}")
            return None

    def download_file(self, filename):
        """Download the generated markdown file"""
        try:
            from google.colab import files
            files.download(filename)
            print(f"📥 Downloaded {filename}")
        except ImportError:
            print("Not running in Colab - file saved locally")
        except Exception as e:
            print(f"Error downloading file: {str(e)}")

# Main execution function
def process_pdf_with_mistral_ocr(pdf_path, api_key="LkXgm8OQ97KxIZDHkLChILFxynZ4FGpX"):
    """Main function to process PDF with Mistral OCR"""

    print("🚀 Starting Mistral OCR Processing...")
    print(f"📄 Processing file: {pdf_path}")

    # Initialize the OCR processor
    ocr_processor = MistralOCRProcessor(api_key)

    # Process the PDF
    success = ocr_processor.process_pdf_pages(pdf_path, max_pages=49)

    if success:
        # Save to markdown
        output_file = ocr_processor.save_to_markdown("bengali_book_ocr.md")

        if output_file:
            # Download the file
            ocr_processor.download_file(output_file)

            # Print summary
            print(f"\n📊 Processing Summary:")
            print(f"Total pages processed: {len(ocr_processor.extracted_pages)}")
            successful_pages = sum(1 for page in ocr_processor.extracted_pages if page['success'])
            print(f"Successfully processed: {successful_pages}")
            print(f"Failed pages: {len(ocr_processor.extracted_pages) - successful_pages}")

            return output_file
    else:
        print("❌ OCR processing failed")
        return None

# Usage example
if __name__ == "__main__":
    
    PDF_PATH = "hsc_bangla.pdf"  

    # Check if file exists
    if os.path.exists(PDF_PATH):
        result = process_pdf_with_mistral_ocr(PDF_PATH)
        if result:
            print(f"✅ Processing completed successfully! Output saved as: {result}")
    else:
        print(f"❌ PDF file not found: {PDF_PATH}")
        print("Please upload your PDF file and update the PDF_PATH variable")