import re
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import hashlib
from datetime import datetime

# [Your complete BengaliTextChunker class here - exactly as you wrote it]
import re
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import hashlib
from datetime import datetime

@dataclass
class Chunk:
    id: str
    text: str
    word_count: int
    metadata: Dict

class BengaliTextChunker:
    def __init__(self, min_chunk_size=200, max_chunk_size=400, overlap_words=50):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_words = overlap_words

        # Bengali and English sentence endings
        self.bengali_sentence_endings = ['।', '?', '!', ':', ';', '।।']
        self.english_sentence_endings = ['.', '?', '!', ':', ';']

        # Section patterns for Bengali textbook
        self.section_patterns = {
            'title': r'^# (.+)$',
            'main_heading': r'^## ([^#\n]+)$',
            'sub_heading': r'^### ([^#\n]+)$',
            'learning_outcomes': r'শিখনফল|Learning Outcomes',
            'pre_assessment': r'প্রাক-মুস্যায়ন|Pre-assessment',
            'vocabulary': r'শব্দার্থ ও টীকা|Vocabulary',
            'main_story': r'মূল গল্প|Main Story',
            'main_discussion': r'মূল আলোচ্য বিষয়',
            'author_intro': r'লেখক পরিচিতি|Author Introduction',
            'text_intro': r'পাঠ পরিচিতি|Text Introduction',
            'questions': r'প্রশ্ন|Questions|বহুনির্বাচনী',
            'creative_questions': r'সৃজনশীল প্রশ্ন|Creative Questions',
            'board_questions': r'বোর্ড পরীক্ষার প্রশ্ন|Board Questions',
            'university_questions': r'বিশ্ববিদ্যালয় ভর্তি পরীক্ষার প্রশ্ন',
            'practice': r'প্র্যাকটিস|Practice',
            'table_content': r'^\|.*\|$',
            'answer_section': r'সমাধান:|উত্তর:|Answer:'
        }

    def extract_pages(self, content: str) -> List[Dict]:
        """Extract content by pages"""
        pages = []
        page_pattern = r'## Page (\d+)\n\n(.*?)(?=## Page \d+|\Z)'
        matches = re.findall(page_pattern, content, re.DOTALL)

        for page_num, page_content in matches:
            pages.append({
                'page_number': int(page_num),
                'content': page_content.strip()
            })
        return pages

    def identify_section_type(self, text: str) -> str:
        """Identify the type of section based on content"""
        text_lower = text.lower()

        # Check for specific Bengali sections
        if re.search(r'শিখনফল', text):
            return 'learning_outcomes'
        elif re.search(r'প্রাক-মুস্যায়ন', text):
            return 'pre_assessment'
        elif re.search(r'শব্দার্থ ও টীকা', text):
            return 'vocabulary_notes'
        elif re.search(r'মূল আলোচ্য বিষয়', text):
            return 'main_discussion'
        elif re.search(r'মূল গল্প', text):
            return 'main_story'
        elif re.search(r'লেখক পরিচিতি', text):
            return 'author_introduction'
        elif re.search(r'পাঠ পরিচিতি', text):
            return 'text_introduction'
        elif re.search(r'সৃজনশীল প্রশ্ন', text):
            return 'creative_questions'
        elif re.search(r'বোর্ড পরীক্ষার প্রশ্ন', text):
            return 'board_questions'
        elif re.search(r'বিশ্ববিদ্যালয় ভর্তি', text):
            return 'university_questions'
        elif re.search(r'বহুনির্বাচনী', text):
            return 'mcq_questions'
        elif re.search(r'প্র্যাকটিস', text):
            return 'practice'
        elif re.search(r'সমাধান:|উত্তর:', text):
            return 'answer_section'
        elif re.search(r'^\|.*\|', text, re.MULTILINE):
            return 'table_content'
        elif re.search(r'^\d+।', text, re.MULTILINE):
            return 'numbered_questions'
        else:
            return 'general_content'

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences considering Bengali and English"""
        sentences = []

        # First, handle special cases like abbreviations
        text = re.sub(r'(\w+)\.(\s+[A-Z])', r'\1।\2', text)  # Convert some periods to Bengali periods

        # Split on sentence boundaries
        sentence_pattern = r'([।।?!:;.]+)'
        parts = re.split(sentence_pattern, text)

        current_sentence = ""
        for i, part in enumerate(parts):
            if part.strip():
                current_sentence += part
                # Check if this part is a sentence ending
                if any(ending in part for ending in self.bengali_sentence_endings + self.english_sentence_endings):
                    if current_sentence.strip():
                        sentences.append(current_sentence.strip())
                        current_sentence = ""

        # Add any remaining text
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        return [s for s in sentences if s.strip()]

    def count_words(self, text: str) -> int:
        """Count words in Bengali and English text"""
        # Remove markdown formatting and special characters
        clean_text = re.sub(r'[#*`\[\](){}|]', '', text)
        clean_text = re.sub(r'\$[^$]*\$', '', clean_text)  # Remove LaTeX

        # Split by whitespace and filter empty strings
        words = [word for word in clean_text.split() if word.strip()]
        return len(words)

    def create_overlap_text(self, sentences: List[str], max_words: int) -> str:
        """Create overlap text from the end of sentences"""
        overlap_text = ""
        word_count = 0

        for sentence in reversed(sentences):
            sentence_words = self.count_words(sentence)
            if word_count + sentence_words <= max_words:
                overlap_text = sentence + " " + overlap_text
                word_count += sentence_words
            else:
                break

        return overlap_text.strip()

    def create_semantic_chunks(self, text: str, metadata: Dict) -> List[Chunk]:
        """Create semantic chunks with proper overlap"""
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = ""
        current_words = 0
        chunk_id = 1

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_words = self.count_words(sentence)

            # If adding this sentence exceeds max limit and we have enough content
            if current_words + sentence_words > self.max_chunk_size and current_words >= self.min_chunk_size:
                # Create chunk
                chunk = Chunk(
                    id=f"page_{metadata.get('page_number', 'unknown')}_chunk_{chunk_id}",
                    text=current_chunk.strip(),
                    word_count=current_words,
                    metadata={
                        **metadata,
                        'chunk_id': chunk_id,
                        'start_sentence_index': max(0, i - len(self.split_into_sentences(current_chunk))),
                        'end_sentence_index': i,
                        'has_overlap': chunk_id > 1
                    }
                )
                chunks.append(chunk)
                chunk_id += 1

                # Create overlap for next chunk
                chunk_sentences = self.split_into_sentences(current_chunk)
                overlap_text = self.create_overlap_text(chunk_sentences, self.overlap_words)

                current_chunk = overlap_text
                current_words = self.count_words(overlap_text)

            # Add current sentence to chunk
            current_chunk += (" " if current_chunk else "") + sentence
            current_words += sentence_words
            i += 1

        # Handle remaining content
        if current_chunk.strip() and current_words >= 50:  # Minimum viable chunk
            chunk = Chunk(
                id=f"page_{metadata.get('page_number', 'unknown')}_chunk_{chunk_id}",
                text=current_chunk.strip(),
                word_count=current_words,
                metadata={
                    **metadata,
                    'chunk_id': chunk_id,
                    'start_sentence_index': max(0, len(sentences) - len(self.split_into_sentences(current_chunk))),
                    'end_sentence_index': len(sentences),
                    'has_overlap': chunk_id > 1,
                    'is_final_chunk': True
                }
            )
            chunks.append(chunk)

        return chunks

    def split_page_into_sections(self, page_content: str) -> List[Dict]:
        """Split page content into semantic sections"""
        sections = []

        # Split by major headings and section breaks
        section_breaks = [
            r'^# [^#].*$',
            r'^## [^#].*$',
            r'^### [^#].*$',
            r'---',
            r'^\| .* \|$'  # Table headers
        ]

        # Split by double newlines first (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', page_content)

        current_section = ""
        current_section_type = "general_content"

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Check if this starts a new major section
            is_new_section = False
            for pattern in section_breaks:
                if re.match(pattern, paragraph, re.MULTILINE):
                    is_new_section = True
                    break

            if is_new_section and current_section.strip():
                # Save current section
                sections.append({
                    'content': current_section.strip(),
                    'section_type': current_section_type
                })
                current_section = ""

            # Update section type
            new_section_type = self.identify_section_type(paragraph)
            if new_section_type != 'general_content':
                current_section_type = new_section_type

            current_section += paragraph + "\n\n"

        # Add final section
        if current_section.strip():
            sections.append({
                'content': current_section.strip(),
                'section_type': current_section_type
            })

        return sections

    def process_content(self, content: str) -> List[Chunk]:
        """Main processing function"""
        all_chunks = []

        # Extract pages
        pages = self.extract_pages(content)

        for page in pages:
            page_content = page['content']
            page_number = page['page_number']

            # Split page into sections
            sections = self.split_page_into_sections(page_content)

            for section_idx, section in enumerate(sections):
                if not section['content'].strip():
                    continue

                metadata = {
                    'page_number': page_number,
                    'section_type': section['section_type'],
                    'section_index': section_idx,
                    'source': 'HSC26_Bangla_1st_paper_Aparichita',
                    'book_title': 'অপরিচিতা',
                    'author': 'রবীন্দ্রনাথ ঠাকুর',
                    'processing_date': datetime.now().isoformat()
                }

                # Create chunks for this section
                section_chunks = self.create_semantic_chunks(section['content'], metadata)
                all_chunks.extend(section_chunks)

        return all_chunks

    def save_chunks_to_markdown(self, chunks: List[Chunk], output_file: str):
        """Save chunks to markdown file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Chunked Content - HSC26 Bangla 1st Paper (অপরিচিতা)\n\n")
            f.write(f"**Total chunks:** {len(chunks)}\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Source:** HSC26 Bengali Textbook - Aparichita by Rabindranath Tagore\n\n")
            f.write("---\n\n")

            current_page = None

            for chunk in chunks:
                page_num = chunk.metadata.get('page_number', 'unknown')

                # Add page separator
                if current_page != page_num:
                    if current_page is not None:
                        f.write("\n---\n\n")
                    f.write(f"## PAGE {page_num} CHUNKS\n\n")
                    current_page = page_num

                # Write chunk
                f.write(f"### Chunk {chunk.id}\n\n")

                # Metadata table
                f.write("**Metadata:**\n\n")
                f.write("| Attribute | Value |\n")
                f.write("|-----------|-------|\n")
                f.write(f"| Chunk ID | `{chunk.id}` |\n")
                f.write(f"| Page Number | {chunk.metadata.get('page_number', 'N/A')} |\n")
                f.write(f"| Section Type | {chunk.metadata.get('section_type', 'N/A')} |\n")
                f.write(f"| Word Count | {chunk.word_count} |\n")
                f.write(f"| Has Overlap | {chunk.metadata.get('has_overlap', False)} |\n")
                f.write(f"| Section Index | {chunk.metadata.get('section_index', 'N/A')} |\n")

                if chunk.metadata.get('is_final_chunk'):
                    f.write(f"| Final Chunk | ✓ |\n")

                f.write("\n**Content:**\n\n")
                f.write(f"{chunk.text}\n\n")
                f.write("---\n\n")

    def save_chunks_to_json(self, chunks: List[Chunk], output_file: str):
        """Save chunks to JSON file for vector database"""
        chunk_data = []
        for chunk in chunks:
            chunk_data.append({
                'id': chunk.id,
                'text': chunk.text,
                'word_count': chunk.word_count,
                'metadata': chunk.metadata,
                'text_hash': hashlib.md5(chunk.text.encode('utf-8')).hexdigest()
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)

    def generate_statistics(self, chunks: List[Chunk]) -> Dict:
        """Generate statistics about the chunks"""
        stats = {
            'total_chunks': len(chunks),
            'total_words': sum(c.word_count for c in chunks),
            'avg_chunk_size': sum(c.word_count for c in chunks) / len(chunks) if chunks else 0,
            'min_chunk_size': min(c.word_count for c in chunks) if chunks else 0,
            'max_chunk_size': max(c.word_count for c in chunks) if chunks else 0,
            'pages_processed': len(set(c.metadata.get('page_number') for c in chunks)),
            'section_types': {}
        }

        # Count section types
        for chunk in chunks:
            section_type = chunk.metadata.get('section_type', 'unknown')
            stats['section_types'][section_type] = stats['section_types'].get(section_type, 0) + 1

        return stats
# Usage example
def main():
    # Read your OCR content
    try:
        with open('bengali_book_ocr.md', 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print("Please save your OCR content as 'bengali_book_ocr.md'")
        return

    # Initialize chunker
    chunker = BengaliTextChunker(
        min_chunk_size=400,
        max_chunk_size=200,
        overlap_words=50
    )

    print("Processing Bengali textbook content...")

    # Process content
    chunks = chunker.process_content(content)

    # Generate statistics
    stats = chunker.generate_statistics(chunks)

    # Save results
    chunker.save_chunks_to_markdown(chunks, 'bengali_chunked_content.md')
    chunker.save_chunks_to_json(chunks, 'bengali_chunked_content.json')

    # Print statistics
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"✓ Created {stats['total_chunks']} chunks")
    print(f"✓ Processed {stats['pages_processed']} pages")
    print(f"✓ Total words: {stats['total_words']:,}")
    print(f"✓ Average chunk size: {stats['avg_chunk_size']:.1f} words")
    print(f"✓ Size range: {stats['min_chunk_size']}-{stats['max_chunk_size']} words")

    print(f"\n=== SECTION BREAKDOWN ===")
    for section_type, count in stats['section_types'].items():
        print(f"✓ {section_type}: {count} chunks")

    print(f"\n=== OUTPUT FILES ===")
    print(f"✓ Markdown output: bengali_chunked_content.md")
    print(f"✓ JSON output: bengali_chunked_content.json")

    # Show sample chunks
    print(f"\n=== SAMPLE CHUNKS ===")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Sample Chunk {i+1} ---")
        print(f"ID: {chunk.id}")
        print(f"Page: {chunk.metadata.get('page_number')}")
        print(f"Section: {chunk.metadata.get('section_type')}")
        print(f"Words: {chunk.word_count}")
        print(f"Preview: {chunk.text[:150]}...")


if __name__ == "__main__":
    main()