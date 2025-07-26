# Bengali Multilingual RAG System

## HSC26 Bangladeshi Bengali 1st Paper

## 🎯 Project Overview

A sophisticated multilingual Retrieval-Augmented Generation (RAG) approach that can recognise and give correponding answers to queries in both **Bengali** and **English** particularly about Rabindranath Tagore's classic story "অপরিচিতা" (Aparichita) by processing the HSC26 Bangladeshi Bengali 1st Paper textbook.

## ✨ Features

- **🌏 Multilingual Support**: Handles queries in Bengali and English
- **📚 Knowledge Base**: HSC26 Bengali textbook processing
- **🧠 Dual Memory System**:
  - **Short-term**: Recent conversation history
  - **Long-term**: Vector database with semantic search
- **🔍 Advanced Chunking**: Bengali-aware text segmentation
- **⚡ REST API**: Complete Flask-based conversation API
- **📊 Evaluation Metrics**: Groundedness and relevance scoring

Pre-requisites:

- Programming Language: Python
- API Keys for: OpenAI, Pinecone, Mistral AI

Setup for installation and use:

- Download/Clone the repository
- cd bengali-rag-system
- Install requirement packages: pip install -r requirements.txt

Run the system step-by-step:

# Step 1: Extract text from PDF (if needed)

python ocr_processor.py

# Step 2: Create text chunks

python chunker.py

# Step 3: Setup vector database

python vector_store.py

# Step 4: Test RAG system

python rag_system.py

# Step 5: Run evaluation

python evaluator.py

# Step 6: Start API

python api.py

# Step 7: Test the api

python api_testing.py

CORE DEPENDENCY:

- openai 1.54.3+ FOR LLM for answer generation
- pinecone-client 5.0.0+ FOR Vector database storage
- sentence-transformers 3.2.0+ FOR Multilingual embeddings
- mistralai 1.9.3+ FOR OCR text extraction

SUPPORTING DEPENDENCIES:

- PyPDF2 FOR PDF processing
- flask, flask-cors FOR REST API framework
- transformers, torch FOR ML model support
- numpy, pandas FOR Data processing
- scikit-learn FOR Similarity calculations
- tqdm FOR Progress bars
- requests FOR HTTP requests

TOOLS USED:

- Vector Database: Pinecone (AWS Serverless)
- LLM Model: OpenAI GPT-4o-mini
- Embedding Model: Multilingual MiniLM-L12-v2
- OCR Engine: Mistral AI OCR API
- Development: Python, Jupyter Notebook, VSCode

SAMPLE QUERIES:

❓ Question: অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
✅ Answer: শম্ভুনাথ সেন
🎯 Expected: শম্ভুনাথ
📊 Status: ✅ CORRECT
🔍 Confidence: 0.843

❓ Question: কল্যাণীর বাবার নাম কী?
✅ Answer: শম্ভুনাথ সেন
🎯 Expected: শম্ভুনাথ সেন
📊 Status: ✅ CORRECT
🔍 Confidence: 0.63

❓ Question: Who is described as handsome by Anupam?
✅ Answer: Shombhunath
🎯 Expected: Shombhunath
📊 Status: ✅ CORRECT
🔍 Confidence: 0.843

API DOCUMENTATION:

- Base URL: http://localhost:5000
- End point:
  GET /api/health for health checking
  POST /api/ask for queries posting

EVALUATION OUTCOME:
🎯 Accuracy: 40.0% (2/5)
🔍 Average Confidence: 0.760
⚡ Total Tokens: 4,467

Q/A:

1. What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?

Ans: I have used Mistral AI OCR API for extracting text from the Bengali PDF because Mistral OCR constitutes excellent support for Bengali scripts, preserves text and structure. However, the formatting challenges that I faced was Mixed scripts as PDF contsined both Bengali and English texts. The structures of the tables added some complications in the extraction process and also maintaining appropriate line breaks i.e. accurate sentence system was another challenge.

2. What chunking strategy did you choose (e.g. paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?

Ans: I have used a chunking associated with sentence-boundary structure. The size of chunking is 200 to 400 words in each chunks and also there is an overlap of 50 words between consecutive chunks. This chunking strategy is aligned with Bengali sentence structures, and thus captures the meaning. The 50-word overlap ensures that there is continuation of contexts in between the chunks.

3. What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?

Ans: I have used sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2. This embedding was trained over 50+ different langugages including Bengali and this model also has the capability to capture semantic similarity. It can capture abstract literary concepts via which it can familiarize with teh meanings being conveyed. It maps Bengali "শম্ভুনাথ" close to English "Shombhunath" thus facilitating the able to deliver correct answer. It also tarcks faily relationships across languages.

4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?

Ans: I implemented a semantic search system using cosine similarity with Pinecone vector database to compare user queries with stored text chunks. When a user asks a question, the system first converts it into a 384-dimensional vector using the same embedding model used for the document chunks. Then, Pinecone calculates cosine similarity scores between the query vector and all stored chunk vectors, which measures semantic meaning rather than just word matching - this is crucial for handling Bengali literature where the same concept might be expressed differently or across languages.

5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?

Ans: To ensure meaningful comparison between user queries and document chunks, I implemented a multi-layered approach that handles both clear and ambiguous questions smoothly. The system uses a fixed similarity threshold of 0.5 to filter out weak matches, then combines multiple relevant chunks while preserving important metadata like page numbers and section types to maintain context. Most importantly, I built in conversation memory that tracks the last few question-answer exchanges, which becomes crucial when users ask vague follow-up questions like "তার সম্পর্কে বলুন" (tell me about him) - in these cases, the system looks back at recent conversation history to understand who "তার" (him) refers to based on previously mentioned characters. When queries are too vague or don't match well with stored content, the system responds gracefully by either returning "প্রাসঙ্গিক তথ্য পাওয়া যায়নি" (no relevant information found) for low similarity scores, using conversation context to resolve pronoun references, asking for clarification when needed, or providing partial information rather than completely failing, ensuring users always get some meaningful response even when their questions are incomplete or unclear.

6. Do the results seem relevant? If not, what might improve them (e.g. better chunking, better embedding model, larger document)?

Ans: The results do seem relevant and the chunking method that has been used, according to my knowledge seems to be the most suitable for this task. The separation of English and Bengali text, or a chunking method constituing a much more fluent flexible flow of contexts among chunks could have given better results.
