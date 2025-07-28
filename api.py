
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import time
import requests

app = Flask(__name__)
CORS(app)

# Global RAG system
rag_system = None

def initialize_rag_system():
    """Initialize the RAG system for API use"""
    global rag_system

    print("🔄 Initializing RAG system...")

    try:
        # Configuration
        PINECONE_API_KEY = "pcsk_2Jqw4c_A9sCqN2EKV3WY7k2KHuL9doLVzjt2PAS3NSkqB6YgToCid4wLpEvB5Vy2u8Yogq"
        INDEX_NAME = "bengali-rag-aparichita"
        OPENAI_API_KEY = "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"

        # Initialize vector store
        vector_store = BengaliVectorStore(
            pinecone_api_key=PINECONE_API_KEY,
            index_name=INDEX_NAME
        )

        # Initialize RAG system with conversation memory
        rag_system = ConversationRAGSystem(
            vector_store=vector_store,
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-4"
        )

        print("✅ RAG System initialized successfully!")
        return True

    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    global rag_system
    return jsonify({
        'status': 'healthy',
        'message': 'Bengali RAG API is running',
        'rag_initialized': rag_system is not None,
        'conversation_history': len(rag_system.conversation_history) if rag_system else 0
    })

@app.route('/api/ask', methods=['POST'])
def ask_question():
    global rag_system

    if rag_system is None:
        return jsonify({'error': 'RAG system not initialized', 'success': False}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided', 'success': False}), 400

        question = data.get('question', '')

        if not question:
            return jsonify({'error': 'Question is required', 'success': False}), 400

        print(f"📥 Received question: {question}")

        # Get answer with memory
        result = rag_system.answer_question_with_memory(question)

        response = {
            'question': question,
            'answer': result['answer'],
            'confidence': result['confidence'],
            'chunks_used': result['chunks_retrieved'],
            'conversation_length': len(rag_system.conversation_history),
            'success': True
        }

        print(f"📤 Sending response: {result['answer'][:100]}...")
        return jsonify(response)

    except Exception as e:
        print(f"❌ Error processing question: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    """Reset conversation history"""
    global rag_system
    if rag_system:
        rag_system.conversation_history = []
        return jsonify({'message': 'Conversation reset', 'success': True})
    else:
        return jsonify({'error': 'RAG system not initialized', 'success': False}), 500

def run_flask():
    """Run Flask in a separate thread"""
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# Initialize and start the API
def start_api():
    """Start the API server"""
    print("🚀 Starting Bengali RAG API...")

    # Initialize RAG system
    if not initialize_rag_system():
        print("❌ Failed to initialize RAG system")
        return None

    # Start Flask in background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Wait for server to start
    time.sleep(5)  # Increased wait time

    # Test if server is running
    try:
        response = requests.get('http://localhost:5000/api/health', timeout=10)
        if response.status_code == 200:
            print("✅ API is running successfully!")
            print("🌐 Local URL: http://localhost:5000")
            print("📋 Available endpoints:")
            print("   GET  /api/health - Health check")
            print("   POST /api/ask - Ask questions")
            print("   POST /api/reset - Reset conversation")
            return True
        else:
            print(f"❌ Server responded with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server not responding: {e}")
        return False

# Start the API
print("🎬 Starting API server...")
if start_api():
    print("\n🎉 API is ready for testing!")
else:
    print("\n❌ Failed to start API")