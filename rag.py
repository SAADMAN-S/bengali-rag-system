# Import required libraries
from openai import OpenAI
import json
from typing import List, Dict, Any, Optional
import time

class SimpleBengaliRAGSystem:
    def __init__(self, vector_store, openai_api_key: str, model_name: str = "gpt-4.1"):
        """
        Initialize simplified RAG system with fixed threshold

        Args:
            vector_store: Your BengaliVectorStore instance
            openai_api_key: OpenAI API key
            model_name: OpenAI model to use
        """
        self.vector_store = vector_store
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.model_name = model_name

        # Fixed similarity threshold
        self.similarity_threshold = 0.5

        # System prompt
        self.system_prompt = """You are an expert Bengali literature scholar specializing in Rabindranath Tagore's "অপরিচিতা" (Aparichita).

INSTRUCTIONS:
1. Answer in the same language as the question (Bengali for Bengali, English for English)
2. Give direct, factual answers based ONLY on the provided context
3. Keep answers brief and to the point
4. For names: give just the name
5. For ages/numbers: give just the number with unit

You are analyzing the Bengali story "অপরিচিতা" with focus on characters like অনুপম, কল্যাণী, মামা, শম্ভুনাথ সেন, etc."""

    def retrieve_context_with_threshold(self, query: str, max_chunks: int = 10) -> List[Dict]:
        """
        Retrieve context using fixed similarity threshold

        Args:
            query: User question
            max_chunks: Maximum number of chunks to retrieve

        Returns:
            List of relevant chunks above threshold
        """
        print(f"🎯 Using similarity threshold: {self.similarity_threshold}")

        try:
            # Get initial results
            initial_results = self.vector_store.search_similar_chunks(query, top_k=max_chunks)

            # Filter by threshold
            filtered_results = [
                chunk for chunk in initial_results
                if chunk['score'] >= self.similarity_threshold
            ]

            print(f"📊 Retrieved {len(filtered_results)} chunks above threshold")
            return filtered_results

        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []

    def prepare_context(self, retrieved_chunks: List[Dict], max_length: int = 2500) -> str:
        """
        Prepare context from retrieved chunks

        Args:
            retrieved_chunks: Retrieved chunks
            max_length: Maximum context length

        Returns:
            Formatted context string
        """
        if not retrieved_chunks:
            return "No relevant context found."

        # Sort by similarity score
        sorted_chunks = sorted(retrieved_chunks, key=lambda x: x['score'], reverse=True)

        context_parts = []
        total_length = 0

        for i, chunk in enumerate(sorted_chunks):
            chunk_text = f"[Context {i+1} - Page {chunk['page_number']}]\n{chunk['text']}\n"

            if total_length + len(chunk_text) > max_length:
                break

            context_parts.append(chunk_text)
            total_length += len(chunk_text)

        return "\n".join(context_parts)

    def generate_answer(self, query: str, context: str) -> Dict[str, Any]:
        """
        Generate answer using OpenAI

        Args:
            query: User question
            context: Retrieved context

        Returns:
            Dictionary with answer and metadata
        """
        try:
            user_prompt = f"""Context from "অপরিচিতা":

{context}

Question: {query}

Answer based on the context:"""

            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=150,
                top_p=0.9
            )

            answer = response.choices[0].message.content.strip()

            return {
                "answer": answer,
                "model_used": self.model_name,
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0
            }

        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                "answer": "দুঃখিত, উত্তর তৈরিতে সমস্যা হয়েছে।",
                "model_used": self.model_name,
                "tokens_used": 0,
                "error": str(e)
            }

    def answer_question(self, query: str, show_context: bool = False) -> Dict[str, Any]:
        """
        Complete RAG pipeline

        Args:
            query: User question
            show_context: Whether to show context

        Returns:
            Result dictionary
        """
        print(f"🔍 Processing: {query}")

        # Step 1: Retrieve context
        retrieved_chunks = self.retrieve_context_with_threshold(query)
        print(retrieved_chunks)

        if not retrieved_chunks:
            return {
                "query": query,
                "answer": "প্রাসঙ্গিক তথ্য পাওয়া যায়নি।",
                "chunks_retrieved": 0,
                "confidence": 0.0,
                "threshold_used": self.similarity_threshold
            }

        # Step 2: Prepare context
        context_text = self.prepare_context(retrieved_chunks)

        # Step 3: Generate answer
        print("🤖 Generating answer...")
        generation_result = self.generate_answer(query, context_text)

        # Calculate confidence
        max_similarity = max(chunk['score'] for chunk in retrieved_chunks) if retrieved_chunks else 0.0

        result = {
            "query": query,
            "answer": generation_result["answer"],
            "chunks_retrieved": len(retrieved_chunks),
            "confidence": round(max_similarity, 3),
            "threshold_used": self.similarity_threshold,
            "model_used": generation_result["model_used"],
            "tokens_used": generation_result.get("tokens_used", 0),
            "context_chunks": retrieved_chunks
        }

        # Show context if requested
        if show_context:
            print(f"\n📄 Context ({len(retrieved_chunks)} chunks):")
            for i, chunk in enumerate(retrieved_chunks, 1):
                print(f"\nChunk {i} (Score: {chunk['score']:.4f}):")
                print(f"Page: {chunk['page_number']}")
                print(f"Text: {chunk['text'][:200]}...")

        return result

    def batch_test(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """
        Test multiple questions

        Args:
            test_cases: List of {"question": str, "expected": str}

        Returns:
            Test results
        """
        results = []
        correct_count = 0
        total_tokens = 0

        print(f"🚀 Testing {len(test_cases)} questions...\n")

        for i, test_case in enumerate(test_cases, 1):
            question = test_case["question"]
            expected = test_case.get("expected", "")

            print(f"=== Test {i}/{len(test_cases)} ===")

            # Get answer
            result = self.answer_question(question, show_context=False)

            # Check if correct
            answer_lower = result['answer'].lower().strip()
            expected_lower = expected.lower().strip()
            is_correct = expected_lower in answer_lower

            if is_correct:
                correct_count += 1
                status = "✅ CORRECT"
            else:
                status = "❌ INCORRECT"

            total_tokens += result.get('tokens_used', 0)

            print(f"❓ Question: {question}")
            print(f"✅ Answer: {result['answer']}")
            print(f"🎯 Expected: {expected}")
            print(f"📊 Status: {status}")
            print(f"🔍 Confidence: {result['confidence']}")
            print(f"📑 Chunks: {result['chunks_retrieved']}")
            print("-" * 50)

            results.append({
                "question": question,
                "expected": expected,
                "actual": result['answer'],
                "correct": is_correct,
                "confidence": result['confidence'],
                "chunks_retrieved": result['chunks_retrieved']
            })

            time.sleep(0.5)  # Rate limiting

        # Summary
        accuracy = (correct_count / len(test_cases)) * 100
        avg_confidence = sum(r['confidence'] for r in results) / len(results)

        print(f"\n📊 SUMMARY:")
        print(f"🎯 Accuracy: {accuracy:.1f}% ({correct_count}/{len(test_cases)})")
        print(f"🔍 Average Confidence: {avg_confidence:.3f}")
        print(f"⚡ Total Tokens: {total_tokens:,}")

        return {
            "accuracy": accuracy,
            "correct": correct_count,
            "total": len(test_cases),
            "avg_confidence": avg_confidence,
            "total_tokens": total_tokens,
            "results": results
        }
class ConversationRAGSystem(SimpleBengaliRAGSystem):
    def __init__(self, vector_store, openai_api_key, model_name="gpt-4o-mini"):
        super().__init__(vector_store, openai_api_key, model_name)
        self.conversation_history = []
        self.max_history = 5  # Keep last 5 exchanges

    def add_to_history(self, question: str, answer: str):
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "timestamp": time.time()
        })
        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

    def get_conversation_context(self) -> str:
        if not self.conversation_history:
            return ""

        context = "Previous conversation:\n"
        for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
            context += f"Q: {exchange['question']}\nA: {exchange['answer']}\n\n"
        return context

    def answer_question_with_memory(self, query: str, show_context: bool = False):
        # Get conversation context
        conv_context = self.get_conversation_context()

        # Modify the system prompt to include conversation history
        enhanced_query = f"{conv_context}Current question: {query}" if conv_context else query

        # Get answer
        result = self.answer_question(enhanced_query, show_context)

        # Add to history
        self.add_to_history(query, result['answer'])

        return result

# Main function
def main_simple_rag():
    """Initialize simple RAG system"""

    # Configuration from your existing code
    PINECONE_API_KEY = "pcsk_2Jqw4c_A9sCqN2EKV3WY7k2KHuL9doLVzjt2PAS3NSkqB6YgToCid4wLpEvB5Vy2u8Yogq"
    INDEX_NAME = "bengali-rag-aparichita"
    OPENAI_API_KEY = "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"

    # Connect to existing vector store
    print("🔗 Connecting to Pinecone...")
    vector_store = BengaliVectorStore(
        pinecone_api_key=PINECONE_API_KEY,
        index_name=INDEX_NAME
    )

   # Initialize RAG system with conversation memory
    print("🚀 Initializing RAG System with Short-term Memory...")
    rag_system = ConversationRAGSystem(
      vector_store=vector_store,
      openai_api_key=OPENAI_API_KEY,
      model_name="gpt-4.1"
    )

    print(f"✅ RAG System Ready! Using threshold: {rag_system.similarity_threshold}")

    # Test cases
    test_cases = [
        {"question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?", "expected": "শম্ভুনাথ"},
        {"question": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?", "expected": "মামাকে"},
        {"question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?", "expected": "১৫ বছর"},
        {"question": "অনুপমের বন্ধুর নাম কী?", "expected": "হরিশ"},
        {"question": "কল্যাণীর বাবার নাম কী?", "expected": "শম্ভুনাথ সেন"}
    ]

    # Run tests
    test_results = rag_system.batch_test(test_cases)

    return rag_system, test_results


def ask_question_with_memory(rag_system, query: str):
    """Ask a question with conversation memory"""
    result = rag_system.answer_question_with_memory(query, show_context=True)

    print(f"\n{'='*50}")
    print(f"❓ Question: {query}")
    print(f"✅ Answer: {result['answer']}")
    print(f"🎯 Confidence: {result['confidence']}")
    print(f"📑 Chunks Used: {result['chunks_retrieved']}")
    print(f"🧠 History Length: {len(rag_system.conversation_history)}")
    print(f"{'='*50}")

    return result

def test_conversation_memory(rag_system):
    """Test conversation memory with follow-up questions"""
    print("\n🧠 Testing Conversation Memory...")

    # First question
    print("\n--- First Question ---")
    ask_question_with_memory(rag_system, "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?")

    # Follow-up question (should understand "তার" refers to the previous answer)
    print("\n--- Follow-up Question ---")
    ask_question_with_memory(rag_system, "তার সম্পর্কে আরও বলুন")

    # Another follow-up
    print("\n--- Another Follow-up ---")
    ask_question_with_memory(rag_system, "তিনি কোথায় থাকতেন?")

    print(f"\n🧠 Total conversation history: {len(rag_system.conversation_history)} exchanges")


# Run the system
if __name__ == "__main__":
    # Initialize the system
    rag_system, test_results = main_simple_rag()

    print(f"\n🎉 RAG System with Short-term Memory Ready!")

    # Test conversation memory
    test_conversation_memory(rag_system)

    print(f"\nUsage: ask_question_with_memory(rag_system, 'Your question here')")