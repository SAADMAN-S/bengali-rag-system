from typing import List, Dict
import time

class BengaliRAGEvaluator:
    def __init__(self, rag_system):
        self.rag_system = rag_system
    
    def evaluate_answer_correctness(self, expected: str, actual: str) -> bool:
        """Use the EXACT same logic as your batch_test method"""
        # This matches your exact logic from batch_test
        answer_lower = actual.lower().strip()
        expected_lower = expected.lower().strip()
        is_correct = expected_lower in answer_lower
        
        return is_correct
    
    def comprehensive_evaluation(self, test_cases: List[Dict]) -> Dict:
        """Run evaluation with the same logic as your batch_test"""
        print("🔍 Starting RAG Evaluation (Using Your Batch Test Logic)...")
        
        results = {
            'total_cases': len(test_cases),
            'accuracy': 0.0,
            'correct_count': 0,
            'avg_confidence': 0.0,
            'detailed_results': []
        }
        
        correct_count = 0
        confidence_scores = []
        total_tokens = 0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n=== Evaluation Test {i}/{len(test_cases)} ===")
            
            question = test_case['question']
            expected = test_case.get('expected', '')
            
            print(f"❓ Question: {question}")
            print(f"🎯 Expected: {expected}")
            
            # Get RAG result using YOUR method
            rag_result = self.rag_system.answer_question_with_memory(question)
            actual_answer = rag_result['answer']
            
            print(f"🤖 Actual: {actual_answer}")
            
            # Use YOUR exact correctness logic
            is_correct = self.evaluate_answer_correctness(expected, actual_answer)
            
            if is_correct:
                correct_count += 1
                status = "✅ CORRECT"
            else:
                status = "❌ INCORRECT"
            
            # Get metrics
            confidence = rag_result.get('confidence', 0.0)
            confidence_scores.append(confidence)
            total_tokens += rag_result.get('tokens_used', 0)
            
            print(f"📊 Status: {status}")
            print(f"🔍 Confidence: {confidence}")
            print(f"📑 Chunks: {rag_result.get('chunks_retrieved', 0)}")
            print("-" * 50)
            
            # Store detailed result
            detailed_result = {
                'test_case_id': i,
                'question': question,
                'expected': expected,
                'actual': actual_answer,
                'correct': is_correct,
                'confidence': confidence,
                'chunks_retrieved': rag_result.get('chunks_retrieved', 0),
                'tokens_used': rag_result.get('tokens_used', 0)
            }
            results['detailed_results'].append(detailed_result)
            
            time.sleep(0.5)  # Same rate limiting as your code
        
        # Calculate final metrics (same as your batch_test)
        accuracy = (correct_count / len(test_cases)) * 100
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        results['accuracy'] = accuracy
        results['correct_count'] = correct_count
        results['avg_confidence'] = avg_confidence
        results['total_tokens'] = total_tokens
        
        # Print summary (same format as your batch_test)
        print(f"\n📊 EVALUATION SUMMARY:")
        print(f"🎯 Accuracy: {accuracy:.1f}% ({correct_count}/{len(test_cases)})")
        print(f"🔍 Average Confidence: {avg_confidence:.3f}")
        print(f"⚡ Total Tokens: {total_tokens:,}")
        
        return results
    
    def compare_with_batch_test(self, test_cases: List[Dict]):
        """Compare evaluation results with your batch_test method"""
        print("🔄 Running Both Methods for Comparison...")
        
        # Run your batch_test method
        print("\n--- YOUR BATCH_TEST METHOD ---")
        batch_results = self.rag_system.batch_test(test_cases)
        
        # Run our evaluation method
        print("\n--- EVALUATION METHOD ---")
        eval_results = self.comprehensive_evaluation(test_cases)
        
        # Compare results
        print(f"\n📊 COMPARISON:")
        print(f"Batch Test Accuracy: {batch_results['accuracy']:.1f}%")
        print(f"Evaluation Accuracy: {eval_results['accuracy']:.1f}%")
        
        return batch_results, eval_results


# Enhanced evaluation with additional metrics (but keeping your core logic)
class EnhancedBengaliRAGEvaluator(BengaliRAGEvaluator):
    def __init__(self, rag_system):
        super().__init__(rag_system)
        # Only import this if needed for additional metrics
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            self.eval_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            self.has_semantic_eval = True
        except ImportError:
            print("⚠️ Sentence transformers not available - using basic evaluation only")
            self.has_semantic_eval = False
    
    def evaluate_groundedness(self, question: str, answer: str, context_chunks: List[Dict]) -> float:
        """Optional: Check if answer is supported by retrieved context"""
        if not self.has_semantic_eval or not context_chunks or not answer.strip():
            return 0.0
        
        try:
            # Combine all context
            combined_context = " ".join([chunk['text'] for chunk in context_chunks])
            
            # Create embeddings
            answer_embedding = self.eval_model.encode([answer], normalize_embeddings=True)
            context_embedding = self.eval_model.encode([combined_context], normalize_embeddings=True)
            
            # Calculate semantic similarity
            similarity = cosine_similarity(answer_embedding, context_embedding)[0][0]
            return float(similarity)
            
        except Exception as e:
            return 0.0
    
    def evaluate_relevance(self, question: str, context_chunks: List[Dict]) -> float:
        """Optional: Check if retrieved documents are relevant to question"""
        if not self.has_semantic_eval or not context_chunks or not question.strip():
            return 0.0
        
        try:
            import numpy as np
            question_embedding = self.eval_model.encode([question], normalize_embeddings=True)
            
            relevance_scores = []
            for chunk in context_chunks:
                chunk_embedding = self.eval_model.encode([chunk['text']], normalize_embeddings=True)
                similarity = cosine_similarity(question_embedding, chunk_embedding)[0][0]
                relevance_scores.append(similarity)
            
            return float(np.mean(relevance_scores))
            
        except Exception as e:
            return 0.0
    
    def comprehensive_evaluation_with_bonus_metrics(self, test_cases: List[Dict]) -> Dict:
        """Enhanced evaluation with bonus metrics but keeping your core accuracy logic"""
        print("🔍 Starting Enhanced RAG Evaluation...")
        
        results = {
            'total_cases': len(test_cases),
            'accuracy': 0.0,
            'correct_count': 0,
            'avg_confidence': 0.0,
            'avg_groundedness': 0.0,
            'avg_relevance': 0.0,
            'detailed_results': []
        }
        
        correct_count = 0
        confidence_scores = []
        groundedness_scores = []
        relevance_scores = []
        total_tokens = 0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n=== Enhanced Test {i}/{len(test_cases)} ===")
            
            question = test_case['question']
            expected = test_case.get('expected', '')
            
            print(f"❓ Question: {question}")
            print(f"🎯 Expected: {expected}")
            
            # Get RAG result
            rag_result = self.rag_system.answer_question_with_memory(question)
            actual_answer = rag_result['answer']
            context_chunks = rag_result.get('context_chunks', [])
            
            print(f"🤖 Actual: {actual_answer}")
            
            # Core accuracy 
            is_correct = self.evaluate_answer_correctness(expected, actual_answer)
            if is_correct:
                correct_count += 1
            
            # Additional metrics (BONUS)
            confidence = rag_result.get('confidence', 0.0)
            confidence_scores.append(confidence)
            
            groundedness = self.evaluate_groundedness(question, actual_answer, context_chunks)
            groundedness_scores.append(groundedness)
            
            relevance = self.evaluate_relevance(question, context_chunks)
            relevance_scores.append(relevance)
            
            total_tokens += rag_result.get('tokens_used', 0)
            
            print(f"✅ Correct: {'Yes' if is_correct else 'No'}")
            print(f"🔍 Confidence: {confidence:.3f}")
            if self.has_semantic_eval:
                print(f"🏠 Groundedness: {groundedness:.3f}")
                print(f"🎯 Relevance: {relevance:.3f}")
            print("-" * 50)
            
            # Store detailed result
            detailed_result = {
                'test_case_id': i,
                'question': question,
                'expected': expected,
                'actual': actual_answer,
                'correct': is_correct,
                'confidence': confidence,
                'groundedness': groundedness,
                'relevance': relevance,
                'chunks_retrieved': rag_result.get('chunks_retrieved', 0)
            }
            results['detailed_results'].append(detailed_result)
            
            time.sleep(0.5)
        
        # Calculate metrics
        import numpy as np
        results['accuracy'] = (correct_count / len(test_cases)) * 100
        results['correct_count'] = correct_count
        results['avg_confidence'] = np.mean(confidence_scores)
        results['avg_groundedness'] = np.mean(groundedness_scores)
        results['avg_relevance'] = np.mean(relevance_scores)
        results['total_tokens'] = total_tokens
        
        # Print enhanced summary
        print(f"\n📊 ENHANCED EVALUATION RESULTS:")
        print(f"🎯 Accuracy: {results['accuracy']:.1f}% ({correct_count}/{len(test_cases)})")
        print(f"🔍 Average Confidence: {results['avg_confidence']:.3f}")
        if self.has_semantic_eval:
            print(f"🏠 Average Groundedness: {results['avg_groundedness']:.3f}")
            print(f"🎯 Average Relevance: {results['avg_relevance']:.3f}")
        print(f"⚡ Total Tokens: {total_tokens:,}")
        
        return results


# Usage functions
def run_basic_evaluation(rag_system):
    """Run basic evaluation matching your batch_test logic"""
    test_cases = [
        {"question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?", "expected": "শম্ভুনাথ"},
        {"question": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?", "expected": "মামাকে"},
        {"question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?", "expected": "১৫ বছর"},
        {"question": "অনুপমের বন্ধুর নাম কী?", "expected": "হরিশ"},
        {"question": "কল্যাণীর বাবার নাম কী?", "expected": "শম্ভুনাথ সেন"}
    ]
    
    evaluator = BengaliRAGEvaluator(rag_system)
    return evaluator.compare_with_batch_test(test_cases)

def run_enhanced_evaluation(rag_system):
    """Run enhanced evaluation with bonus metrics"""
    test_cases = [
        {"question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?", "expected": "শম্ভুনাথ"},
        {"question": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?", "expected": "মামাকে"},
        {"question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?", "expected": "১৫ বছর"},
        {"question": "অনুপমের বন্ধুর নাম কী?", "expected": "হরিশ"},
        {"question": "কল্যাণীর বাবার নাম কী?", "expected": "শম্ভুনাথ সেন"}
    ]
    
    evaluator = EnhancedBengaliRAGEvaluator(rag_system)
    return evaluator.comprehensive_evaluation_with_bonus_metrics(test_cases)

# Test with your existing RAG system
print("🎬 Running Evaluation That Matches Your Logic...")
if 'rag_system' in globals():
    # Run basic evaluation first
    batch_results, eval_results = run_basic_evaluation(rag_system)
    
    print("\n" + "="*60)
    print("🎉 EVALUATION COMPLETE!")
    print("✅ This evaluation uses your EXACT same logic")
    print("✅ Results should match your batch_test method")
    print("="*60)
    
    # Optionally run enhanced evaluation
    print("\n🚀 Running Enhanced Evaluation with Bonus Metrics...")
    enhanced_results = run_enhanced_evaluation(rag_system)
    
else:
    print(" Run your main_simple_rag() function first to initialize rag_system!")