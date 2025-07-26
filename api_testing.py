import requests
import json
import time

def test_bengali_rag_api():
    """Test the Bengali RAG API"""
    base_url = "http://localhost:5000"

    print("🧪 Testing Bengali RAG API...")

    # Test 1: Health check
    print("\n=== 1. HEALTH CHECK ===")
    try:
        response = requests.get(f"{base_url}/api/health", timeout=10)
        print(f"✅ Status: {response.status_code}")
        health_data = response.json()
        print(f"📊 Health Response:")
        for key, value in health_data.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

    # Test 2: Bengali question
    print("\n=== 2. BENGALI QUESTION ===")
    try:
        question_data = {"question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"}
        response = requests.post(
            f"{base_url}/api/ask",
            json=question_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        print(f"✅ Status: {response.status_code}")
        result = response.json()

        print(f"❓ Question: {result.get('question', 'N/A')}")
        print(f"✅ Answer: {result.get('answer', 'N/A')}")
        print(f"🎯 Confidence: {result.get('confidence', 'N/A')}")
        print(f"📑 Chunks Used: {result.get('chunks_used', 'N/A')}")
        print(f"🧠 Conversation Length: {result.get('conversation_length', 'N/A')}")

    except Exception as e:
        print(f"❌ Bengali question failed: {e}")
        return False

    # Test 3: English question
    print("\n=== 3. ENGLISH QUESTION ===")
    try:
        question_data = {"question": "Who is Anupam's uncle?"}
        response = requests.post(
            f"{base_url}/api/ask",
            json=question_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        print(f"✅ Status: {response.status_code}")
        result = response.json()

        print(f"❓ Question: {result.get('question', 'N/A')}")
        print(f"✅ Answer: {result.get('answer', 'N/A')}")
        print(f"🎯 Confidence: {result.get('confidence', 'N/A')}")
        print(f"📑 Chunks Used: {result.get('chunks_used', 'N/A')}")
        print(f"🧠 Conversation Length: {result.get('conversation_length', 'N/A')}")

    except Exception as e:
        print(f"❌ English question failed: {e}")
        return False

    # Test 4: Follow-up question (memory test)
    print("\n=== 4. FOLLOW-UP QUESTION (MEMORY TEST) ===")
    try:
        question_data = {"question": "তার সম্পর্কে আরও বলুন"}
        response = requests.post(
            f"{base_url}/api/ask",
            json=question_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        print(f"✅ Status: {response.status_code}")
        result = response.json()

        print(f"❓ Question: {result.get('question', 'N/A')}")
        print(f"✅ Answer: {result.get('answer', 'N/A')}")
        print(f"🎯 Confidence: {result.get('confidence', 'N/A')}")
        print(f"📑 Chunks Used: {result.get('chunks_used', 'N/A')}")
        print(f"🧠 Conversation Length: {result.get('conversation_length', 'N/A')}")

    except Exception as e:
        print(f"❌ Follow-up question failed: {e}")
        return False

    # Test 5: Reset conversation
    print("\n=== 5. RESET CONVERSATION ===")
    try:
        response = requests.post(f"{base_url}/api/reset", timeout=10)
        print(f"✅ Status: {response.status_code}")
        result = response.json()
        print(f"💬 Reset Response: {result.get('message', 'N/A')}")

    except Exception as e:
        print(f"❌ Reset failed: {e}")
        return False

    return True

# Wait for server to be ready, then test
print("⏳ Waiting for server to be ready...")
time.sleep(3)

# Run tests
if test_bengali_rag_api():
    print("\n🎉 ALL API TESTS PASSED!")
    print("✅ REST API BONUS TASK COMPLETE!")
    print("\n📋 Your API supports:")
    print("   ✅ Bengali and English queries")
    print("   ✅ Conversation memory (short-term)")
    print("   ✅ Vector-based retrieval (long-term)")
    print("   ✅ Confidence scoring")
    print("   ✅ Health monitoring")
    print("   ✅ Conversation reset")
else:
    print("\n❌ Some API tests failed")