"""
from assets.config import config
def check_ollama_connection():
    import requests
    try:
        response = requests.get(config.BASE_URL, timeout=2)
        if response.status_code == 200:
            print(f"✓ Ollama server is running at {config.BASE_URL}")
            return True
    except:
        pass
    
    print(f"⚠ WARNING: Cannot connect to Ollama at {config.BASE_URL}")
    print("The agent will use fallback methods (keyword-based classification)")
    print("\nTo fix this:")
    print("1. Install Ollama from https://ollama.ai")
    print("2. Run 'ollama serve' in a terminal")
    print("3. Run 'ollama pull phi3:3.8b'\n")
    return False

def main():
    check_ollama_connection()

if __name__ == "__main__":
    main()
"""
from agent.dspy_signatures import RouterModule


def test_router():
    router = RouterModule()
    
    test_cases = [
        # RAG cases
        ("According to the product policy, what is the return window for Beverages?", "rag"),
        ("What is the return window days for unopened Beverages per the policy?", "rag"),
        ("Using the KPI definition, what is AOV?", "rag"),  # Just asking for definition
        
        # SQL cases
        ("Top 3 products by total revenue all-time", "sql"),
        ("List all customers with highest orders", "sql"),
        ("How many total orders in the database?", "sql"),
        
        # Hybrid cases
        ("During Summer Beverages 2016, which category had highest quantity?", "hybrid"),
        ("Using the AOV definition, what was the average order value during Winter 2016?", "hybrid"),
        ("What was the total revenue for Beverages during Summer campaign?", "hybrid"),
    ]
    
    print("\n" + "="*70)
    print("ROUTER CLASSIFICATION TESTS")
    print("="*70)
    
    passed = 0
    failed = 0
    
    for question, expected in test_cases:
        result = router._fallback_classify(question)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        print(f"\n{status} Question: {question[:60]}...")
        print(f"  Expected: {expected}")
        print(f"  Got: {result}")
    
    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70)

if __name__ == "__main__":
    test_router()