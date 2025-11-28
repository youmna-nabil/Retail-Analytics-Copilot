from assets.config import config
def check_ollama_connection():
    """Check if Ollama is running before proceeding"""
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