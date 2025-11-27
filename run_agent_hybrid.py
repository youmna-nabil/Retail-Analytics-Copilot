import json
import dspy
import argparse

from agent.graph_hybrid import HybridRetailAgent
from agent.dspy_signatures import RouterModule, NLToSQLModule, SynthesizerModule

from assets.settings import settings
from assets.config import config


def setup_dspy():
    lm = dspy.LM(model=f"ollama_chat/{settings.MODEL}", api_base=config.BASE_URL, max_tokens=settings.MaxTokens,  temperature=settings.Temperature)
    dspy.settings.configure(lm=lm)


def main():
   
    
    batch_file = config.BatchFile
    output_file = config.OutputFile
    
    print(f"Input: {batch_file}")
    print(f"Output: {output_file}")
    
    # Setup DSPy
    setup_dspy()
    
    # Initialize modules
    router = RouterModule()
    nl_to_sql = NLToSQLModule()
    synthesizer = SynthesizerModule()
    
    # Create agent
    agent = HybridRetailAgent(router, nl_to_sql, synthesizer)
    
    # Load questions
    with open(batch_file, 'r') as f:
        content = f.read()
    
    decoder = json.JSONDecoder()
    pos = 0
    questions = []
    while pos < len(content):
        if content[pos].isspace():
            pos += 1
            continue
        try:
            obj, end = decoder.raw_decode(content, pos)
            questions.append(obj)
            pos += end
        except json.JSONDecodeError:
            break  # Stop if no more valid JSON
    
    print(f"\n[INFO] Loaded {len(questions)} questions\n")
    
    # Process each question
    results = []
    
    for idx, q in enumerate(questions):
        question_id = q["id"]
        question_text = q["question"]
        format_hint = q.get("format_hint", "str")
        
        print(f"\n{'=' * 70}")
        print(f"[{idx+1}/{len(questions)}] Processing: {question_id}")
        print(f"{'=' * 70}")
        print(f"Q: {question_text}")
        print(f"Format: {format_hint}")
        
        try:
            # Run agent
            result = agent.run(question_text, format_hint)
            
            # Format output
            output = {
                "id": question_id,
                "final_answer": result["final_answer"],
                "sql": result["sql"],
                "confidence": round(result["confidence"], 2),
                "explanation": result["explanation"],
                "citations": result["citations"]
            }
            
            print(f"\n[SUCCESS] Answer: {result['final_answer']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Citations: {', '.join(result['citations'][:5])}{'...' if len(result['citations']) > 5 else ''}")
            
        except Exception as e:
            print(f"\n[ERROR] {str(e)}")
            output = {
                "id": question_id,
                "final_answer": None,
                "sql": "",
                "confidence": 0.0,
                "explanation": f"Error: {str(e)}",
                "citations": []
            }
        
        results.append(output)
    
    # Write results
    print(f"\n{'=' * 70}")
    print(f"Writing results to {output_file}...")
    print(f"{'=' * 70}")
    
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\n✓ Done! Results written to {output_file}")
    print(f"Processed {len(results)} questions")
    
    # Cleanup
    agent.close()


if __name__ == "__main__":
    main()