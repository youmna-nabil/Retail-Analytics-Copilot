import json
import dspy
import argparse

from agent.graph_hybrid import HybridRetailAgent
from agent.dspy_signatures import RouterModule, NLToSQLModule, SynthesizerModule

from assets.settings import settings
from assets.config import config


def setup_dspy():
    lm = dspy.LM(model=f"ollama/{settings.MODEL}", api_base=config.BASE_URL, max_tokens=settings.MaxTokens,  temperature=settings.Temperature)
    dspy.settings.configure(lm=lm)


def main():
   
    
    batch_file = config.BatchFile
    output_file = config.OutputFile
    
    # Setup DSPy
    setup_dspy()
    
    # Initialize modules
    router = RouterModule()
    nl_to_sql = NLToSQLModule()
    synthesizer = SynthesizerModule()
    
    # Create agent
    agent = HybridRetailAgent(router, nl_to_sql, synthesizer)
    
    questions = []
    with open(batch_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                questions.append(obj)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
                continue
    
    print(f"\n Loaded {len(questions)} questions\n")
    
    # Process each question
    results = []
    
    for idx, q in enumerate(questions):
        question_id = q["id"]
        question_text = q["question"]
        format_hint = q.get("format_hint", "str")
        
        print(f" Processing: {question_id}")

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
    
    # Write results to output file    
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Processed {len(results)} questions")
    
    # Cleanup
    agent.close()


if __name__ == "__main__":
    main()