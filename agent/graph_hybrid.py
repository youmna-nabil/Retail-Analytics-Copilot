from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END
import json
import re

from agent.rag.retrieval import DocumentRetriever, Chunk
from agent.tools.sqlite_tool import SQLiteTool
from agent.dspy_signatures import RouterModule, NLToSQLModule, SynthesizerModule

from assets.settings import settings

class AgentState(TypedDict):
    question: str
    format_hint: str
    query_type: str
    retrieved_chunks: List[Chunk]
    extracted_context: dict
    sql_query: str
    sql_results: dict
    final_answer: Any
    explanation: str
    citations: List[str]
    confidence: float
    repair_count: int
    trace: List[str]
    error: str

class HybridRetailAgent:
    def __init__(self, router: RouterModule, nl_to_sql: NLToSQLModule, synthesizer: SynthesizerModule):
        self.retriever = DocumentRetriever()
        self.db_tool = SQLiteTool()
        self.router = router
        self.nl_to_sql = nl_to_sql
        self.synthesizer = synthesizer
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self.router_node)
        workflow.add_node("retriever", self.retriever_node)
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("nl_to_sql", self.nl_to_sql_node)
        workflow.add_node("executor", self.executor_node)
        workflow.add_node("synthesizer", self.synthesizer_node)
        workflow.add_node("validator", self.validator_node)
        workflow.add_node("repair", self.repair_node)
        
        # Define edges
        workflow.set_entry_point("router")
        
        # Router decides path
        workflow.add_conditional_edges(
            "router",
            self.route_decision,
            {
                "rag": "retriever",
                "sql": "planner",
                "hybrid": "retriever"
            }
        )
        
        # RAG path
        workflow.add_edge("retriever", "planner")
        
        # Planning to SQL generation
        workflow.add_conditional_edges(
            "planner",
            self.needs_sql,
            {
                "yes": "nl_to_sql",
                "no": "synthesizer"
            }
        )
        
        # SQL execution
        workflow.add_edge("nl_to_sql", "executor")
        workflow.add_edge("executor", "synthesizer")
        
        # Validation and repair loop
        workflow.add_edge("synthesizer", "validator")
        workflow.add_conditional_edges(
            "validator",
            self.needs_repair,
            {
                "repair": "repair",
                "done": END
            }
        )
        
        workflow.add_edge("repair", "nl_to_sql")
        
        return workflow.compile()
    
    def router_node(self, state: AgentState) -> AgentState:
        state["trace"].append("ROUTER: Classifying query type")
        query_type = self.router(question=state["question"])
        state["query_type"] = query_type
        state["trace"].append(f"ROUTER: Classified as {query_type}")
        return state
    
    def retriever_node(self, state: AgentState) -> AgentState:
        state["trace"].append("RETRIEVER: Fetching relevant documents")
        chunks = self.retriever.retrieve(state["question"])
        state["retrieved_chunks"] = chunks
        
        # Add document citations
        for chunk in chunks:
            if chunk.id not in state["citations"]:
                state["citations"].append(chunk.id)
        
        state["trace"].append(f"RETRIEVER: Found {len(chunks)} chunks with scores {[round(c.score, 3) for c in chunks]}")
        return state
    
    def planner_node(self, state: AgentState) -> AgentState:
        state["trace"].append("PLANNER: Extracting context and constraints")
        
        context = {}
        chunks_text = "\n\n".join([c.content for c in state.get("retrieved_chunks", [])])
        
        # Extract date ranges with flexible patterns
        context.update(self._extract_date_ranges(chunks_text, state["question"]))
        
        # Extract mentioned categories dynamically
        context.update(self._extract_categories(chunks_text, state["question"]))
        
        # Extract KPI definitions dynamically
        context.update(self._extract_kpi_definitions(chunks_text, state["question"]))
        
        # Extract numeric values for RAG-only queries
        context.update(self._extract_numeric_values(chunks_text, state["question"]))
        
        state["extracted_context"] = context
        state["trace"].append(f"PLANNER: Extracted context: {context}")
        return state
    
    def _extract_date_ranges(self, text: str, question: str) -> dict:
        """Dynamically extract date ranges from text - ENHANCED"""
        context = {}
        
        # Pattern 1: YYYY-MM-DD to YYYY-MM-DD
        date_pattern1 = r'(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})'
        dates = re.findall(date_pattern1, text)
        if dates:
            context["date_range"] = list(dates[0])  # Convert tuple to list
        
        campaign_patterns = [
            (r'Summer\s+Beverages\s+1997', r'Summer\s+Beverages\s+1997.*?(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})'),
            (r'Winter\s+Classics\s+1997', r'Winter\s+Classics\s+1997.*?(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})'),
        ]
        
        for campaign_name_pattern, date_extraction_pattern in campaign_patterns:
            if re.search(campaign_name_pattern, question, re.IGNORECASE):
                match = re.search(date_extraction_pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    context["date_range"] = [match.group(1), match.group(2)]
                    context["campaign_name"] = re.search(campaign_name_pattern, question, re.IGNORECASE).group(0)
                    break
        
        # Pattern 3: Year-only references
            year_match = re.search(r'\b(1997|1998)\b', question)
            if year_match:
                year = year_match.group(1)
                context["year"] = year
                context["date_range"] = [f"{year}-01-01", f"{year}-12-31"]
        
        return context
    
    def _extract_categories(self, text: str, question: str) -> dict:
        context = {}
        
        # Define standard categories from catalog
        standard_categories = [
            'Beverages', 'Condiments', 'Confections', 'Dairy Products', 
            'Grains/Cereals', 'Meat/Poultry', 'Produce', 'Seafood'
        ]
        
        # Check which categories are mentioned in question or text
        mentioned_categories = []
        question_lower = question.lower()
        
        for category in standard_categories:
            if category.lower() in question_lower:
                mentioned_categories.append(category)
        
        if mentioned_categories:
            context["categories"] = mentioned_categories
        
        # Also extract from marketing campaigns
        if 'summer beverages' in question_lower:
            if 'Beverages' not in mentioned_categories:
                mentioned_categories.append('Beverages')
            context["categories"] = mentioned_categories
        
        return context
    
    def _extract_kpi_definitions(self, text: str, question: str) -> dict:
        context = {}
        question_lower = question.lower()
        
        # AOV definition extraction
        if 'aov' in question_lower or 'average order value' in question_lower:
            # Look for AOV formula in text
            aov_pattern = r'AOV\s*=\s*(.+?)(?:\n|##|$)'
            match = re.search(aov_pattern, text, re.IGNORECASE)
            if match:
                context["aov_formula"] = match.group(1).strip()
        
        # Gross Margin definition extraction
        if 'margin' in question_lower or 'gross margin' in question_lower:
            gm_pattern = r'GM\s*=\s*(.+?)(?:\n|##|$)'
            match = re.search(gm_pattern, text, re.IGNORECASE)
            if match:
                context["gm_formula"] = match.group(1).strip()
            
            # Also check for cost approximation note
            if 'cost' in text.lower() and '70%' in text:
                context["cost_approximation"] = "70% of UnitPrice"
        
        return context
    
    def _extract_numeric_values(self, text: str, question: str) -> dict:
        context = {}
        question_lower = question.lower()
        
        # For return policy questions - FIXED LOGIC
        if 'return' in question_lower:
            # Check for beverages specifically
            if 'beverage' in question_lower:
                # Look for both opened and unopened policies
                unopened_pattern = r'Beverages?\s+unopened[:\s]+(\d+)\s+days?'
                opened_pattern = r'Beverages?.*?opened[:\s]+no\s+returns?'
                
                unopened_match = re.search(unopened_pattern, text, re.IGNORECASE)
                if unopened_match:
                    days = int(unopened_match.group(1))
                    context["beverages_return_days"] = days
                    context["return_answer"] = days
            
            # Generic return days extraction for other products
            if 'return_answer' not in context:
                product_types = {
                    'produce': r'Produce[^.]*?(\d+)[–-](\d+)\s+days?',
                    'seafood': r'Seafood[^.]*?(\d+)[–-](\d+)\s+days?',
                    'dairy': r'Dairy[^.]*?(\d+)[–-](\d+)\s+days?',
                    'perishable': r'Perishables[^.]*?(\d+)[–-](\d+)\s+days?',
                    'non-perishable': r'Non-perishables[:\s]+(\d+)\s+days?',
                }
                
                for prod_type, pattern in product_types.items():
                    if prod_type in question_lower:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            if len(match.groups()) == 2:
                                # Range like "3-7 days", use upper bound
                                context["return_answer"] = int(match.group(2))
                            else:
                                context["return_answer"] = int(match.group(1))
                            break
        
        return context
    
    def nl_to_sql_node(self, state: AgentState) -> AgentState:
        state["trace"].append("NL2SQL: Generating SQL query")
        
        context_str = json.dumps(state["extracted_context"])
        
        sql_query = self.nl_to_sql(
            question=state["question"],
            db_schema=self.db_tool.get_schema(),
            context=context_str
        )
        
        state["sql_query"] = sql_query
        state["trace"].append(f"NL2SQL: Generated SQL:\n{sql_query}")
        return state
    
    def executor_node(self, state: AgentState) -> AgentState:
        state["trace"].append("EXECUTOR: Executing SQL query")
        results = self.db_tool.execute_query(state["sql_query"])
        state["sql_results"] = results
        
        if not results['success']:
            state["error"] = results['error']
            state["trace"].append(f"EXECUTOR: FAILED - {results['error']}")
        else:
            row_count = len(results['rows']) if results['rows'] else 0
            state["trace"].append(f"EXECUTOR: Success - {row_count} rows, Data: {results['rows'][:3]}")
        
        return state
    
    def synthesizer_node(self, state: AgentState) -> AgentState:
        state["trace"].append("SYNTHESIZER: Generating final answer")
        
        # Check if we have a direct answer from context extraction
        if state["query_type"] == "rag" and "return_answer" in state["extracted_context"]:
            answer = state["extracted_context"]["return_answer"]
            state["final_answer"] = self._parse_answer(str(answer), state["format_hint"], state["extracted_context"])
            state["explanation"] = "Extracted from policy document"
            state["trace"].append(f"SYNTHESIZER: Direct answer from context: {answer}")
            return state
        
        rag_context = "\n\n".join([c.content for c in state.get("retrieved_chunks", [])])
        
        sql_results = state.get("sql_results", {})
        if sql_results.get("success"):
            sql_results_str = f"Columns: {sql_results['columns']}\nRows: {sql_results['rows']}"
        else:
            sql_results_str = ""
        
        answer, explanation = self.synthesizer(
            question=state["question"],
            format_hint=state["format_hint"],
            rag_context=rag_context,
            sql_results=sql_results_str
        )
        
        state["final_answer"] = self._parse_answer(answer, state["format_hint"], state["extracted_context"])
        state["explanation"] = explanation
        state["trace"].append(f"SYNTHESIZER: Final answer={state['final_answer']}, type={type(state['final_answer']).__name__}")
        return state
    
    def validator_node(self, state: AgentState) -> AgentState:
        state["trace"].append("VALIDATOR: Checking response quality")
        
        is_valid = True
        
        # Check if answer is valid
        if state["final_answer"] is None:
            is_valid = False
            state["error"] = "No valid answer generated"
        elif isinstance(state["final_answer"], str) and state["final_answer"] in ["Unable to determine", "Not found", ""]:
            is_valid = False
            state["error"] = "No valid answer generated"
        elif isinstance(state["final_answer"], dict) and not state["final_answer"]:
            is_valid = False
            state["error"] = "Empty dictionary returned"
        elif isinstance(state["final_answer"], list) and not state["final_answer"]:
            is_valid = False
            state["error"] = "Empty list returned"
        
        # Check SQL success for SQL/hybrid queries
        if state["query_type"] in ["sql", "hybrid"]:
            if not state.get("sql_results", {}).get("success"):
                is_valid = False
                if not state.get("error"):
                    state["error"] = state.get("sql_results", {}).get("error", "SQL failed")
        
        confidence = 0.5
        
        if state.get("retrieved_chunks"):
            avg_score = sum(c.score for c in state["retrieved_chunks"]) / len(state["retrieved_chunks"])
            confidence += min(avg_score * 0.3, 0.3)
        
        if state.get("sql_results", {}).get("success"):
            row_count = len(state["sql_results"].get("rows", []))
            if row_count > 0:
                confidence += 0.2
        
        if state.get("repair_count", 0) > 0:
            confidence -= 0.15 * state["repair_count"]
        
        if state.get("error"):
            confidence -= 0.2
        
        # Boost confidence for successful RAG-only queries
        if state["query_type"] == "rag" and is_valid:
            confidence += 0.15
        
        state["confidence"] = max(0.0, min(1.0, confidence))
        
        state["trace"].append(f"VALIDATOR: Valid={is_valid}, Confidence={state['confidence']:.2f}")
        return state
    
    def repair_node(self, state: AgentState) -> AgentState:
        state["repair_count"] = state.get("repair_count", 0) + 1
        state["trace"].append(f"REPAIR: Attempt {state['repair_count']}/2")
        
        if state.get("error"):
            error_msg = state["error"]
            state["extracted_context"]["error_feedback"] = error_msg
            
            if "no such table" in error_msg.lower():
                match = re.search(r'no such table: (\w+)', error_msg, re.IGNORECASE)
                if match:
                    wrong_table = match.group(1)
                    state["extracted_context"]["table_fix_needed"] = wrong_table
        
        state["error"] = ""
        return state
    
    def route_decision(self, state: AgentState) -> str:
        return state["query_type"]
    
    def needs_sql(self, state: AgentState) -> str:
        if state["query_type"] in ["sql", "hybrid"]:
            return "yes"
        return "no"
    
    def needs_repair(self, state: AgentState) -> str:
        repair_count = state.get("repair_count", 0)
        has_error = bool(state.get("error"))
        
        if has_error and repair_count < 2:
            return "repair"
        return "done"
    
    def _parse_answer(self, answer: str, format_hint: str, context: dict) -> Any:
        answer = str(answer).strip()
        
        # Remove markdown code blocks
        if '```' in answer:
            answer = re.sub(r'```(?:json|python|sql)?\s*', '', answer)
            answer = answer.strip('`').strip()
        
        if format_hint == "int":
            # Priority 1: Direct integer in answer
            try:
                return int(float(answer))
            except:
                pass
            
            # Priority 2: Extract first integer from answer
            match = re.search(r'\b(\d+)\b', answer)
            if match:
                return int(match.group(1))
            
            # Priority 3: Check extracted context
            if "return_answer" in context:
                return int(context["return_answer"])
            
            return 0
        
        elif format_hint == "float":
            try:
                return round(float(answer), 2)
            except:
                pass
            
            match = re.search(r'(\d+\.?\d*)', answer)
            if match:
                return round(float(match.group(1)), 2)
            
            return 0.0
        
        elif format_hint.startswith("{"):
            # Parse dictionary
            try:
                # Clean and find JSON
                answer = answer.replace("'", '"')
                start = answer.find('{')
                end = answer.rfind('}') + 1
                
                if start >= 0 and end > start:
                    json_str = answer[start:end]
                    parsed = json.loads(json_str)
                    
                    # Normalize keys to match expected format
                    expected_keys = re.findall(r'(\w+):', format_hint)
                    if expected_keys:
                        normalized = {}
                        for k, v in parsed.items():
                            key_lower = k.lower()
                            for expected in expected_keys:
                                if expected.lower() == key_lower or expected.lower() in key_lower:
                                    # Convert values to correct types
                                    if ':int' in format_hint and expected in format_hint:
                                        normalized[expected] = int(v) if v is not None else 0
                                    elif ':float' in format_hint and expected in format_hint:
                                        normalized[expected] = float(v) if v is not None else 0.0
                                    else:
                                        normalized[expected] = v
                                    break
                        
                        if normalized:
                            return normalized
                    
                    return parsed
            except Exception as e:
                print(f"Dict parsing error: {e}")
            
            return {}
        
        elif format_hint.startswith("list"):
            try:
                answer = answer.replace("'", '"')
                start = answer.find('[')
                end = answer.rfind(']') + 1
                
                if start >= 0 and end > start:
                    json_str = answer[start:end]
                    parsed = json.loads(json_str)
                    
                    # Ensure list items match expected format
                    if parsed and isinstance(parsed, list):
                        return parsed
            except Exception as e:
                print(f"List parsing error: {e}")            
            return []
        
        return answer
    
    def run(self, question: str, format_hint: str) -> dict:
        initial_state = {
            "question": question,
            "format_hint": format_hint,
            "query_type": "",
            "retrieved_chunks": [],
            "extracted_context": {},
            "sql_query": "",
            "sql_results": {},
            "final_answer": None,
            "explanation": "",
            "citations": [],
            "confidence": 0.0,
            "repair_count": 0,
            "trace": [],
            "error": ""
        }
        
        final_state = self.graph.invoke(initial_state)
        
        print("\n Trace Log:")
        for trace_line in final_state["trace"]:
            print(trace_line)        
        return {
            "final_answer": final_state["final_answer"],
            "sql": final_state.get("sql_query", ""),
            "confidence": final_state["confidence"],
            "explanation": final_state["explanation"],
            "citations": final_state["citations"],
            "trace": final_state["trace"]
        }
    
    def close(self):
        self.db_tool.close()