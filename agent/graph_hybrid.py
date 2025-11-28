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
        state["trace"].append(f"PLANNER: Extracted context: {list(context.keys())}")
        return state
    
    def _extract_date_ranges(self, text: str, question: str) -> dict:
        """Dynamically extract date ranges from text"""
        context = {}
        
        # Pattern 1: YYYY-MM-DD to YYYY-MM-DD
        date_pattern1 = r'(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})'
        dates = re.findall(date_pattern1, text)
        if dates:
            context["date_range"] = dates[0]
        
        # Pattern 2: Named date references in question (e.g., "Summer Beverages 1997")
        # Extract campaign names from question
        campaign_patterns = [
            r'([A-Z][a-z]+\s+[A-Z][a-z]+\s+\d{4})',  # "Summer Beverages 1997"
            r'([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+\d{4})',  # "Winter Classics Campaign 1997"
        ]
        
        for pattern in campaign_patterns:
            campaigns = re.findall(pattern, question)
            if campaigns:
                campaign_name = campaigns[0]
                # Search for this campaign in retrieved text
                campaign_section = re.search(
                    rf'##\s*{re.escape(campaign_name)}.*?Dates:\s*(\d{{4}}-\d{{2}}-\d{{2}})\s+to\s+(\d{{4}}-\d{{2}}-\d{{2}})',
                    text,
                    re.IGNORECASE | re.DOTALL
                )
                if campaign_section:
                    context["date_range"] = (campaign_section.group(1), campaign_section.group(2))
                    context["campaign_name"] = campaign_name
                    break
        
        # Pattern 3: Year-only references
        if "date_range" not in context:
            year_match = re.search(r'\b(19\d{2}|20\d{2})\b', question)
            if year_match:
                year = year_match.group(1)
                context["year"] = year
        
        return context
    
    def _extract_categories(self, text: str, question: str) -> dict:
        """Dynamically extract product categories"""
        context = {}
        
        # Find all capitalized category-like terms from documents
        category_pattern = r'\b([A-Z][a-z]+(?:/[A-Z][a-z]+)?(?:\s+[A-Z][a-z]+)?)\b'
        potential_categories = set(re.findall(category_pattern, text))
        
        # Also check question for mentioned categories
        question_categories = set(re.findall(category_pattern, question))
        
        # Filter to actual categories mentioned in both or explicitly listed
        all_categories = potential_categories.union(question_categories)
        
        # Common category keywords
        category_keywords = ['beverages', 'condiments', 'confections', 'dairy', 'grains', 
                            'cereals', 'meat', 'poultry', 'produce', 'seafood']
        
        found_categories = [cat for cat in all_categories 
                           if any(keyword in cat.lower() for keyword in category_keywords)]
        
        if found_categories:
            context["categories"] = found_categories
        
        return context
    
    def _extract_kpi_definitions(self, text: str, question: str) -> dict:
        """Dynamically extract KPI definitions"""
        context = {}
        text_lower = text.lower()
        question_lower = question.lower()
        
        # Detect which KPIs are mentioned in the question
        kpi_patterns = {
            'aov': r'AOV\s*=\s*(.+?)(?:\n|##|$)',
            'average order value': r'Average Order Value.*?\n.*?=\s*(.+?)(?:\n|##|$)',
            'gm': r'GM\s*=\s*(.+?)(?:\n|##|$)',
            'gross margin': r'Gross Margin.*?\n.*?=\s*(.+?)(?:\n|##|$)',
        }
        
        for kpi_name, pattern in kpi_patterns.items():
            if kpi_name in question_lower:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    definition = match.group(1).strip()
                    # Clean up the definition
                    definition = definition.split('\n')[0].strip()
                    context[f"{kpi_name}_definition"] = definition
        
        return context
    
    def _extract_numeric_values(self, text: str, question: str) -> dict:
        """Extract specific numeric values mentioned in context"""
        context = {}
        question_lower = question.lower()
        
        # For return policy questions
        if 'return' in question_lower and 'days' in question_lower:
            # Find product type mentioned
            product_types = ['beverages', 'perishables', 'produce', 'seafood', 
                           'dairy', 'non-perishables']
            
            for prod_type in product_types:
                if prod_type in question_lower:
                    # Search for "X days" pattern near this product type
                    section = re.search(
                        rf'{prod_type}[^.]*?(\d+)\s*days?',
                        text,
                        re.IGNORECASE
                    )
                    if section:
                        context[f"{prod_type}_return_days"] = section.group(1)
                        break
            
            # Also try general pattern matching
            if not any(k.endswith('_return_days') for k in context.keys()):
                # Look for any "X days" mentions
                days_matches = re.findall(r'(\d+)\s*days?', text)
                if days_matches:
                    context["return_days_found"] = days_matches
        
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
        state["trace"].append(f"NL2SQL: Generated query ({len(sql_query)} chars)")
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
            state["trace"].append(f"EXECUTOR: Success - {row_count} rows returned")
        
        return state
    
    def synthesizer_node(self, state: AgentState) -> AgentState:
        state["trace"].append("SYNTHESIZER: Generating final answer")
        
        rag_context = "\n\n".join([c.content for c in state.get("retrieved_chunks", [])])
        
        sql_results = state.get("sql_results", {})
        if sql_results.get("success"):
            sql_results_str = f"Columns: {sql_results['columns']}\nRows:\n{sql_results['rows']}"
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
        state["trace"].append(f"SYNTHESIZER: Answer type={type(state['final_answer']).__name__}")
        return state
    
    def validator_node(self, state: AgentState) -> AgentState:
        state["trace"].append("VALIDATOR: Checking response quality")
        
        is_valid = True
        
        # Check if answer matches format
        if state["final_answer"] is None or state["final_answer"] == "Unable to determine":
            is_valid = False
            state["error"] = "No valid answer generated"
        
        # Check SQL success for SQL/hybrid queries
        if state["query_type"] in ["sql", "hybrid"]:
            if not state.get("sql_results", {}).get("success"):
                is_valid = False
                if not state.get("error"):
                    state["error"] = state.get("sql_results", {}).get("error", "SQL failed")
        
        # Calculate confidence dynamically
        confidence = 0.5  # Base confidence
        
        # Boost for good retrieval
        if state.get("retrieved_chunks"):
            avg_score = sum(c.score for c in state["retrieved_chunks"]) / len(state["retrieved_chunks"])
            confidence += min(avg_score * 0.3, 0.3)
        
        # Boost for successful SQL
        if state.get("sql_results", {}).get("success"):
            row_count = len(state["sql_results"].get("rows", []))
            if row_count > 0:
                confidence += 0.2
        
        # Penalty for repairs
        if state.get("repair_count", 0) > 0:
            confidence -= 0.15 * state["repair_count"]
        
        # Penalty for errors
        if state.get("error"):
            confidence -= 0.2
        
        state["confidence"] = max(0.0, min(1.0, confidence))
        
        state["trace"].append(f"VALIDATOR: Valid={is_valid}, Confidence={state['confidence']:.2f}")
        return state
    
    def repair_node(self, state: AgentState) -> AgentState:
        """Attempt to repair failed queries"""
        state["repair_count"] = state.get("repair_count", 0) + 1
        state["trace"].append(f"REPAIR: Attempt {state['repair_count']}/2")
        
        # Add error context for next SQL generation
        if state.get("error"):
            error_msg = state["error"]
            state["extracted_context"]["error_feedback"] = error_msg
            
            # Try to fix common errors
            if "no such table" in error_msg.lower():
                # Extract table name and suggest correction
                match = re.search(r'no such table: (\w+)', error_msg, re.IGNORECASE)
                if match:
                    wrong_table = match.group(1)
                    state["extracted_context"]["table_fix_needed"] = wrong_table
        
        # Clear error for retry
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
        
        # Only repair if we have an error and haven't exceeded max attempts
        if has_error and repair_count < 2:
            return "repair"
        return "done"
    
    def _parse_answer(self, answer: str, format_hint: str, context: dict) -> Any:
        """Enhanced parsing with context awareness"""
        answer = str(answer).strip()
        
        # Remove markdown code blocks
        if '```' in answer:
            parts = answer.split('```')
            for part in parts:
                clean_part = part.strip()
                if clean_part and not clean_part.startswith(('json', 'python', 'sql')):
                    answer = clean_part
                    break
        
        if format_hint == "int":
            # Try multiple extraction strategies
            # 1. Direct integer parsing
            try:
                return int(answer)
            except:
                pass
            
            # 2. Extract from text
            match = re.search(r'\b(\d+)\b', answer)
            if match:
                return int(match.group(1))
            
            # 3. Check context for extracted values
            for key, value in context.items():
                if 'days' in key or 'return' in key:
                    try:
                        return int(value)
                    except:
                        pass
            
            return 0
        
        elif format_hint == "float":
            # Try multiple extraction strategies
            try:
                return round(float(answer), 2)
            except:
                pass
            
            match = re.search(r'\b(\d+\.?\d*)\b', answer)
            if match:
                return round(float(match.group(1)), 2)
            
            return 0.0
        
        elif format_hint.startswith("{"):
            # Parse dictionary
            try:
                # Clean up
                answer = answer.replace("'", '"')
                # Find JSON boundaries
                start = answer.find('{')
                end = answer.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = answer[start:end]
                    parsed = json.loads(json_str)
                    
                    # Ensure keys match expected format from format_hint
                    # Extract expected keys from format_hint
                    expected_keys = re.findall(r'(\w+):', format_hint)
                    if expected_keys:
                        # Normalize keys to lowercase
                        normalized = {}
                        for k, v in parsed.items():
                            key_lower = k.lower()
                            # Match to expected keys
                            for expected in expected_keys:
                                if expected.lower() in key_lower or key_lower in expected.lower():
                                    normalized[expected] = v
                        if normalized:
                            return normalized
                    
                    return parsed
            except Exception as e:
                print(f"JSON parse error: {e}")
            
            return {}
        
        elif format_hint.startswith("list"):
            # Parse list
            try:
                answer = answer.replace("'", '"')
                start = answer.find('[')
                end = answer.rfind(']') + 1
                if start >= 0 and end > start:
                    json_str = answer[start:end]
                    return json.loads(json_str)
            except Exception as e:
                print(f"List parse error: {e}")
            
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