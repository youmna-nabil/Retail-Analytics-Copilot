from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END
import json
import re

from agent.rag.retrieval import DocumentRetriever, Chunk
from agent.tools.sqlite_tool import SQLiteTool
from agent.dspy_signatures import RouterModule, NLToSQLModule, SynthesizerModule

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
        self.retriever = DocumentRetriever(top_k=3)
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
        
        state["trace"].append(f"RETRIEVER: Found {len(chunks)} chunks")
        return state
    
    def planner_node(self, state: AgentState) -> AgentState:
        state["trace"].append("PLANNER: Extracting context and constraints")
        
        context = {}
        chunks_text = "\n\n".join([c.content for c in state.get("retrieved_chunks", [])])
        
        # Extract date ranges
        date_pattern = r'(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})'
        dates = re.findall(date_pattern, chunks_text)
        if dates:
            context["date_range"] = dates[0]
        
        # Extract mentioned categories
        categories = ["Beverages", "Condiments", "Confections", "Dairy Products", 
                     "Grains/Cereals", "Meat/Poultry", "Produce", "Seafood"]
        found_categories = [cat for cat in categories if cat in chunks_text]
        if found_categories:
            context["categories"] = found_categories
        
        # Extract KPI definitions
        kpi_text = chunks_text.lower()
        if "aov" in kpi_text:
            aov_def = re.search(r'AOV = (.+?)(?:\n|$)', chunks_text, re.IGNORECASE | re.DOTALL)
            if aov_def:
                context["aov_definition"] = aov_def.group(1).strip()
        if "gross margin" in kpi_text or "gm" in kpi_text:
            gm_def = re.search(r'GM = (.+?)(?:\n|$)', chunks_text, re.IGNORECASE | re.DOTALL)
            if gm_def:
                context["gm_definition"] = gm_def.group(1).strip()
        
        state["extracted_context"] = context
        state["trace"].append(f"PLANNER: Extracted {len(context)} items")
        return state
    
    def nl_to_sql_node(self, state: AgentState) -> AgentState:
        state["trace"].append("NL2SQL: Generating SQL query")
        
        context_str = json.dumps(state["extracted_context"])
        
        sql_query = self.nl_to_sql(
            question=state["question"],
            db_schema=self.db_tool.get_schema(),
            context=context_str
        )
        
        state["sql_query"] = sql_query
        state["trace"].append(f"NL2SQL: Generated: {sql_query[:100]}...")
        return state
    
    def executor_node(self, state: AgentState) -> AgentState:
        state["trace"].append("EXECUTOR: Executing SQL query")
        results = self.db_tool.execute_query(state["sql_query"])
        state["sql_results"] = results
        if not results['success']:
            state["error"] = results['error']
        state["trace"].append(f"EXECUTOR: Success={results['success']}")
        return state
    
    def synthesizer_node(self, state: AgentState) -> AgentState:
        state["trace"].append("SYNTHESIZER: Generating final answer")
        
        rag_context = "\n\n".join([c.content for c in state.get("retrieved_chunks", [])])
        
        sql_results = state.get("sql_results", {})
        if sql_results.get("success"):
            sql_results_str = f"Columns: {sql_results['columns']}\nRows:\n" + "\n".join([str(row) for row in sql_results['rows']])
        else:
            sql_results_str = ""
        
        answer, explanation = self.synthesizer(
            question=state["question"],
            format_hint=state["format_hint"],
            rag_context=rag_context,
            sql_results=sql_results_str
        )
        
        state["final_answer"] = self._parse_answer(answer, state["format_hint"])
        state["explanation"] = explanation
        state["trace"].append("SYNTHESIZER: Done")
        return state
    
    def validator_node(self, state: AgentState) -> AgentState:
        state["trace"].append("VALIDATOR: Checking response quality")
        
        is_valid = True
        
        # Check if answer matches format
        if state["final_answer"] is None:
            is_valid = False
            state["error"] = "No answer generated"
        
        # Check SQL success for SQL/hybrid queries
        if state["query_type"] in ["sql", "hybrid"]:
            if not state.get("sql_results", {}).get("success"):
                is_valid = False
                if not state.get("error"):
                    state["error"] = state.get("sql_results", {}).get("error", "SQL failed")
        
        # Calculate confidence
        confidence = 0.7
        if state.get("retrieved_chunks"):
            avg_score = sum(c.score for c in state["retrieved_chunks"]) / len(state["retrieved_chunks"])
            confidence += avg_score * 0.2
        if state.get("sql_results", {}).get("success"):
            confidence += 0.1
        if state.get("repair_count", 0) > 0:
            confidence -= 0.1 * state["repair_count"]
        
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
    
    def _parse_answer(self, answer: str, format_hint: str) -> Any:
        answer = str(answer).strip()
        
        # Remove markdown code blocks
        if '```' in answer:
            answer = answer.split('```')[1].split('```')[0].strip()
            if answer.startswith('json'):
                answer = answer[4:].strip()
        
        if format_hint == "int":
            # Extract first integer
            match = re.search(r'-?\d+', answer)
            return int(match.group()) if match else 0
        
        elif format_hint == "float":
            # Extract first number
            match = re.search(r'-?\d+\.?\d*', answer)
            return round(float(match.group()), 2) if match else 0.0
        
        elif format_hint.startswith("{") or format_hint.startswith("list["):
            # Try to parse JSON
            try:
                # Clean up common issues
                answer = answer.replace("'", '"')
                return json.loads(answer)
            except:
                # Try to extract from text
                if "{" in answer and "}" in answer:
                    json_str = answer[answer.find("{"):answer.rfind("}")+1]
                    try:
                        return json.loads(json_str)
                    except:
                        pass
                return {}
        
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