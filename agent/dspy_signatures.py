import dspy
from typing import List
import re


class RouterSignature(dspy.Signature):
    question = dspy.InputField(desc="User's question about retail analytics")
    query_type = dspy.OutputField(desc="Classification: 'rag', 'sql', or 'hybrid'")


class RouterModule(dspy.Module):    
    def __init__(self):
        super().__init__()
        self.classifier = dspy.Predict(RouterSignature)  # Changed from ChainOfThought
    
    def forward(self, question: str) -> str:
        try:
            result = self.classifier(question=question)
            query_type = result.query_type.strip().lower()
            
            # Ensure valid output
            if query_type not in ['rag', 'sql', 'hybrid']:
                # Fallback: Use keyword-based classification
                query_type = self._fallback_classify(question)
            
            return query_type
            
        except Exception as e:
            print(f"[WARNING] Router failed: {e}. Using fallback classification.")
            return self._fallback_classify(question)
    
    def _fallback_classify(self, question: str) -> str:
        q_lower = question.lower()
        
        # RAG indicators
        rag_keywords = ['according to', 'policy', 'definition', 'what is the', 'what are the', 
                        'from the document', 'per the', 'as defined in']
        
        # SQL indicators
        sql_keywords = ['total', 'count', 'sum', 'average', 'top', 'list', 'how many',
                        'revenue', 'quantity', 'price']
        
        # Hybrid indicators (both needed)
        hybrid_keywords = ['during', 'within', 'between', 'in the period', 'campaign',
                          'aov', 'kpi', 'margin', 'using the']
        
        rag_score = sum(1 for kw in rag_keywords if kw in q_lower)
        sql_score = sum(1 for kw in sql_keywords if kw in q_lower)
        hybrid_score = sum(1 for kw in hybrid_keywords if kw in q_lower)
        
        # Decision logic
        if hybrid_score > 0 and sql_score > 0:
            return 'hybrid'
        elif rag_score > sql_score:
            return 'rag'
        elif sql_score > 0:
            return 'sql'
        else:
            return 'hybrid'  
    

class NLToSQLSignature(dspy.Signature):
    question = dspy.InputField(desc="Natural language question requiring database query")
    db_schema = dspy.InputField(desc="Database schema with table and column information")
    context = dspy.InputField(desc="Additional context from retrieved documents (dates, KPIs, categories)")
    sql_query = dspy.OutputField(desc="Valid SQLite query to answer the question")


class NLToSQLModule(dspy.Module):    
    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict(NLToSQLSignature)
    
    def forward(self, question: str, db_schema: str, context: str) -> str:
        try:
            result = self.generator(
                question=question,
                db_schema=db_schema,
                context=context
            )
            
            sql = result.sql_query.strip()
            
        except Exception as e:
            print(f"[WARNING] NL2SQL generation failed: {e}. Using template-based SQL.")
            sql = self._generate_template_sql(question, context)
        
        # Clean up common formatting issues
        if sql.startswith('```sql'):
            sql = sql[6:]
        if sql.startswith('```'):
            sql = sql[3:]
        if sql.endswith('```'):
            sql = sql[:-3]
        
        sql = sql.strip()
        
        # Ensure it ends with semicolon
        if sql and not sql.endswith(';'):
            sql += ';'
        
        return sql
    
    def _generate_template_sql(self, question: str, context: str) -> str:
        """Fallback template-based SQL generation"""
        q_lower = question.lower()
        
        # Parse context for date ranges
        date_range = None
        if context:
            import json
            try:
                ctx_dict = json.loads(context)
                date_range = ctx_dict.get('date_range')
            except:
                pass
        
        if 'top' in q_lower and 'category' in q_lower:
            sql = """
            SELECT c.CategoryName, SUM(od.Quantity) as TotalQty
            FROM [Order Details] od
            JOIN Products p ON od.ProductID = p.ProductID
            JOIN Categories c ON p.CategoryID = c.CategoryID
            JOIN Orders o ON od.OrderID = o.OrderID
            """
            if date_range:
                sql += f"WHERE o.OrderDate BETWEEN '{date_range[0]}' AND '{date_range[1]}'\n"
            sql += "GROUP BY c.CategoryName ORDER BY TotalQty DESC LIMIT 1;"
            return sql
        
        elif 'revenue' in q_lower:
            sql = """
            SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as Revenue
            FROM [Order Details] od
            JOIN Products p ON od.ProductID = p.ProductID
            JOIN Orders o ON od.OrderID = o.OrderID
            """
            if date_range:
                sql += f"WHERE o.OrderDate BETWEEN '{date_range[0]}' AND '{date_range[1]}'\n"
            sql += ";"
            return sql
        
        return "SELECT COUNT(*) FROM Orders;"


class SynthesizerSignature(dspy.Signature):
    """Synthesize final answer from context and SQL results."""
    question = dspy.InputField(desc="Original user question")
    format_hint = dspy.InputField(desc="Expected output format (int, float, str, dict, list)")
    rag_context = dspy.InputField(desc="Context from retrieved documents")
    sql_results = dspy.InputField(desc="Results from SQL query execution (columns and rows)")
    answer = dspy.OutputField(desc="Final answer matching format_hint exactly")


class SynthesizerModule(dspy.Module):    
    def __init__(self):
        super().__init__()
        self.synthesizer = dspy.Predict(SynthesizerSignature)
    
    def forward(self, question: str, format_hint: str, rag_context: str, sql_results: str):
        try:
            result = self.synthesizer(
                question=question,
                format_hint=format_hint,
                rag_context=rag_context,
                sql_results=sql_results
            )
            
            answer = result.answer
            explanation = "Answer synthesized from available data."
            
        except Exception as e:
            print(f"[WARNING] Synthesis failed: {e}. Using direct extraction.")
            answer, explanation = self._fallback_synthesis(question, format_hint, 
                                                          rag_context, sql_results)
        
        return answer, explanation
    
    def _fallback_synthesis(self, question: str, format_hint: str, 
                           rag_context: str, sql_results: str) -> tuple:
        
        # For RAG-only questions
        if not sql_results and rag_context:
            if format_hint == "int":
                # Extract first number from context
                numbers = re.findall(r'\b(\d+)\b', rag_context)
                if numbers:
                    return numbers[0], "Extracted from document context"
        
        # For SQL results
        if "Rows:" in sql_results:
            rows_text = sql_results.split("Rows:")[1].strip()
            if rows_text and rows_text != "[]":
                # Try to parse first row
                match = re.search(r'\((.*?)\)', rows_text)
                if match:
                    values = match.group(1).split(',')
                    if format_hint == "int":
                        return values[0].strip().strip("'\""), "Extracted from query results"
                    elif format_hint == "float":
                        return values[0].strip().strip("'\""), "Extracted from query results"
                    elif format_hint.startswith("{"):
                        if len(values) >= 2:
                            return f'{{"category": "{values[0].strip()}", "quantity": {values[1].strip()}}}', "Extracted from query results"
        
        return "Unable to determine", "Insufficient data to answer question"


class Example:
    def __init__(self, question: str, query_type: str):
        self.question = question
        self.query_type = query_type


# Sample training examples for router optimization
ROUTER_TRAIN_EXAMPLES = [
    Example(
        question="According to the product policy, what is the return window for unopened Beverages?",
        query_type="rag"
    ),
    Example(
        question="What is the total revenue from all orders?",
        query_type="sql"
    ),
    Example(
        question="During Summer Beverages 1997, which category had the highest sales?",
        query_type="hybrid"
    ),
    Example(
        question="What is the AOV definition from the KPI docs?",
        query_type="rag"
    ),
    Example(
        question="List top 5 customers by order count",
        query_type="sql"
    ),
    Example(
        question="Using the AOV definition, what was the average order value in 1997?",
        query_type="hybrid"
    ),
    Example(
        question="What categories are mentioned in the catalog?",
        query_type="rag"
    ),
    Example(
        question="Total quantity sold for product ID 42",
        query_type="sql"
    ),
    Example(
        question="Per KPI definition, who was top customer by gross margin during Winter Classics 1997?",
        query_type="hybrid"
    ),
    Example(
        question="What are the return policies for perishables?",
        query_type="rag"
    ),
    Example(
        question="Count of orders in December 1997",
        query_type="sql"
    ),
    Example(
        question="Total revenue from Beverages during the Summer Beverages 1997 campaign dates?",
        query_type="hybrid"
    ),
]