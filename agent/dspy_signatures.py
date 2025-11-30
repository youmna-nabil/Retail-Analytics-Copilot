import dspy
from typing import List
import re
import json


class RouterSignature(dspy.Signature):
    question: str = dspy.InputField(desc="User's question about retail analytics")
    query_type: str = dspy.OutputField(desc="Must be exactly one of: rag, sql, or hybrid. Return ONLY the classification word, nothing else.")


class RouterModule(dspy.Module):    
    def __init__(self):
        super().__init__()
        self.classifier = dspy.ChainOfThought(RouterSignature)
    
    def forward(self, question: str) -> str:
        try:
            result = self.classifier(question=question)
            query_type = str(result.query_type).strip().lower()
            
            # Extract the classification word
            for word in query_type.split():
                word_clean = word.strip('.,!?;:"\'')
                if word_clean in ['rag', 'sql', 'hybrid']:
                    return word_clean
            
            # If no valid classification found, use fallback
            print(f"[WARNING] Could not extract valid query_type from: '{query_type}'. Using fallback.")
            return self._fallback_classify(question)
            
        except Exception as e:
            print(f"[WARNING] Router failed: {e}. Using fallback classification.")
            return self._fallback_classify(question)
    
    def _fallback_classify(self, question: str) -> str:
        q_lower = question.lower()
        
        # PRIORITY 1: RAG INDICATORS (Check FIRST!)
        strong_rag_indicators = [
            'according to',          # "according to the policy"
            'per the',              # "per the definition"
            'as defined in',        # "as defined in the KPI docs"
            'what is the return',   # "what is the return window"
            'return window',        # explicit return policy question
            'return policy',        # explicit policy question
            'policy',               # any policy question
        ]
        
        for indicator in strong_rag_indicators:
            if indicator in q_lower:
                print(f"[ROUTER FALLBACK] Detected RAG indicator: '{indicator}'")
                return 'rag'
        
        # Check for definition questions (KPI definitions)
        if any(term in q_lower for term in ['definition', 'what is aov', 'what is the average order value']):
            if 'kpi' in q_lower or 'defined' in q_lower:
                print(f"[ROUTER FALLBACK] Detected definition question -> RAG")
                return 'rag'
        
        # PRIORITY 2: HYBRID INDICATORS
        
        # Campaign references (need marketing calendar dates + database)
        campaign_indicators = [
            'summer beverages',
            'winter classics',
            'during.*campaign',
        ]
        
        for pattern in campaign_indicators:
            if re.search(pattern, q_lower):
                print(f"[ROUTER FALLBACK] Detected campaign reference -> HYBRID")
                return 'hybrid'
        
        # Questions that combine definition + calculation
        has_definition_reference = any(term in q_lower for term in ['using the', 'per the definition', 'kpi definition'])
        has_calculation_need = any(term in q_lower for term in ['what was', 'calculate', 'total', 'average', 'during'])
        
        if has_definition_reference and has_calculation_need:
            print(f"[ROUTER FALLBACK] Detected definition + calculation -> HYBRID")
            return 'hybrid'
        
        # Questions with time periods that reference calendar
        if 'during' in q_lower and any(year in q_lower for year in ['1997', '2016', '2017']):
            print(f"[ROUTER FALLBACK] Detected time period question -> HYBRID")
            return 'hybrid'
        
        # PRIORITY 3: SQL INDICATORS (Check LAST!)
        pure_sql_indicators = [
            'all-time',             # "top products all-time"
            'all time',             # "revenue all time"
            'top.*by.*revenue',     # "top products by revenue"
            'top.*by.*quantity',    # "top by quantity"
            'list all',             # "list all customers"
            'how many total',       # "how many total orders"
        ]
        
        for pattern in pure_sql_indicators:
            if re.search(pattern, q_lower):
                print(f"[ROUTER FALLBACK] Detected pure SQL query -> SQL")
                return 'sql'
        
        # Generic SQL indicators (only if no RAG/Hybrid detected)
        sql_keywords = ['top', 'sum', 'count', 'total', 'highest', 'lowest', 'list']
        sql_score = sum(1 for kw in sql_keywords if kw in q_lower)
        
        if sql_score >= 2: 
            print(f"[ROUTER FALLBACK] Multiple SQL keywords detected -> SQL")
            return 'sql'
        
        return 'hybrid'

class NLToSQLSignature(dspy.Signature):
    question: str = dspy.InputField()
    db_schema: str = dspy.InputField()
    context: str = dspy.InputField()
    sql_query: str = dspy.OutputField(desc="Return ONLY a valid SQLite SELECT query. No explanations, no markdown, just the SQL query ending with semicolon.")


class NLToSQLModule(dspy.Module):    
    def __init__(self):
        super().__init__()
        self.generator = dspy.ChainOfThought(NLToSQLSignature)
    
    def forward(self, question: str, db_schema: str, context: str) -> str:
        try:
            enhanced_context = f"{context}\n\nIMPORTANT: Generate ONLY the SQL query. Use [Order Details] with brackets for the order details table."
            
            result = self.generator(
                question=question,
                db_schema=db_schema,
                context=enhanced_context
            )
            
            sql = str(result.sql_query).strip()
            
            if len(sql) < 20 or 'SELECT' not in sql.upper():
                print(f"[WARNING] Generated SQL looks invalid, using template.")
                sql = self._generate_template_sql(question, context, db_schema)
            
        except Exception as e:
            print(f"[WARNING] NL2SQL generation failed: {e}. Using template-based SQL.")
            sql = self._generate_template_sql(question, context, db_schema)
        
        sql = self._clean_sql(sql)
        return sql
    
    def _clean_sql(self, sql: str) -> str:
        sql = re.sub(r'```sql\s*', '', sql)
        sql = re.sub(r'```\s*', '', sql)
        
        select_match = re.search(r'(SELECT\s+.+?;)', sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            sql = select_match.group(1)
        
        sql = sql.strip()
        
        if sql and not sql.endswith(';'):
            sql += ';'
        
        return sql
    
    def _generate_template_sql(self, question: str, context: str, db_schema: str) -> str:
        q_lower = question.lower()
        
        # Parse context
        date_range = None
        categories = []
        year = None
        
        try:
            ctx_dict = json.loads(context)
            date_range = ctx_dict.get('date_range')
            categories = ctx_dict.get('categories', [])
            year = ctx_dict.get('year')
        except:
            pass
        
        intent = self._detect_query_intent(q_lower)
        
        if intent['type'] == 'top_category_by_quantity':
            return self._build_top_category_query(date_range, categories)
        elif intent['type'] == 'top_products_by_revenue':
            return self._build_top_products_query(intent['limit'], date_range)
        elif intent['type'] == 'revenue_by_category':
            return self._build_category_revenue_query(categories, date_range)
        elif intent['type'] == 'aov_calculation':
            return self._build_aov_query(date_range)
        elif intent['type'] == 'customer_margin':
            return self._build_customer_margin_query(date_range, year)
        else:
            return "SELECT COUNT(*) as count FROM Orders;"
    
    def _detect_query_intent(self, question: str) -> dict:
        intent = {'type': 'generic', 'limit': None}
        
        if 'category' in question and 'quantity' in question and ('highest' in question or 'top' in question):
            intent['type'] = 'top_category_by_quantity'
        elif 'product' in question and 'revenue' in question and 'top' in question:
            limit_match = re.search(r'top\s*(\d+)', question)
            intent['type'] = 'top_products_by_revenue'
            intent['limit'] = int(limit_match.group(1)) if limit_match else 1
        elif 'revenue' in question and ('category' in question or 'beverages' in question):
            intent['type'] = 'revenue_by_category'
        elif 'aov' in question or 'average order value' in question:
            intent['type'] = 'aov_calculation'
        elif 'customer' in question and 'margin' in question:
            intent['type'] = 'customer_margin'
        
        return intent
    
    def _build_top_category_query(self, date_range, categories):
        sql = """SELECT c.CategoryName as category, SUM(od.Quantity) as quantity
FROM [Order Details] od
JOIN Products p ON od.ProductID = p.ProductID
JOIN Categories c ON p.CategoryID = c.CategoryID
JOIN Orders o ON od.OrderID = o.OrderID"""
        
        conditions = []
        if date_range:
            conditions.append(f"o.OrderDate BETWEEN '{date_range[0]}' AND '{date_range[1]}'")
        
        if conditions:
            sql += "\nWHERE " + " AND ".join(conditions)
        
        sql += "\nGROUP BY c.CategoryName\nORDER BY quantity DESC\nLIMIT 1;"
        return sql
    
    def _build_top_products_query(self, limit, date_range):
        sql = """SELECT p.ProductName as product, 
       ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)), 2) as revenue
FROM [Order Details] od
JOIN Products p ON od.ProductID = p.ProductID
JOIN Orders o ON od.OrderID = o.OrderID"""
        
        if date_range:
            sql += f"\nWHERE o.OrderDate BETWEEN '{date_range[0]}' AND '{date_range[1]}'"
        
        sql += f"\nGROUP BY p.ProductName\nORDER BY revenue DESC\nLIMIT {limit};"
        return sql
    
    def _build_category_revenue_query(self, categories, date_range):
        sql = """SELECT ROUND(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)), 2) as revenue
FROM [Order Details] od
JOIN Products p ON od.ProductID = p.ProductID
JOIN Categories c ON p.CategoryID = c.CategoryID
JOIN Orders o ON od.OrderID = o.OrderID"""
        
        conditions = []
        
        if categories:
            category_list = "', '".join(categories)
            conditions.append(f"c.CategoryName IN ('{category_list}')")
        
        if date_range:
            conditions.append(f"o.OrderDate BETWEEN '{date_range[0]}' AND '{date_range[1]}'")
        
        if conditions:
            sql += "\nWHERE " + " AND ".join(conditions)
        
        sql += ";"
        return sql
    
    def _build_aov_query(self, date_range):
        sql = """SELECT ROUND(
    SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) / COUNT(DISTINCT o.OrderID),
    2
) as aov
FROM [Order Details] od
JOIN Orders o ON od.OrderID = o.OrderID"""
        
        if date_range:
            sql += f"\nWHERE o.OrderDate BETWEEN '{date_range[0]}' AND '{date_range[1]}'"
        
        sql += ";"
        return sql
    
    def _build_customer_margin_query(self, date_range, year):
        sql = """SELECT c.CompanyName as customer, 
       ROUND(SUM((od.UnitPrice - od.UnitPrice * 0.7) * od.Quantity * (1 - od.Discount)), 2) as margin
FROM [Order Details] od
JOIN Orders o ON od.OrderID = o.OrderID
JOIN Customers c ON o.CustomerID = c.CustomerID"""
        
        conditions = []
        
        if date_range:
            conditions.append(f"o.OrderDate BETWEEN '{date_range[0]}' AND '{date_range[1]}'")
        elif year:
            conditions.append(f"strftime('%Y', o.OrderDate) = '{year}'")
        
        if conditions:
            sql += "\nWHERE " + " AND ".join(conditions)
        
        sql += "\nGROUP BY c.CompanyName\nORDER BY margin DESC\nLIMIT 1;"
        return sql


class SynthesizerSignature(dspy.Signature):
    question: str = dspy.InputField()
    format_hint: str = dspy.InputField()
    rag_context: str = dspy.InputField()
    sql_results: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Return ONLY the answer value matching the format_hint. For int: return just the number. For float: return the number. For dict: return JSON object. For list: return JSON array. No explanations.")


class SynthesizerModule(dspy.Module):    
    def __init__(self):
        super().__init__()
        self.synthesizer = dspy.ChainOfThought(SynthesizerSignature)
    
    def forward(self, question: str, format_hint: str, rag_context: str, sql_results: str):
        try:
            result = self.synthesizer(
                question=question,
                format_hint=format_hint,
                rag_context=rag_context,
                sql_results=sql_results
            )
            
            answer = str(result.answer).strip()
            explanation = "Answer synthesized from available data."
            
            if not self._validate_answer_format(answer, format_hint):
                print(f"[WARNING] Answer format mismatch, using fallback extraction.")
                answer, explanation = self._fallback_synthesis(question, format_hint, rag_context, sql_results)
            
        except Exception as e:
            print(f"[WARNING] Synthesis failed: {e}. Using direct extraction.")
            answer, explanation = self._fallback_synthesis(question, format_hint, rag_context, sql_results)
        
        return answer, explanation
    
    def _validate_answer_format(self, answer: str, format_hint: str) -> bool:
        if format_hint == "int":
            return bool(re.search(r'\d+', answer))
        elif format_hint == "float":
            return bool(re.search(r'\d+\.?\d*', answer))
        elif format_hint.startswith("{"):
            return '{' in answer
        elif format_hint.startswith("list"):
            return '[' in answer
        return True
    
    def _fallback_synthesis(self, question: str, format_hint: str, rag_context: str, sql_results: str) -> tuple:        
       
        # Priority 1: SQL results
        if "Rows:" in sql_results and sql_results.split("Rows:")[1].strip() not in ["[]", ""]:
            return self._extract_from_sql_results(question, format_hint, sql_results)
        
        # Priority 2: RAG context
        if rag_context:
            return self._extract_from_context(question, format_hint, rag_context)
        
        return "Unable to determine", "Insufficient data to answer question"
    
    def _extract_from_context(self, question, format_hint, context):
        q_lower = question.lower()
        
        if format_hint == "int":
            # For return policy questions
            if 'return' in q_lower and 'beverage' in q_lower:
                # Look for "Beverages unopened: X days"
                match = re.search(r'Beverages?\s+unopened[:\s]+(\d+)\s+days?', context, re.IGNORECASE)
                if match:
                    return str(match.group(1)), "Extracted from policy document"
                
                # Also try generic pattern
                match = re.search(r'(\d+)\s+days?', context)
                if match:
                    return str(match.group(1)), "Extracted from policy document"
            
            # Generic integer extraction
            numbers = re.findall(r'\b(\d+)\b', context)
            if numbers:
                return numbers[0], "Extracted from document"
        
        elif format_hint == "float":
            match = re.search(r'\b(\d+\.?\d*)\b', context)
            if match:
                return str(match.group(1)), "Extracted from document"
        
        return "Not found in documents", "Unable to extract answer from context"
    
    def _extract_from_sql_results(self, question, format_hint, sql_results):
        try:
            lines = sql_results.split('\n')
            columns_line = next((l for l in lines if l.startswith('Columns:')), None)
            rows_line = next((l for l in lines if l.startswith('Rows:')), None)
            
            if not columns_line or not rows_line:
                return "No results", "Empty SQL results"
            
            # Parse columns
            columns_text = columns_line.replace('Columns:', '').strip()
            columns = eval(columns_text)
            
            # Parse rows
            rows_text = rows_line.replace('Rows:', '').strip()
            import ast
            rows = ast.literal_eval(rows_text)
            
            if not rows:
                return "No results", "Query returned no rows"
            
            if format_hint == "int":
                value = rows[0][0]
                return str(int(value)) if value is not None else "0", "Extracted from query result"
            
            elif format_hint == "float":
                value = rows[0][0]
                return str(round(float(value), 2)) if value is not None else "0.0", "Extracted from query result"
            
            elif format_hint.startswith("{"):
                row = rows[0]
                result_dict = {}
                
                # Match column names to expected keys in format_hint
                expected_keys = re.findall(r'(\w+):', format_hint)
                
                for i, col in enumerate(columns):
                    if i < len(row):
                        value = row[i]
                        col_lower = col.lower()
                        
                        # Map to expected key
                        for expected in expected_keys:
                            if expected.lower() == col_lower or expected.lower() in col_lower or col_lower in expected.lower():
                                # Type conversion based on format_hint
                                if ':int' in format_hint and expected in format_hint:
                                    result_dict[expected] = int(value) if value is not None else 0
                                elif ':float' in format_hint and expected in format_hint:
                                    result_dict[expected] = float(value) if value is not None else 0.0
                                else:
                                    result_dict[expected] = str(value) if value is not None else ""
                                break
                
                return json.dumps(result_dict), "Formatted from query result"
            
            elif format_hint.startswith("list"):
                result_list = []
                expected_keys = re.findall(r'(\w+):', format_hint)
                
                for row in rows:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        if i < len(row):
                            value = row[i]
                            col_lower = col.lower()
                            
                            for expected in expected_keys:
                                if expected.lower() == col_lower or expected.lower() in col_lower or col_lower in expected.lower():
                                    if ':int' in format_hint:
                                        row_dict[expected] = int(value) if value is not None else 0
                                    elif ':float' in format_hint:
                                        row_dict[expected] = float(value) if value is not None else 0.0
                                    else:
                                        row_dict[expected] = str(value) if value is not None else ""
                                    break
                    
                    if row_dict:
                        result_list.append(row_dict)
                
                return json.dumps(result_list), "Formatted from query results"
            
            else:
                return str(rows[0][0]), "Extracted from query result"
        
        except Exception as e:
            return f"Parse error: {str(e)}", "Failed to parse SQL results"