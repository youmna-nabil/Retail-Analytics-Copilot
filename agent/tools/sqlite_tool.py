import sqlite3
from typing import Dict

from assets.config import config

class SQLiteTool:
    def __init__(self):
        self.conn = sqlite3.connect(config.DB_PATH)
        self.cursor = self.conn.cursor()

    def get_schema(self) -> str:
        schema = ""
        tables = self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        for (table_name,) in tables:
            schema += f"Table: {table_name}\n"
            columns = self.cursor.execute(f"PRAGMA table_info('{table_name}');").fetchall()
            for col in columns:
                schema += f"- {col[1]} ({col[2]})\n"
        return schema
    
    def execute_query(self, query: str) -> Dict:
        try:
            self.cursor.execute(query)
            columns = [description[0] for description in self.cursor.description]
            rows = self.cursor.fetchall()
            result= {
                'success': True,
                'columns': columns,
                'rows': rows,
                'error': None
            }
        except Exception as e:
            result= {
                'success': False,
                'columns': None,
                'rows': None,
                'error': str(e)
            }
        return result
    
    def close(self):
        self.conn.close()
