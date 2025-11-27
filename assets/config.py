class Config:
    def __init__(self):
        self.DOCS_DIR = r"docs"
        self.DB_PATH = r"data\northwind.db"
        self.BatchFile = r"sample_questions_hybrid_eval.jsonl"
        self.OutputFile = r"outputs_hybrid.jsonl"

        self.BASE_URL = r"http://127.0.0.1:11434"
      


config = Config()