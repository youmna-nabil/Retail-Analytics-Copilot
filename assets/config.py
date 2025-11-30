class Config:
    def __init__(self):
        self.DOCS_DIR = r"docs"
        self.DB_PATH = r"data\northwind.sqlite"
        self.BatchFile = r"sample_questions_hybrid_eval.jsonl"
        self.OutputFile = r"outputs_hybrid.jsonl"

        self.BASE_URL = "http://127.0.0.1:11435"
      


config = Config()