class Settings:
    def __init__(self):
        self.MODEL = "qwen2.5:7b"
        self.Temperature = 0.3
        self.MaxTokens = 1000
        self.ChunkSize = 300
        self.TopK = 3
settings = Settings()