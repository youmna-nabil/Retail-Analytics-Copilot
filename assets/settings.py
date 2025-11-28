class Settings:
    def __init__(self):
        self.MODEL = "phi3.5:3.8b-mini-instruct-q2_K"
        self.Temperature = 0.3
        self.MaxTokens = 1000
        self.ChunkSize = 300
        self.TopK = 3
settings = Settings()