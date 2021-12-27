import json

class PMSConfig:
    def __init__(self, path):
        self.path = path
        with open(path, "r") as f:
            self.config = json.load(f)
        for k, v in self.config.items():
            setattr(self, k, v)

    def clone(self):
        return PMSConfig(self.path)