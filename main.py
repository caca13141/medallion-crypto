from src.config import Config
from src.core.engine import Engine

def main():
    Config.validate()
    engine = Engine()
    engine.run()

if __name__ == "__main__":
    main()
