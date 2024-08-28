import dotenv
import os


class AppConfig:
    def __init__(self):
        dotenv.load_dotenv()

    def get(self, key):
        return os.getenv(key)

    def get_all(self):
        return os.environ

    def set(self, key, value):
        os.environ[key] = value

    def delete(self, key):
        del os.environ[key]
