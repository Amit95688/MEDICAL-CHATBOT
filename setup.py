import os

FOLDERS = [
    "app",
    "app/api",
    "app/rag",
    "app/utils",
    "data/raw",
    "data/processed",
    "models",
    "tests",
    "docker"
]

FILES = [
    "app/__init__.py",
    "app/main.py",
    "app/config.py",
    "app/api/__init__.py",
    "app/api/routes.py",
    "app/rag/__init__.py",
    "app/rag/loader.py",
    "app/rag/embeddings.py",
    "app/rag/retriever.py",
    "app/rag/generator.py",
    "app/utils/__init__.py",
    "app/utils/helpers.py",
    "tests/test_rag.py",
    "docker/Dockerfile",
    "requirements.txt",
    ".env.example",
    "README.md"
]


def create_structure():
    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)

    for file in FILES:
        if not os.path.exists(file):
            with open(file, "w") as f:
                if file.endswith(".py"):
                    f.write("# Auto-generated file\n")
                elif file == "README.md":
                    f.write("# GenAI Healthcare RAG System\n")
                elif file == "requirements.txt":
                    f.write(
                        "flask\nlangchain\nsentence-transformers\nqdrant-client\npython-dotenv\n"
                    )

    print("✅ Project structure created ")


if __name__ == "__main__":
    create_structure()
