from chunking_system import ChunkingSystem

# Создаем систему
system = ChunkingSystem(
    chunk_size=512,
    chunk_overlap=100,
    model_name="BAAI/bge-base-en-v1.5",
    db_name="star_wars_chunks",
    input_dir="knowledge_base",
    persist_dir="chroma_db"
)

# Выполняем поиск
results = system.query("Jedi training", n_results=5)

for result in results:
    print(f"Релевантность: {result['relevance']:.2%}")
    print(f"Источник: {result['source']}")
    print(f"Текст: {result['content']}\n")
# Выполняем поиск
results = system.query("Empire and rebellion", n_results=5)

for result in results:
    print(f"Релевантность: {result['relevance']:.2%}")
    print(f"Источник: {result['source']}")
    print(f"Текст: {result['content']}\n")
