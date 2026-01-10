from chunking_system import ChunkingSystem

# Создаем систему
system = ChunkingSystem()

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
