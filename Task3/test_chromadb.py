import chromadb
from chromadb.utils import embedding_functions

def init_db():
    # Инициализируем встроенную функцию эмбеддинга
    bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-base-en-v1.5"
    )

    # Создаем клиент с персистентным хранилищем
    client = chromadb.PersistentClient(path="chroma_data/")

    # Создаем коллекцию с BGE моделью
    collection = client.get_or_create_collection(
        name="bge_collection",
        embedding_function=bge_ef,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def save_doc(collection):
    documents = [
        "ChromaDB - это векторная база данных для AI приложений",
        "BGE-Base-v1.5 преобразует текст в 768-мерные эмбеддинги",
        "Семантический поиск находит документы по смыслу, а не по ключевым словам"
    ]

    collection.add(
        documents=documents,
        metadatas=[
            {"source": "doc1"},
            {"source": "doc2"},
            {"source": "doc3"}
        ],
        ids=["id1", "id2", "id3"]
    )

def search(collection):
    results = collection.query(
        query_texts=["как работают эмбеддинги?"],
        n_results=2
    )

    for doc, distance in zip(results["documents"][0], results["distances"][0]):
        print(f"Документ: {doc}")
        print(f"Расстояние: {distance:.4f}\n")

def main():
    collection = init_db()
    # save_doc(collection)
    # search(collection)

    results = collection.get(
        ids=["id1"],
        include=["embeddings", "documents", "metadatas"]  # Добавьте это
    )
    print(results)

if __name__ == '__main__':
    main()