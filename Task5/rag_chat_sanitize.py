import sys
import os
from dotenv import load_dotenv
from langchain_community.llms.yandex import YandexGPT
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# ✅ Загрузка промпта из файла
# PROMPT_FILE = "Task5/prompt.txt"
PROMPT_FILE = "Task5/prompt_upd.txt"

def load_prompt():
    """Загружает промпт из файла prompt.txt"""
    try:
        with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
            template = f.read()
        print(f"✅ Промпт загружен из файла '{PROMPT_FILE}'")
        return template
    except FileNotFoundError:
        print(f"❌ Файл '{PROMPT_FILE}' не найден!")
        print("💡 Создайте файл prompt.txt с вашим промптом")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Ошибка чтения промпта: {e}")
        sys.exit(1)

class ChromaDBRetriever(BaseRetriever):
    """Ретривер ChromaDB с BGE-Base-v1.5"""

    # Определение поля для Pydantic
    collection: object = Field(description="ChromaDB collection")

    # Конфиг для Pydantic, разрешающий arbitrary types
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str) -> list:
        """Поиск релевантных документов с санитизацией"""
        results = self.collection.query(
            query_texts=[query],
            n_results=3
        )

        documents = []
        if results['documents'] and results['documents'][0]:
            for i, doc_text in enumerate(results['documents'][0]):
                distance = results['distances'][0][i]
                similarity = 1 - distance

                doc = Document(
                    page_content=doc_text,
                    metadata={
                        "similarity": f"{similarity:.1%}",
                        "distance": f"{distance:.4f}"
                    }
                )
                documents.append(doc)

        # ✅ Санитизируем найденные документы
        documents = self.sanitize_documents(documents)

        return documents

    def sanitize_documents(self, docs: list) -> list:
        """
        Удаляет потенциально опасный контент из документов.
        Защита от prompt injection атак.
        """
        dangerous_patterns = [
            "ignore all instructions",
            "ignore the above",
            "disregard",
            "output:",
            "execute:",
            "run:",
            "command:",
            "password:",
            "secret:",
            "api_key:",
            "api key:",
            "token:",
            "credential:",
            "bypass",
            "override",
            "disable",
            "don't follow",
            "stop following",
            "forget about",
            "switch mode",
            "jailbreak"
        ]

        filtered_docs = []

        for doc in docs:
            content_lower = doc.page_content.lower()

            # Проверяем, содержит ли документ опасные паттерны
            is_dangerous = any(pattern in content_lower for pattern in dangerous_patterns)

            if is_dangerous:
                print(f"⚠️  Документ заблокирован (обнаружена попытка injection):")
                print(f"   Текст: {doc.page_content[:80]}...")
                continue  # Пропускаем опасный документ

            filtered_docs.append(doc)

        # Если все документы заблокированы, возвращаем пустой список
        if not filtered_docs:
            print("⚠️  Все найденные документы содержат подозрительный контент!")

        return filtered_docs


    def get_relevant_documents(self, query: str) -> list:
        """Публичный метод для получения релевантных документов"""
        return self._get_relevant_documents(query)

def initialize_rag_chain(api_key, folder_id, persist_dir="chroma_db", db_name="star_wars_chunks"):
    """Инициализация RAG цепочки (LangChain 1.2.3)"""

    print("=" * 70)
    print("Инициализация RAG системы (LangChain 1.2.3)")
    print("=" * 70 + "\n")

    print("📄 Загружаю промпт...")
    prompt_template = load_prompt()

    print("📦 Загружаю ChromaDB индекс...")
    client = chromadb.PersistentClient(path=persist_dir)

    try:
        collection = client.get_collection(name=db_name)
        print(f"✅ Индекс загружен: {collection.count()} документов")
        print(f"✅ Используется embedding function: {collection.metadata.get('embedding_function', 'default')}\n")
    except Exception as e:
        print(f"❌ Ошибка при загрузке индекса: {e}")
        print(f"\n💡 Подсказка: Возможно, коллекция '{db_name}' не существует.")
        print(f"   Сначала создайте индекс с помощью ChunkingSystem\n")
        sys.exit(1)

    print("🤖 Загружаю YandexGPT...")
    llm = YandexGPT(
        api_key=api_key,
        folder_id=folder_id
    )
    print("✅ YandexGPT готов\n")

    # Создание ретривера
    retriever = ChromaDBRetriever(collection=collection)

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Функция для форматирования документов
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Создание RAG цепочки через Runnable API
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return {"chain": rag_chain, "retriever": retriever}

def process_query(query, rag_chain, retriever):
    """Обработка одного запроса"""
    print("\n" + "=" * 70)
    print(f"📖 Вопрос: {query}\n")

    try:
        print("🔍 Ищу релевантные документы (используя BGE-Base-v1.5)...")
        print("🤖 Генерирую ответ...\n")

        source_docs = retriever.get_relevant_documents(query)
        result = rag_chain.invoke(query)

        print("💬 Ответ YandexGPT:")
        print(result)

        if source_docs:
            print("\n📚 Найденные документы (отранжированы BGE-Base-v1.5):")
            for i, doc in enumerate(source_docs, 1):
                similarity = doc.metadata.get("similarity", "N/A")
                print(f"\n[Документ {i}] (релевантность: {similarity})")
                print(f"Текст: {doc.page_content[:150]}...")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)

def main():
    api_key = os.getenv("YANDEX_API_KEY")
    folder_id = os.getenv("YANDEX_FOLDER_ID")

    if not api_key or not folder_id:
        print("❌ Ошибка: установите YANDEX_API_KEY и YANDEX_FOLDER_ID в .env")
        sys.exit(1)

    rag = initialize_rag_chain(api_key, folder_id)
    rag_chain = rag["chain"]
    retriever = rag["retriever"]

    print("=" * 70)
    print("RAG Система: ChromaDB + BGE-Base-v1.5 + YandexGPT (LangChain 1.2.3)")
    print("=" * 70 + "\n")

    # Если запрос передан как аргумент
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        process_query(query, rag_chain, retriever)
    else:
        # Интерактивный режим
        print("Введите 'exit' для выхода\n")

        while True:
            query = input("❓ Вопрос: ").strip()

            if not query:
                continue

            if query.lower() == "exit":
                print("\n👋 До встречи!")
                break

            # Обработка запроса
            process_query(query, rag_chain, retriever)

if __name__ == "__main__":
    main()