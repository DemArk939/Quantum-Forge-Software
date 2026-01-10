import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Any

# Импорты LangChain
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

# Загрузка промпта из файла
PROMPT_FILE = "Task7/prompt.txt"

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
            "ignore all instructions", "ignore the above", "disregard", "output:",
            "execute:", "run:", "command:", "password:", "secret:", "api_key:",
            "api key:", "token:", "credential:", "bypass", "override", "disable",
            "don't follow", "stop following", "forget about", "switch mode", "jailbreak"
        ]

        filtered_docs = []
        for doc in docs:
            content_lower = doc.page_content.lower()
            # Проверяем, содержит ли документ опасные паттерны
            is_dangerous = any(pattern in content_lower for pattern in dangerous_patterns)

            if is_dangerous:
                print(f"⚠️ Документ заблокирован (обнаружена попытка injection):")
                print(f" Текст: {doc.page_content[:80]}...")
                continue # Пропускаем опасный документ

            filtered_docs.append(doc)

        # Если все документы заблокированы, возвращаем пустой список
        if not filtered_docs:
            print("⚠️ Все найденные документы содержат подозрительный контент!")

        return filtered_docs

    def get_relevant_documents(self, query: str) -> list:
        """Публичный метод для получения релевантных документов"""
        return self._get_relevant_documents(query)

def initialize_rag_chain(api_key, folder_id, persist_dir="chroma_db", db_name="star_wars_chunks"):
    """Инициализация RAG цепочки (LangChain)"""
    print("=" * 70)
    print("Инициализация RAG системы (LangChain)")
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
        print(f" Сначала создайте индекс с помощью ChunkingSystem\n")
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

class RAGTestLogger:
    """Логгер для автоматизированного тестирования RAG"""

    def __init__(self):
        self.results = []
        self.test_id = 0

        # JSON файл для программной обработки
        self.json_file = "Task7/test_results.json"

    def log_test(self, query: str, rag_chain, retriever):
        """Логирование одного теста"""
        self.test_id += 1
        timestamp = datetime.now().isoformat()

        result = {
            "TestID": self.test_id,
            "Timestamp": timestamp,
            "Query": query,
            "QueryLen": len(query),
            "ChunksFound": "No",
            "ChunkCount": 0,
            "ResponseLen": 0,
            "KeyTermsFound": "0/0",
            "Success": "FAIL",
            "Quality": "",
            "Sources": "",
            "Error": "",
            "Response": ""  # Для JSON
        }

        try:
            # Поиск чанков
            source_docs = retriever.get_relevant_documents(query)
            result["ChunksFound"] = "Yes" if source_docs else "No"
            result["ChunkCount"] = len(source_docs)

            if source_docs:
                sources = [f"[{i+1}] {doc.metadata.get('similarity', 'N/A')}"
                           for i, doc in enumerate(source_docs)]
                result["Sources"] = "; ".join(sources[:3])  # Первые 3 источника

            # Генерация ответа
            response = rag_chain.invoke(query)
            result["Response"] = response
            result["ResponseLen"] = len(response)

            # ✅ Оценка качества ответа
            quality, success = self.evaluate_response(response, source_docs)
            result["Quality"] = quality
            result["Success"] = "PASS" if success else "FAIL"

        except Exception as e:
            result["Error"] = str(e)
            result["Success"] = "ERROR"

        # Сохранение полного результата в JSON
        self.results.append(result)
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        # Консольный вывод
        status = "✅ PASS" if result["Success"] == "PASS" else "❌ FAIL"
        print(f"   {status} | {result['ResponseLen']} симв. | Чанки: {result['ChunkCount']}")

    def evaluate_response(self, response: str, docs: List) -> tuple:
        """Оценка качества ответа по длине и содержанию"""
        response_lower = response.lower()

        # Плохие индикаторы (неопределенность)
        bad_indicators = ["не знаю", "нет информации", "не найдено", "i don't know", "неизвестно", "no information", "do not contain any information", "not provide information", "не нашёл подтверждений"]
        if any(ind in response_lower for ind in bad_indicators):
            return "❌ No Info", False

        # Оценка по длине
        if len(response) < 30:
            return "❌ Too Short", False
        elif len(response) < 100:
            return "⚠️ Short", len(docs) > 0
        elif len(response) < 300:
            return "✅ Good", True
        else:
            return "✅ Excellent", True

    def print_summary(self):
        """Итоговая статистика тестов"""
        if not self.results:
            print("Нет результатов для анализа")
            return

        total = len(self.results)
        success = sum(1 for r in self.results if r["Success"] == "PASS")
        errors = sum(1 for r in self.results if r["Success"] == "ERROR")
        avg_chunks = sum(int(r["ChunkCount"]) for r in self.results) / total
        avg_response_len = sum(r["ResponseLen"] for r in self.results) / total

        print(f"\n{'='*80}")
        print("📊 ИТОГОВАЯ СТАТИСТИКА RAG ТЕСТОВ")
        print(f"{'='*80}")
        print(f"Всего тестов:          {total}")
        print(f"✅ Успешных:           {success}/{total} ({success/total*100:.1f}%)")
        print(f"❌ Ошибок:             {errors}")
        print(f"📚 Среднее чанков:     {avg_chunks:.1f}")
        print(f"💬 Средняя длина ответа:{avg_response_len:.0f} симв.")
        print(f"💾 JSON отчет:         {self.json_file}")
        print(f"{'='*80}")

def run_automated_tests(rag_chain, retriever, logger):
    """Автоматизированный запуск тестов из файла golden_questions.json"""

    # ✅ Загрузка вопросов из файла
    try:
        with open("Task7/golden_questions.json", "r", encoding="utf-8") as f:
            test_questions = json.load(f)
        print(f"✅ Загружено {len(test_questions)} вопросов из golden_questions.json")
    except FileNotFoundError:
        print("❌ Файл golden_questions.json не найден!")
        print("💡 Создайте файл с массивом вопросов в JSON формате")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Ошибка в формате JSON: {e}")
        return

    print(f"\n{'='*80}")
    print("🚀 АВТОМАТИЗИРОВАННЫЕ RAG ТЕСТЫ (Star Wars Saga)")
    print(f"{'='*80}\n")

    for i, question in enumerate(test_questions, 1):
        print(f"[{i:2d}/{len(test_questions)}] {question[:60]}{'...' if len(question)>60 else ''}")
        logger.log_test(question, rag_chain, retriever)

    print()  # Пустая строка перед статистикой
    logger.print_summary()

def process_single_query(query, rag_chain, retriever, logger):
    """Обработка одиночного запроса с логированием"""
    print("\n" + "=" * 70)
    print(f"📖 Вопрос: {query}\n")

    print("🔍 Ищу релевантные документы...")
    print("🤖 Генерирую ответ...\n")

    logger.log_test(query, rag_chain, retriever)
    logger.print_summary()

def main():
    api_key = os.getenv("YANDEX_API_KEY")
    folder_id = os.getenv("YANDEX_FOLDER_ID")

    if not api_key or not folder_id:
        print("❌ Ошибка: установите YANDEX_API_KEY и YANDEX_FOLDER_ID в .env")
        sys.exit(1)

    rag = initialize_rag_chain(api_key, folder_id)
    rag_chain = rag["chain"]
    retriever = rag["retriever"]

    logger = RAGTestLogger()

    print("=" * 70)
    print("RAG Тестировщик: ChromaDB + YandexGPT + Автотесты")
    print("=" * 70 + "\n")

    #Различные режимы запуска
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--test":
            run_automated_tests(rag_chain, retriever, logger)
        else:
            # Одиночный запрос из аргументов
            query = " ".join(sys.argv[1:])
            process_single_query(query, rag_chain, retriever, logger)
    else:
        # Интерактивный режим
        print("Режимы:")
        print("  --test          - автотесты из golden_questions.json")
        print("  'вопрос'        - одиночный тест")
        print("  'exit'          - выход\n")

        while True:
            query = input("❓ Вопрос/команда: ").strip()
            if not query:
                continue
            if query.lower() == "exit":
                print("\n👋 До встречи!")
                break
            if query == "--test":
                run_automated_tests(rag_chain, retriever, logger)
                continue
            # Одиночный тест
            process_single_query(query, rag_chain, retriever, logger)

if __name__ == "__main__":
    main()
