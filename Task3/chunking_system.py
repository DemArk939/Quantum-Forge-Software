#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEXT CHUNKING SYSTEM с CHROMADB и BAAI/bge-base-en-v1.5
Совместимо с ChromaDB 1.2.2+ (новый API)

Полный workflow:
1. Загружает текстовые документы из replaced_texts/
2. Разбивает на логические чанки (512 символов с 100 символами перекрытия)
3. Генерирует эмбединги BAAI/bge-base-en-v1.5
4. Сохраняет в ChromaDB с полной метаинформацией
5. Готово для семантического поиска и RAG

Использование:
    pip install langchain langchain-text-splitters chromadb sentence-transformers
    python3 chunking_system_v2.py
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

# LangChain imports - совместимо с разными версиями
try:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    print("❌ LangChain не установлен правильно")
    print("Установите: pip install langchain langchain-text-splitters")
    exit(1)

# ChromaDB imports
try:
    import chromadb
except ImportError:
    print("❌ ChromaDB не установлен")
    print("Установите: pip install chromadb")
    exit(1)

# Sentence Transformers для эмбедингов
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ sentence-transformers не установлен")
    print("Установите: pip install sentence-transformers")
    exit(1)


# Логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChunkingSystem:
    """
    Полная система разбиения текстов на чанки с сохранением в ChromaDB
    Совместимо с ChromaDB 1.2.2+ (новый PersistentClient API)
    """

    def __init__(self,
                 chunk_size: int = 512,
                 chunk_overlap: int = 100,
                 model_name: str = "BAAI/bge-base-en-v1.5",
                 db_name: str = "star_wars_chunks",
                 input_dir: str = "knowledge_base",
                 persist_dir: str = "chroma_db"):
        """
        Инициализирует систему

        Args:
            chunk_size: размер чанка в символах
            chunk_overlap: перекрытие между чанками для контекста
            model_name: название модели для эмбедингов
            db_name: название коллекции в ChromaDB
            input_dir: директория с текстовыми файлами
            persist_dir: директория для сохранения ChromaDB
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.db_name = db_name
        self.input_dir = input_dir
        self.persist_dir = persist_dir

        logger.info("=" * 80)
        logger.info("🚀 TEXT CHUNKING SYSTEM INITIALIZATION")
        logger.info("=" * 80)

        # Инициализируем RecursiveCharacterTextSplitter
        logger.info(f"📦 Инициализирую TextSplitter")
        logger.info(f"   📏 Размер чанка: {chunk_size} символов")
        logger.info(f"   📐 Перекрытие: {chunk_overlap} символов")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

        # Инициализируем модель для эмбедингов
        logger.info(f"\n🤖 Загружаю модель: {model_name}")
        logger.info(f"   (первый запуск может занять 1-2 минуты)")

        try:
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"   ✅ Модель загружена успешно")
            logger.info(f"   📊 Размерность векторов: {self.embedding_model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise

        # Инициализируем ChromaDB (новый API 1.2+)
        logger.info(f"\n💾 Инициализирую ChromaDB 1.2+")
        logger.info(f"   📁 Директория: {persist_dir}")

        os.makedirs(persist_dir, exist_ok=True)

        try:
            self.chroma_client = chromadb.PersistentClient(path=persist_dir)
            logger.info(f"   ✅ Клиент ChromaDB инициализирован")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации ChromaDB: {e}")
            raise

        # Создаем или получаем коллекцию
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name=db_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"   ✅ Коллекция '{db_name}' готова")
        except Exception as e:
            logger.error(f"❌ Ошибка создания коллекции: {e}")
            raise

        logger.info("\n" + "=" * 80)
        logger.info("✅ СИСТЕМА ИНИЦИАЛИЗИРОВАНА")
        logger.info("=" * 80 + "\n")

    def load_documents(self) -> Dict[str, str]:
        """
        Загружает все .txt файлы из input_dir

        Returns:
            Dict с именами файлов и содержимым
        """
        logger.info(f"📂 ЗАГРУЗКА ДОКУМЕНТОВ")
        logger.info(f"📁 Директория: {self.input_dir}\n")

        documents = {}
        input_path = Path(self.input_dir)

        if not input_path.exists():
            logger.error(f"❌ Директория {self.input_dir} не найдена")
            return documents

        txt_files = sorted(list(input_path.glob("*.txt")))

        if not txt_files:
            logger.warning(f"⚠️  В {self.input_dir} не найдены .txt файлы")
            return documents

        logger.info(f"📊 Найдено файлов: {len(txt_files)}\n")

        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                documents[file_path.name] = content

                file_size = file_path.stat().st_size
                logger.info(f"✅ {file_path.name}")
                logger.info(f"   📏 Размер: {len(content)} символов ({file_size} байт)")

            except Exception as e:
                logger.error(f"❌ Ошибка загрузки {file_path.name}: {e}")

        logger.info("")
        return documents

    def chunk_document(self,
                       text: str,
                       source: str) -> List[Dict]:
        """
        Разбивает документ на чанки с полной метаинформацией

        Args:
            text: содержимое документа
            source: имя источника (файла)

        Returns:
            Список чанков с метаинформацией
        """
        chunks_data = []

        # Разбиваем текст на чанки
        chunks = self.text_splitter.split_text(text)

        # Для каждого чанка сохраняем метаинформацию
        current_pos = 0

        for chunk_idx, chunk in enumerate(chunks):
            # Находим позицию чанка в оригинальном тексте
            start_pos = text.find(chunk, current_pos)
            if start_pos == -1:
                start_pos = current_pos

            end_pos = start_pos + len(chunk)
            current_pos = end_pos

            # Вычисляем структурную информацию
            text_before = text[:start_pos]
            paragraph_num = text_before.count('\n\n')
            line_num = text_before.count('\n')

            # Создаем метаданные чанка
            chunk_metadata = {
                "content": chunk,
                "source": source,
                "chunk_id": f"{source}_{chunk_idx:04d}",
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
                "start_position": start_pos,
                "end_position": end_pos,
                "character_count": len(chunk),
                "word_count": len(chunk.split()),
                "paragraph_number": paragraph_num,
                "line_number": line_num,
                "model": self.model_name,
                "embedding_dimension": self.embedding_model.get_sentence_embedding_dimension(),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "created_at": datetime.now().isoformat(),
                "citation": {
                    "source": source,
                    "chunk": chunk_idx + 1,
                    "total_chunks": len(chunks),
                    "position": f"символы {start_pos}-{end_pos}"
                },
                "preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
            }

            chunks_data.append(chunk_metadata)

        return chunks_data

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Генерирует эмбединги для текстов

        Args:
            texts: список текстов

        Returns:
            Список векторов эмбедингов
        """
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False
        )

        # Конвертируем если нужно
        if hasattr(embeddings, 'tolist'):
            embeddings = embeddings.tolist()
        return embeddings

    def save_to_chromadb(self, all_chunks: List[Dict]) -> None:
        """
        Сохраняет чанки в ChromaDB с эмбедингами

        Args:
            all_chunks: список всех чанков с метаинформацией
        """
        logger.info("💾 СОХРАНЕНИЕ В CHROMADB\n")

        batch_size = 32  # Обрабатываем батчами для эффективности
        total_chunks = len(all_chunks)

        for batch_start in range(0, total_chunks, batch_size):
            batch_end = min(batch_start + batch_size, total_chunks)
            batch = all_chunks[batch_start:batch_end]

            # Извлекаем данные для ChromaDB
            ids = [chunk["chunk_id"] for chunk in batch]
            documents = [chunk["content"] for chunk in batch]
            metadatas = [
                {
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "start_position": chunk["start_position"],
                    "end_position": chunk["end_position"],
                    "paragraph_number": chunk["paragraph_number"],
                    "line_number": chunk["line_number"],
                    "word_count": chunk["word_count"],
                    "preview": chunk["preview"]
                }
                for chunk in batch
            ]

            # Генерируем эмбединги
            embeddings = self.generate_embeddings(documents)

            # Сохраняем в ChromaDB
            try:
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )

                logger.info(f"✅ Сохранено {batch_end}/{total_chunks} чанков")

            except Exception as e:
                logger.error(f"❌ Ошибка сохранения батча: {e}")

        logger.info(f"\n✅ ВСЕ {total_chunks} ЧАНКОВ СОХРАНЕНЫ В CHROMADB\n")

    def save_chunks_to_json(self, all_chunks: List[Dict], output_file: str = "chunks.json") -> None:
        """
        Сохраняет чанки в JSON файл для справки

        Args:
            all_chunks: список всех чанков
            output_file: имя выходного файла
        """
        logger.info(f"📄 СОХРАНЕНИЕ ЧАНКОВ В JSON\n")

        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_chunks": len(all_chunks),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "model": self.model_name,
                "embedding_dimension": self.embedding_model.get_sentence_embedding_dimension(),
                "database": self.db_name
            },
            "chunks": all_chunks
        }

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            file_size = os.path.getsize(output_file) / 1024 / 1024
            logger.info(f"✅ Чанки сохранены: {output_file}")
            logger.info(f"   📦 Размер файла: {file_size:.1f} MB\n")

        except Exception as e:
            logger.error(f"❌ Ошибка сохранения JSON: {e}")

    def get_statistics(self, all_chunks: List[Dict]) -> Dict:
        """
        Вычисляет статистику по чанкам

        Args:
            all_chunks: список всех чанков

        Returns:
            Словарь со статистикой
        """
        stats = {
            "total_chunks": len(all_chunks),
            "total_characters": sum(c["character_count"] for c in all_chunks),
            "total_words": sum(c["word_count"] for c in all_chunks),
            "average_chunk_size": sum(c["character_count"] for c in all_chunks) / len(all_chunks) if all_chunks else 0,
            "average_words_per_chunk": sum(c["word_count"] for c in all_chunks) / len(all_chunks) if all_chunks else 0,
            "sources": len(set(c["source"] for c in all_chunks)),
            "unique_sources": sorted(list(set(c["source"] for c in all_chunks)))
        }
        return stats

    def print_statistics(self, all_chunks: List[Dict]) -> None:
        """Выводит статистику в консоль"""
        stats = self.get_statistics(all_chunks)

        logger.info("=" * 80)
        logger.info("📊 СТАТИСТИКА")
        logger.info("=" * 80)
        logger.info(f"✅ Всего чанков: {stats['total_chunks']}")
        logger.info(f"✅ Всего символов: {stats['total_characters']:,}")
        logger.info(f"✅ Всего слов: {stats['total_words']:,}")
        logger.info(f"✅ Средний размер чанка: {stats['average_chunk_size']:.0f} символов")
        logger.info(f"✅ Средняя длина: {stats['average_words_per_chunk']:.0f} слов")
        logger.info(f"✅ Обработано источников: {stats['sources']}")
        logger.info(f"   Источники: {', '.join(stats['unique_sources'])}")
        logger.info("=" * 80 + "\n")

    def run(self) -> Tuple[List[Dict], Dict]:
        """
        Запускает полный workflow

        Returns:
            Кортеж (список всех чанков, статистика)
        """
        try:
            # 1. Загружаем документы
            documents = self.load_documents()

            if not documents:
                logger.error("❌ Не найдены документы для обработки")
                return [], {}

            # 2. Разбиваем на чанки и генерируем эмбединги
            logger.info("🔪 РАЗБИЕНИЕ НА ЧАНКИ И ОБРАБОТКА\n")

            all_chunks = []
            for source, content in documents.items():
                chunks = self.chunk_document(content, source)
                all_chunks.extend(chunks)
                logger.info(f"✅ {source}: {len(chunks)} чанков создано")

            logger.info("")

            # 3. Сохраняем в ChromaDB
            self.save_to_chromadb(all_chunks)

            # 4. Сохраняем в JSON для справки
            self.save_chunks_to_json(all_chunks)

            # 5. Выводим статистику
            self.print_statistics(all_chunks)

            logger.info("✅ ПРОЦЕСС ЗАВЕРШЕН УСПЕШНО!")
            logger.info("=" * 80 + "\n")

            return all_chunks, self.get_statistics(all_chunks)

        except Exception as e:
            logger.error(f"❌ Критическая ошибка: {e}")
            raise

    def query(self, query_text: str, n_results: int = 3) -> List[Dict]:
        """
        Выполняет семантический поиск в ChromaDB

        Args:
            query_text: текст запроса
            n_results: количество результатов

        Returns:
            Список релевантных чанков
        """
        logger.info(f"\n🔍 СЕМАНТИЧЕСКИЙ ПОИСК")
        logger.info(f"Запрос: '{query_text}'")
        logger.info(f"Количество результатов: {n_results}\n")

        try:
            # Генерируем эмбединг запроса той же моделью (BAAI/bge-base-en-v1.5)
            query_embedding = self.embedding_model.encode(query_text, show_progress_bar=False)

            # Конвертируем если нужно
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()

            # Используем query_embeddings вместо query_texts
            results = self.collection.query(
                query_embeddings=[query_embedding],  # ✅ 768 размерность - совместимо!
                n_results=n_results
            )

            if results and results['documents'] and results['documents'][0]:
                logger.info(f"✅ Найдено {len(results['documents'][0])} релевантных чанков:\n")

                formatted_results = []
                for i, (doc, meta, distance) in enumerate(zip(
                        results['documents'][0],
                        results['metadatas'][0],
                        results['distances'][0]
                ), 1):
                    logger.info(f"[{i}] Источник: {meta['source']} (чанк {meta['chunk_index'] + 1})")
                    logger.info(f"    Релевантность: {1 - distance:.2%}")
                    logger.info(f"    Позиция: {meta['start_position']}-{meta['end_position']}")
                    logger.info(f"    Текст: {meta['preview']}\n")

                    formatted_results.append({
                        "rank": i,
                        "source": meta['source'],
                        "chunk_index": meta['chunk_index'],
                        "relevance": 1 - distance,
                        "content": doc,
                        "metadata": meta
                    })

                return formatted_results
            else:
                logger.info("⚠️  Релевантные чанки не найдены")
                return []

        except Exception as e:
            logger.error(f"❌ Ошибка поиска: {e}")
            return []


def main():
    """Главная функция"""

    # Создаем систему с настройками по умолчанию
    system = ChunkingSystem(
        chunk_size=512,
        chunk_overlap=100,
        model_name="BAAI/bge-base-en-v1.5",
        db_name="star_wars_chunks",
        input_dir="knowledge_base",
        persist_dir="chroma_db"
    )

    # Запускаем полный workflow
    chunks, stats = system.run()

    # Пример: выполняем поиск
    if chunks:
        system.query("Jedi and lightsaber", n_results=3)
        system.query("Empire and rebellion", n_results=3)

    # system.query("Jedi and lightsaber", n_results=3)
    # system.query("Empire and rebellion", n_results=3)


if __name__ == "__main__":
    main()