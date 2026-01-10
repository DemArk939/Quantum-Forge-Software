#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Функциональность:
1. Сканирует папку knowledge_base на новые/изменённые .txt файлы
2. Отслеживает хеши файлов для определения изменений
3. Инкрементально добавляет новые чанки в ChromaDB
4. Удаляет чанки удалённых файлов

Использование:
python3 file_monitor.py
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import logging

# LangChain imports
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
    from chromadb.utils import embedding_functions
except ImportError:
    print("❌ ChromaDB не установлен")
    print("Установите: pip install chromadb")
    exit(1)

# Sentence Transformers
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


class SimpleFileMonitor:
    """Простая система отслеживания файлов и инкрементального обновления ChromaDB"""

    def __init__(self,
                 chunk_size: int = 512,
                 chunk_overlap: int = 100,
                 model_name: str = "BAAI/bge-base-en-v1.5",
                 db_name: str = "star_wars_chunks",
                 knowledge_base_dir: str = "knowledge_base",
                 persist_dir: str = "chroma_db",
                 state_file: str = "file_state.json"):
        """Инициализирует систему"""

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.db_name = db_name
        self.knowledge_base_dir = knowledge_base_dir
        self.persist_dir = persist_dir
        self.state_file = state_file

        logger.info("=" * 80)
        logger.info("🚀 FILE MONITOR INITIALIZATION")
        logger.info("=" * 80)

        # Загружаем состояние файлов
        self.file_state = self._load_state()

        # Инициализируем TextSplitter
        logger.info(f"📦 TextSplitter: chunk_size={chunk_size}, overlap={chunk_overlap}")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

        # Загружаем модель эмбедингов
        logger.info(f"🤖 Загружаю модель: {model_name}")
        try:
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f" ✅ Модель загружена")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise

        # Инициализируем ChromaDB
        logger.info(f"💾 Инициализирую ChromaDB")
        os.makedirs(persist_dir, exist_ok=True)

        try:
            self.chroma_client = chromadb.PersistentClient(path=persist_dir)
            bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name=db_name,
                embedding_function=bge_ef,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f" ✅ ChromaDB готов")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации ChromaDB: {e}")
            raise

        logger.info("=" * 80 + "\n")

    def _load_state(self) -> Dict:
        """Загружает состояние файлов из файла"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Ошибка загрузки состояния: {e}")
                return {"files": {}, "indexed_chunks": {}}
        return {"files": {}, "indexed_chunks": {}}

    def _save_state(self) -> None:
        """Сохраняет состояние в файл"""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.file_state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения состояния: {e}")

    def _get_file_hash(self, file_path: Path) -> str:
        """Вычисляет SHA-256 хеш файла"""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"❌ Ошибка хеша {file_path.name}: {e}")
            return ""

    def _is_file_changed(self, file_name: str, file_path: Path) -> bool:
        """Проверяет, изменился ли файл"""
        current_hash = self._get_file_hash(file_path)
        if not current_hash:
            return False

        stored_hash = self.file_state["files"].get(file_name, {}).get("hash")
        return current_hash != stored_hash

    def _chunk_document(self, text: str, source: str) -> List[Dict]:
        """Разбивает документ на чанки"""
        chunks_data = []
        chunks = self.text_splitter.split_text(text)

        current_pos = 0
        for chunk_idx, chunk in enumerate(chunks):
            start_pos = text.find(chunk, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            end_pos = start_pos + len(chunk)
            current_pos = end_pos

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
                "created_at": datetime.now().isoformat(),
                "preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
            }
            chunks_data.append(chunk_metadata)

        return chunks_data

    def _save_to_chromadb(self, chunks: List[Dict]) -> int:
        """Сохраняет чанки в ChromaDB"""
        if not chunks:
            return 0

        batch_size = 32
        total_saved = 0

        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch = chunks[batch_start:batch_end]

            ids = [chunk["chunk_id"] for chunk in batch]
            documents = [chunk["content"] for chunk in batch]
            metadatas = [
                {
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "start_position": chunk["start_position"],
                    "end_position": chunk["end_position"],
                    "word_count": chunk["word_count"],
                    "preview": chunk["preview"]
                }
                for chunk in batch
            ]

            # Генерируем эмбединги
            embeddings = self.embedding_model.encode(documents, show_progress_bar=False)
            if hasattr(embeddings, 'tolist'):
                embeddings = embeddings.tolist()

            # Сохраняем в ChromaDB
            try:
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
                total_saved += len(batch)
            except Exception as e:
                logger.error(f"❌ Ошибка сохранения батча: {e}")

        return total_saved

    def _delete_from_chromadb(self, chunk_ids: List[str]) -> int:
        """Удаляет чанки из ChromaDB"""
        if not chunk_ids:
            return 0

        deleted = 0
        batch_size = 32

        for batch_start in range(0, len(chunk_ids), batch_size):
            batch_end = min(batch_start + batch_size, len(chunk_ids))
            batch_ids = chunk_ids[batch_start:batch_end]

            try:
                self.collection.delete(ids=batch_ids)
                deleted += len(batch_ids)
            except Exception as e:
                logger.error(f"❌ Ошибка удаления батча: {e}")

        return deleted

    def _process_file(self, file_name: str, file_path: Path) -> tuple:
        """Обрабатывает файл и возвращает (добавлено, удалено)"""
        logger.info(f"\n📄 {file_name}")

        # Удаляем старые чанки
        old_chunk_ids = self.file_state["indexed_chunks"].get(file_name, [])
        deleted = 0
        if old_chunk_ids:
            deleted = self._delete_from_chromadb(old_chunk_ids)

        # Загружаем и разбиваем
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"  ❌ Ошибка загрузки: {e}")
            return 0, deleted

        chunks = self._chunk_document(content, file_name)
        logger.info(f"  🔪 Чанков: {len(chunks)}")

        # Сохраняем в БД
        added = self._save_to_chromadb(chunks)
        logger.info(f"  ✅ Добавлено: {added}, Удалено: {deleted}")

        # Обновляем состояние
        current_hash = self._get_file_hash(file_path)
        self.file_state["files"][file_name] = {
            "hash": current_hash,
            "indexed_at": datetime.now().isoformat(),
            "chunk_count": len(chunks)
        }
        self.file_state["indexed_chunks"][file_name] = [c["chunk_id"] for c in chunks]

        return added, deleted

    def scan(self) -> None:
        """Сканирует папку и обновляет БД"""
        logger.info("=" * 80)
        logger.info("🔍 СКАНИРОВАНИЕ НА ИЗМЕНЕНИЯ")
        logger.info("=" * 80)

        # Получаем файлы на диске
        kb_path = Path(self.knowledge_base_dir)
        if not kb_path.exists():
            logger.warning(f"⚠️ Папка {self.knowledge_base_dir} не найдена")
            return

        current_files = {f.name: f for f in sorted(kb_path.glob("*.txt"))}
        tracked_files = set(self.file_state["files"].keys())

        new_count = 0
        modified_count = 0
        deleted_count = 0
        total_added = 0
        total_deleted = 0

        # Новые и изменённые файлы
        for file_name, file_path in current_files.items():
            if file_name not in tracked_files:
                logger.info(f"\n🆕 НОВЫЙ: {file_name}")
                new_count += 1
                added, deleted = self._process_file(file_name, file_path)
                total_added += added
                total_deleted += deleted
            elif self._is_file_changed(file_name, file_path):
                logger.info(f"\n📝 ИЗМЕНЁН: {file_name}")
                modified_count += 1
                added, deleted = self._process_file(file_name, file_path)
                total_added += added
                total_deleted += deleted
            else:
                logger.info(f"\n⏸️ БЕЗ ИЗМЕНЕНИЙ: {file_name}")

        # Удалённые файлы
        for file_name in tracked_files:
            if file_name not in current_files:
                logger.info(f"\n🗑️ УДАЛЁН: {file_name}")
                deleted_count += 1
                chunk_ids = self.file_state["indexed_chunks"].pop(file_name, [])
                deleted = self._delete_from_chromadb(chunk_ids)
                total_deleted += deleted
                self.file_state["files"].pop(file_name, None)

        # Сохраняем состояние
        self._save_state()

        # Итоги
        logger.info("\n" + "=" * 80)
        logger.info("📊 ИТОГИ")
        logger.info("=" * 80)
        logger.info(f"📁 Новых: {new_count}")
        logger.info(f"📝 Изменённых: {modified_count}")
        logger.info(f"🗑️ Удалённых: {deleted_count}")
        logger.info(f"➕ Добавлено чанков: {total_added}")
        logger.info(f"➖ Удалено чанков: {total_deleted}")
        logger.info("=" * 80 + "\n")


def main():
    """Главная функция"""
    monitor = SimpleFileMonitor(
        chunk_size=512,
        chunk_overlap=100,
        model_name="BAAI/bge-base-en-v1.5",
        knowledge_base_dir="knowledge_base",
        persist_dir="chroma_db"
    )

    monitor.scan()


if __name__ == "__main__":
    main()