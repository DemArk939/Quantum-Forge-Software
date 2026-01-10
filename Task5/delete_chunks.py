#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
УТИЛИТА ДЛЯ УДАЛЕНИЯ ЧАНКОВ ИЗ CHROMADB

Позволяет удалить все чанки определенного файла или по другим критериям
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging
import chromadb

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChromaDBManager:
    """Управление ChromaDB коллекциями и чанками"""

    def __init__(self,
                 persist_dir: str = "./chroma_db",
                 db_name: str = "text_chunks"):
        """
        Инициализирует менеджер

        Args:
            persist_dir: директория хранения ChromaDB
            db_name: название коллекции
        """
        self.persist_dir = persist_dir
        self.db_name = db_name

        logger.info("=" * 80)
        logger.info("📊 CHROMADB MANAGER")
        logger.info("=" * 80)

        try:
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.client.get_or_create_collection(
                name=db_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"✅ Подключено к БД: {persist_dir}")
            logger.info(f"✅ Коллекция: {db_name}\n")
        except Exception as e:
            logger.error(f"❌ Ошибка подключения: {e}")
            raise

    def get_all_chunks_info(self) -> Dict:
        """
        Получает информацию о всех чанках в БД

        Returns:
            Dict с информацией о чанках по источникам
        """
        try:
            # Получаем все чанки из коллекции
            result = self.collection.get()

            chunks_by_source = {}

            if result and result['ids']:
                for chunk_id, metadata in zip(result['ids'], result['metadatas']):
                    source = metadata.get('source', 'unknown')

                    if source not in chunks_by_source:
                        chunks_by_source[source] = []

                    chunks_by_source[source].append({
                        'chunk_id': chunk_id,
                        'chunk_index': metadata.get('chunk_index', 0),
                        'word_count': metadata.get('word_count', 0),
                        'preview': metadata.get('preview', '')
                    })

            return chunks_by_source

        except Exception as e:
            logger.error(f"❌ Ошибка получения информации: {e}")
            return {}

    def print_chunks_info(self):
        """Выводит информацию о чанках в консоль"""
        chunks_info = self.get_all_chunks_info()

        logger.info("=" * 80)
        logger.info("📊 СТАТИСТИКА ЧАНКОВ В БД")
        logger.info("=" * 80)

        total_chunks = 0

        for source, chunks in sorted(chunks_info.items()):
            total_words = sum(c['word_count'] for c in chunks)
            total_chunks += len(chunks)

            logger.info(f"\n📄 {source}")
            logger.info(f"   ✅ Чанков: {len(chunks)}")
            logger.info(f"   ✅ Слов: {total_words:,}")
            logger.info(f"   ✅ Индексы: {chunks[0]['chunk_index']}-{chunks[-1]['chunk_index']}")

        logger.info(f"\n{'=' * 80}")
        logger.info(f"✅ ВСЕГО ЧАНКОВ В БД: {total_chunks}")
        logger.info(f"{'=' * 80}\n")

        return chunks_info

    def delete_chunks_by_source(self, source_filename: str) -> bool:
        """
        Удаляет все чанки определенного файла

        Args:
            source_filename: имя файла (например "Anakin_Skywalker.txt")

        Returns:
            True если успешно, False если ошибка
        """
        logger.info("=" * 80)
        logger.info(f"🗑️  УДАЛЕНИЕ ЧАНКОВ ИЗ: {source_filename}")
        logger.info("=" * 80)

        try:
            # Получаем все чанки
            result = self.collection.get()

            if not result or not result['ids']:
                logger.warning("⚠️  В БД нет чанков")
                return False

            # Находим чанки нужного источника
            ids_to_delete = []

            for chunk_id, metadata in zip(result['ids'], result['metadatas']):
                if metadata.get('source') == source_filename:
                    ids_to_delete.append(chunk_id)

            if not ids_to_delete:
                logger.warning(f"⚠️  Чанки из '{source_filename}' не найдены")
                return False

            # Удаляем чанки
            self.collection.delete(ids=ids_to_delete)

            logger.info(f"\n✅ УСПЕШНО УДАЛЕНО: {len(ids_to_delete)} чанков")
            logger.info(f"   📄 Источник: {source_filename}")
            logger.info("")

            return True

        except Exception as e:
            logger.error(f"❌ Ошибка удаления: {e}")
            return False

    def delete_chunks_by_multiple_sources(self, sources: List[str]) -> Dict[str, bool]:
        """
        Удаляет чанки из нескольких файлов

        Args:
            sources: список имен файлов

        Returns:
            Dict с результатами для каждого источника
        """
        logger.info("=" * 80)
        logger.info(f"🗑️  УДАЛЕНИЕ ЧАНКОВ ИЗ {len(sources)} ФАЙЛОВ")
        logger.info("=" * 80)

        results = {}

        for source in sources:
            success = self.delete_chunks_by_source(source)
            results[source] = success

        logger.info(f"\n✅ РЕЗУЛЬТАТЫ:")
        for source, success in results.items():
            status = "✅ Успешно" if success else "❌ Ошибка/не найдено"
            logger.info(f"   {status}: {source}")

        logger.info("")
        return results

    def delete_all_chunks(self) -> bool:
        """
        Удаляет ВСЕ чанки из коллекции
        (ВНИМАНИЕ: необратимо!)

        Returns:
            True если успешно
        """
        logger.info("=" * 80)
        logger.warning("⚠️  ВНИМАНИЕ: УДАЛЕНИЕ ВСЕХ ЧАНКОВ (НЕОБРАТИМО!)")
        logger.info("=" * 80)

        response = input("\nВы уверены? Введите 'да' для подтверждения: ").strip().lower()

        if response != "да":
            logger.info("❌ Операция отменена")
            return False

        try:
            result = self.collection.get()

            if result and result['ids']:
                self.collection.delete(ids=result['ids'])
                logger.info(f"\n✅ УДАЛЕНО: {len(result['ids'])} чанков\n")
                return True
            else:
                logger.warning("⚠️  В БД нет чанков")
                return False

        except Exception as e:
            logger.error(f"❌ Ошибка удаления: {e}")
            return False

    def delete_and_reindex(self, source_filename: str, chunks_data: List[Dict]) -> bool:
        """
        Удаляет старые чанки источника и добавляет новые
        (для переобработки документа)

        Args:
            source_filename: имя файла
            chunks_data: новые данные чанков с эмбедингами

        Returns:
            True если успешно
        """
        logger.info("=" * 80)
        logger.info(f"🔄 ПЕРЕИНДЕКСИРОВАНИЕ: {source_filename}")
        logger.info("=" * 80)

        # Удаляем старые чанки
        logger.info(f"\n1️⃣  Удаление старых чанков...")
        delete_success = self.delete_chunks_by_source(source_filename)

        if not delete_success:
            logger.warning(f"⚠️  Старых чанков не найдено (OK для первой индексации)")

        # Добавляем новые чанки
        logger.info(f"2️⃣  Добавление новых чанков...")

        try:
            ids = [chunk["chunk_id"] for chunk in chunks_data]
            documents = [chunk["content"] for chunk in chunks_data]
            metadatas = [
                {
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "start_position": chunk["start_position"],
                    "end_position": chunk["end_position"],
                    "word_count": chunk["word_count"],
                    "preview": chunk["preview"]
                }
                for chunk in chunks_data
            ]
            embeddings = [chunk.get("embedding") for chunk in chunks_data]

            # Если нет эмбедингов, нужно их сгенерировать
            if not embeddings[0]:
                logger.warning("⚠️  Эмбединги не предоставлены - используйте generate_embeddings()")
                return False

            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )

            logger.info(f"✅ Добавлено: {len(ids)} новых чанков\n")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка добавления: {e}")
            return False

    def export_collection_to_json(self, output_file: str = "chromadb_export.json") -> bool:
        """
        Экспортирует всю коллекцию в JSON файл (для бэкапа)

        Args:
            output_file: имя выходного файла

        Returns:
            True если успешно
        """
        logger.info(f"\n💾 ЭКСПОРТ В: {output_file}")

        try:
            result = self.collection.get()

            export_data = {
                "collection": self.db_name,
                "timestamp": str(Path(self.persist_dir).stat().st_mtime),
                "total_chunks": len(result['ids']) if result and result['ids'] else 0,
                "data": {
                    "ids": result['ids'] if result else [],
                    "documents": result['documents'] if result else [],
                    "metadatas": result['metadatas'] if result else [],
                }
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            file_size = os.path.getsize(output_file) / 1024 / 1024
            logger.info(f"✅ Экспортировано: {export_data['total_chunks']} чанков")
            logger.info(f"   Размер файла: {file_size:.1f} MB\n")

            return True

        except Exception as e:
            logger.error(f"❌ Ошибка экспорта: {e}")
            return False


def main():
    """Интерактивное меню"""

    manager = ChromaDBManager(
        persist_dir="chroma_db",
        db_name="star_wars_chunks"
    )

    while True:
        logger.info("\n" + "=" * 80)
        logger.info("🛠️  МЕНЮ ОПЕРАЦИЙ")
        logger.info("=" * 80)
        logger.info("1️⃣  - Показать статистику чанков")
        logger.info("2️⃣  - Удалить чанки из одного файла")
        logger.info("3️⃣  - Удалить чанки из нескольких файлов")
        logger.info("4️⃣  - Удалить ВСЕ чанки")
        logger.info("5️⃣  - Экспортировать коллекцию")
        logger.info("6️⃣  - Выход")
        logger.info("=" * 80)

        choice = input("\nВыберите операцию (1-6): ").strip()

        if choice == "1":
            manager.print_chunks_info()

        elif choice == "2":
            chunks_info = manager.get_all_chunks_info()

            if not chunks_info:
                logger.warning("⚠️  В БД нет чанков")
                continue

            logger.info("\n📄 Доступные файлы:")
            for i, source in enumerate(sorted(chunks_info.keys()), 1):
                count = len(chunks_info[source])
                logger.info(f"   {i}. {source} ({count} чанков)")

            filename = input("\nВведите имя файла для удаления: ").strip()

            if filename in chunks_info:
                manager.delete_chunks_by_source(filename)
            else:
                logger.warning(f"⚠️  Файл '{filename}' не найден")

        elif choice == "3":
            chunks_info = manager.get_all_chunks_info()

            if not chunks_info:
                logger.warning("⚠️  В БД нет чанков")
                continue

            logger.info("\n📄 Доступные файлы:")
            for i, source in enumerate(sorted(chunks_info.keys()), 1):
                count = len(chunks_info[source])
                logger.info(f"   {i}. {source} ({count} чанков)")

            filenames = input("\nВведите имена файлов через запятую: ").strip().split(',')
            filenames = [f.strip() for f in filenames]

            manager.delete_chunks_by_multiple_sources(filenames)

        elif choice == "4":
            manager.delete_all_chunks()

        elif choice == "5":
            manager.export_collection_to_json()

        elif choice == "6":
            logger.info("👋 До свидания!")
            break

        else:
            logger.warning("❌ Неверный выбор")


if __name__ == "__main__":
    main()
