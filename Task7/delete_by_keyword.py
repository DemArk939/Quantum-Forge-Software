#!/usr/bin/env python3
"""
Простой скрипт для удаления документов из ChromaDB по ключевым словам
"""

import os
import sys
import chromadb

PERSIST_DIR = "chroma_db"
DB_NAME = "star_wars_chunks"

def delete_by_keyword(keywords, persist_dir=PERSIST_DIR, db_name=DB_NAME):
    """
    Удаляет документы, содержащие ключевые слова
    
    Args:
        keywords (list): Список ключевых слов для поиска
        persist_dir (str): Путь к ChromaDB
        db_name (str): Имя коллекции
    """
    
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_collection(name=db_name)
        
        # Получаем все документы
        all_data = collection.get()
        documents = all_data['documents']
        ids = all_data['ids']
        
        print(f"📦 Всего документов в БД: {len(documents)}\n")
        
        # Находим документы с ключевыми словами
        docs_to_delete = []
        
        for i, doc in enumerate(documents):
            doc_lower = doc.lower()
            
            for keyword in keywords:
                if keyword.lower() in doc_lower:
                    docs_to_delete.append((ids[i], keyword, doc[:100]))
                    break  # Один документ удаляем один раз
        
        if not docs_to_delete:
            print(f"❌ Документы с ключевыми словами {keywords} не найдены")
            return
        
        print(f"🔍 Найдено документов для удаления: {len(docs_to_delete)}\n")
        
        # Выводим список документов
        for i, (doc_id, keyword, preview) in enumerate(docs_to_delete, 1):
            print(f"{i}. [{keyword}] {preview}...")
        
        # Подтверждение
        confirm = input(f"\n⚠️ Удалить {len(docs_to_delete)} документов? (yes/no): ")
        
        if confirm.lower() != 'yes':
            print("❌ Отменено")
            return
        
        # Удаляем документы
        for doc_id, _, _ in docs_to_delete:
            collection.delete(ids=[doc_id])
        
        print(f"\n✅ Удалено документов: {len(docs_to_delete)}")
        print(f"📦 Осталось документов: {collection.count()}")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)

def main():
    """Главная функция"""
    
    print("=" * 70)
    print("🗑️ УДАЛЕНИЕ ДОКУМЕНТОВ ИЗ CHROMADB ПО КЛЮЧЕВЫМ СЛОВАМ")
    print("=" * 70 + "\n")

    keywords = ["Kael Thorn","Ryn Vaelor","Stellar Hawk"]
    
    print(f"\n🔎 Ищу документы со словами: {keywords}\n")
    
    delete_by_keyword(keywords)

if __name__ == "__main__":
    main()
