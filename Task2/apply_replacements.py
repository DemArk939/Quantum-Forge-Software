#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для применения замен терминов из terms_map.json к текстовым файлам
"""

import json
import re
from pathlib import Path
from typing import Dict, List
import os


class TextReplacer:
    """Класс для замены терминов в текстах на основе словаря"""

    def __init__(self, terms_map_path: str = 'terms_map.json'):
        self.terms_map = self.load_terms_map(terms_map_path)
        self.terms_map_path = terms_map_path

    def load_terms_map(self, path: str) -> Dict[str, str]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                terms_map = json.load(f)
            print(f"✅ Загружен словарь с {len(terms_map)} заменами из {path}\n")
            return terms_map
        except FileNotFoundError:
            print(f"❌ Файл {path} не найден")
            return {}
        except json.JSONDecodeError:
            print(f"❌ Ошибка при парсинге JSON файла {path}")
            return {}

    def apply_replacements(self, text: str) -> tuple:
        if not self.terms_map:
            return text, 0

        sorted_terms = sorted(self.terms_map.items(), key=lambda x: len(x[0]), reverse=True)
        replacements_count = 0

        for original, replacement in sorted_terms:
            pattern = r'\b' + re.escape(original) + r'\b'
            matches = len(re.findall(pattern, text, flags=re.IGNORECASE))
            replacements_count += matches
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text, replacements_count

    def process_single_file(self, input_file: Path, output_file: Path) -> tuple:
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                original_text = f.read()

            replaced_text, replacements_count = self.apply_replacements(original_text)

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(replaced_text)

            return True, replacements_count
        except Exception as e:
            print(f"      ❌ Ошибка обработки: {e}")
            return False, 0

    def process_files(self, source_dir: str, output_dir: str) -> Dict[str, any]:
        source_path = Path(source_dir)
        output_path = Path(output_dir)

        output_path.mkdir(parents=True, exist_ok=True)

        if not source_path.exists():
            print(f"❌ Директория {source_dir} не найдена")
            return None

        txt_files = sorted(source_path.glob('*.txt'))

        if not txt_files:
            print(f"⚠️  В {source_dir} не найдены .txt файлы")
            return None

        stats = {
            'total_files': len(txt_files),
            'successful': 0,
            'failed': 0,
            'total_replacements': 0,
            'files_info': []
        }

        print(f"\n{'='*70}")
        print(f"🔄 ПРИМЕНЕНИЕ ЗАМЕН К ФАЙЛАМ")
        print(f"{'='*70}\n")
        print(f"📁 Источник: {source_dir}")
        print(f"📁 Результат: {output_dir}")
        print(f"📊 Файлов к обработке: {len(txt_files)}\n")
        print(f"{'='*70}\n")

        for idx, input_file in enumerate(txt_files, 1):
            output_file = output_path / input_file.name
            print(f"[{idx}/{len(txt_files)}] {input_file.name}")

            success, replacements_count = self.process_single_file(input_file, output_file)

            if success:
                stats['successful'] += 1
                stats['total_replacements'] += replacements_count
                file_size = input_file.stat().st_size
                print(f"      ✅ Выполнено | Замен: {replacements_count} | Размер: {file_size} байт")

                stats['files_info'].append({
                    'filename': input_file.name,
                    'status': 'success',
                    'replacements': replacements_count,
                    'size': file_size,
                    'output_path': str(output_file)
                })
            else:
                stats['failed'] += 1
                stats['files_info'].append({
                    'filename': input_file.name,
                    'status': 'failed',
                    'replacements': 0
                })

        print(f"\n{'='*70}")
        print("✅ ОБРАБОТКА ЗАВЕРШЕНА")
        print(f"{'='*70}\n")

        print(f"📊 СТАТИСТИКА:")
        print(f"   ✅ Успешно обработано: {stats['successful']}/{stats['total_files']}")
        print(f"   ❌ Ошибок: {stats['failed']}/{stats['total_files']}")
        print(f"   🔄 Всего замен выполнено: {stats['total_replacements']}")
        print(f"   📁 Результаты сохранены в: {output_dir}\n")

        self.save_report(output_path, stats)

        return stats

    def save_report(self, output_dir: Path, stats: Dict) -> None:
        report = {
            'total_files': stats['total_files'],
            'successful': stats['successful'],
            'failed': stats['failed'],
            'total_replacements': stats['total_replacements'],
            'replacement_dict_used': self.terms_map_path,
            'replacement_count': len(self.terms_map),
            'files': stats['files_info']
        }

        report_path = output_dir / 'replacement_report.json'

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📄 Отчет сохранен: {report_path}\n")
        except Exception as e:
            print(f"⚠️  Не удалось сохранить отчет: {e}\n")

    def process_single_text(self, text: str) -> str:
        replaced_text, _ = self.apply_replacements(text)
        return replaced_text


def main():
    source_dir = 'source'
    output_dir = 'knowledge_base'
    terms_map_file = 'terms_map.json'

    if not Path(terms_map_file).exists():
        print(f"❌ Файл {terms_map_file} не найден")
        return

    if not Path(source_dir).exists():
        print(f"❌ Директория {source_dir} не найдена")
        return

    replacer = TextReplacer(terms_map_file)

    if not replacer.terms_map:
        print("❌ Не удалось загрузить словарь замен")
        return

    stats = replacer.process_files(source_dir, output_dir)

    if stats:
        output_path = Path(output_dir)
        result_files = list(output_path.glob('*.txt'))

        if result_files:
            print(f"{'='*70}")
            print("🎉 УСПЕШНО!")
            print(f"{'='*70}\n")
            print(f"💾 Обработано файлов: {len(result_files)}")
            print(f"📂 Результаты в: {output_dir}")
            print(f"📊 Всего замен: {stats['total_replacements']}\n")

            print("📋 Обработанные файлы:")
            for f in result_files[:5]:
                size = f.stat().st_size
                print(f"   ✅ {f.name} ({size} байт)")

            if len(result_files) > 5:
                print(f"   ... и еще {len(result_files) - 5} файлов")

            print()


if __name__ == '__main__':
    main()
