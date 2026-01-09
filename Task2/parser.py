#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для скачивания страниц и извлечения чистого текста с правильной структурой
Оптимизирован для Fandom/Wiki сайтов (Wikipedia, Fandom и т.д.)
Чтение URL только из файла urls.txt
"""

import requests
from bs4 import BeautifulSoup
import os
import re
from pathlib import Path
from typing import List, Optional, Dict
import json
import time


class WikiScraper:
    """Класс для скачивания и очистки текста с Wiki-сайтов"""

    def __init__(self, output_dir: str = "downloaded_pages", timeout: int = 15):
        """
        :param output_dir: директория для сохранения текстов
        :param timeout: время ожидания ответа сервера (сек)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

    def download_page(self, url: str) -> Optional[str]:
        """Скачивает HTML-код страницы по URL"""
        try:
            print(f"📥 Скачиваю: {url}")
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            response.encoding = 'utf-8'
            return response.text
        except requests.exceptions.Timeout:
            print(f"❌ Ошибка: Время ожидания истекло для {url}")
            return None
        except requests.exceptions.ConnectionError:
            print(f"❌ Ошибка: Не удается подключиться к {url}")
            return None
        except requests.exceptions.HTTPError as e:
            print(f"❌ Ошибка HTTP ({e.response.status_code}): {url}")
            return None
        except Exception as e:
            print(f"❌ Непредвиденная ошибка: {e}")
            return None

    def get_page_title(self, soup: BeautifulSoup) -> str:
        """Извлекает заголовок страницы"""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
            # Очищаем от лишнего (например " | Fandom" или " - Wikipedia")
            title = re.sub(r'\s*[||-]\s*(Fandom|Wikipedia|Wiki|FANDOM).*$', '', title)
            return title if title else "untitled"

        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)

        return "untitled"

    def remove_unwanted_elements(self, soup: BeautifulSoup) -> None:
        """Удаляет ненужные элементы из страницы"""

        unwanted_selectors = [
            'script',
            'style',
            'noscript',
            '[data-nosnippet]',
            '.navbox',
            '.toc',
            '#toc',
            '.reflist',
            '.reference',
            '.mw-editsection',
            '.navframe',
            'footer',
            '.footer',
            'nav',
            '.navigation',
            '[role="navigation"]',
            'aside',
            '.aside',
            '.advertisement',
            '.ads',
            '[class*="ad-"]',
            '[class*="advertisement"]',
            '.infobox',
            '.infobox-image',
            '.coordinates',
            'sup.reference',
            '.mw-headline-number',
            '[class*="cite"]',
        ]

        for selector in unwanted_selectors:
            for element in soup.select(selector):
                element.decompose()

        # Удаляем HTML комментарии
        for comment in soup.find_all(string=lambda text: isinstance(text, str) and text.strip().startswith('<!--')):
            comment.extract()

    def extract_text_from_element(self, element) -> str:
        """
        Извлекает текст из элемента, сохраняя пробелы между словами и ссылками
        """
        text_parts = []

        for content in element.children:
            if isinstance(content, str):
                # Обычный текст
                text = str(content).strip()
                if text:
                    text_parts.append(text)
            else:
                # Элемент (тег)
                if content.name == 'a':
                    # Для ссылок берем только текст
                    link_text = content.get_text(strip=True)
                    if link_text:
                        text_parts.append(link_text)
                elif content.name in ['br', 'hr']:
                    # Переносы строк пропускаем в контексте абзаца
                    pass
                else:
                    # Для других тегов берем текст рекурсивно
                    inner_text = self.extract_text_from_element(content)
                    if inner_text:
                        text_parts.append(inner_text)

        # Объединяем с сохранением пробелов
        result = ' '.join(text_parts)
        # Удаляем множественные пробелы
        result = re.sub(r' +', ' ', result)
        return result.strip()

    def extract_clean_text(self, soup: BeautifulSoup) -> str:
        """
        Извлекает основной текст со страницы, сохраняя структуру абзацев
        """

        # Пытаемся найти основной контент
        main_content = None
        selectors = [
            '#mw-content-text',  # Wikipedia/Fandom
            '.mw-parser-output',  # Wikipedia
            'main',
            'article',
            '[role="main"]',
            '.article-content',
            '#content',
        ]

        for selector in selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.find('body') or soup

        # Создаем копию для работы
        content_copy = BeautifulSoup(str(main_content), 'html.parser')

        # Удаляем еще раз ненужные элементы из контента
        self.remove_unwanted_elements(content_copy)

        # Получаем текст с сохранением структуры абзацев
        paragraphs = []

        for p_tag in content_copy.find_all('p'):
            # Используем специальный метод для извлечения текста со ссылками
            text = self.extract_text_from_element(p_tag)

            # Удаляем номера сносок [1], [2] и т.д.
            text = re.sub(r'\[\d+\]', '', text)

            # Удаляем [edit] и подобные теги
            text = re.sub(r'\[edit\]|\[редактировать\]', '', text)

            # Очищаем от лишних пробелов (на случай если что-то осталось)
            text = re.sub(r' +', ' ', text)
            text = text.strip()

            # Добавляем только непустые абзацы с достаточной длиной
            if text and len(text) > 10:
                paragraphs.append(text)

        # Объединяем абзацы с одним переносом строки между ними
        result = '\n'.join(paragraphs)

        # Финальная очистка - удаляем множественные переносы
        result = re.sub(r'\n\n+', '\n', result)

        return result.strip()

    def sanitize_filename(self, filename: str, max_length: int = 200) -> str:
        """Очищает имя файла от недопустимых символов"""
        invalid_chars = r'[/\\?%*:|"<>\x7F\x00-\x1F]'
        safe_name = re.sub(invalid_chars, '_', filename)
        safe_name = re.sub(r'_+', '_', safe_name)
        safe_name = safe_name.strip('_')

        if len(safe_name) > max_length:
            safe_name = safe_name[:max_length]

        return safe_name

    def process_url(self, url: str) -> Dict[str, Optional[str]]:
        """
        Скачивает страницу и извлекает чистый текст

        :param url: URL страницы
        :return: словарь с результатами
        """
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        result = {
            'url': url,
            'title': None,
            'text': None,
            'file_path': None,
            'status': 'pending',
            'text_length': 0
        }

        # Скачиваем страницу
        html = self.download_page(url)
        if not html:
            result['status'] = 'error'
            return result

        # Парсим HTML
        soup = BeautifulSoup(html, 'html.parser')

        # Извлекаем заголовок
        page_title = self.get_page_title(soup)
        result['title'] = page_title

        print(f"📄 Заголовок: {page_title}")

        # Удаляем ненужные элементы
        self.remove_unwanted_elements(soup)

        # Извлекаем чистый текст
        clean_text = self.extract_clean_text(soup)

        result['text'] = clean_text
        result['text_length'] = len(clean_text)

        print(f"📊 Символов: {len(clean_text)}")

        # Создаем имя файла на основе title
        safe_filename = self.sanitize_filename(page_title)
        if not safe_filename:
            safe_filename = "page"

        filename = f"{safe_filename}.txt"
        filepath = self.output_dir / filename

        # Обработка дубликатов
        counter = 1
        while filepath.exists():
            filename = f"{safe_filename}_{counter}.txt"
            filepath = self.output_dir / filename
            counter += 1

        # Сохраняем в файл
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(clean_text)
            result['file_path'] = str(filepath)
            result['status'] = 'success'
            print(f"✅ Сохранено: {filepath}\n")
        except Exception as e:
            print(f"❌ Ошибка при сохранении: {e}")
            result['status'] = 'save_error'

        return result

    def process_urls(self, urls: List[str]) -> Dict[str, Dict]:
        """Обрабатывает список URL-адресов"""
        print(f"\n{'='*70}")
        print(f"🚀 Начинаю парсинг {len(urls)} страниц...")
        print(f"{'='*70}\n")

        results = {}
        for idx, url in enumerate(urls, 1):
            print(f"[{idx}/{len(urls)}]")
            result = self.process_url(url)
            results[url] = result
            time.sleep(2)

        return results

    def generate_report(self, results: Dict[str, Dict],
                        report_file: str = "parsing_report.json") -> None:
        """Создает отчет в JSON формате"""
        report = {
            'total_urls': len(results),
            'successful': 0,
            'failed': 0,
            'total_text_length': 0,
            'details': []
        }

        for url, data in results.items():
            if data['status'] == 'success':
                report['successful'] += 1
                report['total_text_length'] += data['text_length']
            else:
                report['failed'] += 1

            report['details'].append({
                'url': url,
                'title': data['title'],
                'status': data['status'],
                'file': data['file_path'],
                'text_length': data['text_length']
            })

        report_path = self.output_dir / report_file
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"📊 Отчет сохранен: {report_path}\n")

    def print_summary(self, results: Dict[str, Dict]) -> None:
        """Выводит сводку результатов"""
        print(f"\n{'='*70}")
        print("📋 СВОДКА РЕЗУЛЬТАТОВ")
        print(f"{'='*70}\n")

        successful = sum(1 for r in results.values() if r['status'] == 'success')
        failed = len(results) - successful
        total_length = sum(r['text_length'] for r in results.values())

        print(f"✅ Успешно обработано: {successful}/{len(results)}")
        print(f"❌ Ошибок: {failed}/{len(results)}")
        print(f"📊 Всего символов: {total_length}\n")

        for url, data in results.items():
            status_icon = '✅' if data['status'] == 'success' else '❌'
            print(f"{status_icon} {url}")
            if data['title']:
                print(f"   📄 {data['title']}")
            if data['text_length']:
                print(f"   📊 {data['text_length']} символов")
            if data['file_path']:
                print(f"   💾 {data['file_path']}")
            print()


def load_urls_from_file(filename: str) -> List[str]:
    """Загружает список URL из текстового файла (один URL на строку)"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]

        if urls:
            print(f"📋 Загружено {len(urls)} URL из файла {filename}\n")
        else:
            print(f"⚠️  Файл {filename} пуст")

        return urls
    except FileNotFoundError:
        print(f"❌ Файл {filename} не найден")
        print("📝 Создайте файл urls.txt со списком URL (один на строку)")
        return []


def main():
    """Главная функция - чтение URL только из файла"""

    # Загружаем URL из файла urls.txt
    urls = load_urls_from_file('urls.txt')

    if not urls:
        print("\n❌ Нет URL для обработки")
        print("📝 Создайте файл urls.txt и добавьте URL по одному на строку:")
        print("   https://starwars.fandom.com/wiki/Anakin_Skywalker")
        print("   https://starwars.fandom.com/wiki/Luke_Skywalker")
        return

    # Создаем scraper и обрабатываем URL
    scraper = WikiScraper(output_dir="source")
    results = scraper.process_urls(urls)
    scraper.print_summary(results)
    scraper.generate_report(results)

    # Вывести превью первых 800 символов успешно обработанных файлов
    # for url, data in results.items():
    #     if data['status'] == 'success' and data['text']:
    #         print(f"{'='*70}")
    #         print(f"📝 ПРЕВЬЮ: {data['title']}")
    #         print(f"{'='*70}\n")
    #         print(data['text'][:800])
    #         print("\n[Текст обрезан для просмотра...]\n")

if __name__ == '__main__':
    main()
