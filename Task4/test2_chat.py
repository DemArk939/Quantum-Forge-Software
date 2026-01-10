import sys
from langchain_community.llms.yandex import YandexGPT
import os
from dotenv import load_dotenv

# Загрузка переменных окружения из .env
load_dotenv()

def main():
    # Получение ключей из переменных окружения
    api_key = os.getenv("YANDEX_API_KEY")
    folder_id = os.getenv("YANDEX_FOLDER_ID")

    # Проверка наличия ключей
    if not api_key or not folder_id:
        print("❌ Ошибка: переменные окружения YANDEX_API_KEY и YANDEX_FOLDER_ID не установлены")
        print("Установите их в файле .env или в переменных окружения")
        sys.exit(1)

    # Инициализация YandexGPT
    llm = YandexGPT(
        api_key=api_key,
        folder_id=folder_id
    )

    # Получение запроса из аргументов командной строки
    if len(sys.argv) > 1:
        # Если аргументы переданы, объединяем их в один запрос
        user_input = " ".join(sys.argv[1:])
    else:
        # Если аргументов нет, запрашиваем ввод у пользователя
        print("=" * 60)
        print("YandexGPT Чат-бот")
        print("=" * 60)
        print("Введите ваш вопрос (или 'exit' для выхода):\n")
        user_input = input("Вы: ").strip()

        if user_input.lower() == "exit":
            print("До встречи!")
            sys.exit(0)

    if not user_input:
        print("❌ Ошибка: запрос не может быть пустым")
        sys.exit(1)

    print(f"\n🤖 YandexGPT обрабатывает запрос...\n")

    try:
        # Отправка запроса к YandexGPT
        response = llm.invoke(user_input)

        print(f"Вопрос: {user_input}")
        print(f"\nОтвет YandexGPT:\n{response}")
        print("\n" + "=" * 60)

    except Exception as e:
        print(f"❌ Ошибка при обращении к YandexGPT: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
