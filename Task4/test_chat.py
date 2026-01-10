import os
from dotenv import load_dotenv
from langchain_community.llms.yandex import YandexGPT

load_dotenv()

# Способ 1: С API ключом (Access Key)
llm = YandexGPT(
    api_key=os.getenv("YANDEX_API_KEY"),
    folder_id=os.getenv("YANDEX_FOLDER_ID")
)

# Или напрямую:
# llm = YandexGPT(
#     api_key="",
#     folder_id=""
# )

# Тест
response = llm.invoke("Привет!")
print(response)
