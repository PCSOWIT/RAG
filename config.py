import os
from dotenv import load_dotenv

load_dotenv()   # 把 .env 里的键值对塞进"环境变量"

class Settings:
    VOLC_AK = os.getenv("VOLC_AK")
    VOLC_SK = os.getenv("VOLC_SK")
    ARK_API_KEY = os.getenv("ARK_API_KEY")
    ARK_ENDPOINT_ID = os.getenv("ARK_ENDPOINT_ID")
    RTC_APP_ID = os.getenv("RTC_APP_ID")
    RTC_APP_KEY = os.getenv("RTC_APP_KEY")
    SERVER_URL = os.getenv("SERVER_URL", "")        # 第二个参数=取不到时的默认值
    KB_COLLECTION_NAME = os.getenv("KB_COLLECTION_NAME")
    KB_PROJECT_NAME = os.getenv("KB_PROJECT_NAME", "default")
    VOLC_ACCOUNT_ID = os.getenv("VOLC_ACCOUNT_ID")

settings = Settings()   # 全局唯一实例，别的文件直接 import 来用
