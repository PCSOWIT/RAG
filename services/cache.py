"""
检索结果缓存层（Redis）

作用：同一个问题问第二遍，直接返回上次的检索结果，
      跳过向量化（调 ollama，最慢的一步）和数据库查询。

两个关键设计：
1. 空结果也缓存（值存成 __EMPTY__），防止同一个查不到的问题反复穿透到底层
2. Redis 挂了不影响主流程：连续失败就自动降级成"无缓存"，服务照常跑
"""
import os

import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))   # 有结果：存 1 小时
EMPTY_TTL = int(os.getenv("EMPTY_TTL", "300"))    # 空结果：只存 5 分钟
EMPTY_MARK = "__EMPTY__"


class RagCache:
    def __init__(self):
        self.r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        self.available = True

    @staticmethod
    def _key(query, category):
        # 带上品类，避免"多少钱"这种泛问被不同品类的结果串味
        return f"rag:{category or 'all'}:{query}"

    def get(self, query, category):
        """命中返回字符串（可能是空串），未命中返回 None"""
        if not self.available:
            return None
        try:
            value = self.r.get(self._key(query, category))
            if value is None:
                return None
            return "" if value == EMPTY_MARK else value
        except Exception as exc:
            print(f"⚠️ [Redis] 读取失败，本次降级为无缓存：{exc}")
            self.available = False
            return None

    def set(self, query, category, value):
        if not self.available:
            return
        try:
            if value:
                self.r.set(self._key(query, category), value, ex=CACHE_TTL)
            else:
                self.r.set(self._key(query, category), EMPTY_MARK, ex=EMPTY_TTL)
        except Exception as exc:
            print(f"⚠️ [Redis] 写入失败，本次降级为无缓存：{exc}")
            self.available = False

    def clear(self):
        """商品数据更新后调用，删掉所有检索缓存"""
        if not self.available:
            return 0
        try:
            keys = list(self.r.scan_iter(match="rag:*", count=100))
            if not keys:
                return 0
            return self.r.delete(*keys)
        except Exception as exc:
            print(f"⚠️ [Redis] 清空失败：{exc}")
            return 0


rag_cache = RagCache()
