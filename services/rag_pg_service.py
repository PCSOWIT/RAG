import asyncio
import time
import httpx
import psycopg

from services.cache import rag_cache

# 品类识别规则：用户问题里出现这些词，就只在该品类里检索
CATEGORY_RULES = {
    "phone": ["手机", "iphone", "小米", "华为mate", "oppo", "vivo", "一加", "红米", "安卓"],
    "laptop": ["笔记本", "电脑", "本子", "笔电", "macbook", "mac", "拯救者", "matebook", "xps", "游戏本"],
    "audio": ["耳机", "音箱", "airpods", "索尼", "bose", "漫步者", "jbl", "降噪"],
    "pad": ["平板", "ipad", "matepad"],
}

# 明确不在本店经营范围的话题：命中就直接拒答，不给模型任何资料
# 光靠相似度阈值挡不住这类问题——它们和商品描述有语义交集，会被误召回
OUT_OF_SCOPE_WORDS = ["课程", "培训", "学费", "报名", "学制", "上课"]


class PgRagService:
    def __init__(self):
        self.dsn = "host=127.0.0.1 port=5433 dbname=ai_kb user=postgres"
        self.ollama_url = "http://127.0.0.1:11434/api/embed"
        self.model = "bge-m3"

    async def _embed(self, text: str) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self.ollama_url,
                json={"model": self.model, "input": [text]},
                timeout=60.0,
            )
            resp.raise_for_status()
            vector = resp.json()["embeddings"][0]
        return "[" + ",".join(map(str, vector)) + "]"

    @staticmethod
    def detect_category(query: str) -> str | None:
        q = query.lower()
        for cat, words in CATEGORY_RULES.items():
            if any(w in q for w in words):
                return cat
        return None

    def _vector_search(self, vec_str: str, top_k: int, category: str | None = None) -> list:
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                if category:
                    cur.execute(
                        "SELECT content, category, 1 - (embedding <=> %s::halfvec) AS score "
                        "FROM kb_chunks WHERE category = %s "
                        "ORDER BY embedding <=> %s::halfvec LIMIT %s",
                        (vec_str, category, vec_str, top_k),
                    )
                else:
                    cur.execute(
                        "SELECT content, category, 1 - (embedding <=> %s::halfvec) AS score "
                        "FROM kb_chunks ORDER BY embedding <=> %s::halfvec LIMIT %s",
                        (vec_str, vec_str, top_k),
                    )
                return cur.fetchall()

    def _keyword_search(self, query: str, top_k: int, category: str | None = None) -> list:
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                if category:
                    cur.execute(
                        "SELECT content, category, similarity(content, %s) AS score "
                        "FROM kb_chunks WHERE category = %s "
                        "ORDER BY score DESC LIMIT %s",
                        (query, category, top_k),
                    )
                else:
                    cur.execute(
                        "SELECT content, category, similarity(content, %s) AS score "
                        "FROM kb_chunks ORDER BY score DESC LIMIT %s",
                        (query, top_k),
                    )
                return cur.fetchall()

    @staticmethod
    def _rrf_fuse(vec_rows: list, kw_rows: list, k: int = 60) -> list:
        scores = {}
        info = {}
        for rank, row in enumerate(vec_rows, start=1):
            key = row[0]
            info[key] = {"row": row, "vec": row[2], "kw": 0.0}
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        for rank, row in enumerate(kw_rows, start=1):
            key = row[0]
            if key in info:
                info[key]["kw"] = row[2]
            else:
                info[key] = {"row": row, "vec": 0.0, "kw": row[2]}
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        order = sorted(scores, key=lambda c: scores[c], reverse=True)
        return [(info[c], scores[c]) for c in order]

    def _hybrid_search(self, vec_str: str, query: str, top_k: int, category: str | None,
                       recall_n: int = 10, min_score: float = 0.52, kw_min: float = 0.05) -> list:
        t0 = time.perf_counter()
        vec_rows = self._vector_search(vec_str, recall_n, category)
        t1 = time.perf_counter()
        kw_rows = self._keyword_search(query, recall_n, category)
        t2 = time.perf_counter()

        vec_rows = [r for r in vec_rows if r[2] >= min_score]
        kw_rows = [r for r in kw_rows if r[2] >= kw_min]

        fused = self._rrf_fuse(vec_rows, kw_rows)
        print(f"   ⏱ 向量 {1000 * (t1 - t0):.1f}ms / 关键词 {1000 * (t2 - t1):.1f}ms / "
              f"召回 向量{len(vec_rows)}条 关键词{len(kw_rows)}条")
        return fused[:top_k]

    async def retrieve(self, query: str, top_k: int = 3, category: str | None = None,
                       min_score: float = 0.52) -> str:
        t_start = time.perf_counter()
        try:
            if any(w in query for w in OUT_OF_SCOPE_WORDS):
                print(f"⚠️ [PgRagService] 超出经营范围（{query}），拒答")
                return ""

            if category is None:
                category = self.detect_category(query)

            # 先查缓存：命中就直接返回，跳过向量化和数据库查询
            cached = rag_cache.get(query, category)
            if cached is not None:
                label = "空结果" if cached == "" else f"{len(cached)}字"
                print(f"⚡ [PgRagService] 缓存命中（{label}）｜总 {1000 * (time.perf_counter() - t_start):.2f}ms")
                return cached

            vec_str = await self._embed(query)
            t_embed = time.perf_counter()
            rows = await asyncio.to_thread(
                self._hybrid_search, vec_str, query, top_k, category, 10, min_score
            )
            t_total = time.perf_counter()

            if not rows:
                print(f"⚠️ [PgRagService] 无相关内容（品类={category}），拒答")
                rag_cache.set(query, category, "")
                return ""

            print(f"✅ [PgRagService] 命中 {len(rows)} 条｜品类={category}｜"
                  f"向量化 {1000 * (t_embed - t_start):.0f}ms｜总 {1000 * (t_total - t_start):.0f}ms")
            for item, rrf in rows:
                content, cat, _ = item["row"]
                print(f"   [{cat}] 向量 {item['vec']:.3f}｜关键词 {item['kw']:.3f}｜"
                      f"RRF {rrf:.4f}｜{content[:24]}")
            result = "\n\n".join(item[0]["row"][0] for item in rows)
            rag_cache.set(query, category, result)
            return result
        except Exception as e:
            print(f"❌ [PgRagService] 异常: {e}")
            return ""


pg_rag_service = PgRagService()
