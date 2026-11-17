
import httpx
import psycopg
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

OLLAMA_URL = "http://127.0.0.1:11434/api/embed"
MODEL_NAME = "bge-m3"
DSN = "host=127.0.0.1 port=5433 dbname=ai_kb user=postgres"

def get_embeddings(texts: list) -> list:
    resp = httpx.post(OLLAMA_URL, json={"model": MODEL_NAME, "input": texts}, timeout=120)
    resp.raise_for_status()
    return resp.json()["embeddings"]

def split_text(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", ""],
    )
    return splitter.split_text(text)


def split_lines(text: str) -> list:
    return [line.strip() for line in text.splitlines() if line.strip()]


def main():
    doc_dir = Path(__file__).parent.parent / "docs"
    docs = []
    for f in sorted(doc_dir.glob("*.txt")):
        raw = f.read_text(encoding="utf-8")
        chunks = split_text(raw) if f.stem == "course" else split_lines(raw)
        for c in chunks:
            docs.append({"source": f.name, "category": f.stem, "content": c})

    print(f"切成 {len(docs)} 段")

    vectors = get_embeddings([d["content"] for d in docs])

    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            for d, v in zip(docs, vectors):
                vec_str = "[" + ",".join(map(str, v)) + "]"
                cur.execute(
                    "INSERT INTO kb_chunks (content, source, category, embedding) VALUES (%s, %s, %s, %s::halfvec)",
                    (d["content"], d["source"], d["category"], vec_str),
                )
        conn.commit()

    print(f"插入了 {len(docs)} 条")

if __name__ == "__main__":
    main()

