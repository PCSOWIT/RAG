import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from services.rag_pg_service import pg_rag_service

# (问题, 期望品类, 答案里应出现的关键词)  期望品类为 None 表示应当拒答
CASES = [
    ("哪个手机最便宜", "phone", "红米K80"),
    ("iPhone 16 多少钱", "phone", "iPhone 16"),
    ("小米15什么配置", "phone", "小米15"),
    ("有没有便宜点的安卓机", "phone", None),
    ("华为Mate 70 支持卫星通信吗", "phone", "Mate 70"),

    ("MacBook Air 多少钱", "laptop", "MacBook Air"),
    ("打游戏买哪个笔记本", "laptop", None),
    ("有没有轻薄一点的电脑", "laptop", None),
    ("拯救者什么显卡", "laptop", "拯救者"),

    ("哪个耳机降噪好", "audio", None),
    ("AirPods Pro 卖多少钱", "audio", "AirPods"),
    ("有没有便宜的音箱", "audio", None),

    ("iPad 多少钱", "pad", "iPad"),
    ("平板哪个适合学生", "pad", None),
    ("MatePad 屏幕多大", "pad", "MatePad"),

    ("Python数据分析课程多少钱", None, None),
    ("学费怎么交", None, None),

    ("你们食堂几点开门", None, None),
    ("你们老板是谁", None, None),
    ("今天天气怎么样", None, None),

    # ---- 第二轮扩充：口语简称 / 品牌简称 / 库里没有的品牌 ----
    ("想买个打游戏的本子", "laptop", None),
    ("苹果手机多少钱", "phone", "iPhone"),
    ("三星的手机怎么样", "phone", None),
    ("推荐个适合学生用的平板", "pad", None),
    ("索尼的耳机降噪行不行", "audio", "索尼"),
    ("荣耀手机值得买吗", "phone", None),
    ("小米15和iPhone16哪个好", "phone", None),
    ("笔记本能打原神吗", "laptop", None),
    ("你们卖车吗", None, None),
    ("帮我推荐个办公用的电脑", "laptop", None),
]


async def main():
    total = len(CASES)
    passed = 0
    recall_total = 0
    recall_pass = 0
    reject_total = 0
    reject_pass = 0
    t_all = time.perf_counter()

    for q, expect_cat, expect_kw in CASES:
        t0 = time.perf_counter()
        content = await pg_rag_service.retrieve(q, top_k=3)
        cost = (time.perf_counter() - t0) * 1000
        is_reject_case = expect_cat is None

        if is_reject_case:
            reject_total += 1
            ok = content == ""
            if ok:
                reject_pass += 1
                print(f"PASS  {q:<26} 正确拒答（{cost:.0f}ms）")
            else:
                print(f"FAIL  {q:<26} 应拒答却返回了 {len(content)} 字（{cost:.0f}ms）")
        else:
            recall_total += 1
            got_cat = pg_rag_service.detect_category(q)
            cat_ok = got_cat == expect_cat
            kw_ok = (expect_kw is None) or (expect_kw in content)
            ok = content != "" and cat_ok and kw_ok
            if ok:
                recall_pass += 1
                print(f"PASS  {q:<26} 品类={got_cat} 命中（{cost:.0f}ms）")
            else:
                reason = []
                if content == "":
                    reason.append("未召回")
                if not cat_ok:
                    reason.append(f"品类错({got_cat})")
                if not kw_ok:
                    reason.append(f"缺关键词({expect_kw})")
                print(f"FAIL  {q:<26} {'/'.join(reason)}（{cost:.0f}ms）")

        passed += 1 if ok else 0

    t_cost = (time.perf_counter() - t_all) * 1000
    print("\n" + "=" * 56)
    print(f"总命中率     {passed}/{total} = {passed / total * 100:.1f}%")
    print(f"召回命中率   {recall_pass}/{recall_total} = {recall_pass / recall_total * 100:.1f}%")
    print(f"拒答准确率   {reject_pass}/{reject_total} = {reject_pass / reject_total * 100:.1f}%")
    print(f"平均单次耗时 {t_cost / total:.0f}ms（含 ollama 首次加载）")
    print("=" * 56)


if __name__ == "__main__":
    asyncio.run(main())
