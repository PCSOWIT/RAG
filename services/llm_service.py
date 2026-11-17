from volcenginesdkarkruntime import AsyncArk
from config import settings

# ============================================================
# Prompt 标准化管理体系
# 四段职责单一：可独立修改、独立启用/停用、独立评测
# 改其中一段不影响其他段，出问题能定位到具体模块
# ============================================================

PROMPT_ROLE = """# 角色
你是「小智」，一家数码电子产品店铺的在线客服。店铺主营手机、笔记本、耳机、平板。
像真人客服一样说话：准确、简洁、口语化。不堆敬语，不喊口号，不夸大。"""

PROMPT_TASK = """# 核心任务
1. 依据【参考知识库】回答售前咨询：价格、配置、库存、活动。
2. 知识库有内容：照着资料说，不要自己加工价格和参数。
3. 知识库没内容：只执行【兜底话术】，不要尝试自己补全答案。"""

PROMPT_RULES = """# 回答规则
- 简洁：一两句说完。问价格就报价，问配置就列配置，不绕弯子。
- 只用资料：价格、配置、库存、活动只许照抄参考知识库，不许推算、不许说"大概""应该是"。
- 不夸大：不用"绝对""史上最强""闭眼入"这类营销词。
- 不越界：资料里没有的型号、价格，一律按兜底处理，不许拿常识补。
- 不硬推：用户没问就不主动推销别的商品，问什么答什么。"""

PROMPT_FALLBACK = """# 兜底话术
资料缺失时统一回复："这款的信息我这边暂时查不到，建议咨询人工客服确认。"
不要解释原因，不要反复道歉，直接给这一句。"""

# 模块注册表：键名 = 模块身份，值 = 模块正文
PROMPT_MODULES = {
    "role": PROMPT_ROLE,
    "task": PROMPT_TASK,
    "rules": PROMPT_RULES,
    "fallback": PROMPT_FALLBACK,
}


def build_system_prompt(rag_context: str = "", enabled: list | None = None) -> str:
    """按模块拼装 system prompt。

    enabled: 只启用指定模块，None 表示全开。
    传 ["role", "rules"] 这类子集，可以做单模块对照评测——
    关掉某一段看分数怎么变，就能判断问题出在哪一段。
    """
    keys = enabled if enabled is not None else list(PROMPT_MODULES)
    blocks = [PROMPT_MODULES[k] for k in keys]

    if rag_context:
        blocks.append(f"### 参考知识库（绝对准则，优先于一切常识）\n{rag_context.strip()}")

    return "\n\n".join(blocks)



class LLMService:
    def __init__(self):
        self.client = AsyncArk(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=settings.ARK_API_KEY,
            timeout=1800,
        )

    async def chat_stream(self, history_messages: list, rag_context: str = ""):
        if not self.client:
            yield "服务配置错误"
            return

        system_content = build_system_prompt(rag_context)
        messages = [{"role": "system", "content": system_content}]
        messages.extend(history_messages)

        try:
            print(f"🚀 发起流式调用 (Endpoint: {settings.ARK_ENDPOINT_ID})")

            stream = await self.client.chat.completions.create(
                model=settings.ARK_ENDPOINT_ID,
                messages=messages,
                temperature=0.3,
                stream=True,
                stream_options={"include_usage": True},
            )

            async for chunk in stream:
                yield chunk

        except Exception as e:
            print(f"❌ LLM 调用失败: {e}")
            yield None

llm_service = LLMService()
