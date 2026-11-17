import httpx
from config import settings
from services.utils import Signer
import asyncio
import httpx
import psycopg

class RagService:
    def __init__(self):
        self.ak = settings.VOLC_AK
        self.sk = settings.VOLC_SK
        self.collection_name = settings.KB_COLLECTION_NAME
        self.project_name = settings.KB_PROJECT_NAME
        self.account_id = settings.VOLC_ACCOUNT_ID

        self.host = "api-knowledgebase.mlp.cn-beijing.volces.com"
        self.region = "cn-north-1"
        self.service = "air"

    async def retrieve(self, query: str) -> str:
        if not self.ak or not self.sk or not self.account_id:
            print("⚠️ [RagService] 配置缺失")
            return ""

        path = "/api/knowledge/collection/search_knowledge"

        body = {
            "project": self.project_name,
            "name": self.collection_name,
            "query": query,
            "limit": 1,
            "pre_processing": {
                "need_instruction": True,
                "return_token_usage": True,
                "messages": [{"role": "user", "content": query}]
            },
            "post_processing": {
                "get_attachment_link": True
            }
        }

        headers = {
            "Host": self.host,
            "Content-Type": "application/json",
            "V-Account-Id": self.account_id
        }

        request_data = {
            "method": "POST",
            "path": path,
            "headers": headers,
            "body": body,
            "params": {}
        }

        try:
            signer = Signer(request_data, service=self.service, region=self.region)
            signer.add_authorization({
                "accessKeyId": self.ak,
                "secretKey": self.sk
            })

            url = f"http://{self.host}{path}"

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    headers=request_data["headers"],
                    json=body,
                    timeout=10.0
                )

            if resp.status_code != 200:
                print(f"❌ [RagService] 请求失败: {resp.status_code}, {resp.text}")
                return ""

            data = resp.json()
            result_list = data.get("data", {}).get("result_list", [])
            contents = [item.get("content", "") for item in result_list if item.get("content")]

            if not contents:
                print("⚠️ [RagService] 未检索到匹配的知识内容")
                return ""

            context_text = "\n\n".join(contents)
            print(f"✅ [RagService] 成功提取 {len(contents)} 条知识内容")
            return context_text

        except Exception as e:
            print(f"❌ [RagService] 异常: {e}")
            return ""


rag_service = RagService()
