from fastapi import FastAPI
from config import settings
import time
from config import settings
from services.token_build import AccessToken, PRIVILEGES
import uvicorn
import httpx
from fastapi import FastAPI, Request
from services.utils import Signer
import uuid
import json
from fastapi.responses import StreamingResponse
import os
from services.rag_service import rag_service
from services.rag_pg_service import pg_rag_service
if os.getenv("RAG_BACKEND", "volc") == "pg":
    rag_service = pg_rag_service
from services.llm_service import llm_service
from fastapi.middleware.cors import CORSMiddleware
from volcenginesdkarkruntime import AsyncArk



app = FastAPI()          # 创建应用实例
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")            # 注册一个 GET 路由
def hello():
    return {"msg": "服务跑起来了"}

@app.get("/test-config")
def test_config():
    return {"AppId": settings.RTC_APP_ID, "有SK": bool(settings.VOLC_SK)}

@app.post("/api/chat_callback")
async def chat_callback(request: Request):
    try:
        data = await request.json()
    except:
        return {"text": ""}

    print(f"======================== 流式请求 {data}")

    messages = data.get("messages", [])

    if not messages or messages[-1].get("role") != "user":
        print("⚠️ 忽略：非用户主动发言")
        return {"text": ""}

    async def generate_sse():
        rag_content = await rag_service.retrieve(messages[-1].get("content", ""))
        stream_iterator = llm_service.chat_stream(messages, rag_content)

        response_id = str(uuid.uuid4())
        created = int(time.time())
        model_name = settings.ARK_ENDPOINT_ID

        async for chunk in stream_iterator:
            if chunk and chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                content = delta.content
                if content:
                    sse_payload = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "finish_reason": None,
                            "delta": {"role": "assistant", "content": content},
                        }],
                        "stream_options": {"include_usage": True},
                    }
                    yield f"data: {json.dumps(sse_payload, ensure_ascii=False)}\n\n"

        final_payload = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "delta": {},
            }],
            "stream_options": {"include_usage": True},
        }
        yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.post("/getScenes")
async def get_scenes():
    room_id = "ChatRoom01"
    user_id = "Huoshan01"

    token_builder = AccessToken(settings.RTC_APP_ID, settings.RTC_APP_KEY, room_id, user_id)
    token_builder.add_privilege(PRIVILEGES["PrivSubscribeStream"], 0)
    token_builder.add_privilege(PRIVILEGES["PrivPublishStream"], 0)
    token_builder.expire_time(int(time.time()) + 3600 * 24)
    token = token_builder.serialize()

    return {
        "ResponseMetadata": {"Action": "getScenes"},
        "Result": {
            "scenes": [
                {
                    "scene": {
                        "id": "Custom",
                        "name": "自定义助手",
                        "botName": "AiAgent",
                        "icon": "https://lf3-rtc-demo.volccdn.com/obj/rtc-aigc-assets/DoubaoAvatar.png",
                        "isInterruptMode": True,
                        "isVision": False,
                        "isScreenMode": False,
                        "isAvatarScene": None,
                        "avatarBgUrl": None,
                    },
                    "rtc": {
                        "AppId": settings.RTC_APP_ID,
                        "RoomId": room_id,
                        "UserId": user_id,
                        "Token": token,
                    },
                    "VoiceChat": {},
                }
            ]
        },
    }

@app.post("/proxy")
async def proxy(request: Request):
    action = request.query_params.get("Action")
    version = request.query_params.get("Version", "2025-06-01")

    try:
        incoming_body = await request.json()
        print(f"DEBUG: 收到前端请求 {action}, Body: {incoming_body}")
    except Exception as e:
        print(e)

    target_app_id = settings.RTC_APP_ID
    target_room_id = "ChatRoom01"
    target_user_id = "Huoshan01"

    request_body = {}

    if action == "StartVoiceChat":
        request_body = {
            "AppId": target_app_id,
            "RoomId": target_room_id,
            "TaskId": "ChatTask01",
            "AgentConfig": {
                "TargetUserId": [target_user_id],
                "WelcomeMessage": "你好，我是小智，这家店的在线客服。想了解哪款的价格或配置，直接问我就行。",
                "UserId": "AiAgent",
                "EnableConversationStateCallback": True,
            },
            "Config": {
                "ASRConfig": {
                    "Provider": "volcano",
                    "ProviderParams": {
                        "Mode": "bigmodel",
                        "ApiResourceId": "volc.seedasr.sauc.duration",
                        "StreamMode": 2
                    },
                },
                "TTSConfig": {
                    "Provider": "volcano_bidirection",
                    "ProviderParams": {
                        "audio": {
                            "voice_type": "zh_female_vv_uranus_bigtts",
                            "speech_rate": 0
                        },
                        "ResourceId": "seed-tts-2.0",
                        "additions": {
                            "enable_latex_tn": True,
                            "disable_markdown_filter": True,
                            "enable_language_detector": False
                        }
                    },
                },
                "LLMConfig": {
                    "Mode": "CustomLLM",
                    "Url": f"{settings.SERVER_URL}/api/chat_callback",
                    "Method": "POST",
                    "ApiType": "https" if str(settings.SERVER_URL).startswith("https") else "http",
                },
                "InterruptMode": 0,
            },
        }
    elif action == "StopVoiceChat":
        request_body = {
            "AppId": target_app_id,
            "RoomId": target_room_id,
            "TaskId": "ChatTask01",
        }
    else:
        request_body = incoming_body

    host = "rtc.volcengineapi.com"
    open_api_request_data = {
        "method": "POST",
        "path": "/",
        "params": {"Action": action, "Version": version},
        "headers": {"Host": host, "Content-Type": "application/json"},
        "body": request_body,
    }

    account_config = {"accessKeyId": settings.VOLC_AK, "secretKey": settings.VOLC_SK}
    signer = Signer(open_api_request_data, "rtc")
    signer.add_authorization(account_config)

    url = f"https://{host}?Action={action}&Version={version}"

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=open_api_request_data["headers"], json=request_body, timeout=30.0)
        result = resp.json()
        print(f"DEBUG: 火山引擎返回结果: {result}")
        return result


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=3001, reload=True)
