import os
from fastapi import FastAPI, Request, HTTPException
import httpx
from typing import Dict, Deque, Optional
from collections import defaultdict, deque
from openai import OpenAI

# WhatsApp Cloud API + FastAPI + Groq (OpenAI-compatible)
# - Verificação de webhook (GET /webhook): responde hub.challenge
# - Recebimento de mensagens (POST /webhook)
# - Resposta com LLM da Groq
# - Histórico curto em memória por usuário

# === Variáveis de ambiente ===
VERIFY_TOKEN = os.environ.get("APP_VERIFY_TOKEN", "")          # token de verificação do webhook (definido por você)
WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN", "")          # token Bearer da Cloud API (Meta)
PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")  # ID do número no WhatsApp Business
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL_ID = os.environ.get("MODEL_ID", "llama-3.1-8b-instant")

if not VERIFY_TOKEN:
    raise RuntimeError("Faltou APP_VERIFY_TOKEN no ambiente.")
if not WHATSAPP_TOKEN:
    raise RuntimeError("Faltou WHATSAPP_TOKEN no ambiente.")
if not PHONE_NUMBER_ID:
    raise RuntimeError("Faltou WHATSAPP_PHONE_NUMBER_ID no ambiente.")
if not GROQ_API_KEY:
    raise RuntimeError("Faltou GROQ_API_KEY no ambiente.")

SIMULATE = (os.environ.get("WHATSAPP_TOKEN", "") == "FAKE")

# Cliente OpenAI compatível apontando para Groq
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)

app = FastAPI(title="WhatsApp LLM Bot (Groq)")

# Histórico curto em memória por remetente (número de telefone)
History = Dict[str, Deque[dict]]
history: History = defaultdict(lambda: deque(maxlen=10))

GRAPH_BASE = "https://graph.facebook.com/v19.0"

async def send_whatsapp_text(to_phone: str, text: str):
    if SIMULATE:
        print(f"[SIMULATE] Responder para {to_phone}: {text[:120]}...")
        return
    url = f"{GRAPH_BASE}/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_phone,
        "type": "text",
        "text": {"body": text},
    }
    async with httpx.AsyncClient(timeout=20) as http:
        r = await http.post(url, headers=headers, json=payload)
        r.raise_for_status()

@app.get("/")
def health():
    return {"status": "ok"}

# Verificação do webhook (setup via Meta > Webhooks)
@app.get("/webhook")
async def verify(mode: Optional[str] = None, challenge: Optional[str] = None, hub_verify_token: Optional[str] = None, **kwargs):
    # Meta envia como: hub.mode, hub.verify_token, hub.challenge
    if hub_verify_token != VERIFY_TOKEN:
        raise HTTPException(403, "Invalid verify token")
    # challenge pode vir como string; devolver como texto é suficiente
    return challenge or "OK"

@app.post("/webhook")
async def incoming(request: Request):
    data = await request.json()
    # Estrutura: entry[0].changes[0].value.messages[0]
    try:
        entry = data.get("entry", [])[0]
        change = entry.get("changes", [])[0]
        value = change.get("value", {})
    except Exception:
        return {"ok": True}

    messages = value.get("messages", [])
    if not messages:
        return {"ok": True}

    message = messages[0]
    from_phone = message.get("from")  # ex.: "5511999999999"
    text_body = ""

    if message.get("type") == "text":
        text_body = (message.get("text") or {}).get("body", "").strip()
    else:
        if from_phone:
            await send_whatsapp_text(from_phone, "No momento só entendo mensagens de texto. Envie sua pergunta 🙂")
        return {"ok": True}

    # Comandos simples
    low = text_body.lower()
    if low.startswith("/start"):
        await send_whatsapp_text(from_phone, "Olá! Sou um bot no WhatsApp usando Llama 3.1 (Groq). Mande sua pergunta.")
        return {"ok": True}
    if low.startswith("/reset"):
        history[from_phone].clear()
        await send_whatsapp_text(from_phone, "Histórico limpo. Pode continuar!")
        return {"ok": True}
    if low.startswith("/help"):
        await send_whatsapp_text(from_phone, "Comandos: /help, /start, /reset")
        return {"ok": True}

    # Contexto + LLM
    msgs = list(history[from_phone])
    msgs.append({"role": "user", "content": text_body})

    try:
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": "Responda em português do Brasil, de forma objetiva e útil."}
            ] + msgs,
            temperature=0.6,
            max_tokens=512,
        )
        answer = completion.choices[0].message.content or "Desculpe, não consegui responder agora."
    except Exception:
        answer = "Ops! Tive um problema ao falar com o modelo. Tente novamente em alguns segundos."

    # persiste histórico e responde
    history[from_phone].append({"role": "user", "content": text_body})
    history[from_phone].append({"role": "assistant", "content": answer})

    await send_whatsapp_text(from_phone, answer)
    return {"ok": True}
