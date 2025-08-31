# WhatsApp LLM Bot (FastAPI + Groq + WhatsApp Cloud API)

Starter para criar um bot no WhatsApp usando a **Cloud API da Meta** com resposta via LLM (Groq).

## Passos
1) Meta for Developers → habilite **WhatsApp Cloud API** e obtenha:
   - `WHATSAPP_PHONE_NUMBER_ID`
   - `WHATSAPP_TOKEN`
2) Groq Cloud → gere `GROQ_API_KEY`.
3) Deploy (Render):
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Variáveis: `APP_VERIFY_TOKEN`, `WHATSAPP_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID`, `GROQ_API_KEY`, opcional `MODEL_ID`
4) Webhook (Meta > WhatsApp > Configuration):
   - Callback URL: `https://SEU-SERVICO.onrender.com/webhook`
   - Verify Token: igual a `APP_VERIFY_TOKEN`
   - Verifique e assine o tópico `messages`.
5) Teste com o número de teste da Meta. Comandos: `/help`, `/start`, `/reset`.
