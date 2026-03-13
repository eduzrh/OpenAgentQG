"""
Minimal API test. Run: export OPENAI_API_KEY=sk-xxx; python test_api.py
"""
import sys
import httpx
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_API_BASE

if not OPENAI_API_KEY:
    print("Set OPENAI_API_KEY and retry.")
    sys.exit(1)

client = OpenAI(
    base_url=OPENAI_API_BASE,
    api_key=OPENAI_API_KEY,
    http_client=httpx.Client(base_url=OPENAI_API_BASE, follow_redirects=True),
)

if __name__ == "__main__":
    print("Request: POST .../chat/completions")
    try:
        r = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
        )
        print("OK:", (r.choices[0].message.content or "")[:80])
    except Exception as e:
        print("Error:", e)
