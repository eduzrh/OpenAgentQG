"""
LLM client for OpenAgentQG. Compatible with API proxies. Auto-retry on failure.
"""
import time
import httpx
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError

import tokens_cal
from config import OPENAI_API_BASE, OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL

MAX_LLM_RETRIES = 3
RETRY_BACKOFF = [2, 4, 8]  # seconds
RATE_LIMIT_BACKOFF = [10, 25, 60]  # 429 backoff
HTTP_TIMEOUT = 120.0  # seconds


def get_client():
    """Chat completion with base_url, api_key, httpx.Client."""
    base = OPENAI_API_BASE or "https://api.openai.com/v1"
    return OpenAI(
        base_url=base,
        api_key=OPENAI_API_KEY,
        timeout=HTTP_TIMEOUT,
        http_client=httpx.Client(
            base_url=base,
            follow_redirects=True,
            timeout=HTTP_TIMEOUT,
        ),
    )


# Module-level client (lazy init)
_client = None


def client():
    global _client
    if _client is None:
        _client = get_client()
    return _client


def chat(messages, model=None, temperature=0.7):
    """
    Single chat completion; updates tokens_cal.global_tokens.
    遇连接/超时错误自动重试，避免中转断开导致整批失败。
    Returns: (content_str, usage_total_tokens)
    """
    last_err = None
    for attempt in range(MAX_LLM_RETRIES):
        try:
            resp = client().chat.completions.create(
                model=model or LLM_MODEL,
                messages=messages,
                temperature=temperature,
                stream=False,
            )
            usage = getattr(resp, "usage", None)
            total = usage.total_tokens if usage else 0
            tokens_cal.update_add_var(total)
            content = (resp.choices[0].message.content or "").strip()
            return content, total
        except RateLimitError as e:
            last_err = e
            if attempt < MAX_LLM_RETRIES - 1:
                delay = RATE_LIMIT_BACKOFF[min(attempt, len(RATE_LIMIT_BACKOFF) - 1)]
                time.sleep(delay)
            else:
                raise
        except (APIConnectionError, APITimeoutError, httpx.RemoteProtocolError) as e:
            last_err = e
            if attempt < MAX_LLM_RETRIES - 1:
                delay = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                time.sleep(delay)
    raise last_err


def embed(text, model=None):
    """
    Get embedding for a single text. Uses OpenAI embeddings API (same base_url/api_key as chat).
    Returns: list of floats (embedding vector), or None on failure.
    """
    emb_model = model or EMBEDDING_MODEL
    if not (text or "").strip():
        return None
    try:
        resp = client().embeddings.create(
            model=emb_model,
            input=(text or "").strip()[:8192],
        )
        if resp.data and len(resp.data) > 0:
            vec = resp.data[0].embedding
            if getattr(resp, "usage", None) and resp.usage.total_tokens:
                tokens_cal.update_add_var(resp.usage.total_tokens)
            return vec
    except Exception:
        pass
    return None


def embed_batch(texts, model=None):
    """
    Get embeddings for multiple texts in one API call (OpenAI supports batch).
    Returns: list of list of floats (or None for failed items).
    """
    emb_model = model or EMBEDDING_MODEL
    inputs = [(t or "").strip()[:8192] for t in texts if (t or "").strip()]
    if not inputs:
        return []
    try:
        resp = client().embeddings.create(
            model=emb_model,
            input=inputs,
        )
        if not resp.data:
            return [None] * len(texts)
        # resp.data is in same order as input
        vecs = [d.embedding for d in resp.data]
        if getattr(resp, "usage", None) and resp.usage.total_tokens:
            tokens_cal.update_add_var(resp.usage.total_tokens)
        # Map back to original texts (skip empty)
        out = []
        j = 0
        for t in texts:
            if not (t or "").strip():
                out.append(None)
            else:
                out.append(vecs[j] if j < len(vecs) else None)
                j += 1
        return out
    except Exception:
        return [None] * len(texts)


def chat_with_logprobs(messages, model=None):
    """
    Chat completion with logprobs for entropy computation (meta-neural virtual nodes).
    Auto-retry on connection/timeout.
    Returns: (content_str, logprobs_per_token_or_none, total_tokens)
    """
    last_err = None
    for attempt in range(MAX_LLM_RETRIES):
        try:
            resp = client().chat.completions.create(
                model=model or LLM_MODEL,
                messages=messages,
                temperature=0.0,
                stream=False,
                logprobs=True,
                top_logprobs=5,
            )
            usage = getattr(resp, "usage", None)
            total = usage.total_tokens if usage else 0
            tokens_cal.update_add_var(total)
            content = (resp.choices[0].message.content or "").strip()
            logprobs = getattr(resp.choices[0].message, "logprobs", None)
            return content, logprobs, total
        except RateLimitError as e:
            last_err = e
            if attempt < MAX_LLM_RETRIES - 1:
                delay = RATE_LIMIT_BACKOFF[min(attempt, len(RATE_LIMIT_BACKOFF) - 1)]
                time.sleep(delay)
            else:
                raise
        except (APIConnectionError, APITimeoutError, httpx.RemoteProtocolError) as e:
            last_err = e
            if attempt < MAX_LLM_RETRIES - 1:
                delay = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                time.sleep(delay)
    raise last_err
