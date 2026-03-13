# Token counter for LLM API calls (compatible with MTKGA-main usage)
import threading
global_tokens = 0
_lock = threading.Lock()


def update_add_var(new_val):
    global global_tokens
    with _lock:
        global_tokens = global_tokens + new_val


def get_tokens():
    global global_tokens
    return global_tokens


def reset_tokens():
    global global_tokens
    global_tokens = 0
