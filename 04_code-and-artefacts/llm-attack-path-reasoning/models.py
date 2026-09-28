"""
Model clients.

One function per provider, all with the same signature:

    call(prompt, model) -> (text, input_tokens, output_tokens)

The point of this file is the tutor's second request: the conclusions currently
rest on one model, so the runner has to be able to address several without the
experiment code knowing which one it is talking to. Adding a provider means
adding a function here and one line in CLIENTS.

Free or near-free options, in the order worth trying:

  ollama    fully local, no key, no quota. The only option that removes the
            quota constraint reported in Section 4.10 of the paper, and the
            only one that is reproducible without an account. Use it for the
            large runs.
  gemini    free tier, no card. Ten requests a minute, so pass --delay 7.
  openai    any OpenAI-compatible endpoint. Set OPENAI_BASE_URL to point at
            OpenRouter, Groq, DeepSeek, Mistral or a local vLLM server. This
            is the cheapest way to reach a third model family.
  anthropic paid.
"""

import json
import os
import urllib.request


class ModelError(RuntimeError):
    pass


def _post(url, body, headers, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


# ------------------------------------------------------------------ providers

def call_mock(prompt, model):
    """Deterministic stand-in so the whole pipeline, including C2, can be run
    and debugged without a key and without spending quota. It reads the
    scenario back out of the prompt and answers correctly with a probability
    that falls as the prompt grows, which is enough to exercise every branch."""
    import hashlib
    import re
    ids = re.findall(r"\b(?:VULN-\d{3}|CVE-\d{4}-\d+)\b", prompt)
    ids = list(dict.fromkeys(ids))
    h = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
    if not ids:
        return '{"chain": []}', len(prompt) // 4, 20
    pick = ids[h % len(ids)]
    if "Answer with one identifier" in prompt:          # C2 turn
        return json.dumps({"next": pick, "reasoning": "mock"}), len(prompt) // 4, 20
    n = 1 + h % 4
    return (json.dumps({"chain": ids[:n], "reasoning": "mock"}),
            len(prompt) // 4, 20 + 6 * n)


def call_gemini(prompt, model):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ModelError("GEMINI_API_KEY is not set")
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        # Recent Flash models spend part of the output budget on internal
        # reasoning before emitting text, so a small budget can be exhausted
        # before any answer appears.
        "generationConfig": {"temperature": 0, "maxOutputTokens": 8192},
        # Synthetic scenarios written in offensive-security vocabulary. Relax
        # to the high-confidence threshold so legitimate prompts are not
        # blocked; anything actually blocked is reported rather than silently
        # recorded as an empty answer.
        "safetySettings": [{"category": c, "threshold": "BLOCK_ONLY_HIGH"}
                           for c in ("HARM_CATEGORY_HARASSMENT",
                                     "HARM_CATEGORY_HATE_SPEECH",
                                     "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                     "HARM_CATEGORY_DANGEROUS_CONTENT")],
    }
    data = _post(url, body, {"Content-Type": "application/json",
                             "x-goog-api-key": key})
    cands = data.get("candidates", [])
    text = ""
    if cands:
        text = "".join(p.get("text", "")
                       for p in cands[0].get("content", {}).get("parts", []))
    if not text:
        reason = cands[0].get("finishReason", "unknown") if cands else "no candidate"
        blocked = data.get("promptFeedback", {}).get("blockReason")
        raise ModelError(f"empty response (finishReason={reason}"
                         + (f", blockReason={blocked}" if blocked else "") + ")")
    u = data.get("usageMetadata", {})
    return text, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)


def call_ollama(prompt, model):
    """Local inference through Ollama.

    num_ctx is set explicitly and never left to the default. Ollama's default
    context is small, commonly 2048 tokens, and the prompts of this study reach
    about 2070 tokens at sixty distractors. Once the prompt exceeds the window
    it is truncated from the front, which is where the instructions and the
    initial state sit, so the model would be asked to find a path without being
    told where it starts. That would corrupt exactly the cells the study is
    about, and it would leave no trace in the output.

    num_gpu is the number of transformer layers offloaded to the GPU. Ollama
    decides it automatically from the available VRAM, and the automatic choice
    is usually right, so it is only overridden when OLLAMA_NUM_GPU is set.
    """
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    options = {
        "temperature": 0,
        "num_predict": 1024,
        "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "4096")),
    }
    if os.environ.get("OLLAMA_NUM_GPU"):
        options["num_gpu"] = int(os.environ["OLLAMA_NUM_GPU"])

    body = {"model": model, "prompt": prompt, "stream": False,
            "options": options}
    data = _post(f"{host}/api/generate", body,
                 {"Content-Type": "application/json"}, timeout=900)

    used = data.get("prompt_eval_count", 0)
    if used and used >= options["num_ctx"] - 64:
        raise ModelError(
            f"prompt filled the context window ({used} of "
            f"{options['num_ctx']} tokens): raise OLLAMA_NUM_CTX before "
            f"trusting this run")
    return data.get("response", ""), used, data.get("eval_count", 0)


def call_openai(prompt, model):
    """Any OpenAI-compatible chat endpoint."""
    key = os.environ.get("OPENAI_API_KEY")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not key:
        raise ModelError("OPENAI_API_KEY is not set")
    body = {"model": model, "temperature": 0, "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}]}
    data = _post(f"{base}/chat/completions", body,
                 {"Content-Type": "application/json",
                  "Authorization": f"Bearer {key}"})
    text = data["choices"][0]["message"]["content"]
    u = data.get("usage", {})
    return text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def call_anthropic(prompt, model):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ModelError("ANTHROPIC_API_KEY is not set")
    body = {"model": model, "max_tokens": 2048, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    data = _post("https://api.anthropic.com/v1/messages", body,
                 {"Content-Type": "application/json",
                  "x-api-key": key, "anthropic-version": "2023-06-01"})
    text = "".join(b.get("text", "") for b in data.get("content", [])
                   if b.get("type") == "text")
    u = data.get("usage", {})
    return text, u.get("input_tokens", 0), u.get("output_tokens", 0)


CLIENTS = {
    "mock": call_mock,
    "gemini": call_gemini,
    "ollama": call_ollama,
    "openai": call_openai,
    "anthropic": call_anthropic,
}


def get_client(provider):
    if provider not in CLIENTS:
        raise ModelError(f"unknown provider {provider!r}; "
                         f"choose from {', '.join(CLIENTS)}")
    return CLIENTS[provider]


def list_gemini_models():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("GEMINI_API_KEY is not set.")
        return
    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models",
        headers={"x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    print("Models this key can use (usable ones marked *):\n")
    for m in data.get("models", []):
        name = m.get("name", "").replace("models/", "")
        usable = "generateContent" in m.get("supportedGenerationMethods", [])
        print(f"{' *' if usable else '  '} {name}")
    print("\nPrefer a flash model: pro models usually require billing.")
