"""
Base agent class with shared AWS Bedrock / Anthropic API access.
Supports local vLLM endpoint via USE_LOCAL_VLLM env var.
"""

import json
import os
import re
import time
from typing import Any, Dict, Optional

import config

# Retry settings for network resilience
MAX_RETRIES = 5
RETRY_BASE_DELAY = 10  # seconds, doubles each retry (sleeps 10, 20, 40, 80; the 5th failure raises)

# Local vLLM endpoint (set USE_LOCAL_VLLM=true to use)
USE_LOCAL_VLLM = os.environ.get("USE_LOCAL_VLLM", "").lower() in ("true", "1", "yes")
VLLM_ENDPOINT = os.environ.get("VLLM_ENDPOINT", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "")
VLLM_THINKING = os.environ.get("VLLM_THINKING", "on").lower() != "off"
# Per-request timeout (seconds) for the local vLLM client; the SDK default of 600 s is too
# short for 16k-token generations under batched load. Env-overridable, logged in run_config.
VLLM_REQUEST_TIMEOUT = float(os.environ.get("VLLM_REQUEST_TIMEOUT") or 1800)


class BaseAgent:
    """
    Provides a shared call_llm() interface used by all four agents.
    Routes to a local vLLM server (USE_LOCAL_VLLM env var), else to AWS Bedrock or the direct Anthropic API per config.
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if USE_LOCAL_VLLM:
            from openai import OpenAI
            self._client = OpenAI(base_url=VLLM_ENDPOINT, api_key="unused", timeout=VLLM_REQUEST_TIMEOUT)
        elif config.USE_AWS_BEDROCK:
            import boto3
            self._client = boto3.client("bedrock-runtime", region_name=config.AWS_REGION)
        else:
            import anthropic
            self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

        return self._client

    def call_llm(
        self,
        prompt: str,
        system_prompt: str = "You are a mathematical research assistant with expertise in formal mathematics.",
        max_tokens: int = 1024,
        temperature: float = None,
    ) -> str:
        """
        Call the configured LLM (Claude via Bedrock/Anthropic, or a local vLLM model) and return the text response.

        Args:
            prompt: User message
            system_prompt: System message
            max_tokens: Max tokens in response
            temperature: Sampling temperature (defaults to config.AGENT_TEMPERATURE)

        Returns:
            Response text string
        """
        if temperature is None:
            temperature = config.AGENT_TEMPERATURE

        for attempt in range(MAX_RETRIES):
            try:
                client = self._get_client()

                if USE_LOCAL_VLLM:
                    user_content = prompt
                    if not VLLM_THINKING:
                        user_content = prompt + " /no_think"
                    response = client.chat.completions.create(
                        model=VLLM_MODEL or "default",
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                    )
                    text = response.choices[0].message.content
                    # Strip <think>...</think> blocks (complete or truncated)
                    if "<think>" in text:
                        if "</think>" in text:
                            text = text.split("</think>", 1)[-1].strip()
                        else:
                            # Truncated thinking - remove everything from <think> onward
                            text = text.split("<think>", 1)[0].strip()
                    return text
                elif config.USE_AWS_BEDROCK:
                    body = json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": prompt}],
                    })
                    response = client.invoke_model(modelId=config.MODEL_ID, body=body)
                    result = json.loads(response["body"].read())
                    return result["content"][0]["text"]
                else:
                    import anthropic
                    message = client.messages.create(
                        model=config.MODEL_NAME,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system_prompt,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return message.content[0].text

            except Exception as e:
                msg = str(e)
                # vLLM context-length overflow (HTTP 400 "maximum context length is N ..."):
                # the prompt plus the requested output exceeds the window. Refit max_tokens
                # to the room left and retry with the same prompt, so the model still sees
                # the full augmentation context. Any parsing failure falls through to the
                # normal retry below.
                if USE_LOCAL_VLLM and (
                    "maximum context length" in msg or "reduce the length" in msg
                ):
                    try:
                        m_ctx = re.search(r"maximum context length is (\d+)", msg)
                        m_in = re.search(r"contains at least (\d+)", msg)
                        if m_ctx and m_in:
                            window = int(m_ctx.group(1))
                            reported_in = int(m_in.group(1))
                            room = window - reported_in
                            # The reported input size is a lower bound, so keep a margin:
                            # 1024 tokens when there is room, half the room when it is tight.
                            margin = min(1024, room // 2) if room > 0 else 1024
                            fitted = room - margin
                            # Refit only if the new budget is positive and smaller than the
                            # current one; otherwise fall through to the normal retry.
                            if 0 < fitted < max_tokens:
                                self.log(
                                    f"Context overflow: refitting max_tokens "
                                    f"{max_tokens} -> {fitted} and retrying"
                                )
                                max_tokens = fitted
                                continue
                    except Exception:
                        pass  # parsing failed -> fall through to normal retry/backoff

                delay = RETRY_BASE_DELAY * (2 ** attempt)
                is_last = attempt == MAX_RETRIES - 1
                error_type = type(e).__name__
                if is_last:
                    self.log(f"API call failed after {MAX_RETRIES} retries: {error_type}: {e}")
                    raise
                self.log(f"API call failed ({error_type}), retrying in {delay}s... (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(delay)
                # Reset client in case connection is stale
                self._client = None

        # Reached only if a refit `continue` lands on the final iteration.
        raise RuntimeError("call_llm exhausted all retries without a response")

    def call_llm_json(
        self,
        prompt: str,
        system_prompt: str = "You are a mathematical research assistant. Always respond with valid JSON.",
        max_tokens: int = 1024,
    ) -> Optional[Dict]:
        """
        Call LLM and parse JSON from response. Returns None on parse failure.
        """
        response = self.call_llm(prompt, system_prompt, max_tokens)
        return self._extract_json(response)

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON object from LLM response, handling markdown fences."""
        # Try direct parse first
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding a JSON object anywhere in the text
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    verbose = True

    def log(self, message: str):
        if self.verbose:
            print(f"[{self.agent_name}] {message}")
