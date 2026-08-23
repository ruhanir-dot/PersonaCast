from __future__ import annotations
import json
import random
import re
import threading
import time
from typing import TypeVar
from pydantic import BaseModel, ValidationError
from collections import deque
from .. import config

from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError

T = TypeVar("T", bound=BaseModel) # T generic type variable bound to basemodel
_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.DOTALL) # to extract Json blocks using regex, compile once for easy usage 

def _retry_after_seconds(error: Exception | None) -> float | None:
    """
    Reads Retry-After header (if there is one) from api returns how many seconds to wait

    Adjusted to use read HTTP headers, and JSON body
    """

    if error is None: 
        return None
    
    response = getattr(error, "response", None)

    if response is not None: 
        header = response.headers.get("retry-after") # will hold string value of error, or None
        
        if header: 
            try: 
                return float(header) 
            except ValueError: 
                pass 
        
    body = getattr(error,"body", None)

    # gemini wraps error body in a list so need to extract from list  
    if isinstance(body, list) and body:
        body = body[0]

    if isinstance(body, dict):
        metadata  = body.get("metadata") or {}

        if "retry_after_seconds" in metadata:
            return float(metadata["retry_after_seconds"])

        # gemini nests the hint as error.details[].retryDelay = "55s"
        error_obj = body.get("error") or {} # get error object from body object
        
        for detail in error_obj.get("details", []): # iterate through details provided
            delay = detail.get("retryDelay") if isinstance(detail, dict) else None # if the detail is a dict, grab retry delay
            if delay:
                try:
                    return float(str(delay).rstrip("s")) # strip s 
                except ValueError:
                    pass

    return None

def _extract_json(text): 
    """
    handiling markdown fences strip and return JSON string

    First try to find fenced block, else goes back to _JSON_BLOCK regex 
    """
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip() # extract block from capturning group (.*?)
    
    match = _JSON_BLOCK.search(text)

    return match.group(0) if match else text.strip()

### lookup tables for the strict schema method
_SCHEMA_ANNOTATIONS = ("default", "title")
_NAME_MAPS = ("properties", "$defs")

def _strict_schema(node):
    """
    we take in the pydantic describing itself proceeding model_json_schema() 
    from here we rewrite the description into something the decoder can compile 
    """
    if isinstance(node, list):
        return [_strict_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    out = {}
    for key, value in node.items():
        if key in _SCHEMA_ANNOTATIONS:
            continue
        if key in _NAME_MAPS and isinstance(value, dict):
            # var key is the key in out dict that maps to another dict key is name which are from value.items()
            # values are recirsive calling of strict schema on inner values
            out[key] = {name: _strict_schema(sub) for name, sub in value.items()}
        else:
            out[key] = _strict_schema(value)

    if out.get("type") == "object" and isinstance(out.get("properties"), dict):
        out["additionalProperties"] = False
        out["required"] = list(out["properties"].keys())

    return out


class BudgetExceeded(RuntimeError):
    pass


class LLMClient:
    def __init__(self, model = None, *, on_stage = None):
        self.model = model or config.MODEL
        self._client = OpenAI(
            base_url = config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            max_retries=0,
            timeout=config.LLM_REQUEST_TIMEOUT_SECONDS
        )

        self.on_stage = on_stage # progress and status callback

        self._calls: deque[float] = deque() # call double ended queue for sliding window of 60 
        self._last_call_at = 0.0
        self._lock = threading.Lock() # instance of threading lock object don;t have to instantiate again to lock something

        self._native_json: bool | None = None

        self.stats = {"calls": 0, "rate_wait": 0.0, "api_time": 0.0, "http_retries": 0, "json_retries": 0} # per llm clienet instance cumulative since construction 

    def snapshot(self) -> dict:
        return dict(self.stats) # snapshot of stats

    def since(self, before: dict) -> dict:
        """
        allows us to find time diff between snapshot at time t subtrated from snapshot at time t-1 to get 
        how much time a response took 
        """
        return {k: round(self.stats[k] - before.get(k, 0), 3) for k in self.stats}

    def _note(self, message: str) -> None:
        """
        relay message 
        """
        if self.on_stage:
            self.on_stage(message)

    def _expire(self, now: float) -> None:
        """
        remove any old call timestamps in the 60 second window 
        self._calls is double ended queue append right, pop left FIFO
        """
        window = config.LLM_RPM_WINDOW_SECONDS
        while self._calls and now - self._calls[0] >= window: # while the deque and now to the call time stamp of the oldest call is within a 60 second window we can pop left
            self._calls.popleft()

    def _admit(self, priority: str = "critical") -> bool:
        """
        priority mechanism, for admitting calls in 
        """
        while True:
            message = None
            with self._lock: # while thread lock 
                now = time.monotonic()
                self._expire(now) # expire any stale calls 
                free = config.LLM_MAX_RPM - len(self._calls) # checking free slots

                if priority == "background" and free <= config.LLM_BACKGROUND_RESERVE: # check if priority of call isi low and there is only calls for important reeve calls 
                    return False

                if free > 0: # if we have space in our slots
                    floor_wait = config.LLM_MIN_INTERVAL_SECONDS - (now - self._last_call_at) # required gap between calls - elapsed gap between calls 
                    if floor_wait <= 0: # if gap is ssatisfied, admit the call 
                        self._calls.append(now) 
                        self._last_call_at = now
                        return True
                    wait = floor_wait
                else: # else then wait 
                    wait = config.LLM_RPM_WINDOW_SECONDS - (now - self._calls[0])
                    message = f"rate limit — waiting {wait:.0f}s for the window to clear"

            if message:
                self._note(message)
            slept = max(wait, 0.01)
            time.sleep(slept)
            self.stats["rate_wait"] += slept


    def complete(self, system, user, temperature = 0.7, *, priority: str = "critical", response_format = None):
        """
        complete main method usage 
        """
        last_error = None
        for attempt in range(config.LLM_MAX_RETRIES): # for atempt in range of max retrues 
            if not self._admit(priority if attempt == 0 else "critical"):  # message to display a background call not admitted 
                raise BudgetExceeded("background call refused, rate limit window too full")
            try:
                ### http call to openai endpoint system + user 
                kwargs = {"response_format": response_format} if response_format else {}
                _api_t0 = time.perf_counter()
                response = self._client.chat.completions.create(
                    model = self.model,
                    messages = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    temperature = temperature,
                    **kwargs,
                )
                self.stats["api_time"] += time.perf_counter() - _api_t0
                self.stats["calls"] += 1
                return response.choices[0].message.content or ""
            except (RateLimitError, APIConnectionError) as error: # retry without raising error 
                last_error = error
            except APIStatusError as error: # raising error 
                if error.status_code >= 500:
                    last_error = error
                else:
                    raise

            self.stats["api_time"] += time.perf_counter() - _api_t0
            self.stats["http_retries"] += 1
            retry_after = _retry_after_seconds(last_error)
            sleep = retry_after if retry_after is not None else (config.LLM_BACKOFF_BASE_SECONDS * (2 ** attempt) + random.random())
            time.sleep(sleep)

        raise RuntimeError(f"LLM call failed after {config.LLM_MAX_RETRIES} retries") from last_error

    def structured(self, system, user, schema: type[T], temperature = 0.7, *, priority: str = "critical") -> T:
        """
        retrying for strict json schema 
        
        """
        if self._native_json is not False:
            try:
                raw = self.complete(
                    system, user, temperature=temperature, priority=priority,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema.__name__,
                            "schema": _strict_schema(schema.model_json_schema()),
                            "strict": True,
                        },
                    },
                )
                parsed = schema.model_validate_json(_extract_json(raw))
                self._native_json = True
                return parsed
            except BudgetExceeded:
                raise
            except (APIStatusError, TypeError, ValidationError, json.JSONDecodeError) as err:
                if self._native_json is True:
                    raise
                self._native_json = False
                self._note(f"native json mode unavailable ({type(err).__name__}), using prompt schema")

        schema_hint = json.dumps(schema.model_json_schema(), indent= 2)
        sys_with_schema = (
            f"{system}\n\nRespond with ONLY valid JSON matching this schema (no prose, no "
            f"markdown fences):\n{schema_hint}"
        )

        nudge = ""
        last_err = None

        for _ in range(config.LLM_MAX_RETRIES):
            raw = self.complete(sys_with_schema, user + nudge, temperature= temperature,
                                priority=priority)
            try:
                return schema.model_validate_json(_extract_json(raw))
            except (ValidationError, json.JSONDecodeError) as err:
                last_err = err
                self.stats["json_retries"] += 1
                nudge = (
                    f"\n\nYour previous reply did not validate: {err}\n"
                    "Return ONLY the corrected JSON."
                )
        raise RuntimeError(f"LLM returned malformed JSON {config.LLM_MAX_RETRIES}x") from last_err
