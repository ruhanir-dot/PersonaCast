from __future__ import annotations
import json
import random
import re
import time
from typing import TypeVar
from pydantic import BaseModel, ValidationError
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

class LLMClient:
    """
    OpenRouter-backed structured-output client.
    """
    def __init__(self, model = None): 
        self.model = model or config.MODEL
        self._client = OpenAI(
            base_url = config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            ### we are doing own retry and backoff handling 
            max_retries=0,
            timeout=config.LLM_REQUEST_TIMEOUT_SECONDS
        )

        self._last_call_at = 0.0 # timestamp initialization 

    def _throttle(self):
        """
        before every API call check if enough time has passed since our last call
        defined in `config.LLM_REQUEST_TIMEOUT_SECONDS`

        if not we sleep through difference, to prevent hitting rate limits
        """
        gap = config.LLM_MIN_INTERVAL_SECONDS

        if gap <= 0: 
            return 
        
        seconds_since_last_call = time.monotonic() - self._last_call_at
        wait = gap - seconds_since_last_call # calculate wait time till nect call
        if wait > 0: 
            time.sleep(wait)
        
        self._last_call_at = time.monotonic() # reset timer 


    def complete(self, system, user, temperature = 0.7): 
        last_error = None 
        for attempt in range(config.LLM_MAX_RETRIES): # loop for max retries
            self._throttle() # method to garuntee proper time passed between last API call
            try: 
                response = self._client.chat.completions.create(
                    model = self.model,
                    messages = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ], 
                    temperature = temperature
                ) # llm client with our params
                return response.choices[0].message.content or "" # if sucess extracts the string response
            except (RateLimitError, APIConnectionError) as error: # if rate limit error
                last_error = error # set last error to thsi error object
            except APIStatusError as error: # catch HTTP error codes
                if error.status_code >= 500: # 500 and huigher server error
                    last_error = error # set last exception object as last error 
                else: 
                    raise # 
            
            retry_after = _retry_after_seconds(last_error) # call helpeer  function to check how many seconds to wait
            sleep = retry_after if retry_after is not None else (config.LLM_BACKOFF_BASE_SECONDS * (2 ** attempt) + random.random())
            # if server provided retry after we use that else, we use exponential backoff
            time.sleep(sleep) # sleep for that time and retry attempt till for loop expedning

        raise RuntimeError(f"LLM call failed after {config.LLM_MAX_RETRIES} retries") from last_error # raise error if nothing worked
    
    def structured(self, system, user, schema: type[T], temperature = 0.7 ) -> T: 
        """
        to wrap complete to get pydantic objects back instead of raw strings
        
        append Pydantic schema as JSON to system prompt so model knows shape to return
        calls complete, parses and valudate respinse with `model_validate_json`, if validation fails add nudge
        to next attempt to tell model what went wrong, out of reties then RunTime Error
        """
        schema_hint = json.dumps(schema.model_json_schema(), indent= 2) # extract pydantic structure convert to json
        sys_with_schema = (
            f"{system}\n\nRespond with ONLY valid JSON matching this schema (no prose, no "
            f"markdown fences):\n{schema_hint}"
        ) # compliance instructions given 

        nudge = "" 
        last_error = None 

        for _ in range(config.LLM_MAX_RETRIES): # retry loop for formatting issues
            raw = self.complete(sys_with_schema, user + nudge, temperature= temperature)
            try:
                return schema.model_validate_json(_extract_json(raw)) # strip md blocks or conversational text
            except (ValidationError, json.JSONDecodeError) as err:
                last_err = err
                nudge = (
                    f"\n\nYour previous reply did not validate: {err}\n"
                    "Return ONLY the corrected JSON."
                )
        raise RuntimeError(f"LLM returned malformed JSON {config.LLM_MAX_RETRIES}x") from last_err


        








