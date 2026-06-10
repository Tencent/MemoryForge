import json
import logging
import asyncio
import sys
import time
import datetime
from typing import Type, TypeVar, Optional, List, Dict, Any

from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    stop_never,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryCallState
)

# Use litellm unified API, supports async calls
# Before running, ensure installation: pip install litellm pydantic tenacity httpx
import litellm
from litellm import acompletion

# Initialize logger
logger = logging.getLogger(__name__)

# Define generic type for Pydantic type inference
T = TypeVar('T', bound=BaseModel)

import re as _re

def _strip_trailing_commas(s: str) -> str:
    """Remove trailing commas before } or ] to tolerate non-strict JSON from LLMs.

    Handles patterns like:
        {"key": "value",}   →  {"key": "value"}
        [1, 2, 3,]          →  [1, 2, 3]
    Uses a simple regex that is safe for typical LLM JSON output.
    """
    return _re.sub(r',\s*([}\]])', r'\1', s)

class LLMGenerationError(Exception):
    """Custom error raised during LLM generation or parsing"""
    pass


class ContentPolicyError(Exception):
    """Non-retryable error: prompt flagged by content policy (e.g. OpenAI invalid_prompt).

    This is intentionally NOT in the tenacity retry list so that content-policy
    violations fail fast instead of burning 10 retry attempts.
    """
    pass


def _is_content_policy_error(exc: Exception) -> bool:
    """Return True if *exc* is a content-policy / invalid-prompt rejection."""
    err_str = str(exc).lower()
    return (
        "contentpolicyviolation" in err_str
        or "invalid_prompt" in err_str
        or "flagged as potentially violating" in err_str
        or "content_policy_violation" in err_str
    )


# ---------------------------------------------------------------------------
# Module 10: Adaptive Rate Limiter (AIMD-style)
# ---------------------------------------------------------------------------

class AdaptiveRateLimiter:
    """AIMD-style rate limiter: additive increase, multiplicative decrease.
    
    Replaces the fixed RPM throttle. Starts at `initial_rpm` and:
    - On success: slowly increases RPM (additive increase)
    - On 429 rate limit: halves RPM (multiplicative decrease)
    """

    def __init__(
        self,
        initial_rpm: int = 30,
        min_rpm: int = 5,
        max_rpm: int = 120,
        increase_step: int = 2,
        decrease_factor: float = 0.5,
        success_threshold: int = 10,
    ):
        self._current_rpm: float = float(initial_rpm)
        self._min_rpm: int = min_rpm
        self._max_rpm: int = max_rpm
        self._increase_step: int = increase_step
        self._decrease_factor: float = decrease_factor
        self._success_threshold: int = success_threshold
        self._consecutive_successes: int = 0
        self._request_timestamps: List[float] = []
        self._lock: Optional[asyncio.Lock] = None

    @property
    def current_rpm(self) -> float:
        return self._current_rpm

    async def acquire(self) -> None:
        """Wait until a request slot is available within current RPM."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            now = time.time()
            # Clean old timestamps
            self._request_timestamps = [
                ts for ts in self._request_timestamps if now - ts < 60
            ]
            effective_rpm = int(self._current_rpm)
            if len(self._request_timestamps) >= effective_rpm:
                oldest = self._request_timestamps[0]
                wait_time = 60 - (now - oldest) + 0.5
                if wait_time > 0:
                    logger.debug(
                        f"AdaptiveRateLimiter: waiting {wait_time:.1f}s "
                        f"(current RPM={effective_rpm})"
                    )
                    await asyncio.sleep(wait_time)
            self._request_timestamps.append(time.time())

    def on_success(self) -> None:
        """Record a successful request. Slowly increase RPM."""
        self._consecutive_successes += 1
        if self._consecutive_successes >= self._success_threshold:
            old_rpm = self._current_rpm
            self._current_rpm = min(
                self._current_rpm + self._increase_step,
                self._max_rpm,
            )
            self._consecutive_successes = 0
            if self._current_rpm != old_rpm:
                logger.info(
                    f"AdaptiveRateLimiter: RPM increased {old_rpm:.0f} -> {self._current_rpm:.0f}"
                )

    def on_rate_limit(self, retry_after: Optional[float] = None) -> float:
        """Record a rate limit hit. Halve RPM and return wait time."""
        old_rpm = self._current_rpm
        self._current_rpm = max(
            self._current_rpm * self._decrease_factor,
            self._min_rpm,
        )
        self._consecutive_successes = 0
        logger.warning(
            f"AdaptiveRateLimiter: RPM decreased {old_rpm:.0f} -> {self._current_rpm:.0f} "
            f"(rate limited)"
        )
        return retry_after or 5.0


# ---------------------------------------------------------------------------
# Module 9: CALM v2 — Cost-Aware LLM Mixing Router
# ---------------------------------------------------------------------------

class CALMRouterV2:
    """Cost-Aware LLM Mixing: route tasks to appropriate model tiers.
    
    Tier 1 (Economy):  gpt-5.4-mini  — structural judgment, validation, no creativity
    Tier 2 (Balanced): gpt-5.4-mini  — understanding + moderate creativity
    Tier 3 (Premium):  gpt-5.2       — core narrative creativity, factual accuracy, persona coherence
    """

    TASK_TIER_MAP: Dict[str, int] = {
        # ═══ Tier 1: Economy — structural judgment, no creativity ═══
        "step3_persona_validation": 1,       # P2/P3 persona validation
        "event_resolution": 1,               # P3 event resolution
        "step4_persona_validation": 1,       # P2 batch persona validation

        # P1a Tier 1 (pure structural/deterministic tasks)
        "p1a_social_context": 1,             # Social context inference
        "p1a_transition_dates": 1,           # Transition date hints
        "p1a_plan_validation": 1,            # Plan validation report
        "p1a_error_attribution": 1,          # Error attribution

        # P1a Tier 2 (upgraded: require reasoning to catch grade drift & over-segmentation)
        "p1a_plan_review": 2,                # Plan review — needs reasoning to catch grade drift
        "p1a_plan_judgement": 2,             # Plan judgement — needs reasoning to detect over-segmentation
        "p1a_duration_validation": 2,        # Duration validation — needs social context understanding

        # P2 new Tier 1
        "p2_participant_sufficiency": 1,     # Participant sufficiency check

        # P3 new Tier 1
        "p3_pool_supplement": 1,             # Pool supplement decision

        # ═══ Tier 2: Balanced — understanding + moderate creativity ═══
        "step2_participant_selection": 2,    # P2 participant selection
        "interaction_history": 2,            # P2 interaction history
        "uper_reflection": 2,                # P2 reflection
        "era_context": 2,                    # P3 era context
        "character_style": 2,                # P3 character style
        "refined_summary": 2,                # Refined summary

        # P1a new Tier 2
        "p1a_period_refresh": 2,             # Period content refresh
        "p1a_year_enrichment": 2,            # Year enrichment (key_life_path)

        # P1b new Tier 2
        "p1b_basic_identity": 2,             # Basic identity generation
        "p1b_initial_memories": 2,           # Initial memories
        "p1b_temporal_briefs": 2,            # Temporal briefs
        "p1b_full_profile": 2,               # Full profile (supporting)

        # P2 new Tier 2
        "p2_year_enrichment": 2,             # Year enrichment
        "p2_new_participant": 2,             # New participant generation

        # P3 new Tier 2
        "p3_participant_refinement": 2,      # Participant refinement

        # ═══ Tier 3: Premium — core narrative creativity + factual accuracy ═══
        "step1a_event_framework": 3,         # P2 low-res framework
        "step1b_high_res_outline": 3,        # P2 high-res outline
        "director_selection": 3,             # P3 director selection
        "agent_action": 3,                   # P3 agent action
        "post_scene_memory": 3,              # P3 post-scene memory

        # Upgraded from Tier 2 → Tier 3 (require persona coherence & factual accuracy)
        "persona_update": 3,                 # P2/P1b persona update — must respect location/timeline
        "scene_outline": 3,                  # P3 scene outline — needs creative + factual accuracy
        "period_summary": 3,                 # P2 period summary — must be factually consistent
        "life_summary": 3,                   # P2 life summary — global coherence critical
        "p1a_persona_pathway": 3,            # Persona pathway inference — core planning
        "p1b_target_brief": 3,               # Target initial brief — persona accuracy
        "p1b_target_profile": 3,             # Full profile (target) — persona accuracy

        # P1a new Tier 3
        "p1a_milestone_plan": 3,             # Milestone planning
        "p1a_forward_period": 3,             # Forward period generation
        "p1a_backward_period": 3,            # Backward period generation

        # M4+M5: LR+Outline merged generation tasks
        "step1a_lr_with_outlines": 3,        # P2 LR+outline merged (replaces step1a+step1b)
        "step2_unified_pool_for_lr_and_outlines": 3,  # P2 unified pool for LR+outline

        # M3: One-shot script generation
        "m3_full_script": 3,                 # M3 full script generation
        "m3_review_script": 3,               # M3 script review
        "m3_correct_script": 3,              # M3 script correction
        # Parallel Protagonist Mode
        "parallel_protagonist_skeleton": 3,  # Director skeleton generation (Tier 3: narrative creativity)
        "parallel_protagonist_turn": 3,      # Protagonist turn generation (Tier 3: persona coherence)

        # ═══ Previously missing task_types (added to avoid silent Tier-2 fallback) ═══

        # P0: Persona refinement
        "p0_refinement": 1,                  # P0.5 one-shot persona field inference (structural)

        # P1.5: Milestone planning
        "p1_5_milestone_identification": 3,  # Milestone identification — core planning
        "p1_5_skeleton_generation": 3,       # Milestone skeleton generation — core planning

        # P1d: Persona current state
        "p1d_persona_current_state": 2,      # Persona current state inference per period

        # P2: Batch/alternate naming variants
        "step1a_batch_event_framework": 3,   # Batch event framework (same tier as step1a_event_framework)
        "step1a_cross_lp_batch_event_framework": 3,  # Cross-LP batch LR framework (strongest model: coherence + factual accuracy)
        "step2_batch_participant_selection": 2,  # Batch participant selection (same tier as step2_participant_selection)
        "step2_unified_pool_lr_outlines": 3, # Unified pool for LR+outlines (alt name variant)

        # P2: Summary tasks
        "lr_refined_summary": 2,             # LR refined summary
        "multi_summary": 2,                  # Multi-period summary

        # P3: Scene planning & beat narration
        "p3_scene_planning": 3,              # P3 scene planning — creative + factual accuracy
        "beat_narration": 3,                 # Beat narration — core narrative creativity
        "beat_completion_check": 1,          # Beat completion check — structural judgment

        # M3: alternate naming
        "m3_generate_full_script": 3,        # M3 full script generation (alt name for m3_full_script)

        # ═══ v2 new helpers (added to avoid silent default_model fallback) ═══
        "p2_persona_context_signals": 1,     # P2 cultural/occupation context classifier (structural)
        "p2_outline_salience": 1,            # P2 outline salience batch scorer (structural)
        "p3_attitude_classification": 1,     # P3 negative-attitude identity-core classifier (structural)
        "p0_format_classification": 1,       # P0 SimulatorArena vs PersonaGym format classifier (structural)
    }

    DEFAULT_TIER_MODELS: Dict[int, str] = {
        1: "gpt-5.4-mini",
        2: "gpt-5.4-mini",
        3: "gpt-5.2",
    }

    def __init__(self, tier_models: Optional[Dict[int, str]] = None):
        self._tier_models = tier_models or self.DEFAULT_TIER_MODELS.copy()

    def get_model(self, task_type: str, importance: float = 0.5) -> str:
        """Route to appropriate model based on task type and importance."""
        tier = self.TASK_TIER_MAP.get(task_type, 2)
        # Upgrade tier for high-importance events
        if importance >= 0.8 and tier < 3:
            tier = min(tier + 1, 3)
        return self._tier_models.get(tier, self._tier_models[2])

    def set_tier_model(self, tier: int, model: str) -> None:
        """Override the model for a specific tier."""
        self._tier_models[tier] = model


# ---------------------------------------------------------------------------
# Per-call timeout: prevent indefinite TCP-level stalls
# ---------------------------------------------------------------------------
# Maximum seconds to wait for a single LLM API call before raising TimeoutError.
# Set to None to disable (not recommended for production).
_SINGLE_CALL_TIMEOUT_SECONDS: int = 120


# ---------------------------------------------------------------------------
# Smart wait strategy: intelligent sleep based on error type
# ---------------------------------------------------------------------------
_RETRY_AFTER_BUFFER_S = 2          # extra seconds on top of Retry-After
_FALLBACK_EXPONENTIAL = wait_exponential(multiplier=1, min=2, max=60)


def _extract_retry_after(exc: BaseException) -> Optional[float]:
    """
    Try to extract the Retry-After value (in seconds) from a RateLimitError
    or any HTTP response that carries the header.
    Returns None if the header is missing or unparseable.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def _smart_wait(retry_state: RetryCallState) -> float:
    """
    Intelligent tenacity wait callback — sleep duration is derived from the
    actual error returned by the server, not a fixed schedule.

    Decision table
    ──────────────
    RateLimitError (429) + Retry-After header
        → wait exactly (Retry-After + buffer) seconds
          Rationale: the server told us exactly when to retry; honour it.

    RateLimitError (429) without Retry-After header
        → exponential backoff (2s … 60s)
          Rationale: we don't know the quota window; back off gradually.

    ServiceUnavailableError (503) / InternalServerError (500)
        → exponential backoff starting at 5s (min=5, max=120)
          Rationale: server is overloaded; give it more breathing room than
          a simple rate-limit.

    APIConnectionError / asyncio.TimeoutError / LLMGenerationError
    (wraps TCP-level stall or per-call timeout)
        → short fixed wait (3s) then retry
          Rationale: connection issues are often transient; retry quickly
          but don't hammer the server.

    All other retryable errors
        → standard exponential backoff (2s … 60s)
    """
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    attempt = retry_state.attempt_number  # 1-based

    if exc is None:
        return _FALLBACK_EXPONENTIAL(retry_state)

    # ── 429 Rate Limit ────────────────────────────────────────────────────
    if isinstance(exc, litellm.RateLimitError):
        retry_after = _extract_retry_after(exc)
        if retry_after is not None and retry_after > 0:
            wait_seconds = retry_after + _RETRY_AFTER_BUFFER_S
            logger.info(
                "[smart_wait] Rate-limited (429). Server Retry-After=%ss → "
                "sleeping %.1fs (attempt #%d).",
                retry_after, wait_seconds, attempt,
            )
            return wait_seconds
        # No header: exponential backoff
        wait_seconds = min(2 * (2 ** (attempt - 1)), 60)
        logger.info(
            "[smart_wait] Rate-limited (429, no Retry-After header) → "
            "sleeping %.1fs (attempt #%d).",
            wait_seconds, attempt,
        )
        return wait_seconds

    # ── 503 Service Unavailable / 500 Internal Server Error ──────────────
    if isinstance(exc, (litellm.ServiceUnavailableError, litellm.InternalServerError)):
        wait_seconds = min(5 * (2 ** (attempt - 1)), 120)
        logger.info(
            "[smart_wait] Server error (%s) → sleeping %.1fs (attempt #%d).",
            type(exc).__name__, wait_seconds, attempt,
        )
        return wait_seconds

    # ── Connection / Timeout errors ───────────────────────────────────────
    if isinstance(exc, (litellm.APIConnectionError, litellm.Timeout, asyncio.TimeoutError)):
        wait_seconds = min(3 * attempt, 30)
        logger.info(
            "[smart_wait] Connection/timeout error (%s) → sleeping %.1fs (attempt #%d).",
            type(exc).__name__, wait_seconds, attempt,
        )
        return wait_seconds

    # ── LLMGenerationError (wraps TCP stall timeout or other failures) ────
    if isinstance(exc, LLMGenerationError):
        err_msg = str(exc)
        if "timed out" in err_msg:
            # TCP-level stall: exponential backoff to give server more recovery time
            wait_seconds = min(10 * (2 ** (attempt - 1)), 120)
            logger.info(
                "[smart_wait] LLM call timed out → sleeping %.1fs before retry (attempt #%d).",
                wait_seconds, attempt,
            )
            return wait_seconds
        # Other LLMGenerationError: standard exponential
        return _FALLBACK_EXPONENTIAL(retry_state)

    # ── Fallback: standard exponential backoff ────────────────────────────
    return _FALLBACK_EXPONENTIAL(retry_state)

class AsyncLLMClient:
    """
    Professional asynchronous LLM client.
    Supports high-concurrency API calls with built-in exponential backoff retry and structured (JSON Schema) output parsing.
    Fully compatible with standard providers (OpenAI, Anthropic) and enterprise internal custom-compatible endpoints.
    """
    
    def __init__(
        self, 
        default_model: str = "gpt-5.1", 
        api_base: Optional[str] = None, 
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        # ── New parameters ──
        llm_log_path: Optional[str] = None,
        log_prompt_chars: int = 0,
        log_response_chars: int = 0,
        rpm_limit: int = 15,
        enable_calm_routing: bool = False,
        calm_tier_models: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize the asynchronous client.
        
        :param default_model: Default model name (e.g., "gpt-5.1")
:param api_base: Custom API base URL, e.g., "https://api.openai.com/v1"
        :param api_key: API authentication token (the part after Bearer)
        :param temperature: Default sampling temperature; can be set globally at init time
        :param llm_log_path: Path to write full LLM call records (JSONL)
        :param log_prompt_chars: Max chars of prompt to show in pipeline log
        :param log_response_chars: Max chars of response to show in pipeline log
        :param rpm_limit: Client-side RPM limit (0=disabled). Used as initial RPM
                          for the adaptive rate limiter.
        :param enable_calm_routing: Enable CALM v2 cost-aware model routing
        :param calm_tier_models: Override default tier-to-model mapping for CALM
        """
        self.default_model = default_model
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature

        # LLM call logging
        self._call_counter: int = 0
        self._llm_log_path: Optional[str] = llm_log_path
        self._log_prompt_chars: int = log_prompt_chars
        self._log_response_chars: int = log_response_chars

        # Adaptive RPM throttling (Module 10)
        initial_rpm = max(rpm_limit, 15) if rpm_limit > 0 else 30
        self._rate_limiter = AdaptiveRateLimiter(
            initial_rpm=initial_rpm,
            min_rpm=5,
            max_rpm=120,
        )
        # Keep legacy field for backward compatibility
        self._rpm_limit: int = rpm_limit

        # CALM v2 model routing (Module 9)
        self._enable_calm_routing: bool = enable_calm_routing
        self._calm_router = CALMRouterV2(tier_models=calm_tier_models)
        
        # litellm global config
        litellm.set_verbose = False
        litellm.drop_params = True  # Auto-drop unsupported params for the current model

    async def _throttle(self) -> None:
        """Ensure we don't exceed RPM limit using adaptive rate limiter."""
        if self._rpm_limit <= 0:
            return
        await self._rate_limiter.acquire()

    def _resolve_model(self, model: Optional[str], task_type: Optional[str], importance: float = 0.5) -> Optional[str]:
        """Resolve the model to use based on explicit model, task_type, or default.
        
        Priority: explicit model > CALM routing > default_model
        """
        if model is not None:
            return model
        if task_type and self._enable_calm_routing:
            return self._calm_router.get_model(task_type, importance)
        return None  # Will fall back to default_model

    def _log_llm_call(
        self,
        call_id: int,
        method: str,
        model: str,
        temperature: float,
        system_prompt: Optional[str],
        user_prompt: str,
        response_text: str,
        duration_ms: float,
        response_model_name: str = "",
    ) -> None:
        """Log LLM call to pipeline_log (summary) and llm_calls.jsonl (full)."""
        # Truncate for pipeline log (0 = no truncation)
        pc = self._log_prompt_chars
        rc = self._log_response_chars
        sp_full = system_prompt or ""
        sp_trunc = sp_full[:pc] if pc > 0 else sp_full
        up_trunc = user_prompt[:pc] if pc > 0 else user_prompt
        resp_trunc = response_text[:rc] if rc > 0 else response_text

        sp_label = f"first {pc} chars" if pc > 0 else "full"
        up_label = f"first {pc} chars" if pc > 0 else "full"
        resp_label = f"first {rc} chars" if rc > 0 else "full"

        logger.info(
            f"\n┌─────────────────────────────────────────────────────────\n"
            f"│ LLM Call #{call_id}  [{method}]\n"
            f"│ Model: {model}  |  Temperature: {temperature}"
            + (f"  |  Response Model: {response_model_name}" if response_model_name else "")
            + f"\n├─────────────────────────────────────────────────────────\n"
            f"│ SYSTEM PROMPT ({sp_label}):\n"
            f"│   {sp_trunc}\n│\n"
            f"│ USER PROMPT ({up_label}):\n"
            f"│   {up_trunc}\n"
            f"├─────────────────────────────────────────────────────────\n"
            f"│ RESPONSE ({resp_label}):\n"
            f"│   {resp_trunc}\n│\n"
            f"│ Duration: {duration_ms:.0f} ms\n"
            f"└─────────────────────────────────────────────────────────"
        )

        # Write full record to JSONL
        if self._llm_log_path:
            record = {
                "call_id": call_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "method": method,
                "model": model,
                "temperature": temperature,
                "response_model": response_model_name,
                "system_prompt": system_prompt or "",
                "user_prompt": user_prompt,
                "response_text": response_text,
                "duration_ms": round(duration_ms, 3),
            }
            try:
                import pathlib
                pathlib.Path(self._llm_log_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self._llm_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                logger.exception("Failed to write LLM call log")

    def _build_messages(self, prompt: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Build the standard message body format"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _get_model_and_kwargs(self, model: Optional[str] = None) -> tuple[str, dict]:
        """
        Resolve model name and auth parameters.
        For custom OpenAI-compatible endpoints, litellm requires specifying custom_llm_provider
        or prefixing the model with 'openai/'.
        """
        target_model = model or self.default_model
        
        # If a custom api_base is provided and the model has no specific prefix, default to OpenAI-compatible protocol
        if self.api_base and not target_model.startswith(("openai/", "anthropic/", "gemini/")):
            target_model = f"openai/{target_model}"
            
        kwargs = {}
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
            
        return target_model, kwargs

    async def _execute_completion(self, target_model: str, messages: list, temperature: float, **kwargs) -> str:
        """
        Core call logic: handle standard non-streaming requests, with fallback to aggregate
        streaming chunks when the server forces SSE responses.
        Resolves issues where enterprise proxies or gateways forcibly return chunk (SSE) data
        causing parsing errors.

        Per-call timeout: wrapped with asyncio.wait_for(_SINGLE_CALL_TIMEOUT_SECONDS) to
        prevent indefinite TCP-level stalls. On timeout, raises LLMGenerationError which
        is caught by the outer tenacity retry and triggers _smart_wait.
        """
        timeout = _SINGLE_CALL_TIMEOUT_SECONDS

        async def _do_non_streaming() -> str:
            kw = dict(kwargs)
            kw["stream"] = False
            response = await acompletion(
                model=target_model,
                messages=messages,
                temperature=temperature,
                **kw
            )
            return response.choices[0].message.content

        async def _do_streaming() -> str:
            kw = dict(kwargs)
            kw["stream"] = True
            response_stream = await acompletion(
                model=target_model,
                messages=messages,
                temperature=temperature,
                **kw
            )
            full_text = ""
            async for chunk in response_stream:
                delta = chunk.choices[0].delta
                # Compatibility: Taiji gateway delta may be dict or Object
                if isinstance(delta, dict):
                    content = delta.get("content", "")
                else:
                    content = getattr(delta, "content", "")
                if content:
                    full_text += content
            return full_text

        try:
            if timeout is not None:
                result = await asyncio.wait_for(_do_non_streaming(), timeout=timeout)
            else:
                result = await _do_non_streaming()
            return result
        except asyncio.TimeoutError:
            logger.warning(
                f"[_execute_completion] Call timed out after {timeout}s "
                f"(model={target_model}). Will be retried by tenacity."
            )
            raise LLMGenerationError(f"LLM call timed out after {timeout}s")
        except Exception as e:
            error_msg = str(e)
            # Content-policy / invalid-prompt: non-retryable, fail fast
            if _is_content_policy_error(e):
                logger.warning(
                    f"[_execute_completion] Content policy violation (non-retryable): {e}"
                )
                raise ContentPolicyError(str(e)) from e
            # Catch parsing errors caused by proxy servers forcing SSE/chunk responses
            if "chunk" in error_msg or "reverse proxy" in error_msg or "Expecting value" in error_msg or "JSON" in error_msg:
                logger.warning("Detected API forcing streaming (SSE) responses, automatically switching to stream aggregation mode...")
                try:
                    if timeout is not None:
                        result = await asyncio.wait_for(_do_streaming(), timeout=timeout)
                    else:
                        result = await _do_streaming()
                    return result
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[_execute_completion] Streaming fallback also timed out after {timeout}s."
                    )
                    raise LLMGenerationError(f"LLM streaming call timed out after {timeout}s")
            # Other network or API errors are re-raised for outer tenacity retry handling
            raise

    @retry(
        stop=stop_after_attempt(10),
        wait=_smart_wait,
        retry=retry_if_exception_type((
            litellm.Timeout,
            litellm.APIConnectionError,
            litellm.RateLimitError,
            litellm.ServiceUnavailableError,
            litellm.InternalServerError,
            LLMGenerationError,
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        task_type: Optional[str] = None,
        importance: float = 0.5,
        **kwargs
    ) -> str:
        """
        Asynchronously generate plain text responses.
        
        :param task_type: CALM task type for automatic model routing (e.g. 'event_resolution')
        :param importance: Event importance 0-1, used to upgrade model tier when >= 0.8
        """
        actual_temperature = temperature if temperature is not None else self.temperature
        resolved_model = self._resolve_model(model, task_type, importance)
        target_model, auth_kwargs = self._get_model_and_kwargs(resolved_model)
        messages = self._build_messages(prompt, system_prompt)
        call_kwargs = {**auth_kwargs, **kwargs}

        await self._throttle()
        self._call_counter += 1
        call_id = self._call_counter
        start_time = time.time()

        try:
            result = await self._execute_completion(target_model, messages, actual_temperature, **call_kwargs)
            duration_ms = (time.time() - start_time) * 1000
            self._rate_limiter.on_success()
            self._log_llm_call(
                call_id=call_id, method="generate_text", model=target_model,
                temperature=actual_temperature, system_prompt=system_prompt,
                user_prompt=prompt, response_text=result, duration_ms=duration_ms,
            )
            return result
        except litellm.RateLimitError as e:
            retry_after = _extract_retry_after(e)
            self._rate_limiter.on_rate_limit(retry_after)
            raise
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.exception(
                f"LLM Call #{call_id} failed after {duration_ms:.0f}ms"
            )
            raise LLMGenerationError(f"Generation failed: {str(e)}")

    @retry(
        stop=stop_after_attempt(10),
        wait=_smart_wait,
        retry=retry_if_exception_type((
            litellm.Timeout,
            litellm.APIConnectionError,
            litellm.RateLimitError,
            litellm.ServiceUnavailableError,
            litellm.InternalServerError,
            LLMGenerationError,
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    async def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Async generation that accepts a full messages list (supports multi-turn history).
        Unlike generate_text(), this method does not rebuild the message body,
        making it suitable for scenarios that require passing a complete conversation history.

        :param messages: Full message list, e.g. [{"role": "system/user/assistant", "content": "..."}]
        :param model: Optional, overrides the default model
        :param temperature: Optional, overrides the default temperature
        """
        actual_temperature = temperature if temperature is not None else self.temperature
        resolved_model = self._resolve_model(model, None, 0.5)
        target_model, auth_kwargs = self._get_model_and_kwargs(resolved_model)
        call_kwargs = {**auth_kwargs, **kwargs}

        await self._throttle()
        self._call_counter += 1
        call_id = self._call_counter
        start_time = time.time()

        # Extract system and last user prompt for logging
        system_prompt_for_log = next(
            (m["content"] for m in messages if m["role"] == "system"), None
        )
        user_prompt_for_log = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )

        try:
            result = await self._execute_completion(target_model, messages, actual_temperature, **call_kwargs)
            duration_ms = (time.time() - start_time) * 1000
            self._rate_limiter.on_success()
            self._log_llm_call(
                call_id=call_id, method="generate_with_messages", model=target_model,
                temperature=actual_temperature, system_prompt=system_prompt_for_log,
                user_prompt=user_prompt_for_log, response_text=result, duration_ms=duration_ms,
            )
            return result
        except litellm.RateLimitError as e:
            retry_after = _extract_retry_after(e)
            self._rate_limiter.on_rate_limit(retry_after)
            raise
        except litellm.ContentPolicyViolationError as e:
            logger.warning(f"Content policy violation (will not retry, continuing): {e}")
            raise LLMGenerationError(f"Generation failed: {str(e)}")
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.exception(
                f"LLM Call #{call_id} failed after {duration_ms:.0f}ms"
            )
            raise LLMGenerationError(f"Generation failed: {str(e)}")

    @retry(
        stop=stop_after_attempt(10),
        wait=_smart_wait,
        retry=retry_if_exception_type((
            litellm.Timeout,
            litellm.APIConnectionError,
            litellm.RateLimitError,
            litellm.ServiceUnavailableError,
            litellm.InternalServerError,
            LLMGenerationError,
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        task_type: Optional[str] = None,
        importance: float = 0.5,
        _max_validation_retries: int = 10,
        **kwargs
    ) -> T:
        """
        Asynchronously generate structured data strictly following a Pydantic model (JSON Schema).
        
        :param task_type: CALM task type for automatic model routing
        :param importance: Event importance 0-1, used to upgrade model tier when >= 0.8
        :param _max_validation_retries: Max retries for ValidationError/JSONDecodeError with
               error-feedback injected into the prompt (default 3). These retries are handled
               internally and do NOT consume tenacity retry slots (which are reserved for
               network/rate-limit errors).
        """
        actual_temperature = temperature if temperature is not None else self.temperature
        resolved_model = self._resolve_model(model, task_type, importance)
        target_model, auth_kwargs = self._get_model_and_kwargs(resolved_model)
        
        schema_json = response_model.model_json_schema()
        schema_prompt = (
            f"\n\nPlease strictly follow the JSON Schema below to output your result. Do not output any extra markdown markers (e.g., ```json), "
            f"only a valid JSON string:\n{json.dumps(schema_json, ensure_ascii=False, indent=2)}"
        )
        
        enhanced_system_prompt = (system_prompt or "You are a precise data extraction and analysis AI.") + schema_prompt
        # Use a mutable list so we can append error-feedback turns in the retry loop
        messages = self._build_messages(prompt, enhanced_system_prompt)

        call_kwargs = {**auth_kwargs, **kwargs}
        # response_format={"type":"json_object"} is only supported by OpenAI-compatible
        # endpoints. Gemini models use a different generation_config and will raise
        # BadRequestError if this param is passed. Skip it for Gemini targets.
        _is_gemini = "gemini" in target_model.lower()
        if not _is_gemini:
            call_kwargs["response_format"] = {"type": "json_object"}

        response_model_name = response_model.__name__
        content = ""  # Keep in outer scope for error logging

        for attempt in range(1, _max_validation_retries + 1):
            await self._throttle()
            self._call_counter += 1
            call_id = self._call_counter
            start_time = time.time()

            try:
                content = await self._execute_completion(target_model, messages, actual_temperature, **call_kwargs)
                duration_ms = (time.time() - start_time) * 1000

                # Log the raw LLM call
                self._log_llm_call(
                    call_id=call_id, method="generate_structured", model=target_model,
                    temperature=actual_temperature, system_prompt=system_prompt,
                    user_prompt=prompt, response_text=content, duration_ms=duration_ms,
                    response_model_name=response_model_name,
                )

                # Clean up Markdown markers (error-tolerant handling)
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                # ── Gemini schema-wrapper unwrap ──────────────────────────────
                # Gemini flash-lite occasionally echoes the JSON Schema wrapper,
                # returning {"description": "...", "properties": {...actual fields...}}
                # instead of the flat object. Detect and unwrap before validation.
                try:
                    _raw_obj = json.loads(content)
                    if (
                        isinstance(_raw_obj, dict)
                        and "properties" in _raw_obj
                        and isinstance(_raw_obj["properties"], dict)
                        # Heuristic: the wrapper has "description" or "title" at top level
                        # but the actual model fields are nested under "properties"
                        and ("description" in _raw_obj or "title" in _raw_obj)
                        # Guard: the top-level keys should NOT match the target model's fields
                        # (if they do, this is a legitimate response, not a wrapper)
                        and not any(
                            k in response_model.model_fields
                            for k in _raw_obj
                            if k not in ("description", "title", "properties", "type")
                        )
                    ):
                        logger.debug(
                            f"[generate_structured] Detected Gemini schema-wrapper for "
                            f"{response_model_name}; unwrapping 'properties' to flat object."
                        )
                        content = json.dumps(_raw_obj["properties"], ensure_ascii=False)
                except (json.JSONDecodeError, TypeError):
                    pass  # Not valid JSON yet; let model_validate_json handle the error
                # ── End unwrap ────────────────────────────────────────────────

                # Validate and convert to strongly-typed object
                # Pre-clean trailing commas (Gemini flash-lite occasionally emits them)
                cleaned_content = _strip_trailing_commas(content.strip())
                try:
                    parsed_data = response_model.model_validate(json.loads(cleaned_content))
                except (ValidationError, json.JSONDecodeError):
                    # Fallback: try original content in case cleaning broke something
                    parsed_data = response_model.model_validate_json(content.strip())
                self._rate_limiter.on_success()
                return parsed_data

            except (ValidationError, json.JSONDecodeError) as parse_err:
                duration_ms = (time.time() - start_time) * 1000
                if attempt < _max_validation_retries:
                    # Inject error feedback as an assistant+user turn so the model
                    # knows exactly what went wrong and can fix it.
                    error_detail = str(parse_err)
                    logger.warning(
                        f"[generate_structured] Attempt {attempt}/{_max_validation_retries} "
                        f"validation failed for {response_model_name}: {error_detail[:300]}. "
                        f"Retrying with error feedback..."
                    )
                    # Append the bad response as assistant turn, then a correction request
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Your previous output failed JSON validation with the following error:\n"
                            f"{error_detail}\n\n"
                            f"Please carefully fix ALL the above issues and output a corrected, "
                            f"valid JSON that strictly conforms to the required schema. "
                            f"Output only the JSON string, no markdown."
                        ),
                    })
                else:
                    # Final attempt also failed — raise as LLMGenerationError so tenacity
                    # does NOT retry (parse errors are not transient network issues).
                    logger.error(
                        f"[generate_structured] All {_max_validation_retries} validation attempts "
                        f"failed for {response_model_name}. Last error: {parse_err}. "
                        f"Last content: {content[:500]}"
                    )
                    raise LLMGenerationError(
                        f"Structured generation failed after {_max_validation_retries} attempts "
                        f"(validation error): {parse_err}"
                    )

            except litellm.RateLimitError as e:
                retry_after = _extract_retry_after(e)
                self._rate_limiter.on_rate_limit(retry_after)
                raise
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.exception(
                    f"LLM Call #{call_id} failed after {duration_ms:.0f}ms"
                )
                raise LLMGenerationError(f"Structured generation failed: {str(e)}")

        # Should never reach here, but satisfy type checker
        raise LLMGenerationError(f"generate_structured exhausted all attempts for {response_model_name}")

# --- Usage example ---
async def main():
    import os
    
    API_BASE = os.environ.get("OPENAI_API_BASE", "")
    API_KEY = os.environ.get("OPENAI_API_KEY", "")
    MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o")
    if not API_BASE or not API_KEY:
        print("Set OPENAI_API_BASE and OPENAI_API_KEY environment variables")
        sys.exit(1)
    
    client = AsyncLLMClient(
        default_model=MODEL_NAME,
        api_base=API_BASE,
        api_key=API_KEY
    )
    
    # --- Test 1: Concurrent plain text request ---
    logger.info("Sending text request...")
    text_result = await client.generate_text("Hello, please briefly introduce yourself.")
    logger.info("Text Result: %s", text_result)
    
    # --- Test 2: Structured output request ---
    class ResolutionResult(BaseModel):
        is_event_boundary: bool
        event_boundary_score: float
        reasoning: str
        
    logger.info("\nSending structured request...")
    structured_result = await client.generate_structured(
        prompt="On the way to work today, the protagonist ran into an elementary school classmate they hadn't seen in years. They chatted for a long time, and the protagonist felt deeply moved.",
        response_model=ResolutionResult,
        system_prompt="Determine whether the above daily record constitutes an important event boundary."
    )
    logger.info("Structured Result:")
    logger.info(f"Is Boundary: {structured_result.is_event_boundary}")
    logger.info(f"Score: {structured_result.event_boundary_score}")
    logger.info(f"Reasoning: {structured_result.reasoning}")

if __name__ == "__main__":
    # Run the async event loop
    asyncio.run(main())