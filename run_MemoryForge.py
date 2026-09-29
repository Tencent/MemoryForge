#!/usr/bin/env python3
"""
MemoryForge — Unified Pipeline Entry Point
============================================
Synthesizes a complete Autobiographical Memory Base M_π = (L, G, E)
from a single brief persona description.

Usage:
    PYTHONPATH=. python run_MemoryForge.py --persona "a 35-year-old scientist in New York"
    PYTHONPATH=. python run_MemoryForge.py --persona "a 19-year-old farmer from New York who is a fitness freak" --model gpt-5.1
    PYTHONPATH=. python run_MemoryForge.py --persona "..." --temperature 0.7 --recency-window 5 --childhood-amnesia-age 3
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# ── Path setup ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
SYNTH_DIR = PROJECT_ROOT / "lifelong_synth"
P0_DIR = SYNTH_DIR / "simulation_p0_persona_settings"
SCHEMA_PATH = SYNTH_DIR / "configs" / "persona_schema.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

sys.path.insert(0, str(PROJECT_ROOT))

from lifelong_synth.performance_tracker import PerformanceTracker
from llm.client import AsyncLLMClient
from lifelong_synth.simulation_p1_initialisation.life_period_planner import (
    DevelopmentAwareLifePeriodPlanner,
)
from lifelong_synth.simulation_p1_initialisation.participant_pool import (
    ParticipantPoolManager,
)
from lifelong_synth.simulation_p1_initialisation.definition import Participant
from lifelong_synth.simulation_p2_event_organiser.event_organiser import EventOrganiser
from lifelong_synth.simulation_p3_multi_resolution_simulation.high_res_event_simulator import (
    HighResEventSimulator,
)
from lifelong_synth.simulation_p4_memory_organiser.memory_manager import MemoryManager
from lifelong_synth.simulation_p1_initialisation.key_life_path_generator import (
    generate_key_life_path_cache,
)
from lifelong_synth.simulation_p1_initialisation.milestone_planner import (
    generate_milestone_plan,
    load_milestone_plan,
)

logger = logging.getLogger("memoryforge")


# ═══════════════════════════════════════════════════════════════
# Custom Exception
# ═══════════════════════════════════════════════════════════════

class PipelineError(Exception):
    """Pipeline execution error with stage information."""

    def __init__(self, message: str, stage: str):
        super().__init__(message)
        self.stage = stage


# ═══════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════

def setup_logging(output_dir: Path, verbose: bool = False) -> None:
    """Configure dual-output logging: console + file."""
    log_path = output_dir / "pipeline_log.txt"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_path), encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    for noisy_logger in [
        "LiteLLM", "litellm", "openai", "openai._base_client",
        "httpx", "httpcore",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    """Save data to a JSON file with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_run_id(now: Optional[datetime] = None) -> str:
    """Generate a millisecond-level run_id."""
    ts = (now or datetime.now()).strftime("run_%Y%m%d_%H%M%S_%f")[:-3]
    return ts


def sanitize_persona_name(persona: str) -> str:
    """Sanitize persona string for use as directory name."""
    name = persona[:80].strip()
    # Replace filesystem-unsafe characters
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(ch, '_')
    return name


def load_pool_from_json(
    pool_path: Path, llm_client: AsyncLLMClient
) -> ParticipantPoolManager:
    """Restore ParticipantPoolManager from a previously saved JSON file."""
    pool_data = load_json(pool_path)
    pool = ParticipantPoolManager(llm_client=llm_client, save_path=str(pool_path))

    pool._simulation_start_date = pool_data.get("simulation_start_date", "")
    pool._simulation_end_date = pool_data.get("simulation_end_date", "")

    participants_list = pool_data.get("participants", [])
    if isinstance(participants_list, list):
        for pdata in participants_list:
            p = Participant(**pdata)
            pool._participants[p.participant_id] = p
    elif isinstance(participants_list, dict):
        for pid, pdata in participants_list.items():
            p = Participant(**pdata)
            pool._participants[pid] = p

    # Both save() lists and to_json() mappings must resume after existing IDs.
    for pid in pool._participants:
        if pid.startswith("P_") and pid != "P_TARGET":
            try:
                num = int(pid.split("_")[1])
                if num >= pool._next_id_counter:
                    pool._next_id_counter = num + 1
            except ValueError:
                pass

    logger.info(f"  Pool restored: {pool.count()} participants")
    return pool


class ResumeState:
    """Detect existing artifacts to determine which stage to resume from."""

    def __init__(self, plan_path: Path, pool_path: Path, memory_path: Path, refined_sample_path: Optional[Path] = None):
        self.has_plan = plan_path.exists()
        self.has_pool = pool_path.exists()
        self.has_memory = memory_path.exists()
        self.has_refined_sample = refined_sample_path.exists() if refined_sample_path else False

    @property
    def resume_stage(self) -> str:
        if not self.has_refined_sample:
            return "P0.5"
        if not self.has_plan:
            return "P1a"
        if not self.has_pool:
            return "P1b"
        if not self.has_memory:
            return "P2"
        return "P3"


# ═══════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════

async def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full MemoryForge simulation pipeline."""
    if not args.api_base:
        print("ERROR: --api-base is required. Set OPENAI_API_BASE env var or pass --api-base.")
        sys.exit(1)
    if not args.api_key:
        print("ERROR: --api-key is required. Set OPENAI_API_KEY env var or pass --api-key.")
        sys.exit(1)

    start_time = time.time()
    started_at = datetime.now().isoformat()

    # ── Construct persona config from one-line persona string ──
    persona_config = {"persona_brief_text": args.persona}
    sample_name = sanitize_persona_name(args.persona)

    # ── Resolve output paths ──
    output_root = Path(args.output_dir).resolve()
    sample_root = output_root / sample_name
    run_id = args.run_id or generate_run_id()
    output_dir = sample_root / run_id

    plan_path = output_dir / "life_plan.json"
    pool_path = output_dir / "persona_pool.json"
    memory_path = output_dir / "memory_base.json"
    hr_events_dir = output_dir / "high_res_events"
    perf_metrics_path = output_dir / "perf_metrics.jsonl"
    run_metadata_path = output_dir / "run_metadata.json"
    llm_log_path = output_dir / "llm_calls.jsonl"

    for path in [sample_root, output_dir, hr_events_dir]:
        path.mkdir(parents=True, exist_ok=True)

    # Create latest symlink
    latest_path = sample_root / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    latest_path.symlink_to(output_dir, target_is_directory=True)

    setup_logging(output_dir, verbose=args.verbose)
    tracker = PerformanceTracker(str(perf_metrics_path))

    logger.info("=" * 60)
    logger.info("MemoryForge — Autobiographical Memory Synthesis")
    logger.info(f"  Persona: {args.persona}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Recency window: {args.recency_window} years")
    logger.info(f"  Childhood amnesia age: {args.childhood_amnesia_age}")
    logger.info(f"  Run ID: {run_id}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Resume: {args.resume}")
    logger.info("=" * 60)

    # ── Write run metadata ──
    save_json({
        "run_id": run_id,
        "persona": args.persona,
        "model": args.model,
        "temperature": args.temperature,
        "recency_window": args.recency_window,
        "childhood_amnesia_age": args.childhood_amnesia_age,
        "started_at": started_at,
        "status": "running",
    }, run_metadata_path)

    llm_client = AsyncLLMClient(
        default_model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
        llm_log_path=str(llm_log_path),
        rpm_limit=args.rpm_limit,
    )

    refined_sample_path = output_dir / "refined_sample.json"
    resume_state = ResumeState(
        plan_path=plan_path,
        pool_path=pool_path,
        memory_path=memory_path,
        refined_sample_path=refined_sample_path,
    )

    effective_resume = args.resume
    if not effective_resume and args.run_id and resume_state.resume_stage not in ("P0.5", "P1a"):
        effective_resume = True
        logger.info(f"Auto-resume enabled: will resume from stage {resume_state.resume_stage}")

    processed_high_res_count = 0
    simulated_high_res_count = 0

    try:
        # ── P0.5: Validate & Auto-Refine Persona Config ──
        from lifelong_synth.simulation_p0_persona_settings.sample_refiner import SampleRefiner

        if effective_resume and refined_sample_path.exists():
            logger.info("[ContextGenerator] Resume: loading existing refined sample...")
            persona_config = load_json(refined_sample_path)
        else:
            logger.info("[ContextGenerator] Validating & auto-refining persona config...")
            try:
                with tracker.track("pipeline.P0_5", run_id=run_id):
                    refiner = SampleRefiner(llm_client=llm_client)
                    refined_config = await refiner.refine(persona_config)
                    save_json(refined_config, refined_sample_path)
                    persona_config = refined_config
                logger.info(f"[ContextGenerator] ✅ Refined sample saved: {refined_sample_path}")
            except Exception as e:
                raise PipelineError(f"Persona config refinement failed: {e}", stage="P0.5") from e

        with tracker.track("pipeline.total", run_id=run_id):
            # ── P1a: Life Period Plan ──
            if effective_resume and plan_path.exists():
                logger.info("[LifeOrganizer::PeriodPlan] Resume: loading existing life plan...")
                life_plan = load_json(plan_path)
            else:
                logger.info("[LifeOrganizer::PeriodPlan] Generating life period plan...")
                try:
                    with tracker.track("pipeline.P1a", run_id=run_id):
                        planner = DevelopmentAwareLifePeriodPlanner(
                            llm_client=llm_client,
                            schema_path=str(SCHEMA_PATH),
                            recency_window_years=args.recency_window,
                            childhood_amnesia_age=args.childhood_amnesia_age,
                        )
                        life_plan = await planner.generate_plan(persona_config)
                        save_json(life_plan, plan_path)
                    logger.info(f"[LifeOrganizer::PeriodPlan] Saved: {plan_path}")
                except Exception as e:
                    raise PipelineError(f"Life plan generation failed: {e}", stage="P1a") from e

            life_periods = life_plan.get("life_periods", [])
            logger.info(f"[LifeOrganizer::PeriodPlan] ✅ Complete: {len(life_periods)} periods")

            birth_date = (
                life_plan.get("global_summary", {})
                .get("timeline_anchor", {})
                .get("derived_birth_date", "")
            )
            if birth_date:
                persona_config["derived_birth_date"] = birth_date

            # ── P1b: Participant Pool ──
            if effective_resume and pool_path.exists():
                logger.info("[ContextGenerator::SocialNetwork] Resume: loading existing participant pool...")
                pool = load_pool_from_json(pool_path, llm_client)
            else:
                logger.info("[ContextGenerator::SocialNetwork] Initializing participant pool...")
                try:
                    with tracker.track("pipeline.P1b", run_id=run_id):
                        pool = ParticipantPoolManager(
                            llm_client=llm_client,
                            save_path=str(pool_path),
                            performance_tracker=tracker,
                        )
                        logger.info("[ContextGenerator::SocialNetwork]   Step 1: Registering target persona...")
                        pool.register_target_persona(persona_config, life_plan)

                        logger.info("[ContextGenerator::SocialNetwork]   Step 1.5: Building temporal briefs...")
                        await pool.generate_target_temporal_briefs(persona_config, life_plan)

                        logger.info("[ContextGenerator::SocialNetwork]   Steps 2-7: Initializing supporting characters...")
                        await pool.initialize_supporting_characters(
                            persona_config,
                            life_plan,
                            max_concurrency=3,
                        )

                        pool.save(str(pool_path))
                    logger.info(f"[ContextGenerator::SocialNetwork] Saved: {pool_path}")
                except Exception as e:
                    raise PipelineError(
                        f"Participant pool initialization failed: {e}", stage="P1b"
                    ) from e

            logger.info(f"[ContextGenerator::SocialNetwork] ✅ Complete: {pool.count()} participants")

            # ── P1c: Key Life Path Cache ──
            key_life_path_cache_path = output_dir / "year_enrichment_key_life_path_cache.json"
            if effective_resume and key_life_path_cache_path.exists():
                logger.info("[ContextGenerator::SocialContext] Resume: key life path cache already exists")
            else:
                logger.info("[ContextGenerator::SocialContext] Generating key life path cache...")
                try:
                    with tracker.track("pipeline.P1c", run_id=run_id):
                        await generate_key_life_path_cache(
                            life_plan=life_plan,
                            persona_config=persona_config,
                            output_dir=str(output_dir),
                            llm_client=llm_client,
                        )
                    logger.info(f"[ContextGenerator::SocialContext] ✅ Key life path cache saved")
                except Exception as e:
                    logger.warning(f"[ContextGenerator::SocialContext] Key life path cache generation failed (non-fatal): {e}")

            # ── P1.5: Milestone Plan ──
            milestone_plan_path = output_dir / "milestone_plan.json"
            milestone_plan = {}
            if effective_resume and milestone_plan_path.exists():
                logger.info("[LifeOrganizer::Milestone] Resume: loading existing milestone plan...")
                milestone_plan = load_milestone_plan(str(output_dir))
            else:
                logger.info("[LifeOrganizer::Milestone] Generating milestone HR pre-plan...")
                try:
                    with tracker.track("pipeline.P1_5", run_id=run_id):
                        milestone_plan = await generate_milestone_plan(
                            life_plan=life_plan,
                            persona_config=persona_config,
                            output_dir=str(output_dir),
                            llm_client=llm_client,
                        )
                    logger.info(f"[LifeOrganizer::Milestone] ✅ Complete: {len(milestone_plan)} milestones planned")
                except Exception as e:
                    logger.warning(f"[LifeOrganizer::Milestone] Milestone planning failed (non-fatal): {e}")
                    milestone_plan = {}

            # ── P2+P3: Event Generation & Simulation ──
            logger.info("[MultiResSimulator] Starting event generation...")
            memory = MemoryManager(
                memory_base_path=str(memory_path),
                llm_client=llm_client,
                performance_tracker=tracker,
            )
            memory.set_run_id(run_id)

            if memory.period_count == 0:
                memory.import_life_periods_from_plan(life_plan)
                logger.info(f"[MultiResSimulator] Imported {memory.period_count} life periods into memory")

            organiser = EventOrganiser(
                llm_client=llm_client,
                memory_manager=memory,
                participant_pool=pool,
                target_persona=persona_config,
                participant_pool_path=str(pool_path),
                memory_base_path=str(memory_path),
                performance_tracker=tracker,
                run_id=run_id,
                milestone_plan=milestone_plan,
            )

            try:
                with tracker.track("pipeline.P2", run_id=run_id):
                    # ── Global Year Enrichment Pre-generation ──
                    logger.info("[MultiResSimulator] Pre-generating year enrichments...")
                    with tracker.track("pipeline.P2.global_enrichment", run_id=run_id):
                        new_enrichments = await organiser.pregenerate_all_year_enrichments(
                            life_plan=life_plan,
                            persona_config=persona_config,
                        )
                    logger.info(f"[MultiResSimulator] Global enrichment complete: {new_enrichments} new entries")

                    # ── P1d: Persona current state cache ──
                    logger.info("[MultiResSimulator::PersonaState] Pre-generating persona current state cache...")
                    await organiser.pregenerate_all_persona_current_states(
                        life_plan=life_plan,
                        persona_config=persona_config,
                        output_dir=str(output_dir),
                    )
                    logger.info("[MultiResSimulator::PersonaState] ✅ Persona current state cache complete")

                    # ── Build segment iteration from derived_memory_plan ──
                    derived_memory_plan = life_plan.get("derived_memory_plan", {})
                    segments = derived_memory_plan.get("segments", [])
                    period_lookup = {p["period_id"]: p for p in life_periods}

                    if segments:
                        last_seg_for_parent: Dict[str, str] = {}
                        for seg in segments:
                            last_seg_for_parent[seg["parent_period_id"]] = seg["segment_id"]
                        last_seg_ids = set(last_seg_for_parent.values())

                        logger.info(
                            f"[MultiResSimulator] Using derived_memory_plan: {len(segments)} segments "
                            f"from {len(last_seg_for_parent)} canonical periods"
                        )
                    else:
                        logger.info("[MultiResSimulator] No derived_memory_plan found; falling back to canonical life_periods")
                        segments = []
                        last_seg_ids = set()

                    stop_early = False

                    if segments:
                        # ── Cross-LP LR-only batch grouping ──
                        MAX_CROSS_LP_BATCH = 8
                        seg_groups = []
                        j = 0
                        while j < len(segments):
                            seg = segments[j]
                            is_lr_only = (
                                seg.get("max_detail_events", 0) == 0
                                and seg.get("max_outline_events", 0) == 0
                            )
                            if is_lr_only:
                                batch_segs = [seg]
                                j += 1
                                while (
                                    j < len(segments)
                                    and segments[j].get("max_detail_events", 0) == 0
                                    and segments[j].get("max_outline_events", 0) == 0
                                    and len(batch_segs) < MAX_CROSS_LP_BATCH
                                ):
                                    batch_segs.append(segments[j])
                                    j += 1
                                seg_groups.append({"type": "lr_batch", "segs": batch_segs})
                            else:
                                seg_groups.append({"type": "single", "segs": [seg]})
                                j += 1

                        prev_parent_id = None
                        for group in seg_groups:
                            if stop_early:
                                break

                            if group["type"] == "lr_batch" and len(group["segs"]) > 1:
                                batch_segs = group["segs"]
                                logger.info(
                                    f"[MultiResSimulator::LowRes-Batch] Processing {len(batch_segs)} "
                                    f"consecutive LR-only segments in one batch"
                                )
                                HighResEventSimulator.clear_caches()
                                try:
                                    with tracker.track("pipeline.P2.cross_lp_batch", run_id=run_id):
                                        batch_results = await organiser.process_lr_only_batch_segments(
                                            segments=batch_segs,
                                            life_plan=life_plan,
                                            persona_config=persona_config,
                                            hr_events_dir=str(hr_events_dir),
                                        )
                                except Exception:
                                    logger.exception("[MultiResSimulator::LowRes-Batch] Batch failed; falling back to sequential")
                                    batch_results = []
                                    for seg in batch_segs:
                                        parent_period_id = seg["parent_period_id"]
                                        parent_period = period_lookup.get(parent_period_id)
                                        if not parent_period:
                                            continue
                                        try:
                                            r = await organiser.process_period_atomic(
                                                period=parent_period,
                                                life_plan=life_plan,
                                                persona_config=persona_config,
                                                max_turns=args.max_turns,
                                                skip_p3=args.skip_p3,
                                                hr_events_dir=str(hr_events_dir),
                                                segment_override=seg,
                                                is_last_segment_of_period=(seg["segment_id"] in last_seg_ids),
                                            )
                                            batch_results.append(r)
                                        except Exception:
                                            logger.exception(f"[MultiResSimulator] Fallback failed for {seg['segment_id']}")

                                for r in batch_results:
                                    processed_high_res_count += int(r.get("high_res_count", 0) or 0)
                                    simulated_high_res_count += int(r.get("simulated_high_res_count", 0) or 0)
                                memory.save(str(memory_path))
                                pool.save(str(pool_path))

                            else:
                                seg = group["segs"][0]
                                parent_period_id = seg["parent_period_id"]
                                segment_id = seg.get("segment_id", "?")
                                parent_period = period_lookup.get(parent_period_id)
                                if not parent_period:
                                    logger.warning(
                                        f"[MultiResSimulator] Segment {segment_id} references unknown "
                                        f"parent_period_id={parent_period_id}; skipping"
                                    )
                                    continue

                                if parent_period_id != prev_parent_id:
                                    HighResEventSimulator.clear_caches()
                                    prev_parent_id = parent_period_id

                                is_last = segment_id in last_seg_ids
                                logger.info(
                                    f"[MultiResSimulator] Processing segment {segment_id} "
                                    f"(parent={parent_period_id}, last={is_last})..."
                                )
                                try:
                                    with tracker.track(
                                        f"pipeline.P2.{segment_id}",
                                        run_id=run_id,
                                        period_id=parent_period_id,
                                    ):
                                        period_result = await organiser.process_period_atomic(
                                            period=parent_period,
                                            life_plan=life_plan,
                                            persona_config=persona_config,
                                            max_turns=args.max_turns,
                                            skip_p3=args.skip_p3,
                                            hr_events_dir=str(hr_events_dir),
                                            segment_override=seg,
                                            is_last_segment_of_period=is_last,
                                        )
                                except Exception:
                                    logger.exception(
                                        f"[MultiResSimulator] ❌ Segment {segment_id} failed; continuing"
                                    )
                                    memory.save(str(memory_path))
                                    pool.save(str(pool_path))
                                    continue

                                processed_high_res_count += int(period_result.get("high_res_count", 0) or 0)
                                simulated_high_res_count += int(period_result.get("simulated_high_res_count", 0) or 0)
                                logger.info(
                                    f"[MultiResSimulator]   Segment {segment_id} complete: "
                                    f"events={len(period_result.get('committed_event_ids', []))}, "
                                    f"high_res={period_result.get('high_res_count', 0)}"
                                )
                                memory.save(str(memory_path))
                                pool.save(str(pool_path))

                    else:
                        # Fallback: iterate canonical life_periods directly
                        for period in life_periods:
                            period_id = period.get("period_id", "LP?")
                            HighResEventSimulator.clear_caches()
                            logger.info(f"[MultiResSimulator] Processing period {period_id}...")
                            try:
                                with tracker.track(f"pipeline.P2.{period_id}", run_id=run_id):
                                    period_result = await organiser.process_period_atomic(
                                        period=period,
                                        life_plan=life_plan,
                                        persona_config=persona_config,
                                        max_turns=args.max_turns,
                                        skip_p3=args.skip_p3,
                                        hr_events_dir=str(hr_events_dir),
                                    )
                            except Exception:
                                logger.exception(f"[MultiResSimulator] ❌ Period {period_id} failed; continuing")
                                memory.save(str(memory_path))
                                pool.save(str(pool_path))
                                continue

                            processed_high_res_count += int(period_result.get("high_res_count", 0) or 0)
                            simulated_high_res_count += int(period_result.get("simulated_high_res_count", 0) or 0)
                            memory.save(str(memory_path))
                            pool.save(str(pool_path))

                    memory.save(str(memory_path))
                    pool.save(str(pool_path))

            except Exception as e:
                memory.save(str(memory_path))
                pool.save(str(pool_path))
                raise PipelineError(f"Event generation failed: {e}", stage="P2") from e

            logger.info(
                f"[MultiResSimulator] ✅ Complete: {memory.event_count} events, "
                f"{simulated_high_res_count} high-res simulated"
            )

            # ── Final save ──
            with tracker.track("pipeline.final_save", run_id=run_id):
                memory.save(str(memory_path))
                pool.save(str(pool_path))

            # ── Export M_π = (L, G, E) format ──
            logger.info("[Export] Generating M_π three-layer memory format...")
            memory_forge_output = memory.export_memory_forge_format()
            memory_forge_output["metadata"]["persona_brief"] = args.persona
            m_pi_path = output_dir / "M_pi.json"
            save_json(memory_forge_output, m_pi_path)
            logger.info(f"[Export] M_π exported to {m_pi_path}")
            logger.info(f"  L (Lifetime Periods): {memory_forge_output['metadata']['total_periods']}")
            logger.info(f"  G (General Events): {memory_forge_output['metadata']['total_general_events']}")
            logger.info(f"  E (Event-Specific): {memory_forge_output['metadata']['total_specific_experiences']}")

        # ── Update run metadata ──
        save_json({
            "run_id": run_id,
            "persona": args.persona,
            "model": args.model,
            "temperature": args.temperature,
            "recency_window": args.recency_window,
            "childhood_amnesia_age": args.childhood_amnesia_age,
            "started_at": started_at,
            "status": "success",
            "total_events": memory.event_count,
            "high_res_simulated": simulated_high_res_count,
        }, run_metadata_path)

    except Exception as e:
        failed_stage = e.stage if isinstance(e, PipelineError) else "unknown"
        save_json({
            "run_id": run_id,
            "persona": args.persona,
            "model": args.model,
            "started_at": started_at,
            "status": "failed",
            "failed_stage": failed_stage,
            "error": str(e),
        }, run_metadata_path)
        raise

    elapsed_s = round(time.time() - start_time, 1)
    logger.info("=" * 60)
    logger.info("MemoryForge Pipeline Complete!")
    logger.info(f"  Persona: {args.persona}")
    logger.info(f"  Life periods: {len(life_periods)}")
    logger.info(f"  Total events: {memory.event_count}")
    logger.info(f"  High-res simulations: {simulated_high_res_count}")
    logger.info(f"  Total time: {elapsed_s}s")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  M_π export: {m_pi_path}")
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MemoryForge — Autobiographical Memory Synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  PYTHONPATH=. python run_MemoryForge.py --persona "a 35-year-old scientist in New York"
  PYTHONPATH=. python run_MemoryForge.py --persona "a 29-year-old farmer" --temperature 0.7
  PYTHONPATH=. python run_MemoryForge.py --persona "..." --recency-window 5 --childhood-amnesia-age 3
  PYTHONPATH=. python run_MemoryForge.py --persona "..." --skip-p3  # skip high-res simulation
        """,
    )

    # === PRIMARY INPUT ===
    parser.add_argument(
        "--persona",
        required=True,
        type=str,
        help="Brief persona description (one sentence), e.g. 'a 30-year-old scientist in New York'",
    )

    # === MODEL CONFIGURATION ===
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-5.1"),
        help="LLM model name (default: gpt-5.1)",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_API_BASE", ""),
        help="API base URL (required; or set OPENAI_API_BASE env var)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="API access key (required; or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature for ALL calls (default: 0.0)",
    )

    # === MEMORYFORGE-SPECIFIC PARAMETERS ===
    parser.add_argument(
        "--recency-window",
        type=int,
        default=5,
        help="Year recency window for high-res event concentration (default: 5)",
    )
    parser.add_argument(
        "--childhood-amnesia-age",
        type=int,
        default=3,
        help="Age threshold for childhood amnesia exclusion (default: 3)",
    )

    # === SIMULATION PARAMETERS ===
    parser.add_argument("--max-turns", type=int, default=22, help="Max interaction turns for P3 (default: 22)")
    parser.add_argument("--output-dir", default="./output", help="Output root directory (default: ./output)")
    parser.add_argument("--run-id", help="Explicit run_id; if omitted, auto-generated")
    parser.add_argument("--skip-p3", action="store_true", help="Skip high-resolution simulation")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--rpm-limit", type=int, default=300, help="RPM limit (default: 300, 0=disabled)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug-level logging")

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    try:
        asyncio.run(run_pipeline(args))
    except PipelineError as e:
        logger.exception(f"Pipeline failed at stage {e.stage}: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
