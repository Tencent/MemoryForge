"""Exercise participant identity across the pipeline's supported pool formats."""

import json
import tempfile
import unittest
from pathlib import Path

from lifelong_synth.simulation_p1_initialisation.participant_pool import (
    ParticipantPoolManager,
)
from run_MemoryForge import load_pool_from_json


class PoolResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_resume_preserves_participants_and_persistence(self):
        for serialization in ("to_json", "save"):
            with self.subTest(serialization=serialization):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "pool.json"
                    original = ParticipantPoolManager.from_dict({
                        "simulation_start_date": "2000-01-01",
                        "simulation_end_date": "2020-12-31",
                    })
                    for name in ("Alice", "Bob"):
                        await original.add_participant(
                            name, "friend", f"{name}'s background",
                            extend_profile=False,
                        )
                    expected = original.to_dict()["participants"]
                    if serialization == "to_json":
                        path.write_text(original.to_json(), encoding="utf-8")
                    else:
                        original.save(str(path))

                    client = object()
                    restored = load_pool_from_json(path, client)
                    self.assertIs(restored.llm, client)
                    added = await restored.add_participant(
                        "Carol", "colleague", "A new colleague",
                        extend_profile=False,
                    )
                    self.assertEqual(added.participant_id, "P_003")
                    self.assertEqual(restored.count(), 3)
                    for pid, participant in expected.items():
                        self.assertEqual(
                            restored.to_dict()["participants"][pid], participant,
                        )

                    # The restored default save path and simulation dates survive.
                    restored.save()
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual(saved["simulation_start_date"], "2000-01-01")
                    self.assertEqual(saved["simulation_end_date"], "2020-12-31")
                    self.assertEqual(saved["total_count"], 3)
                    resumed_again = load_pool_from_json(path, client)
                    next_added = await resumed_again.add_participant(
                        "David", "friend", "Another friend", extend_profile=False,
                    )
                    self.assertEqual(next_added.participant_id, "P_004")
                    self.assertEqual(resumed_again.count(), 4)

    async def test_resume_uses_highest_numeric_id_in_either_format(self):
        ids = ("P_TARGET", "P_012", "P_002", "P_legacy", "external")
        participants = {
            pid: {
                "participant_id": pid,
                "persona_name_text": pid,
                "relationship_towards_the_main_character": "friend",
                "role": "friend",
                "persona_brief_text": "An existing participant",
                "appear_period": "LP1",
                "interactions_history_with_the_main_character": [],
            }
            for pid in ids
        }
        for payload in (participants, list(participants.values())):
            with self.subTest(format=type(payload).__name__):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "pool.json"
                    path.write_text(
                        json.dumps({"participants": payload}), encoding="utf-8",
                    )
                    restored = load_pool_from_json(path, None)
                    added = await restored.add_participant(
                        "New friend", "friend", "A new participant",
                        extend_profile=False,
                    )
                    self.assertEqual(added.participant_id, "P_013")
                    self.assertEqual(restored.count(), len(ids) + 1)
                    self.assertTrue(all(restored.get_participant(pid) for pid in ids))

    async def test_empty_pool_starts_with_first_id(self):
        for payload in ({}, []):
            with self.subTest(format=type(payload).__name__):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "pool.json"
                    path.write_text(
                        json.dumps({"participants": payload}), encoding="utf-8",
                    )
                    restored = load_pool_from_json(path, None)
                    added = await restored.add_participant(
                        "First friend", "friend", "The first participant",
                        extend_profile=False,
                    )
                    self.assertEqual(added.participant_id, "P_001")
                    self.assertEqual(restored.count(), 1)


if __name__ == "__main__":
    unittest.main()
