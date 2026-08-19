import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_pass_at_k_experiment.py"
SPEC = importlib.util.spec_from_file_location("run_pass_at_k_experiment", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


class PassAtKExperimentTest(unittest.TestCase):
    def test_load_selected_tasks_accepts_swesmith_only_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection = root / "selection.json"
            functions = root / "functions.jsonl"
            swesmith = root / "swesmith.jsonl"
            selection.write_text(
                json.dumps(
                    {
                        "splits": {
                            "train": {
                                "count": 1,
                                "swesmith_py": {"instance_ids": ["repo-task"]},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            functions.write_text("", encoding="utf-8")
            swesmith.write_text(
                json.dumps({"instance_id": "repo-task", "split": "train"}) + "\n",
                encoding="utf-8",
            )

            tasks = experiment.load_selected_tasks(
                selection,
                functions,
                swesmith,
                split="train",
            )

            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["experiment_dataset"], "swesmith_py")

    def test_resume_compatible_hash_allows_worker_change_only(self) -> None:
        base = {
            "config_sha256": "first",
            "model": {"name": "model", "temperature": 0.5},
            "execution_policy": {"workers": 4, "model_timeout_sec": 180},
        }
        more_workers = {
            **base,
            "config_sha256": "second",
            "execution_policy": {"workers": 12, "model_timeout_sec": 180},
        }
        changed_temperature = {
            **more_workers,
            "model": {"name": "model", "temperature": 0.7},
        }
        self.assertEqual(
            experiment.resume_compatible_hash(base), experiment.resume_compatible_hash(more_workers)
        )
        self.assertNotEqual(
            experiment.resume_compatible_hash(base), experiment.resume_compatible_hash(changed_temperature)
        )

    def test_rollout_command_passes_protocol_repair_policy(self) -> None:
        args = SimpleNamespace(
            swesmith_tasks="repo.jsonl",
            evalplus_tasks="function.jsonl",
            repo_max_steps=40,
            function_max_steps=20,
            edit_checkpoint_step=10,
            readonly_reminder_interval=3,
            model="demo",
            model_url="https://example.test",
            api_key_env="KEY",
            context_max_tokens=64000,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.95,
            protocol_retries=2,
            protocol_retry_temperature=0.0,
            thinking_mode="disabled",
            reasoning_effort="low",
            model_timeout_sec=180,
        )

        command = experiment.rollout_command(
            args,
            {"experiment_dataset": "mbppplus", "instance_id": "MBPP/1"},
            Path("rollout"),
            Path("workspace"),
        )

        self.assertIn("--protocol-retries", command)
        self.assertEqual(command[command.index("--protocol-retries") + 1], "2")
        self.assertIn("--protocol-retry-temperature", command)
        self.assertEqual(command[command.index("--protocol-retry-temperature") + 1], "0.0")
        self.assertEqual(command[command.index("--edit-checkpoint-step") + 1], "10")
        self.assertEqual(command[command.index("--readonly-reminder-interval") + 1], "3")

    def test_function_dataset_selects_both_function_benchmarks(self) -> None:
        tasks = [
            {"experiment_dataset": "mbppplus", "instance_id": "MBPP/1"},
            {"experiment_dataset": "humanevalplus", "instance_id": "HumanEval/0"},
            {"experiment_dataset": "swesmith_py", "instance_id": "repo-task"},
        ]

        selected = experiment.filter_tasks_by_dataset(tasks, "function")

        self.assertEqual([task["instance_id"] for task in selected], ["MBPP/1", "HumanEval/0"])

    def test_concurrent_jobs_run_all_samples_and_refresh_in_main_thread(self) -> None:
        args = SimpleNamespace(workers=2, samples=1)
        jobs = [
            {"task": {"instance_id": "one"}, "sample_index": 1, "task_index": 1, "task_count": 2},
            {"task": {"instance_id": "two"}, "sample_index": 1, "task_index": 2, "task_count": 2},
        ]

        with (
            patch.object(experiment, "run_sample", side_effect=lambda *args, **kwargs: kwargs),
            patch.object(experiment, "refresh_summary", return_value={}) as refresh,
        ):
            experiment.run_jobs_concurrently(args, Path("output"), jobs)

        self.assertEqual(refresh.call_count, 2)

    def test_missing_patch_model_failure_is_a_valid_failed_sample(self) -> None:
        result = experiment.classify_sample_result(
            task={"instance_id": "demo", "experiment_dataset": "swesmith_py", "repo": "a/b"},
            record={"status": "max_steps", "patch": "", "metadata": {}, "edit_apply": {}},
            verifier={"benchmark_resolved": None, "verifier_status": "blocked_missing_patch"},
            sample_index=1,
            attempt_index=1,
            split="evaluation",
            runner_returncode=0,
            verifier_returncode=2,
            elapsed_sec=1.0,
            attempt_dir=Path("attempt"),
        )

        self.assertTrue(result["sample_valid"])
        self.assertFalse(result["benchmark_resolved"])
        self.assertEqual(result["verifier_status"], "model_failure_max_steps")
        self.assertEqual(result["harness"]["version"], "lottie_code_agent_harness_v3")
        self.assertEqual(
            result["harness"]["readonly_budget_policy"]["version"],
            "append_nonmod3_action_reset_v1",
        )

    def test_missing_verifier_result_remains_infrastructure_invalid(self) -> None:
        result = experiment.classify_sample_result(
            task={"instance_id": "demo", "experiment_dataset": "mbppplus", "repo": "a/b"},
            record={"status": "resolved", "patch": "diff", "metadata": {}, "edit_apply": {}},
            verifier=None,
            sample_index=1,
            attempt_index=1,
            split="evaluation",
            runner_returncode=0,
            verifier_returncode=1,
            elapsed_sec=1.0,
            attempt_dir=Path("attempt"),
        )

        self.assertFalse(result["sample_valid"])
        self.assertIsNone(result["benchmark_resolved"])
        self.assertEqual(result["verifier_status"], "verifier_missing_result")


if __name__ == "__main__":
    unittest.main()
