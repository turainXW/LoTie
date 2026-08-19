from __future__ import annotations

import ast
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class FunctionGoldResult:
    instance_id: str
    dataset: str
    split: str
    status: str
    gold_sanity_passed: bool
    elapsed_sec: float
    compile_returncode: int | None
    reference_returncode: int | None
    stub_returncode: int | None
    output: dict[str, str]


@dataclass
class FunctionVerifyResult:
    instance_id: str
    benchmark_resolved: bool | None
    verifier_status: str
    patch_present: bool
    returncode: int | None
    elapsed_sec: float
    output: str
    metadata: dict[str, Any]


def load_dataset_server_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path_value in paths:
        path = Path(path_value)
        if path.name.endswith(".jsonl.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                source_rows = [json.loads(line) for line in handle if line.strip()]
            for row in source_rows:
                rows[str(row["task_id"])] = row
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        source_rows = data.get("rows", data) if isinstance(data, dict) else data
        for item in source_rows:
            row = item.get("row", item)
            rows[str(row["task_id"])] = row
    return list(rows.values())


def materialize_selected_tasks(
    selection_path: str | Path,
    mbpp_sources: Iterable[str | Path],
    humaneval_sources: Iterable[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    mbpp_rows = {_mbpp_numeric_id(row["task_id"]): row for row in load_dataset_server_rows(mbpp_sources)}
    humaneval_rows = {str(row["task_id"]): row for row in load_dataset_server_rows(humaneval_sources)}
    tasks: list[dict[str, Any]] = []
    missing: list[str] = []

    for split_name, split in selection["splits"].items():
        for task_id in split["mbppplus"]["task_ids"]:
            row = mbpp_rows.get(int(task_id))
            if row is None:
                missing.append(f"MBPP/{task_id}")
                continue
            tasks.append(_materialize_mbpp(row, split_name))
        for task_id in split["humanevalplus"]["task_ids"]:
            row = humaneval_rows.get(str(task_id))
            if row is None:
                missing.append(str(task_id))
                continue
            tasks.append(_materialize_humaneval(row, split_name))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, tasks)
    manifest = {
        "format": "lottie_evalplus_tasks_v1",
        "selection": str(Path(selection_path).resolve()),
        "output": str(output.resolve()),
        "tasks": len(tasks),
        "missing": missing,
        "dataset_counts": _counts(tasks, "dataset"),
        "split_counts": _counts(tasks, "split"),
        "gold_visible_to_agent": False,
        "official_comparable": False,
    }
    _write_json(output.with_suffix(".manifest.json"), manifest)
    return manifest


def setup_shared_venv(
    python_executable: str | Path,
    venv_path: str | Path,
    timeout_sec: int = 900,
) -> dict[str, Any]:
    python_path = Path(python_executable)
    target = Path(venv_path)
    if not python_path.is_file():
        raise FileNotFoundError(f"Python executable not found: {python_path}")
    fingerprint = hashlib.sha256(f"{python_path.resolve()}|numpy>=1.24,<3".encode()).hexdigest()
    marker = target / ".lottie-ready.json"
    if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get("fingerprint") == fingerprint:
        return _venv_summary(target, "ready_cached")
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    create = subprocess.run(
        [str(python_path), "-m", "venv", str(target)],
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )
    if create.returncode:
        raise RuntimeError(create.stderr or create.stdout or "venv creation failed")
    install = subprocess.run(
        [str(_venv_python(target)), "-m", "pip", "install", "--upgrade", "pip", "numpy>=1.24,<3"],
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )
    if install.returncode:
        raise RuntimeError(install.stderr or install.stdout or "dependency installation failed")
    marker.write_text(json.dumps({"fingerprint": fingerprint}, indent=2) + "\n", encoding="utf-8")
    return _venv_summary(target, "ready")


def run_gold_sanity(
    tasks_path: str | Path,
    output_path: str | Path,
    python_executable: str | Path,
    *,
    split: str | None = None,
    dataset: str | None = None,
    limit: int | None = None,
    timeout_sec: int = 30,
) -> list[FunctionGoldResult]:
    # Keep the venv launcher path intact; resolving its symlink bypasses the venv.
    python_executable = Path(os.path.abspath(python_executable))
    tasks = _read_jsonl(Path(tasks_path))
    if split:
        tasks = [task for task in tasks if task["split"] == split]
    if dataset:
        tasks = [task for task in tasks if task["dataset"] == dataset]
    if limit is not None:
        tasks = tasks[:limit]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results: list[FunctionGoldResult] = []
    with output.open("w", encoding="utf-8") as handle:
        for task in tasks:
            result = sanity_one_task(task, python_executable, timeout_sec=timeout_sec)
            results.append(result)
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            handle.flush()
    return results


def sanity_one_task(
    task: dict[str, Any],
    python_executable: str | Path,
    *,
    timeout_sec: int = 30,
) -> FunctionGoldResult:
    started = time.monotonic()
    output: dict[str, str] = {}
    compile_returncode: int | None = None
    reference_returncode: int | None = None
    stub_returncode: int | None = None
    status = "gold_sanity_passed"
    try:
        reference_source = _evaluation_source(task, task["reference_solution"])
        stub_source = _evaluation_source(task, task["starter_code"])
        with tempfile.TemporaryDirectory(prefix="lottie-evalplus-") as tmp:
            root = Path(tmp)
            reference_file = root / "reference.py"
            stub_file = root / "stub.py"
            reference_file.write_text(reference_source, encoding="utf-8")
            stub_file.write_text(stub_source, encoding="utf-8")
            compiled = _run([str(python_executable), "-m", "py_compile", str(reference_file)], root, timeout_sec)
            compile_returncode = compiled.returncode
            output["compile"] = _combined_output(compiled)
            if compiled.returncode:
                status = "invalid_reference_compile_failed"
            else:
                reference = _run([str(python_executable), "-I", str(reference_file)], root, timeout_sec)
                reference_returncode = reference.returncode
                output["reference"] = _combined_output(reference)
                if reference.returncode:
                    status = "invalid_reference_tests_failed"
                else:
                    stub = _run([str(python_executable), "-I", str(stub_file)], root, timeout_sec)
                    stub_returncode = stub.returncode
                    output["stub"] = _combined_output(stub)
                    if stub.returncode == 0:
                        status = "invalid_tests_did_not_catch_stub"
    except subprocess.TimeoutExpired as exc:
        status = "blocked_timeout"
        output["timeout"] = f"timeout after {exc.timeout}s"
    except (KeyError, SyntaxError, ValueError) as exc:
        status = "invalid_task_format"
        output["error"] = str(exc)
    return FunctionGoldResult(
        instance_id=task["instance_id"],
        dataset=task["dataset"],
        split=task["split"],
        status=status,
        gold_sanity_passed=status == "gold_sanity_passed",
        elapsed_sec=round(time.monotonic() - started, 3),
        compile_returncode=compile_returncode,
        reference_returncode=reference_returncode,
        stub_returncode=stub_returncode,
        output={key: value[-4000:] for key, value in output.items()},
    )


def build_runnable_tasks(
    tasks_path: str | Path,
    sanity_results_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    tasks = _read_jsonl(Path(tasks_path))
    results = {item["instance_id"]: item for item in _read_jsonl(Path(sanity_results_path))}
    runnable = [task for task in tasks if results.get(task["instance_id"], {}).get("gold_sanity_passed") is True]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, runnable)
    manifest = {
        "format": "lottie_evalplus_runnable_v1",
        "tasks_in": len(tasks),
        "tasks_out": len(runnable),
        "excluded": len(tasks) - len(runnable),
        "dataset_counts": _counts(runnable, "dataset"),
        "split_counts": _counts(runnable, "split"),
        "verification_scope": {
            "reference": "reference implementation compiles and executes all official base and plus inputs",
            "negative_control": "target function replaced by NotImplementedError and must fail",
            "official_comparable": False,
        },
    }
    _write_json(output.with_suffix(".manifest.json"), manifest)
    return manifest


def materialize_function_workspaces(
    tasks_path: str | Path,
    output_tasks_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    tasks = _read_jsonl(Path(tasks_path))
    root = Path(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    enriched: list[dict[str, Any]] = []
    for task in tasks:
        repo_path = root / task["instance_id"].replace("/", "__")
        if repo_path.exists():
            shutil.rmtree(repo_path)
        repo_path.mkdir(parents=True)
        (repo_path / "solution.py").write_text(task["starter_code"], encoding="utf-8")
        (repo_path / "README.md").write_text(str(task["problem_statement"]).rstrip() + "\n", encoding="utf-8")
        item = dict(task)
        item.update(
            {
                "repo": f"evalplus/{task['dataset']}",
                "base_commit": hashlib.sha256(task["starter_code"].encode()).hexdigest(),
                "local_repo_path": str(repo_path.resolve()),
                "test_command": "python -m py_compile solution.py",
                "patch": "",
                "bug_patch": "",
                "workspace_entry_file": "solution.py",
            }
        )
        enriched.append(item)
    output = Path(output_tasks_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, enriched)
    manifest = {
        "format": "lottie_evalplus_agent_tasks_v1",
        "tasks": len(enriched),
        "output": str(output.resolve()),
        "repo_root": str(root.resolve()),
        "dataset_counts": _counts(enriched, "dataset"),
        "split_counts": _counts(enriched, "split"),
        "gold_in_workspace": False,
    }
    _write_json(output.with_suffix(".manifest.json"), manifest)
    return manifest


def verify_function_patch_records(
    records_path: str | Path,
    tasks_path: str | Path,
    output_path: str | Path,
    python_executable: str | Path,
    *,
    timeout_sec: int = 30,
    limit: int | None = None,
) -> dict[str, Any]:
    records = _read_jsonl(Path(records_path))
    if limit is not None:
        records = records[:limit]
    tasks = {task["instance_id"]: task for task in _read_jsonl(Path(tasks_path))}
    python_path = Path(os.path.abspath(python_executable))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results: list[FunctionVerifyResult] = []
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            task = tasks.get(record.get("instance_id"))
            result = verify_one_function_record(record, task, python_path, timeout_sec=timeout_sec)
            results.append(result)
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            handle.flush()
    statuses = _counts((asdict(result) for result in results), "verifier_status")
    return {
        "total": len(results),
        "resolved": sum(result.benchmark_resolved is True for result in results),
        "unresolved": sum(result.benchmark_resolved is False for result in results),
        "blocked": sum(result.benchmark_resolved is None for result in results),
        "status_counts": statuses,
        "output": str(output.resolve()),
    }


def verify_one_function_record(
    record: dict[str, Any],
    task: dict[str, Any] | None,
    python_executable: str | Path,
    *,
    timeout_sec: int = 30,
) -> FunctionVerifyResult:
    started = time.monotonic()
    instance_id = str(record.get("instance_id") or "")
    patch = str(record.get("patch") or "")
    if task is None:
        return _function_verify_result(instance_id, None, "blocked_missing_task", bool(patch.strip()), None, started, "")
    repo_path = Path(str(task.get("local_repo_path") or ""))
    python_path = Path(python_executable)
    if not repo_path.is_dir() or not python_path.is_file():
        return _function_verify_result(
            instance_id,
            None,
            "blocked_missing_environment",
            bool(patch.strip()),
            None,
            started,
            "",
            {"repo_path": str(repo_path), "python": str(python_path)},
        )
    if not patch.strip():
        return _function_verify_result(instance_id, False, "unresolved_no_edit", False, None, started, "")
    with tempfile.TemporaryDirectory(prefix="lottie-function-verify-") as tmp:
        workspace = Path(tmp) / "repo"
        shutil.copytree(repo_path, workspace, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        applied = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=workspace,
            input=patch,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        if applied.returncode:
            return _function_verify_result(
                instance_id,
                False,
                "unresolved_patch_apply_failed",
                True,
                applied.returncode,
                started,
                _combined_output(applied),
            )
        candidate_path = workspace / str(task.get("workspace_entry_file") or "solution.py")
        if not candidate_path.is_file():
            return _function_verify_result(instance_id, False, "unresolved_missing_solution", True, None, started, "")
        evaluation = workspace / ".lottie_hidden_eval.py"
        evaluation.write_text(_candidate_evaluation_source(task, candidate_path.read_text(encoding="utf-8")), encoding="utf-8")
        try:
            completed = _run([str(python_path), "-I", str(evaluation)], workspace, timeout_sec)
        except subprocess.TimeoutExpired:
            return _function_verify_result(instance_id, False, "unresolved_timeout", True, None, started, "timeout")
        return _function_verify_result(
            instance_id,
            completed.returncode == 0,
            "resolved" if completed.returncode == 0 else "unresolved_tests_failed",
            True,
            completed.returncode,
            started,
            _combined_output(completed)[-8000:],
        )


def _function_verify_result(
    instance_id: str,
    resolved: bool | None,
    status: str,
    patch_present: bool,
    returncode: int | None,
    started: float,
    output: str,
    metadata: dict[str, Any] | None = None,
) -> FunctionVerifyResult:
    return FunctionVerifyResult(
        instance_id=instance_id,
        benchmark_resolved=resolved,
        verifier_status=status,
        patch_present=patch_present,
        returncode=returncode,
        elapsed_sec=round(time.monotonic() - started, 3),
        output=output,
        metadata={"official_comparable": False, **(metadata or {})},
    )


def _materialize_mbpp(row: dict[str, Any], split: str) -> dict[str, Any]:
    reference = str(row.get("canonical_solution") or row.get("code") or "")
    entry_point = str(row.get("entry_point") or "") or _infer_mbpp_entry_point(
        reference, row.get("test_list", []), str(row.get("test") or "")
    )
    starter = _replace_function_body(reference, entry_point)
    return {
        "instance_id": f"MBPP/{_mbpp_numeric_id(row['task_id'])}",
        "dataset": "mbppplus",
        "split": split,
        "task_type": "function_implementation",
        "problem_statement": str(row["prompt"]),
        "entry_point": entry_point,
        "starter_code": starter,
        "reference_solution": reference,
        "hidden_tests": str(row.get("test") or ""),
        "base_inputs": list(row.get("base_input") or []),
        "plus_inputs": list(row.get("plus_input") or []),
        "atol": row.get("atol", 0),
        "assertion": str(row.get("assertion") or ""),
        "contract": str(row.get("contract") or ""),
        "base_tests": list(row.get("test_list") or []),
        "test_imports": list(row.get("test_imports") or []),
        "gold_policy": "reference_solution_plus_embedded_evalplus_tests",
        "gold_visible_to_agent": False,
        "python_version": "3.10",
        "official_comparable": False,
    }


def _materialize_humaneval(row: dict[str, Any], split: str) -> dict[str, Any]:
    reference = str(row["prompt"]) + str(row["canonical_solution"])
    entry_point = str(row["entry_point"])
    starter = _replace_function_body(reference, entry_point)
    return {
        "instance_id": str(row["task_id"]),
        "dataset": "humanevalplus",
        "split": split,
        "task_type": "function_implementation",
        "problem_statement": str(row["prompt"]),
        "entry_point": entry_point,
        "starter_code": starter,
        "reference_solution": reference,
        "hidden_tests": str(row.get("test") or ""),
        "base_inputs": list(row.get("base_input") or []),
        "plus_inputs": list(row.get("plus_input") or []),
        "atol": row.get("atol", 0),
        "assertion": str(row.get("assertion") or ""),
        "contract": str(row.get("contract") or ""),
        "base_tests": [],
        "test_imports": [],
        "gold_policy": "canonical_solution_plus_embedded_evalplus_tests",
        "gold_visible_to_agent": False,
        "python_version": "3.10",
        "official_comparable": False,
    }


def _evaluation_source(task: dict[str, Any], candidate_source: str) -> str:
    test_source = str(task["hidden_tests"])
    inputs = [*task.get("base_inputs", []), *task.get("plus_inputs", [])]
    if inputs:
        if task["dataset"] == "mbppplus":
            inputs = _deserialize_mbpp_inputs(task["instance_id"], inputs)
        assertion = str(task.get("assertion") or "")
        invocation = (
            f"\n\n_evalplus_inputs = {inputs!r}\n"
            "for _evalplus_input in _evalplus_inputs:\n"
            f"    {task['entry_point']}(*_evalplus_input)\n"
        )
        return candidate_source.rstrip() + "\n\n" + assertion.rstrip() + invocation
    suffix = ""
    if task["dataset"] == "humanevalplus":
        suffix = f"\n\ncheck({task['entry_point']})\n"
    return candidate_source.rstrip() + "\n\n" + test_source.rstrip() + suffix


def _candidate_evaluation_source(task: dict[str, Any], candidate_source: str) -> str:
    inputs = [*task.get("base_inputs", []), *task.get("plus_inputs", [])]
    if task["dataset"] == "mbppplus":
        inputs = _deserialize_mbpp_inputs(task["instance_id"], inputs)
    return f'''import copy
import json
import math
import numpy as np

candidate_ns = {{}}
gold_ns = {{}}
exec(compile({candidate_source!r}, "candidate.py", "exec"), candidate_ns)
exec(compile({str(task["reference_solution"])!r}, "gold.py", "exec"), gold_ns)
entry_point = {str(task["entry_point"])!r}
dataset = {str(task["dataset"])!r}
inputs = {inputs!r}
atol = {task.get("atol", 0)!r}

set_oracles = {{"similar_elements", "find_char_long", "common_in_nested_lists", "extract_singly", "larg_nnum", "intersection_array", "find_dissimilar", "Diff"}}
not_none_oracles = {{"check_str", "text_match_three", "text_starta_endb"}}

def is_floats(value):
    if isinstance(value, float):
        return True
    if isinstance(value, (list, tuple)) and value:
        return all(isinstance(item, float) for item in value)
    if isinstance(value, np.ndarray):
        return value.dtype in (np.float32, np.float64)
    return False

def poly(xs, x):
    return sum(coeff * math.pow(x, i) for i, coeff in enumerate(xs))

def surface_area(base_edge, height):
    slant_height = math.sqrt((base_edge / 2) ** 2 + height**2)
    return round(base_edge**2 + 4 * (base_edge * slant_height) / 2)

def digit_distance(num1, num2):
    left, right = str(num1), str(num2)
    size = max(len(left), len(right))
    return sum(abs(int(a) - int(b)) for a, b in zip(left.zfill(size), right.zfill(size)))

def exact_equal(out, exp):
    try:
        value = out == exp
        if isinstance(value, np.ndarray):
            return bool(np.all(value))
        return bool(value)
    except BaseException:
        return False

def matches(out, exp, inp):
    if dataset == "mbppplus":
        if entry_point == "are_equivalent":
            return True
        if entry_point == "sum_div" and out == 0:
            return True
        if entry_point == "surface_Area" and abs(out - surface_area(*inp)) <= atol:
            return True
        if entry_point == "digit_distance_nums" and out == digit_distance(*inp):
            return True
        if entry_point in set_oracles:
            return set(out) == set(exp)
        if entry_point in not_none_oracles:
            return out == exp if isinstance(out, bool) else exp == (out is not None)
    if dataset == "humanevalplus" and entry_point == "find_zero":
        return abs(poly(*inp, out)) <= atol
    if exact_equal(out, exp):
        return True
    tolerance = 1e-6 if atol == 0 and is_floats(exp) else atol
    if tolerance == 0 or type(out) is not type(exp):
        return False
    if isinstance(exp, (list, tuple)) and len(out) != len(exp):
        return False
    try:
        return bool(np.allclose(out, exp, rtol=1e-7, atol=tolerance))
    except BaseException:
        return False

candidate = candidate_ns[entry_point]
gold = gold_ns[entry_point]
for index, original_input in enumerate(inputs):
    try:
        expected = gold(*copy.deepcopy(original_input))
        actual = candidate(*copy.deepcopy(original_input))
        if not matches(actual, expected, original_input):
            raise AssertionError(f"mismatch at input {{index}}: expected={{expected!r}} actual={{actual!r}}")
    except BaseException as exc:
        print(json.dumps({{"status": "failed", "index": index, "error": repr(exc)}}))
        raise
print(json.dumps({{"status": "passed", "tests": len(inputs)}}))
'''


def _infer_mbpp_entry_point(reference: str, test_list: list[str], hidden_tests: str) -> str:
    tree = ast.parse(reference)
    definitions = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not definitions:
        raise ValueError("MBPP reference contains no top-level function")
    calls: list[str] = []
    for source in [*test_list, hidden_tests]:
        try:
            test_tree = ast.parse(source)
        except SyntaxError:
            continue
        calls.extend(node.func.id for node in ast.walk(test_tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name))
    for name in definitions:
        if name in calls:
            return name
    raise ValueError(f"Unable to infer MBPP entry point from {definitions}")


def _mbpp_numeric_id(value: Any) -> int:
    return int(str(value).rsplit("/", 1)[-1])


def _deserialize_mbpp_inputs(task_id: str, inputs: list[Any]) -> list[Any]:
    """Mirror EvalPlus 0.3.1's JSON input adapters without importing model backends."""
    numeric_id = _mbpp_numeric_id(task_id)
    tuple_args = {
        2, 116, 132, 143, 222, 261, 273, 394, 399, 421, 424, 429,
        470, 560, 579, 596, 616, 630, 726, 740, 744, 809,
    }
    nested_tuple_args = {63, 64, 70, 94, 120, 237, 272, 299, 400, 409, 417, 438, 473, 614, 780}
    if numeric_id in tuple_args:
        return [[tuple(value) for value in item] for item in inputs]
    if numeric_id in nested_tuple_args:
        return [[[tuple(value) for value in group] for group in item] for item in inputs]
    if numeric_id in {75, 413, 444, 753}:
        return [[[tuple(value) for value in item[0]], item[1]] for item in inputs]
    if numeric_id in {106, 750}:
        return [[item[0], tuple(item[1])] for item in inputs]
    if numeric_id == 115:
        return [[[{*value} if isinstance(value, list) and value else {} for value in item[0]]] for item in inputs]
    if numeric_id == 124:
        return [[float(item[0]), complex(item[1])] for item in inputs]
    if numeric_id in {250, 405, 446, 617, 720, 763, 808}:
        return [[tuple(item[0]), item[1]] for item in inputs]
    if numeric_id in {259, 401, 445}:
        return [[tuple(tuple(value) for value in group) for group in item] for item in inputs]
    if numeric_id == 278:
        return [[tuple(tuple(value) if isinstance(value, list) else value for value in item[0])] for item in inputs]
    if numeric_id == 307:
        return [[tuple(item[0]), item[1], item[2]] for item in inputs]
    if numeric_id == 722:
        return [[{key: tuple(value) for key, value in item[0].items()}, *item[1:]] for item in inputs]
    if numeric_id == 252:
        return [[complex(item[0])] for item in inputs]
    if numeric_id in {580, 615, 791}:
        def recursive_tuple(value: Any) -> Any:
            return tuple(recursive_tuple(item) for item in value) if isinstance(value, list) else value

        return [recursive_tuple(item) for item in inputs]
    return inputs


def _replace_function_body(source: str, entry_point: str) -> str:
    tree = ast.parse(source)
    replaced = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            node.body = [
                ast.Raise(
                    exc=ast.Call(func=ast.Name(id="NotImplementedError", ctx=ast.Load()), args=[], keywords=[]),
                    cause=None,
                )
            ]
            replaced = True
            break
    if not replaced:
        raise ValueError(f"Target function not found: {entry_point}")
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _run(command: list[str], cwd: Path, timeout_sec: int) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout_sec, env=env)


def _combined_output(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stdout or "") + (result.stderr or "")


def _venv_python(venv_path: Path) -> Path:
    return venv_path / "bin" / "python"


def _venv_summary(venv_path: Path, status: str) -> dict[str, Any]:
    version = subprocess.run(
        [str(_venv_python(venv_path)), "--version"], text=True, capture_output=True, check=True
    )
    return {
        "status": status,
        "venv_path": str(venv_path.resolve()),
        "python": (version.stdout or version.stderr).strip(),
        "venv_bytes": sum(path.stat().st_size for path in venv_path.rglob("*") if path.is_file()),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _counts(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        result[value] = result.get(value, 0) + 1
    return result
