from __future__ import annotations

import ast
import hashlib
import json
import re
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


IGNORE_DIRS = {
    ".git",
    ".mswea",
    ".codeagent",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "refs/open_source",
}

LANG_BY_SUFFIX = {
    ".py": "python",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".sh": "shell",
    ".txt": "text",
}


@dataclass
class FileContext:
    path: str
    language: str
    role: str
    lines: int
    bytes: int
    sha1: str = ""
    module: str = ""
    imports: list[str] = field(default_factory=list)
    imported_modules: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    symbol_locations: list[dict[str, Any]] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepoMap:
    repo_path: str
    files: list[FileContext]
    generated_at: float = 0.0
    version: str = "repo_context_v2"
    root_packages: list[str] = field(default_factory=list)
    module_to_path: dict[str, str] = field(default_factory=dict)
    import_graph: dict[str, list[str]] = field(default_factory=dict)
    reverse_import_graph: dict[str, list[str]] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_path": self.repo_path,
            "files": [file.to_dict() for file in self.files],
            "generated_at": self.generated_at,
            "version": self.version,
            "root_packages": self.root_packages,
            "module_to_path": self.module_to_path,
            "import_graph": self.import_graph,
            "reverse_import_graph": self.reverse_import_graph,
            "stats": self.stats,
        }


def build_repo_map(repo_path: str | Path, *, max_file_bytes: int = 200_000) -> RepoMap:
    repo = Path(repo_path).resolve()
    files = []
    for path in sorted(repo.rglob("*")):
        if not path.is_file() or _ignored(path, repo):
            continue
        rel = path.relative_to(repo).as_posix()
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > max_file_bytes:
            files.append(
                FileContext(
                    path=rel,
                    language=LANG_BY_SUFFIX.get(path.suffix, "unknown"),
                    role=_infer_role(rel),
                    lines=0,
                    bytes=size,
                    sha1="",
                    module=_module_name(rel),
                )
            )
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        files.append(_summarize_file(rel, text, size))
    module_to_path = {file_ctx.module: file_ctx.path for file_ctx in files if file_ctx.module}
    import_graph = _build_import_graph(files)
    reverse_import_graph = _reverse_graph(import_graph)
    return RepoMap(
        repo_path=str(repo),
        files=files,
        generated_at=time.time(),
        root_packages=_root_packages(files),
        module_to_path=module_to_path,
        import_graph=import_graph,
        reverse_import_graph=reverse_import_graph,
        stats=_repo_stats(files),
    )


def save_repo_map(repo_map: RepoMap, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(repo_map.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_repo_map(path: str | Path) -> RepoMap:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    files = [_file_context_from_dict(file_data) for file_data in data.get("files", [])]
    import_graph = data.get("import_graph") or _build_import_graph(files)
    return RepoMap(
        repo_path=data["repo_path"],
        files=files,
        generated_at=float(data.get("generated_at", 0.0)),
        version=str(data.get("version", "repo_context_v1")),
        root_packages=list(data.get("root_packages") or _root_packages(files)),
        module_to_path=dict(data.get("module_to_path") or {file_ctx.module: file_ctx.path for file_ctx in files if file_ctx.module}),
        import_graph={key: list(value) for key, value in import_graph.items()},
        reverse_import_graph={
            key: list(value) for key, value in (data.get("reverse_import_graph") or _reverse_graph(import_graph)).items()
        },
        stats=dict(data.get("stats") or _repo_stats(files)),
    )


def compact_repo_context(
    repo_map: RepoMap,
    task: str,
    *,
    max_files: int = 12,
    max_chars: int = 5000,
) -> str:
    result = search_repo_context(repo_map, task, max_files=max_files, max_snippets=0, max_chars=max_chars)
    selected = [_file_by_path(repo_map, item["path"]) for item in result["files"]]
    lines = [
        "Repository context map:",
        f"- repo: {repo_map.repo_path}",
        f"- version: {repo_map.version}",
        f"- files: {repo_map.stats.get('file_count', len(repo_map.files))}",
        f"- root_packages: {', '.join(repo_map.root_packages[:8]) or 'unknown'}",
        "- use this map to choose files; read full files only when needed",
    ]
    for file_ctx in [item for item in selected if item is not None]:
        parts = [
            f"- {file_ctx.path}",
            f"role={file_ctx.role}",
            f"lang={file_ctx.language}",
            f"lines={file_ctx.lines}",
        ]
        if file_ctx.symbols:
            parts.append("symbols=" + ", ".join(file_ctx.symbols[:8]))
        if file_ctx.imports:
            parts.append("imports=" + ", ".join(file_ctx.imports[:6]))
        dependents = repo_map.reverse_import_graph.get(file_ctx.module, []) if file_ctx.module else []
        if dependents:
            parts.append("imported_by=" + ", ".join(dependents[:4]))
        if file_ctx.headings:
            parts.append("headings=" + " | ".join(file_ctx.headings[:4]))
        lines.append("; ".join(parts))
    text = "\n".join(lines)
    return text[:max_chars].rstrip()


def search_repo_context(
    repo_map: RepoMap,
    query: str,
    *,
    max_files: int = 12,
    max_snippets: int = 6,
    max_chars: int = 8000,
    include_tests: bool = True,
) -> dict[str, Any]:
    """Return task-ranked file candidates plus compact evidence snippets."""
    ranked = sorted(
        repo_map.files,
        key=lambda file_ctx: score_file_for_task(file_ctx, query),
        reverse=True,
    )
    if not include_tests:
        ranked = [file_ctx for file_ctx in ranked if file_ctx.role != "test"]
    selected = ranked[:max_files]
    snippets: list[dict[str, Any]] = []
    if max_snippets > 0:
        for file_ctx in selected:
            if len(snippets) >= max_snippets:
                break
            snippets.extend(_snippets_for_file(Path(repo_map.repo_path), file_ctx, query, limit=max_snippets - len(snippets)))

    files = []
    for file_ctx in selected:
        dependents = repo_map.reverse_import_graph.get(file_ctx.module, []) if file_ctx.module else []
        imports = repo_map.import_graph.get(file_ctx.module, []) if file_ctx.module else []
        files.append(
            {
                "path": file_ctx.path,
                "score": round(score_file_for_task(file_ctx, query), 3),
                "role": file_ctx.role,
                "language": file_ctx.language,
                "module": file_ctx.module,
                "lines": file_ctx.lines,
                "bytes": file_ctx.bytes,
                "symbols": file_ctx.symbols[:12],
                "imports": imports[:12],
                "imported_by": dependents[:12],
                "reason": _reason_for_match(file_ctx, query, dependents),
            }
        )
    result = {
        "repo_path": repo_map.repo_path,
        "version": repo_map.version,
        "query": query,
        "stats": repo_map.stats,
        "root_packages": repo_map.root_packages,
        "files": files,
        "snippets": snippets[:max_snippets],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if len(text) <= max_chars:
        return result
    result["snippets"] = []
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if len(text) <= max_chars:
        return result
    result["files"] = result["files"][: max(1, max_files // 2)]
    return result


def score_file_for_task(file_ctx: FileContext, task: str) -> float:
    task_terms = set(_terms(task))
    haystack_terms = set(_terms(" ".join([file_ctx.path, file_ctx.role, file_ctx.language, file_ctx.module])))
    haystack_terms.update(_terms(" ".join(file_ctx.symbols + file_ctx.imports + file_ctx.imported_modules + file_ctx.headings)))
    score = len(task_terms & haystack_terms) * 3.0
    path = file_ctx.path.lower()
    task_lower = task.lower()
    path_terms = set(_terms(file_ctx.path))
    symbol_terms = set(_terms(" ".join(file_ctx.symbols)))
    if task_terms & path_terms:
        score += len(task_terms & path_terms) * 2.0
    if task_terms & symbol_terms:
        score += len(task_terms & symbol_terms) * 2.5
    if "readme" in path or path.endswith(".md"):
        score += 0.5
    if "test" in task_lower and ("test" in path or file_ctx.role == "test"):
        score += 4.0
    if "cli" in task_lower and ("script" in path or file_ctx.role == "entrypoint"):
        score += 4.0
    if "permission" in task_lower or "权限" in task:
        if "guard" in path or "mode" in " ".join(file_ctx.symbols).lower():
            score += 5.0
    if "web" in task_lower or "download" in task_lower or "下载" in task:
        if "web" in path or "download" in " ".join(file_ctx.symbols).lower():
            score += 5.0
    return score


def _summarize_file(rel_path: str, text: str, size: int) -> FileContext:
    suffix = Path(rel_path).suffix
    language = LANG_BY_SUFFIX.get(suffix, "unknown")
    imports: list[str] = []
    imported_modules: list[str] = []
    symbols: list[str] = []
    symbol_locations: list[dict[str, Any]] = []
    headings: list[str] = []

    if language == "python":
        imports, imported_modules, symbols, symbol_locations = _python_imports_symbols(text)
    elif language == "markdown":
        headings = [line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#")][:12]

    return FileContext(
        path=rel_path,
        language=language,
        role=_infer_role(rel_path),
        lines=text.count("\n") + (1 if text else 0),
        bytes=size,
        sha1=hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest(),
        module=_module_name(rel_path),
        imports=imports,
        imported_modules=imported_modules,
        symbols=symbols,
        symbol_locations=symbol_locations,
        headings=headings,
    )


def _python_imports_symbols(text: str) -> tuple[list[str], list[str], list[str], list[dict[str, Any]]]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text)
    except SyntaxError:
        return [], [], [], []
    imports: list[str] = []
    imported_modules: list[str] = []
    symbols: list[str] = []
    symbol_locations: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.append(alias.name)
                imports.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.append(node.module)
                imports.append(node.module.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(node.name)
            symbol_locations.append(
                {
                    "name": node.name,
                    "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                    "line": int(getattr(node, "lineno", 0)),
                    "end_line": int(getattr(node, "end_lineno", getattr(node, "lineno", 0))),
                }
            )
    return sorted(set(imports))[:20], sorted(set(imported_modules))[:40], symbols[:40], symbol_locations[:60]


def _infer_role(rel_path: str) -> str:
    name = Path(rel_path).name.lower()
    path = rel_path.lower()
    if name.startswith("test_") or "/tests/" in path or path.startswith("tests/"):
        return "test"
    if name in {"readme.md", "readme_zh.md"}:
        return "overview"
    if path.startswith("scripts/") or name.endswith(".sh"):
        return "entrypoint"
    if "config" in path or name.endswith((".yaml", ".yml", ".toml", ".json")):
        return "config"
    if "web" in path or "tool" in path:
        return "tooling"
    if "context" in path or "skill" in path:
        return "context"
    return "source"


def _ignored(path: Path, repo: Path) -> bool:
    rel = path.relative_to(repo).as_posix()
    parts = set(path.relative_to(repo).parts)
    if parts & IGNORE_DIRS:
        return True
    return any(rel == ignored or rel.startswith(f"{ignored}/") for ignored in IGNORE_DIRS)


def _terms(text: str) -> list[str]:
    terms: list[str] = []
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}", text):
        lowered = raw.lower()
        terms.append(lowered)
        if "_" in lowered:
            terms.extend(part for part in lowered.split("_") if len(part) >= 2)
    return terms


def _file_context_from_dict(data: dict[str, Any]) -> FileContext:
    allowed = set(FileContext.__dataclass_fields__)
    filtered = {key: value for key, value in data.items() if key in allowed}
    if "module" not in filtered:
        filtered["module"] = _module_name(str(filtered.get("path", "")))
    if "sha1" not in filtered:
        filtered["sha1"] = ""
    if "imported_modules" not in filtered:
        filtered["imported_modules"] = list(filtered.get("imports", []))
    if "symbol_locations" not in filtered:
        filtered["symbol_locations"] = []
    return FileContext(**filtered)


def _module_name(rel_path: str) -> str:
    path = Path(rel_path)
    if path.suffix != ".py":
        return ""
    parts = list(path.with_suffix("").parts)
    if len(parts) > 1 and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return ""
    return ".".join(parts)


def _build_import_graph(files: list[FileContext]) -> dict[str, list[str]]:
    modules = {file_ctx.module for file_ctx in files if file_ctx.module}
    graph: dict[str, list[str]] = {}
    for file_ctx in files:
        if not file_ctx.module:
            continue
        matched: set[str] = set()
        for imported in file_ctx.imported_modules or file_ctx.imports:
            candidates = {imported}
            if "." not in imported and "." in file_ctx.module:
                package = file_ctx.module.rsplit(".", 1)[0]
                candidates.add(f"{package}.{imported}")
            for module in modules:
                if any(candidate == module or module.startswith(f"{candidate}.") for candidate in candidates):
                    matched.add(module)
        graph[file_ctx.module] = sorted(matched)
    return graph


def _reverse_graph(graph: dict[str, list[str]]) -> dict[str, list[str]]:
    reverse: dict[str, list[str]] = {}
    for source, targets in graph.items():
        for target in targets:
            reverse.setdefault(target, []).append(source)
    return {key: sorted(set(value)) for key, value in reverse.items()}


def _root_packages(files: list[FileContext]) -> list[str]:
    roots = set()
    for file_ctx in files:
        if file_ctx.module:
            roots.add(file_ctx.module.split(".")[0])
    return sorted(roots)


def _repo_stats(files: list[FileContext]) -> dict[str, Any]:
    by_role: dict[str, int] = {}
    by_language: dict[str, int] = {}
    for file_ctx in files:
        by_role[file_ctx.role] = by_role.get(file_ctx.role, 0) + 1
        by_language[file_ctx.language] = by_language.get(file_ctx.language, 0) + 1
    return {
        "file_count": len(files),
        "python_file_count": by_language.get("python", 0),
        "test_file_count": by_role.get("test", 0),
        "by_role": by_role,
        "by_language": by_language,
    }


def _file_by_path(repo_map: RepoMap, rel_path: str) -> FileContext | None:
    for file_ctx in repo_map.files:
        if file_ctx.path == rel_path:
            return file_ctx
    return None


def _snippets_for_file(repo: Path, file_ctx: FileContext, query: str, *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or file_ctx.lines == 0 or file_ctx.bytes > 120_000:
        return []
    path = repo / file_ctx.path
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    query_terms = set(_terms(query))
    candidates: list[int] = []
    for idx, line in enumerate(lines, start=1):
        if query_terms & set(_terms(line)):
            candidates.append(idx)
    for symbol in file_ctx.symbol_locations:
        name_terms = set(_terms(str(symbol.get("name", ""))))
        if query_terms & name_terms:
            candidates.append(int(symbol.get("line", 1)))
    if not candidates and file_ctx.symbol_locations:
        candidates.append(int(file_ctx.symbol_locations[0].get("line", 1)))
    if not candidates:
        candidates.append(1)
    snippets = []
    used: set[tuple[int, int]] = set()
    for line_no in candidates:
        start = max(1, line_no - 3)
        end = min(len(lines), line_no + 5)
        key = (start, end)
        if key in used:
            continue
        used.add(key)
        body = "\n".join(f"{idx}: {lines[idx - 1]}" for idx in range(start, end + 1))
        snippets.append({"path": file_ctx.path, "start": start, "end": end, "text": body})
        if len(snippets) >= limit:
            break
    return snippets


def _reason_for_match(file_ctx: FileContext, query: str, dependents: list[str]) -> str:
    query_terms = set(_terms(query))
    matched = []
    for label, text in [
        ("path", file_ctx.path),
        ("symbols", " ".join(file_ctx.symbols)),
        ("imports", " ".join(file_ctx.imported_modules or file_ctx.imports)),
        ("headings", " ".join(file_ctx.headings)),
    ]:
        overlap = sorted(query_terms & set(_terms(text)))
        if overlap:
            matched.append(f"{label}:{','.join(overlap[:4])}")
    if dependents:
        matched.append(f"imported_by:{len(dependents)}")
    return "; ".join(matched) or f"role:{file_ctx.role}"
