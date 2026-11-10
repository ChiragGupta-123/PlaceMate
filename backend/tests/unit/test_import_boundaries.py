# filepath: backend/tests/unit/test_import_boundaries.py
"""
Architectural import-boundary tests (Step 3.6b).

These tests read ``app/`` with ``ast`` only: no module under ``app`` is
imported or executed, so no candidate code, harness text or provider code
ever runs here.

Rules enforced (as the code stands today):
  1. ``tools/code_sandbox.py`` imports only stdlib (plus, if it ever needs
     them, ``app.schemas`` / ``app.core``); never httpx, judge0 or app.graph.
  2. ``tools/fake_sandbox.py`` imports only stdlib and ``code_sandbox``.
  3. Only ``tools/judge0.py`` and real LLM adapter modules
     (``tools/llm_<provider>.py``) may import or use httpx.  The port
     (``tools/llm_client.py``) and the fake (``tools/fake_llm.py``) never may.
  4. Only ``tools/judge0.py`` builds the RapidAPI protocol headers;
     ``core/config.py`` is the one other file allowed to *name* rapidapi,
     because that is where the SecretStr credential and its help URL live.
  5. Nothing under ``app/graph/`` imports ``app.tools`` or httpx.
  6. No ``exec()`` / ``eval()`` / ``os.system()`` / ``subprocess.*`` call
     anywhere under ``app/``, and no module imports ``subprocess`` (which
     would allow an aliased call to slip past call-site scanning).
  7. Only real LLM adapters (``tools/llm_<provider>.py``) may import an
     LLM SDK; ``llm_client.py`` and ``fake_llm.py`` never may.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterable, List, Set

APP_DIR = Path(__file__).resolve().parents[2] / "app"

RAPIDAPI_HEADERS = ("x-rapidapi-key", "x-rapidapi-host")
RAPIDAPI_FILES = {"tools/judge0.py", "core/config.py"}
BANNED_BUILTINS = {"exec", "eval"}
BANNED_DOTTED = {"os.system", "builtins.exec", "builtins.eval"}

# LLM adapters are ``tools/llm_<provider>.py``.  The port and the fake are
# NOT adapters even though ``llm_client.py`` shares the prefix.
LLM_PORT_FILE = "tools/llm_client.py"
LLM_FAKE_FILE = "tools/fake_llm.py"
LLM_NON_ADAPTER_FILES = {LLM_PORT_FILE, LLM_FAKE_FILE}

# Third-party LLM SDK roots.  Add new providers here in Step 4.4.
LLM_SDK_ROOTS = (
    "openai",
    "anthropic",
    "groq",
    "mistralai",
    "cohere",
    "litellm",
    "ollama",
    "google.generativeai",
    "vertexai",
    "langchain_openai",
    "langchain_anthropic",
)


# ── AST helpers (no app imports) ─────────────────────────────────────

def _app_files() -> List[Path]:
    files = sorted(APP_DIR.rglob("*.py"))
    assert files, f"no Python files found under {APP_DIR}"
    return files


def _rel(path: Path) -> str:
    return path.relative_to(APP_DIR).as_posix()


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _package_parts(path: Path) -> List[str]:
    parts = list(path.relative_to(APP_DIR).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return parts


def _imported_modules(path: Path, tree: ast.Module) -> Set[str]:
    """Fully-qualified module names from every Import/ImportFrom node."""
    package = _package_parts(path)
    modules: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level]
                prefix = ".".join(base)
                modules.add(f"{prefix}.{node.module}" if node.module else prefix)
            elif node.module:
                modules.add(node.module)
    return modules


def _module_uses(tree: ast.Module, name: str) -> List[int]:
    """Line numbers where ``name`` is used as a value (docstrings don't count)."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == name
    ]


def _is_module(module: str, root: str) -> bool:
    return module == root or module.startswith(root + ".")


def _is_stdlib(module: str) -> bool:
    return module.split(".")[0] in sys.stdlib_module_names


def _app_imports(modules: Iterable[str]) -> Set[str]:
    return {m for m in modules if m.split(".")[0] == "app"}


def _is_llm_adapter(rel: str) -> bool:
    """``tools/llm_<provider>.py``, but not the port or the fake."""
    return (
        rel.startswith("tools/llm_")
        and rel.endswith(".py")
        and rel not in LLM_NON_ADAPTER_FILES
    )


def _may_import_httpx(rel: str) -> bool:
    return rel == "tools/judge0.py" or _is_llm_adapter(rel)


def _is_llm_sdk(module: str) -> bool:
    return any(_is_module(module, root) for root in LLM_SDK_ROOTS)


# ── 1. code_sandbox.py is provider- and graph-agnostic ───────────────

def test_code_sandbox_imports_only_stdlib_schemas_and_core() -> None:
    path = APP_DIR / "tools" / "code_sandbox.py"
    modules = _imported_modules(path, _parse(path))

    for module in modules:
        if _is_stdlib(module):
            continue
        if _is_module(module, "app.schemas") or _is_module(module, "app.core"):
            continue
        raise AssertionError(
            f"code_sandbox.py imports '{module}': only stdlib, app.schemas "
            f"and app.core are allowed"
        )

    forbidden = [m for m in modules if _is_module(m, "httpx")
                 or _is_module(m, "app.tools.judge0") or _is_module(m, "app.graph")]
    assert forbidden == [], f"code_sandbox.py must not import {forbidden}"


# ── 2. fake_sandbox.py depends on code_sandbox only ──────────────────

def test_fake_sandbox_imports_only_stdlib_and_code_sandbox() -> None:
    path = APP_DIR / "tools" / "fake_sandbox.py"
    modules = _imported_modules(path, _parse(path))

    allowed_app = {"app.tools", "app.tools.code_sandbox"}
    offenders = sorted(_app_imports(modules) - allowed_app)
    assert offenders == [], (
        f"fake_sandbox.py imports {offenders}; only code_sandbox is allowed"
    )

    non_stdlib = [m for m in modules if not _is_stdlib(m) and m not in allowed_app]
    assert non_stdlib == [], f"fake_sandbox.py imports non-stdlib {non_stdlib}"
    assert not any("judge0" in m for m in modules)


# ── 3. httpx is confined to judge0.py ────────────────────────────────

def test_only_judge0_and_llm_adapters_import_httpx() -> None:
    offenders: List[str] = []
    for path in _app_files():
        modules = _imported_modules(path, _parse(path))
        if any(_is_module(m, "httpx") for m in modules):
            if not _may_import_httpx(_rel(path)):
                offenders.append(_rel(path))

    assert offenders == [], (
        f"httpx imported outside tools/judge0.py or an LLM adapter: {offenders}"
    )

    judge0 = APP_DIR / "tools" / "judge0.py"
    judge0_modules = _imported_modules(judge0, _parse(judge0))
    assert any(_is_module(m, "httpx") for m in judge0_modules), (
        "judge0.py no longer imports httpx — the boundary test would be vacuous"
    )


def test_only_judge0_and_llm_adapters_use_the_httpx_module_object() -> None:
    """Docstring/comment mentions are fine; real ``httpx.X`` usage is not."""
    offenders = {}
    for path in _app_files():
        lines = _module_uses(_parse(path), "httpx")
        if lines and not _may_import_httpx(_rel(path)):
            offenders[_rel(path)] = lines

    assert offenders == {}, (
        f"httpx is used as a module outside tools/judge0.py or an LLM adapter: "
        f"{offenders}"
    )
    assert _module_uses(_parse(APP_DIR / "tools" / "judge0.py"), "httpx")


def test_llm_port_and_fake_never_import_httpx() -> None:
    for name in (LLM_PORT_FILE, LLM_FAKE_FILE):
        path = APP_DIR / name
        tree = _parse(path)
        modules = _imported_modules(path, tree)

        assert not any(_is_module(m, "httpx") for m in modules), (
            f"{name} imports httpx; only adapters may"
        )
        assert _module_uses(tree, "httpx") == [], (
            f"{name} uses the httpx module; only adapters may"
        )


# ── 3b. LLM port / fake / adapter boundaries ─────────────────────────

def test_llm_client_imports_only_stdlib_core_and_pydantic() -> None:
    path = APP_DIR / "tools" / "llm_client.py"
    modules = _imported_modules(path, _parse(path))

    for module in modules:
        if _is_stdlib(module):
            continue
        if _is_module(module, "app.core") or _is_module(module, "pydantic"):
            continue
        raise AssertionError(
            f"llm_client.py imports '{module}': only stdlib, app.core "
            f"and pydantic are allowed"
        )

    forbidden = [
        m for m in modules
        if _is_module(m, "app.tools")
        or _is_module(m, "app.graph")
        or _is_module(m, "httpx")
        or _is_llm_sdk(m)
    ]
    assert forbidden == [], f"llm_client.py must not import {forbidden}"


def test_fake_llm_imports_only_stdlib_and_llm_client() -> None:
    path = APP_DIR / "tools" / "fake_llm.py"
    modules = _imported_modules(path, _parse(path))

    allowed_app = {"app.tools", "app.tools.llm_client"}
    offenders = sorted(_app_imports(modules) - allowed_app)
    assert offenders == [], (
        f"fake_llm.py imports {offenders}; only llm_client is allowed"
    )

    non_stdlib = [m for m in modules if not _is_stdlib(m) and m not in allowed_app]
    assert non_stdlib == [], f"fake_llm.py imports non-stdlib {non_stdlib}"
    assert not any(_is_llm_sdk(m) for m in modules)


def test_only_llm_adapters_import_an_llm_sdk() -> None:
    offenders: dict[str, List[str]] = {}
    for path in _app_files():
        rel = _rel(path)
        modules = _imported_modules(path, _parse(path))
        sdks = sorted(m for m in modules if _is_llm_sdk(m))
        if sdks and not _is_llm_adapter(rel):
            offenders[rel] = sdks

    assert offenders == {}, (
        f"LLM SDKs imported outside tools/llm_<provider>.py: {offenders}"
    )


def test_llm_sdk_classifier_recognises_known_and_rejects_app_modules() -> None:
    assert _is_llm_sdk("openai")
    assert _is_llm_sdk("openai.resources.chat")
    assert _is_llm_sdk("anthropic")
    assert _is_llm_sdk("google.generativeai")
    assert _is_llm_sdk("langchain_openai")
    assert not _is_llm_sdk("httpx")
    assert not _is_llm_sdk("app.tools.llm_client")
    assert not _is_llm_sdk("openai_helpers")


def test_llm_adapter_path_classifier() -> None:
    assert _is_llm_adapter("tools/llm_openai.py")
    assert not _is_llm_adapter("tools/llm_client.py")
    assert not _is_llm_adapter("tools/fake_llm.py")
    assert not _is_llm_adapter("tools/judge0.py")


# ── 4. RapidAPI protocol confined to judge0.py ───────────────────────

def test_only_judge0_builds_rapidapi_headers() -> None:
    offenders: List[str] = []
    for path in _app_files():
        text = path.read_text(encoding="utf-8").lower()
        if any(header in text for header in RAPIDAPI_HEADERS):
            offenders.append(_rel(path))

    assert offenders == ["tools/judge0.py"], (
        f"RapidAPI headers referenced outside judge0.py: {offenders}"
    )


def test_rapidapi_is_only_named_in_judge0_and_config() -> None:
    """``core/config.py`` owns the credential fields and help URL by design."""
    mentions: Set[str] = set()
    for path in _app_files():
        if "rapidapi" in path.read_text(encoding="utf-8").lower():
            mentions.add(_rel(path))

    assert mentions == RAPIDAPI_FILES, (
        f"unexpected rapidapi references: {sorted(mentions - RAPIDAPI_FILES)}; "
        f"missing: {sorted(RAPIDAPI_FILES - mentions)}"
    )


# ── 5. app/graph never reaches into app/tools or httpx ───────────────

def test_graph_does_not_import_tools_or_httpx() -> None:
    graph_files = sorted((APP_DIR / "graph").rglob("*.py"))
    assert graph_files, "no files under app/graph"

    for path in graph_files:
        modules = _imported_modules(path, _parse(path))
        bad = [
            m for m in modules
            if _is_module(m, "app.tools") or _is_module(m, "httpx")
        ]
        assert bad == [], f"{_rel(path)} imports {bad} (layering violation)"

        uses = _module_uses(_parse(path), "httpx")
        assert uses == [], f"{_rel(path)} uses the httpx module at lines {uses}"


# ── 5b. app/services never reaches into app/graph ────────────────────

def test_services_do_not_import_graph() -> None:
    service_files = sorted((APP_DIR / "services").rglob("*.py"))
    assert service_files, "no files under app/services"

    for path in service_files:
        modules = _imported_modules(path, _parse(path))
        bad = [m for m in modules if _is_module(m, "app.graph")]
        assert bad == [], f"{_rel(path)} imports {bad} (layering violation)"


# ── 6. no dynamic code execution anywhere under app/ ────────────────

def _banned_calls(tree: ast.Module) -> List[str]:
    hits: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func

        if isinstance(func, ast.Name) and func.id in BANNED_BUILTINS:
            hits.append(f"line {node.lineno}: {func.id}()")
        elif isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name):
                dotted = f"{func.value.id}.{func.attr}"
                if dotted in BANNED_DOTTED or dotted.startswith("subprocess."):
                    hits.append(f"line {node.lineno}: {dotted}()")
            elif (
                isinstance(func.value, ast.Attribute)
                and isinstance(func.value.value, ast.Name)
            ):
                dotted = f"{func.value.value.id}.{func.value.attr}.{func.attr}"
                if dotted.startswith("subprocess."):
                    hits.append(f"line {node.lineno}: {dotted}()")
    return hits


def test_no_exec_eval_os_system_or_subprocess_calls() -> None:
    offenders: dict[str, List[str]] = {}
    for path in _app_files():
        hits = _banned_calls(_parse(path))
        if hits:
            offenders[_rel(path)] = hits

    assert offenders == {}, f"dynamic execution under app/: {offenders}"


def test_no_module_imports_subprocess() -> None:
    """Call-site scanning alone could miss an aliased ``subprocess`` import."""
    offenders = sorted(
        _rel(path)
        for path in _app_files()
        if any(
            _is_module(m, "subprocess")
            for m in _imported_modules(path, _parse(path))
        )
    )
    assert offenders == [], f"subprocess imported under app/: {offenders}"
