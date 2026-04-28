#!/usr/bin/env python3
"""
repair.py — generate controller -> run pytest -> build report -> inject AUTO_REPAIR -> retry

ENHANCED VERSION with:
- Improved AUTO_REPAIR feedback (categorized failures, actionable guidance)
- Better prompt injection
- CSV metrics logging for plot_metrics.py visualization
- Support for multiple runs with run_id for fair comparisons

Usage (from repo root):
  # Single run with automatic run_id
  python tools/repair.py \
    --code-gen tools/code_gen.py \
    --prompt tools/prompt.py \
    --world worlds/empty.wbt \
    --out controllers/obs_avoidance/obs_avoidance.py \
    --tests tests \
    --model gpt-4o \
    --max-iters 10

  # Multiple runs with explicit run IDs for comparison
  python tools/repair.py --model gpt-4o --run-id codegen_4o_1 --csv-log metrics_codegen.csv ...
  
  # Then visualize all runs together
  python plot_metrics.py --csv metrics_codegen.csv --save-dir plots
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -------------------------
# CSV Metrics Logging
# -------------------------

@dataclass
class IterationMetrics:
    """Metrics for a single repair iteration"""
    run_id: str
    iter: int
    passed: int
    failed: int
    errors: int
    bad: int
    best_bad: int
    t_gen_s: float
    t_pytest_s: float
    t_parse_s: float
    t_prompt_s: float
    t_llm_s: float
    t_total_s: float
    merged_rules_chars: int
    rules_chars: int
    failed_tests: str


class MetricsLogger:
    """Handles CSV logging of iteration metrics"""
    
    def __init__(self, csv_path: Path, run_id: str):
        self.csv_path = csv_path
        self.run_id = run_id
        self.metrics_history: List[IterationMetrics] = []
        self.best_bad_so_far = float('inf')
        
        # Create CSV with headers if it doesn't exist
        self.csv_existed = csv_path.exists()
        if not self.csv_existed:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'run_id', 'iter', 'passed', 'failed', 'errors', 'bad', 'best_bad',
                    't_gen_s', 't_pytest_s', 't_parse_s', 't_prompt_s', 't_llm_s', 't_total_s',
                    'merged_rules_chars', 'rules_chars', 'failed_tests'
                ])
    
    def log_iteration(self, metrics: IterationMetrics) -> None:
        """Append iteration metrics to CSV"""
        # Update best_bad
        self.best_bad_so_far = min(self.best_bad_so_far, metrics.bad)
        metrics.best_bad = int(self.best_bad_so_far)
        metrics.run_id = self.run_id
        
        self.metrics_history.append(metrics)
        
        # Append to CSV
        with self.csv_path.open('a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.run_id,
                metrics.iter,
                metrics.passed,
                metrics.failed,
                metrics.errors,
                metrics.bad,
                metrics.best_bad,
                f"{metrics.t_gen_s:.4f}",
                f"{metrics.t_pytest_s:.4f}",
                f"{metrics.t_parse_s:.4f}",
                f"{metrics.t_prompt_s:.4f}",
                f"{metrics.t_llm_s:.4f}",
                f"{metrics.t_total_s:.4f}",
                metrics.merged_rules_chars,
                metrics.rules_chars,
                metrics.failed_tests,
            ])
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for this run"""
        if not self.metrics_history:
            return {}
        
        total_time = sum(m.t_total_s for m in self.metrics_history)
        final_metrics = self.metrics_history[-1]
        
        return {
            "run_id": self.run_id,
            "iterations": len(self.metrics_history),
            "final_passed": final_metrics.passed,
            "final_failed": final_metrics.failed,
            "final_errors": final_metrics.errors,
            "final_bad": final_metrics.bad,
            "best_bad": int(self.best_bad_so_far),
            "total_time_s": total_time,
            "avg_iter_time_s": total_time / len(self.metrics_history),
            "success": final_metrics.bad == 0,
        }


# -------------------------
# Small utilities
# -------------------------

def ts() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def clamp(s: str, n: int) -> str:
    """Truncate string to n chars, keeping start and end."""
    s = s or ""
    if len(s) <= n:
        return s
    half = max(1, n // 2)
    return s[:half] + "\n\n...[TRUNCATED]...\n\n" + s[-half:]


def run(cmd: List[str], cwd: Path, env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    """Run a command and return (returncode, stdout+stderr)."""
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out, _ = p.communicate()
    return int(p.returncode or 0), (out or "")


def generate_run_id(model: str, csv_path: Optional[Path]) -> str:
    """Generate a unique run_id based on model and existing runs."""
    if model.startswith("claude"):
        # e.g. claude-opus-4-5-20250514 -> claude_opus_4p5
        model_short = model.split("-20")[0]  # strip date suffix
        model_short = model_short.replace("-", "_").replace(".", "p")
    else:
        model_short = model.replace("gpt-", "").replace(".", "p")
    
    run_num = 1
    if csv_path and csv_path.exists():
        try:
            with csv_path.open('r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                run_ids = {row.get('run_id', '') for row in reader}
                prefix = f"codegen_{model_short}_"
                existing = [rid for rid in run_ids if rid.startswith(prefix)]
                if existing:
                    nums = []
                    for rid in existing:
                        m = re.search(r'_(\d+)$', rid)
                        if m:
                            nums.append(int(m.group(1)))
                    if nums:
                        run_num = max(nums) + 1
        except Exception:
            pass
    
    return f"codegen_{model_short}_{run_num}"


# -------------------------
# Start/goal parsing + world patching
# -------------------------

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def extract_start_goal_xy(user_prompt: str) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Extract start and goal coordinates from user prompt."""
    up = (user_prompt or "").strip()
    if not up:
        return None, None

    start_xy = None
    goal_xy = None

    ms = re.search(rf"(?i)\bstart(?:ing)?\s*at\s*\(\s*({_FLOAT})\s*,\s*({_FLOAT})\s*\)", up)
    if ms:
        try:
            start_xy = (float(ms.group(1)), float(ms.group(2)))
        except Exception:
            start_xy = None

    mg = re.search(rf"(?i)\bgoal\s*at\s*\(\s*({_FLOAT})\s*,\s*({_FLOAT})\s*\)", up)
    if mg:
        try:
            goal_xy = (float(mg.group(1)), float(mg.group(2)))
        except Exception:
            goal_xy = None

    return start_xy, goal_xy


def _patch_translation_in_def_block(world_text: str, def_name: str, xy: Tuple[float, float]) -> str:
    """Patch translation in a DEF block."""
    x, y = xy
    pattern = re.compile(rf"(?s)(\bDEF\s+{re.escape(def_name)}\b.*?\btranslation\s+)({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})")
    m = pattern.search(world_text)
    if not m:
        return world_text
    z = m.group(4)
    repl = f"{m.group(1)}{x:.6g} {y:.6g} {z}"
    return world_text[:m.start()] + repl + world_text[m.end():]


def _patch_translation_in_named_node(world_text: str, name_value: str, xy: Tuple[float, float]) -> str:
    """Patch translation in a named node."""
    x, y = xy
    name_pat = re.compile(rf'(?s)\bname\s+"{re.escape(name_value)}"\b')
    nm = name_pat.search(world_text)
    if not nm:
        return world_text

    window_start = nm.start()
    window_end = min(len(world_text), nm.end() + 2000)
    window = world_text[window_start:window_end]

    trans_pat = re.compile(rf"(?s)(\btranslation\s+)({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})")
    tm = trans_pat.search(window)
    if not tm:
        return world_text

    z = tm.group(4)
    repl = f"{tm.group(1)}{x:.6g} {y:.6g} {z}"
    new_window = window[:tm.start()] + repl + window[tm.end():]
    return world_text[:window_start] + new_window + world_text[window_end:]


def patch_world_text(world_text: str, start_xy: Optional[Tuple[float, float]], goal_xy: Optional[Tuple[float, float]]) -> str:
    """Patch start/goal positions in world text."""
    txt = world_text or ""
    try:
        if goal_xy is not None:
            new_txt = _patch_translation_in_def_block(txt, "GOAL", goal_xy)
            if new_txt == txt:
                new_txt = _patch_translation_in_named_node(txt, "GOAL", goal_xy)
            txt = new_txt

        if start_xy is not None:
            new_txt = _patch_translation_in_def_block(txt, "START", start_xy)
            if new_txt == txt:
                new_txt = _patch_translation_in_def_block(txt, "ROBOT", start_xy)
            if new_txt == txt:
                new_txt = _patch_translation_in_named_node(txt, "START", start_xy)
            txt = new_txt
    except Exception:
        return world_text
    return txt


def make_patched_world_copy(src_world: Path, dst_world: Path, user_prompt: str) -> Tuple[Path, str]:
    """Create a patched copy of the world file."""
    txt = read_text(src_world)
    start_xy, goal_xy = extract_start_goal_xy(user_prompt)
    if start_xy is None and goal_xy is None:
        write_text(dst_world, txt)
        return dst_world, "No start/goal found in user prompt; copied world unchanged."

    patched = patch_world_text(txt, start_xy, goal_xy)
    write_text(dst_world, patched)

    note = "Patched world copy"
    note += f" start={start_xy}" if start_xy is not None else " start=(unchanged)"
    note += f" goal={goal_xy}" if goal_xy is not None else " goal=(unchanged)"
    return dst_world, note


# -------------------------
# JUnit parsing + reporting
# -------------------------

@dataclass
class JUnitFailure:
    suite: str
    test: str
    classname: str
    message: str
    text: str
    file: str = ""
    line: str = ""


@dataclass
class JUnitSummary:
    tests: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    time: float = 0.0


def _guess_file_line_from_text(txt: str) -> Tuple[str, str]:
    """Extract file:line from traceback text."""
    m = re.search(r"(^|\n)([^ \n\t:]+\.py):(\d+):", txt)
    if not m:
        return ("", "")
    return (m.group(2), m.group(3))


def parse_junit(junit_path: Path) -> Tuple[JUnitSummary, List[JUnitFailure]]:
    """Parse JUnit XML and extract summary + failures."""
    summary = JUnitSummary()
    failures: List[JUnitFailure] = []
    if not junit_path.exists():
        return summary, failures

    try:
        root = ET.fromstring(read_text(junit_path))
    except Exception:
        return summary, failures

    suites = []
    if root.tag == "testsuite":
        suites = [root]
    else:
        suites = list(root.findall(".//testsuite"))

    for s in suites:
        try:
            summary.tests += int(s.attrib.get("tests", "0") or "0")
            summary.failures += int(s.attrib.get("failures", "0") or "0")
            summary.errors += int(s.attrib.get("errors", "0") or "0")
            summary.skipped += int(s.attrib.get("skipped", "0") or "0")
            summary.time += float(s.attrib.get("time", "0") or "0")
        except Exception:
            pass

        suite_name = s.attrib.get("name", "")

        for tc in s.findall(".//testcase"):
            classname = tc.attrib.get("classname", "")
            testname = tc.attrib.get("name", "")
            for f in tc.findall("failure"):
                msg = f.attrib.get("message", "") or ""
                text = (f.text or "").strip()
                file_, line_ = _guess_file_line_from_text(text)
                failures.append(
                    JUnitFailure(
                        suite=suite_name,
                        test=testname,
                        classname=classname,
                        message=msg,
                        text=text,
                        file=file_,
                        line=line_,
                    )
                )
            for e in tc.findall("error"):
                msg = e.attrib.get("message", "") or ""
                text = (e.text or "").strip()
                file_, line_ = _guess_file_line_from_text(text)
                failures.append(
                    JUnitFailure(
                        suite=suite_name,
                        test=testname,
                        classname=classname,
                        message=msg,
                        text=text,
                        file=file_,
                        line=line_,
                    )
                )

    return summary, failures


def render_report_md(
    iter_k: int,
    gen_rc: int,
    py_rc: int,
    summary: JUnitSummary,
    fails: List[JUnitFailure],
    pytest_log: str,
    controller_path: Path,
    world_path_used: Path,
    world_note: str,
    user_prompt: str,
) -> str:
    """Render a markdown report for the iteration."""
    lines: List[str] = []
    lines.append(f"# Repair Report — iteration {iter_k}")
    lines.append("")
    lines.append(f"- code_gen exit: `{gen_rc}`")
    lines.append(f"- pytest exit: `{py_rc}`")
    lines.append("")
    lines.append("## User prompt")
    lines.append("")
    lines.append("```")
    lines.append((user_prompt or "").strip())
    lines.append("```")
    lines.append("")
    lines.append("## World used")
    lines.append("")
    lines.append(f"- `{world_path_used}`")
    lines.append(f"- note: {world_note}")
    lines.append("")
    lines.append("## JUnit summary")
    lines.append("")
    lines.append(f"- tests: {summary.tests}")
    lines.append(f"- failures: {summary.failures}")
    lines.append(f"- errors: {summary.errors}")
    lines.append(f"- skipped: {summary.skipped}")
    lines.append(f"- time: {summary.time:.3f}s")
    lines.append("")
    lines.append("## Failing tests")
    lines.append("")
    if not fails:
        lines.append("_No failing testcases parsed from junitxml (see raw pytest.log)._")
    else:
        for i, f in enumerate(fails, 1):
            loc = f"{f.file}:{f.line}" if (f.file and f.line) else ""
            lines.append(f"### {i}. {f.classname}::{f.test}")
            if loc:
                lines.append(f"- location: `{loc}`")
            if f.message:
                lines.append(f"- message: `{clamp(f.message, 300)}`")
            if f.text:
                lines.append("")
                lines.append("Traceback snippet:")
                lines.append("")
                snippet = clamp(f.text, 1500)
                lines.append("```")
                lines.append(snippet)
                lines.append("```")
            lines.append("")

    lines.append("## Pytest log (tail)")
    lines.append("")
    tail = "\n".join((pytest_log or "").splitlines()[-80:])
    lines.append("```")
    lines.append(tail)
    lines.append("```")
    lines.append("")
    lines.append("## Controller path")
    lines.append("")
    lines.append(f"- `{controller_path}`")
    return "\n".join(lines)


# -------------------------
# Prompt patching + wrapper
# -------------------------

# Improved AUTO_REPAIR section - clearer, more actionable
AUTO_REPAIR_SECTION = r"""
[REPAIR MODE - ITERATION FEEDBACK]
Your previous attempt failed tests. Study the failures below and fix them.

$AUTO_REPAIR

REPAIR CHECKLIST (verify each item):
[ ] All 6 functions exist at module level: parse_world, build_grid, astar, compute_wheel_speeds, run_episode, main
[ ] Webots imports (from controller import ...) are ONLY inside run_episode() and main()
[ ] No markdown fences (```) anywhere in output
[ ] parse_world returns dict with keys: plane, start, goal, obstacles, bounds
[ ] start and goal are dicts with "x" and "y" keys (float values)
[ ] obstacles is list of dicts, each with "x", "y", "sx", "sy" keys
[ ] bounds is [x_min, x_max, y_min, y_max] where x_min < x_max and y_min < y_max
[ ] build_grid returns 2D list of integers (0=free, 1=occupied)
[ ] astar returns list of (i, j) tuples forming path from start to goal
[ ] compute_wheel_speeds returns tuple of two floats (left_speed, right_speed)

Output ONLY the corrected Python file. No explanations, no markdown.
"""


def ensure_prompt_has_auto_repair_slot(prompt_path: Path) -> bool:
    """Ensure tools/prompt.py contains $AUTO_REPAIR placeholder."""
    txt = read_text(prompt_path)
    if "$AUTO_REPAIR" in txt or "${AUTO_REPAIR}" in txt:
        return False

    m = re.search(r"PROMPT\s*=\s*Template\(\s*r?([\"']{3})", txt)
    if not m:
        patched = txt + "\n\n# NOTE: repair.py could not auto-inject AUTO_REPAIR into PROMPT template.\n"
        patched += "# Please manually add $AUTO_REPAIR inside your PROMPT Template definition.\n"
        write_text(prompt_path, patched)
        print(f"[repair] ⚠ Could not auto-inject AUTO_REPAIR. Added note to {prompt_path}")
        return True

    quote = m.group(1)
    start = m.end()

    end_idx = txt.find(quote, start)
    if end_idx == -1:
        print(f"[repair] ⚠ Could not find closing quote for PROMPT template", file=sys.stderr)
        return False

    injected = txt[:end_idx] + AUTO_REPAIR_SECTION + "\n" + txt[end_idx:]
    write_text(prompt_path, injected)
    print(f"[repair] ✓ Injected AUTO_REPAIR slot into {prompt_path}")
    return True


def make_wrapper_prompt(wrapper_path: Path, base_prompt_path: Path, auto_repair_text: str) -> None:
    """Create a wrapper prompt module that injects AUTO_REPAIR."""
    ar = clamp(auto_repair_text, 14000).replace('"""', r'\"\"\"')
    code = f'''# AUTO-GENERATED by tools/repair.py (do not edit)
from __future__ import annotations

import importlib.util
from string import Template

BASE_PROMPT_PATH = {repr(str(base_prompt_path))}

def _load_base():
    spec = importlib.util.spec_from_file_location("base_prompt_mod", BASE_PROMPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base prompt: {{BASE_PROMPT_PATH}}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "PROMPT"):
        raise RuntimeError("Base prompt must define PROMPT")
    return mod.PROMPT

_BASE = _load_base()
_FILLED = _BASE.safe_substitute(AUTO_REPAIR="""
{ar}
""")
PROMPT = Template(_FILLED)
'''
    write_text(wrapper_path, code)


# -------------------------
# Controller lint for AUTO_REPAIR
# -------------------------

def controller_lint(controller_text: str) -> Dict[str, Any]:
    """Analyze controller for common issues."""
    req = ["parse_world", "build_grid", "astar", "run_episode", "main", "compute_wheel_speeds"]
    missing = []
    for name in req:
        if re.search(rf"^\s*def\s+{re.escape(name)}\s*\(", controller_text, flags=re.MULTILINE) is None:
            missing.append(name)

    top_level_webots_import = False
    first_def = re.search(r"^\s*(def|class)\s+", controller_text, flags=re.MULTILINE)
    cutoff = first_def.start() if first_def else len(controller_text)
    header = controller_text[:cutoff]
    if re.search(r"^\s*from\s+controller\s+import\s+", header, flags=re.MULTILINE):
        top_level_webots_import = True
    if re.search(r"^\s*import\s+controller\b", header, flags=re.MULTILINE):
        top_level_webots_import = True

    has_markdown_fence = "```" in controller_text
    
    # Check for syntax errors
    syntax_error = None
    try:
        ast.parse(controller_text)
    except SyntaxError as e:
        syntax_error = f"Line {e.lineno}: {e.msg}"

    return {
        "missing_functions": missing,
        "top_level_webots_import": top_level_webots_import,
        "has_markdown_fence": has_markdown_fence,
        "syntax_error": syntax_error,
    }


def categorize_failure(f: JUnitFailure) -> str:
    """Categorize a test failure for better feedback."""
    test_name = f.test.lower()
    classname = f.classname.lower()
    
    if 'static' in classname or 'static' in test_name:
        return 'static'
    if 'parse_world' in test_name or 'parse_world' in classname:
        if 'obstacle' in test_name:
            return 'parse_obstacles'
        if 'start' in test_name or 'goal' in test_name:
            return 'parse_positions'
        if 'bounds' in test_name:
            return 'parse_bounds'
        return 'parse_contract'
    if 'grid' in test_name or 'build_grid' in classname:
        return 'grid'
    if 'astar' in test_name or 'path' in test_name:
        return 'astar'
    if 'wheel' in test_name or 'speed' in test_name or 'motion' in test_name:
        return 'wheel_speeds'
    return 'other'


def get_actionable_hint(f: JUnitFailure, category: str) -> str:
    """Get actionable hint for a failure."""
    test_name = f.test.lower()
    
    hints = {
        'static': {
            'markdown': "Remove all ``` from output - output ONLY Python code",
            'import': "Move 'from controller import ...' INSIDE run_episode() and main()",
            'syntax': "Fix Python syntax error",
            'function': "Add missing function at module level",
            'default': "Check static requirements: valid Python, no markdown, proper imports",
        },
        'parse_positions': {
            'start_position': "Robot start: find E-puck node, extract translation X Y Z, use x=X, y=Y",
            'goal_position': "Goal: find DEF GOAL or name \"GOAL\", extract translation X Y Z",
            'default': "Check coordinate extraction from world file",
        },
        'parse_obstacles': {
            'count': "Must find at least 5 obstacles: parse both WoodenBox AND Rock nodes",
            'fields': "Each obstacle needs: x, y (position), sx, sy (size) - all floats",
            'wooden_box': "WoodenBox: translation gives x,y; size field gives sx,sy",
            'default': "Parse WoodenBox (use size field) and Rock (use scale field) nodes",
        },
        'parse_bounds': {
            'default': "bounds = [x_min, x_max, y_min, y_max] must contain start and goal",
        },
        'parse_contract': {
            'default': "parse_world must return dict with: plane, start, goal, obstacles, bounds",
        },
        'grid': {
            'dimension': "grid must be 2D list with width/height matching bounds/resolution",
            'obstacle': "Mark cells overlapping obstacles as non-zero",
            'default': "build_grid returns 2D list, obstacles marked as 1, free as 0",
        },
        'astar': {
            'start': "Path must start at start_cell",
            'end': "Path must end at goal_cell",
            'connected': "Each step must be adjacent (8-connectivity)",
            'obstacle': "Path must not go through obstacle cells",
            'default': "A* must return valid path from start to goal avoiding obstacles",
        },
        'wheel_speeds': {
            'forward': "Waypoint ahead: both wheels positive",
            'left': "Waypoint to left: right wheel > left wheel",
            'right': "Waypoint to right: left wheel > right wheel",
            'blocked': "Front blocked (high prox): backup (both negative) or turn (opposite signs)",
            'default': "Return (left_speed, right_speed) based on heading error and proximity",
        },
        'other': {
            'default': "Check test requirements",
        },
    }
    
    cat_hints = hints.get(category, hints['other'])
    
    # Try to find specific hint
    for key, hint in cat_hints.items():
        if key != 'default' and key in test_name:
            return hint
    
    return cat_hints.get('default', "Check implementation")


def build_auto_repair_text(
    iter_k: int,
    summary: JUnitSummary,
    fails: List[JUnitFailure],
    pytest_log: str,
    controller_text: str,
) -> str:
    """
    Build actionable AUTO_REPAIR feedback for the LLM.
    
    IMPROVED: Categorizes failures and provides specific hints.
    """
    lint = controller_lint(controller_text)
    
    sections = []
    sections.append(f"═══ ITERATION {iter_k} FAILED ═══")
    sections.append(f"Tests: {summary.tests} total | {summary.failures} failed | {summary.errors} errors")
    sections.append("")
    
    # Critical lint issues first (most likely to cause all tests to fail)
    critical_issues = []
    
    if lint['syntax_error']:
        critical_issues.append(f"🔴 SYNTAX ERROR: {lint['syntax_error']}")
    
    if lint['has_markdown_fence']:
        critical_issues.append("🔴 MARKDOWN FENCES DETECTED: Remove all ``` - output ONLY Python code")
    
    if lint['top_level_webots_import']:
        critical_issues.append("🔴 TOP-LEVEL WEBOTS IMPORT: Move 'from controller import' INSIDE run_episode() and main()")
    
    if lint['missing_functions']:
        critical_issues.append(f"🔴 MISSING FUNCTIONS: {', '.join(lint['missing_functions'])}")
        critical_issues.append("   These must be defined at module level with exact names")
    
    if critical_issues:
        sections.append("CRITICAL ISSUES (fix these first):")
        sections.extend(critical_issues)
        sections.append("")
    
    # Categorize failures
    categorized: Dict[str, List[Tuple[JUnitFailure, str]]] = {}
    for f in fails:
        cat = categorize_failure(f)
        hint = get_actionable_hint(f, cat)
        if cat not in categorized:
            categorized[cat] = []
        categorized[cat].append((f, hint))
    
    # Priority order for categories
    priority = ['static', 'parse_contract', 'parse_positions', 'parse_obstacles', 
                'parse_bounds', 'grid', 'astar', 'wheel_speeds', 'other']
    
    for cat in priority:
        if cat not in categorized:
            continue
        
        cat_fails = categorized[cat]
        cat_name = {
            'static': 'STATIC ANALYSIS',
            'parse_contract': 'PARSE_WORLD CONTRACT',
            'parse_positions': 'PARSE_WORLD POSITIONS',
            'parse_obstacles': 'PARSE_WORLD OBSTACLES',
            'parse_bounds': 'PARSE_WORLD BOUNDS',
            'grid': 'BUILD_GRID',
            'astar': 'ASTAR',
            'wheel_speeds': 'COMPUTE_WHEEL_SPEEDS',
            'other': 'OTHER',
        }.get(cat, cat.upper())
        
        sections.append(f"▸ {cat_name} FAILURES ({len(cat_fails)}):")
        
        # Show up to 3 failures per category with hints
        for f, hint in cat_fails[:3]:
            sections.append(f"  • {f.test}")
            sections.append(f"    → {hint}")
            
            # Include assertion details if available
            if f.message and 'assert' in f.message.lower():
                # Extract the assertion for context
                msg_short = clamp(f.message, 100)
                sections.append(f"    Assertion: {msg_short}")
        
        if len(cat_fails) > 3:
            sections.append(f"  ... and {len(cat_fails) - 3} more {cat_name.lower()} failures")
        
        sections.append("")
    
    # Add specific expected values for position tests (hardcoded from test file)
    if 'parse_positions' in categorized:
        sections.append("EXPECTED VALUES (from test file):")
        sections.append("  • Robot start: x ≈ -0.423352, y ≈ 0.688226 (E-puck translation)")
        sections.append("  • Goal: x ≈ 0.54125, y ≈ -0.0579839 (DEF GOAL translation)")
        sections.append("")
    
    if 'parse_obstacles' in categorized:
        sections.append("EXPECTED OBSTACLES:")
        sections.append("  • At least 5 obstacles (3 WoodenBox + 2 Rock)")
        sections.append("  • WoodenBox at (0.194803, 0.0218687) with size (0.2, 0.2)")
        sections.append("  • Each obstacle: {x, y, sx, sy} all positive floats")
        sections.append("")
    
    # Truncated pytest output for additional context
    pytest_tail = "\n".join((pytest_log or "").splitlines()[-30:])
    sections.append("PYTEST OUTPUT (last 30 lines):")
    sections.append("```")
    sections.append(pytest_tail)
    sections.append("```")
    
    # Controller snippet (first part showing structure)
    if controller_text:
        # Show imports and function definitions
        lines = controller_text.splitlines()
        important_lines = []
        for i, line in enumerate(lines[:100]):
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ', 'def ', 'class ', 'if __name__')):
                important_lines.append(f"{i+1:3d}: {line}")
        
        if important_lines:
            sections.append("")
            sections.append("YOUR CODE STRUCTURE (imports + function defs):")
            sections.append("\n".join(important_lines[:30]))
    
    return "\n".join(sections)


# -------------------------
# Main loop
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Iterative code generation with pytest feedback")
    ap.add_argument("--code-gen", required=True, help="Path to code_gen.py")
    ap.add_argument("--prompt", required=True, help="Path to prompt.py (defines PROMPT=Template(...))")
    ap.add_argument("--world", required=True, help="Path to Webots world file (.wbt)")
    ap.add_argument("--out", required=True, help="Output path for generated controller")
    ap.add_argument("--tests", default="tests", help="Path to tests directory")

    ap.add_argument("--model", default="gpt-4o", help="OpenAI model name")
    ap.add_argument("--max-output-tokens", type=int, default=9000, help="Max tokens for LLM output")
    ap.add_argument("--max-iters", type=int, default=10, help="Maximum repair iterations")
    ap.add_argument("--python-exe", default=sys.executable, help="Python executable")

    ap.add_argument("--logs-dir", default="agent_logs", help="Directory for logs")
    ap.add_argument("--project-root", default="", help="Project root (defaults to parent of tools/)")
    ap.add_argument("--user-prompt", default="", help="User prompt with start/goal coordinates")
    
    # CSV logging arguments
    ap.add_argument("--csv-log", default="metrics.csv", help="CSV file for metrics")
    ap.add_argument("--run-id", default="", help="Run ID for CSV logging")
    
    ap.add_argument("--debug", action="store_true", help="Enable debug output")
    args = ap.parse_args()

    # Resolve paths
    tools_dir = Path(__file__).resolve().parent
    repo_root = Path(args.project_root).resolve() if args.project_root else tools_dir.parent

    code_gen = (repo_root / args.code_gen).resolve() if not Path(args.code_gen).is_absolute() else Path(args.code_gen).resolve()
    base_prompt = (repo_root / args.prompt).resolve() if not Path(args.prompt).is_absolute() else Path(args.prompt).resolve()
    world_src = (repo_root / args.world).resolve() if not Path(args.world).is_absolute() else Path(args.world).resolve()
    out = (repo_root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    tests_dir = (repo_root / args.tests).resolve() if not Path(args.tests).is_absolute() else Path(args.tests).resolve()
    logs_dir = (repo_root / args.logs_dir).resolve() if not Path(args.logs_dir).is_absolute() else Path(args.logs_dir).resolve()
    
    # Verify required files exist
    for p in [code_gen, base_prompt, world_src, tests_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    # Setup CSV logging
    csv_path = (repo_root / args.csv_log).resolve() if not Path(args.csv_log).is_absolute() else Path(args.csv_log).resolve()
    
    # Generate or use provided run_id
    run_id = args.run_id if args.run_id else generate_run_id(args.model, csv_path)
    
    metrics_logger = MetricsLogger(csv_path, run_id)
    
    print(f"[repair] CSV logging: {csv_path}")
    print(f"[repair] Run ID: {run_id}")
    print(f"[repair] Model: {args.model}")

    # Create run-specific logs directory
    logs_dir = logs_dir / run_id
    logs_dir.mkdir(parents=True, exist_ok=True)

    # One-time backup of prompt
    backup = base_prompt.with_suffix(base_prompt.suffix + ".base")
    if not backup.exists():
        write_text(backup, read_text(base_prompt))
        print(f"[repair] ✓ Created backup: {backup}")

    # Ensure $AUTO_REPAIR exists in prompt template
    try:
        changed = ensure_prompt_has_auto_repair_slot(base_prompt)
        if args.debug and changed:
            print(f"[repair] Modified prompt file to include AUTO_REPAIR slot", file=sys.stderr)
    except Exception as e:
        print(f"[repair] ⚠ WARNING: could not patch prompt: {e}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()

    auto_repair = ""

    if args.debug:
        print(f"[repair] repo_root={repo_root}", file=sys.stderr)
        print(f"[repair] code_gen={code_gen}", file=sys.stderr)
        print(f"[repair] base_prompt={base_prompt}", file=sys.stderr)
        print(f"[repair] world_src={world_src}", file=sys.stderr)
        print(f"[repair] out={out}", file=sys.stderr)
        print(f"[repair] tests={tests_dir}", file=sys.stderr)
        print(f"[repair] logs={logs_dir}", file=sys.stderr)
        if (args.user_prompt or "").strip():
            print(f"[repair] user_prompt={args.user_prompt.strip()}", file=sys.stderr)

    for k in range(1, args.max_iters + 1):
        print(f"\n{'='*60}")
        print(f"[repair] Iteration {k}/{args.max_iters} (run_id={run_id})")
        print(f"{'='*60}")

        iter_start_time = time.time()

        iter_dir = logs_dir / f"iter_{k:02d}_{ts()}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # Timing markers
        t_start_parse = time.time()

        # 0) Patch world copy for this iteration
        world_used = iter_dir / "world_patched.wbt"
        world_used, world_note = make_patched_world_copy(world_src, world_used, args.user_prompt)
        write_text(iter_dir / "world_patch_note.txt", world_note)

        t_parse_s = time.time() - t_start_parse
        t_start_prompt = time.time()

        # 1) Create wrapper prompt that injects AUTO_REPAIR
        wrapper_prompt = iter_dir / "prompt_wrapper.py"
        make_wrapper_prompt(wrapper_prompt, base_prompt, auto_repair)

        if args.debug:
            print(f"[repair] Created wrapper prompt: {wrapper_prompt}", file=sys.stderr)

        t_prompt_s = time.time() - t_start_prompt
        t_start_gen = time.time()

        # 2) Generate controller
        gen_cmd = [
            args.python_exe, str(code_gen),
            "--template", str(wrapper_prompt),
            "--world", str(world_used),
            "--out", str(out),
            "--model", args.model,
            "--max-output-tokens", str(args.max_output_tokens),
        ]
        if (args.user_prompt or "").strip():
            gen_cmd += ["--user-prompt", args.user_prompt]
        if args.debug:
            gen_cmd.append("--debug")

        print(f"[repair] Running code_gen...")
        gen_rc, gen_log = run(gen_cmd, cwd=repo_root)
        write_text(iter_dir / "code_gen.log", gen_log)
        write_text(iter_dir / "code_gen_rc.txt", str(gen_rc))

        t_gen_s = time.time() - t_start_gen

        # Extract LLM time from code_gen log if available
        t_llm_s = 0.0
        llm_match = re.search(r"LLM call took:\s*([\d.]+)\s*s", gen_log)
        if llm_match:
            try:
                t_llm_s = float(llm_match.group(1))
            except ValueError:
                pass

        if gen_rc != 0 or not out.exists():
            print(f"[repair] ✗ Code generation failed (rc={gen_rc})")
            
            # Log failure metrics
            metrics = IterationMetrics(
                run_id=run_id,
                iter=k,
                passed=0,
                failed=0,
                errors=1,
                bad=999,  # Large penalty for generation failure
                best_bad=0,
                t_gen_s=t_gen_s,
                t_pytest_s=0.0,
                t_parse_s=t_parse_s,
                t_prompt_s=t_prompt_s,
                t_llm_s=t_llm_s,
                t_total_s=time.time() - iter_start_time,
                merged_rules_chars=len(auto_repair),
                rules_chars=len(auto_repair),
                failed_tests="GENERATOR_FAILED",
            )
            metrics_logger.log_iteration(metrics)
            
            # Build informative error feedback
            auto_repair = f"""═══ ITERATION {k}: CODE GENERATION FAILED ═══

The code generator could not produce a valid controller file.

Exit code: {gen_rc}

Generator output:
{clamp(gen_log, 8000)}

COMMON CAUSES:
1. Output contained markdown fences (```) - remove ALL markdown
2. Output was not valid Python syntax
3. Missing required functions at module level

REQUIREMENTS:
- Output ONLY Python code, no markdown, no explanations
- Define these 6 functions at module level:
  • parse_world(world_text, params_text) -> dict
  • build_grid(bounds, obstacles, resolution, inflation) -> 2D list
  • astar(grid, start_cell, goal_cell) -> list
  • compute_wheel_speeds(pose_xy, yaw, waypoint_xy, prox) -> tuple
  • run_episode(...) -> dict
  • main() -> None
- Import Webots ONLY inside run_episode() and main()
- Include: if __name__ == "__main__": main()

Try again with a complete, valid Python controller.
"""
            write_text(iter_dir / "auto_repair.txt", auto_repair)
            continue

        controller_text = read_text(out)
        write_text(iter_dir / "controller_generated.py", controller_text)
        print(f"[repair] ✓ Controller generated: {out}")

        t_start_pytest = time.time()

        # 3) Run pytest with junit report
        env = os.environ.copy()
        env["CONTROLLER_PATH"] = str(out)
        env["WORLD_PATH"] = str(world_used)

        junit = iter_dir / "pytest_junit.xml"
        py_cmd = [
            args.python_exe, "-m", "pytest", "-q", str(tests_dir),
            "--junitxml", str(junit),
        ]

        print(f"[repair] Running pytest...")
        py_rc, py_log = run(py_cmd, cwd=repo_root, env=env)
        write_text(iter_dir / "pytest.log", py_log)
        write_text(iter_dir / "pytest_rc.txt", str(py_rc))

        t_pytest_s = time.time() - t_start_pytest

        # Keep "latest" pointers
        write_text(logs_dir / "latest_pytest.log", py_log)
        write_text(logs_dir / "latest_pytest_rc.txt", str(py_rc))

        # 3b) Parse junit + generate report
        junit_summary, junit_failures = parse_junit(junit)
        
        # Calculate metrics
        passed = junit_summary.tests - junit_summary.failures - junit_summary.errors
        failed = junit_summary.failures
        errors = junit_summary.errors
        bad = failed + errors
        
        # Format failed tests for CSV
        failed_tests_list = []
        for f in junit_failures:
            test_id = f"{f.classname}::{f.test}" if f.classname else f.test
            if test_id:
                failed_tests_list.append(test_id)
        failed_tests_str = ";".join(failed_tests_list) if failed_tests_list else ""
        
        t_total_s = time.time() - iter_start_time
        
        # Log iteration metrics
        metrics = IterationMetrics(
            run_id=run_id,
            iter=k,
            passed=passed,
            failed=failed,
            errors=errors,
            bad=bad,
            best_bad=0,
            t_gen_s=t_gen_s,
            t_pytest_s=t_pytest_s,
            t_parse_s=t_parse_s,
            t_prompt_s=t_prompt_s,
            t_llm_s=t_llm_s,
            t_total_s=t_total_s,
            merged_rules_chars=len(auto_repair),
            rules_chars=len(auto_repair),
            failed_tests=failed_tests_str,
        )
        metrics_logger.log_iteration(metrics)
        
        report_md = render_report_md(
            k, gen_rc, py_rc,
            junit_summary, junit_failures,
            py_log, out,
            world_used, world_note,
            args.user_prompt,
        )
        write_text(iter_dir / "report.md", report_md)
        write_json(
            iter_dir / "report.json",
            {
                "iteration": k,
                "run_id": run_id,
                "code_gen_rc": gen_rc,
                "pytest_rc": py_rc,
                "junit_summary": asdict(junit_summary),
                "failures": [asdict(f) for f in junit_failures],
                "world_used": str(world_used),
                "world_note": world_note,
                "user_prompt": (args.user_prompt or "").strip(),
                "metrics": {
                    "passed": passed,
                    "failed": failed,
                    "errors": errors,
                    "bad": bad,
                    "t_gen_s": t_gen_s,
                    "t_pytest_s": t_pytest_s,
                    "t_parse_s": t_parse_s,
                    "t_prompt_s": t_prompt_s,
                    "t_llm_s": t_llm_s,
                    "t_total_s": t_total_s,
                },
            },
        )
        write_text(logs_dir / "latest_report.md", report_md)

        if py_rc == 0:
            print(f"\n{'='*60}")
            print(f"[repair] 🎉 SUCCESS at iteration {k}")
            print(f"{'='*60}")
            print(f"[repair] Controller: {out}")
            print(f"[repair] Logs: {iter_dir}")
            print(f"[repair] Metrics CSV: {csv_path}")
            
            # Print summary
            summary = metrics_logger.get_summary()
            print(f"\n[repair] Run Summary:")
            print(f"[repair]   Run ID: {summary['run_id']}")
            print(f"[repair]   Iterations: {summary['iterations']}")
            print(f"[repair]   Final: passed={summary['final_passed']} failed={summary['final_failed']} errors={summary['final_errors']}")
            print(f"[repair]   Best bad count: {summary['best_bad']}")
            print(f"[repair]   Total time: {summary['total_time_s']:.1f}s")
            print(f"[repair]   Avg iteration time: {summary['avg_iter_time_s']:.1f}s")
            
            return

        print(f"[repair] ✗ Pytest failed ({junit_summary.failures} failures, {junit_summary.errors} errors)")
        print(f"[repair] Timing: gen={t_gen_s:.1f}s pytest={t_pytest_s:.1f}s total={t_total_s:.1f}s")
        print(f"[repair] Bad count: {bad} (best so far: {metrics.best_bad})")

        # 4) Update AUTO_REPAIR for next iteration
        auto_repair = build_auto_repair_text(k, junit_summary, junit_failures, py_log, controller_text)
        write_text(iter_dir / "auto_repair.txt", auto_repair)

    print(f"\n{'='*60}")
    print(f"[repair] ✗ FAILED after {args.max_iters} iterations.")
    print(f"{'='*60}")
    print(f"[repair] Last controller: {out}")
    print(f"[repair] See logs in: {logs_dir}")
    print(f"[repair] Metrics CSV: {csv_path}")
    
    # Print final summary
    summary = metrics_logger.get_summary()
    print(f"\n[repair] Run Summary:")
    print(f"[repair]   Run ID: {summary['run_id']}")
    print(f"[repair]   Iterations: {summary['iterations']}")
    print(f"[repair]   Final: passed={summary['final_passed']} failed={summary['final_failed']} errors={summary['final_errors']}")
    print(f"[repair]   Best bad count: {summary['best_bad']}")
    print(f"[repair]   Total time: {summary['total_time_s']:.1f}s")
    print(f"[repair]   Avg iteration time: {summary['avg_iter_time_s']:.1f}s")
    
    raise SystemExit(2)


if __name__ == "__main__":
    main()