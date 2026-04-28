#!/usr/bin/env python3
"""
code_gen.py - Generate a Webots controller using LLM

Key improvements:
- Better markdown fence stripping
- Syntax error detection with clear feedback
- Saves raw output for debugging
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import re
import sys
import time
from pathlib import Path
from string import Template
from typing import Optional, Tuple

from openai import OpenAI

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


# -------------------------
# I/O helpers
# -------------------------

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences - handles multiple formats."""
    t = (text or "").strip()
    
    # Pattern 1: ```python ... ``` or ``` ... ```
    if t.startswith("```"):
        lines = t.splitlines()
        # Remove first line (```python or ```)
        lines = lines[1:]
        # Remove trailing ``` lines
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    
    # Pattern 2: Code block in middle of text - extract largest block
    if "```" in t:
        blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', t, re.DOTALL)
        if blocks:
            t = max(blocks, key=len).strip()
    
    # Ensure trailing newline
    if t and not t.endswith("\n"):
        t += "\n"
    
    return t


def check_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Check Python syntax. Returns (is_valid, error_message)."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        # Get context around error
        lines = code.splitlines()
        line_num = e.lineno or 1
        start = max(0, line_num - 3)
        end = min(len(lines), line_num + 2)
        
        context_lines = []
        for i in range(start, end):
            marker = ">>> " if i == line_num - 1 else "    "
            context_lines.append(f"{i+1:4d}{marker}{lines[i]}")
        context = "\n".join(context_lines)
        
        return False, f"Line {line_num}: {e.msg}\n\n{context}"


# -------------------------
# Prompt template loading
# -------------------------

def load_template_from_py(template_py: Path) -> Template:
    """Load PROMPT Template from a Python module."""
    spec = importlib.util.spec_from_file_location("controller_template_mod", str(template_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import template module: {template_py}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "PROMPT"):
        raise RuntimeError(f"{template_py} must define PROMPT (string.Template).")

    prompt_obj = module.PROMPT
    if not hasattr(prompt_obj, "safe_substitute"):
        raise RuntimeError("PROMPT must be a string.Template (needs .safe_substitute).")
    return prompt_obj


# -------------------------
# World text patching
# -------------------------

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def extract_start_goal_xy(user_prompt: str) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Parse start/goal from user prompt."""
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
            pass

    mg = re.search(rf"(?i)\bgoal\s*at\s*\(\s*({_FLOAT})\s*,\s*({_FLOAT})\s*\)", up)
    if mg:
        try:
            goal_xy = (float(mg.group(1)), float(mg.group(2)))
        except Exception:
            pass

    return start_xy, goal_xy


def _patch_translation_in_def_block(world_text: str, def_name: str, xy: Tuple[float, float]) -> str:
    x, y = xy
    pattern = re.compile(rf"(?s)(\bDEF\s+{re.escape(def_name)}\b.*?\btranslation\s+)({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})")
    m = pattern.search(world_text)
    if not m:
        return world_text
    z = m.group(4)
    repl = f"{m.group(1)}{x:.6g} {y:.6g} {z}"
    return world_text[:m.start()] + repl + world_text[m.end():]


def _patch_translation_in_named_node(world_text: str, name_value: str, xy: Tuple[float, float]) -> str:
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


# -------------------------
# OpenAI call
# -------------------------

def call_openai_code(model: str, prompt: str, max_output_tokens: int) -> Tuple[str, float]:
    """Call OpenAI API. Returns (content, elapsed_seconds)."""
    client = OpenAI()

    system = (
        "You generate Python code. Output ONLY the complete Python file. "
        "No markdown fences. No explanations. No text before or after the code."
    )

    start_time = time.time()
    
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=max_output_tokens,
    )

    elapsed = time.time() - start_time
    content = resp.choices[0].message.content
    
    if content is None:
        raise RuntimeError("OpenAI returned empty response.")
    
    return content, elapsed


def call_anthropic_code(model: str, prompt: str, max_output_tokens: int) -> Tuple[str, float]:
    """Call Anthropic API. Returns (content, elapsed_seconds)."""
    if Anthropic is None:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    client = Anthropic()

    system = (
        "You generate Python code. Output ONLY the complete Python file. "
        "No markdown fences. No explanations. No text before or after the code."
    )

    start_time = time.time()

    resp = client.messages.create(
        model=model,
        max_tokens=max_output_tokens,
        system=system,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    elapsed = time.time() - start_time
    content = resp.content[0].text if resp.content else None

    if content is None:
        raise RuntimeError("Anthropic returned empty response.")

    return content, elapsed


# -------------------------
# Prompt building
# -------------------------

def build_prompt(
    template_py: Optional[Path],
    prompt_file: Optional[Path],
    world_path: Path,
    user_prompt: str = "",
) -> str:
    """Build the final prompt."""
    world_text = read_text(world_path)

    start_xy, goal_xy = extract_start_goal_xy(user_prompt)
    if start_xy is not None or goal_xy is not None:
        world_text = patch_world_text(world_text, start_xy, goal_xy)

    if prompt_file:
        base = read_text(prompt_file)
        filled = Template(base).safe_substitute(
            WORLD_WBT=str(world_path),
            WORLD_WBT_TEXT=world_text,
        )
    elif template_py:
        tmpl = load_template_from_py(template_py)
        filled = tmpl.safe_substitute(
            WORLD_WBT=str(world_path),
            WORLD_WBT_TEXT=world_text,
        )
    else:
        raise RuntimeError("Provide either --template or --prompt-file.")

    up = (user_prompt or "").strip()
    if up:
        filled = up + "\n\n" + filled

    return filled


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Webots Python controller.")
    ap.add_argument("--world", required=True, help="Webots world file (.wbt)")
    ap.add_argument("--out", required=True, help="Output controller .py path")
    ap.add_argument("--template", default="", help="Python module with PROMPT=Template(...)")
    ap.add_argument("--prompt-file", default="", help="Plain text prompt file")
    ap.add_argument("--model", default="gpt-4o", help="OpenAI model name")
    ap.add_argument("--max-output-tokens", type=int, default=9000)
    ap.add_argument("--user-prompt", default="", help="Optional user prompt")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    world_path = Path(args.world).resolve()
    out_path = Path(args.out).resolve()
    template_py = Path(args.template).resolve() if args.template else None
    prompt_file = Path(args.prompt_file).resolve() if args.prompt_file else None

    # Validate inputs
    if not world_path.exists():
        raise FileNotFoundError(f"Missing world: {world_path}")
    if template_py and not template_py.exists():
        raise FileNotFoundError(f"Missing template: {template_py}")
    if prompt_file and not prompt_file.exists():
        raise FileNotFoundError(f"Missing prompt file: {prompt_file}")
    if not template_py and not prompt_file:
        raise RuntimeError("Provide either --template or --prompt-file.")

    # Build prompt
    prompt = build_prompt(template_py, prompt_file, world_path, user_prompt=args.user_prompt)

    if args.debug:
        print(f"[code_gen] model={args.model}", file=sys.stderr)
        print(f"[code_gen] world={world_path}", file=sys.stderr)
        print(f"[code_gen] out={out_path}", file=sys.stderr)

    # Call LLM
    print(f"[code_gen] Calling LLM API ({args.model})...", file=sys.stderr)
    if args.model.startswith("claude"):
        raw, elapsed = call_anthropic_code(args.model, prompt, args.max_output_tokens)
    else:
        raw, elapsed = call_openai_code(args.model, prompt, args.max_output_tokens)
    print(f"[code_gen] LLM call took: {elapsed:.1f}s", file=sys.stderr)

    # Clean output
    controller_py = strip_code_fences(raw)

    # Check syntax (but still write file either way)
    is_valid, syntax_error = check_syntax(controller_py)
    
    # Always write the file so pytest can run
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(controller_py, encoding="utf-8")

    if not is_valid:
        # Save raw output for debugging
        raw_path = out_path.with_suffix(".raw.txt")
        raw_path.write_text(raw, encoding="utf-8")
        
        print(f"[code_gen] ⚠ SYNTAX ERROR: {syntax_error}", file=sys.stderr)
        print(f"[code_gen] Raw output saved: {raw_path}", file=sys.stderr)
        print(f"[code_gen] ✗ Wrote (with errors): {out_path}")
        # Exit with error so repair.py knows generation had issues
        raise RuntimeError(f"Syntax error: {syntax_error}")
    else:
        print(f"[code_gen] ✓ Wrote: {out_path}")


if __name__ == "__main__":
    main()