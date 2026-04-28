#!/usr/bin/env python3
"""
code_agent.py

Generates (or edits) controller.py using an OpenAI model.

Modes:
- generate (default): produce a fresh controller.py from controller_template.py + robot.py + params.json (+ images)
- edit: modify an existing controller.py using pytest failure summaries, WITHOUT changing controller_template.py

Typical usage (generate):
  python code_agent.py --template controller_template.py --map occupancy.png --params params.json \
    --robot DDR.png --robotpy robot.py --out controller.py --model gpt-4.1 --max-output-tokens 9000

Edit usage (called by loop_agent.py):
  python code_agent.py --mode edit --controller-in controller.py --failures-json .last_failures.json \
    --template controller_template.py --map occupancy.png --params params.json --robot DDR.png --robotpy robot.py \
    --out controller.py --model gpt-4.1 --max-output-tokens 9000


python loop_agent.py --project . --iters 20 --edit-retries 3 --model gpt-5.2

"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI


# -----------------------------
# 1) Small file utilities
# -----------------------------
def read_text(path: str) -> str:
    """Read a UTF-8 text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def ensure_parent_dir(path: str) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    try:
        if os.path.exists(path):
            return json.loads(read_text(path))
    except Exception:
        pass
    return {}


def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def file_sha256(path: str) -> str:
    """Hash a file so we can cache uploads by content (not by filename)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------
# 2) Load PROMPT Template
# -----------------------------
def load_prompt_template(template_path: str):
    """
    controller_template.py is a Python file that defines PROMPT as a string.Template.
    We import it and return that PROMPT object.
    """
    spec = importlib.util.spec_from_file_location("controller_template_mod", template_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import template module: {template_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "PROMPT"):
        raise RuntimeError(f"{template_path} must define PROMPT")

    prompt_obj = mod.PROMPT
    if not hasattr(prompt_obj, "safe_substitute"):
        raise RuntimeError("PROMPT must be a string.Template (needs .safe_substitute).")

    return prompt_obj


# -----------------------------
# 3) Upload images once (File ID cache)
# -----------------------------
def get_image_file_id(client: OpenAI, image_path: str, cache: Dict[str, str]) -> str:
    """
    Returns a file_id for this image.
    - If we've already uploaded the *same bytes* before, reuse the cached file_id.
    - Otherwise, upload and store in cache.
    """
    digest = file_sha256(image_path)
    if digest in cache:
        return cache[digest]

    with open(image_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="vision")

    cache[digest] = uploaded.id
    return uploaded.id


# -----------------------------
# 4) Clean + sanity check model output
# -----------------------------
def strip_code_fences(text: str) -> str:
    """If model returns ```python ... ```, remove the fences."""
    t = (text or "").strip()
    if t.startswith("```"):
        ls = t.splitlines()
        if ls and ls[0].startswith("```"):
            ls = ls[1:]
        if ls and ls[-1].startswith("```"):
            ls = ls[:-1]
        t = "\n".join(ls).strip()
    return t + ("\n" if t and not t.endswith("\n") else "")


def sanity_check_controller(py_text: str) -> None:
    """Basic checks so we fail fast if output is clearly wrong."""
    if "```" in py_text:
        raise RuntimeError("Output contains markdown fences (```), forbidden.")
    if "def main" not in py_text:
        raise RuntimeError("Missing `def main` in controller.py.")
    if '__name__ == "__main__"' not in py_text:
        raise RuntimeError("Missing __main__ guard in controller.py.")
    if "from robot import" not in py_text:
        raise RuntimeError("Missing `from robot import ...` import in controller.py.")


def _read_failures_payload(failures_json: str, failures_text: str) -> str:
    if failures_json:
        try:
            raw = read_text(failures_json)
            obj = json.loads(raw)
            return json.dumps(obj, indent=2)[:12000]
        except Exception:
            try:
                return read_text(failures_json)[:12000]
            except Exception:
                return ""
    return (failures_text or "")[:12000]


def _append_edit_block(base_prompt: str, controller_text: str, failures_payload: str, extra: str) -> str:
    extra = (extra or "").strip()
    return (
        base_prompt.rstrip()
        + "\n\n"
        + "============================================================\n"
        + "EDIT MODE\n"
        + "============================================================\n"
        + "You MUST edit the existing controller.py to pass the failing pytest tests.\n"
        + "Output ONLY the full updated controller.py (no markdown, no explanations).\n"
        + "You MUST preserve all constraints stated above (imports, function structure, etc.).\n"
        + "You MAY change any part of the file as needed, but keep the public API expected by tests.\n"
        + ("\nAdditional edit instructions:\n" + extra + "\n" if extra else "")
        + "\n<BEGIN_EXISTING_CONTROLLER>\n"
        + controller_text.rstrip()
        + "\n<END_EXISTING_CONTROLLER>\n"
        + "\n<BEGIN_PYTEST_FAILURES>\n"
        + (failures_payload.rstrip() if failures_payload else "(no failure details provided)")
        + "\n<END_PYTEST_FAILURES>\n"
        + "\nNow output the complete updated controller.py.\n"
    )


# -----------------------------
# 5) Main program
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--robot", default="DDR.png")
    ap.add_argument("--robotpy", default="robot.py")
    ap.add_argument("--out", default="controller.py")

    ap.add_argument("--mode", choices=["generate", "edit"], default="generate",
                    help="generate: create a new controller.py; edit: modify existing controller.py using failures")
    ap.add_argument("--controller-in", default="",
                    help="In edit mode, path to existing controller.py (defaults to --out)")
    ap.add_argument("--failures-json", default="",
                    help="In edit mode, path to JSON (or text) file with pytest failure summaries")
    ap.add_argument("--failures-text", default="",
                    help="In edit mode, failure summary text (alternative to --failures-json)")
    ap.add_argument("--edit-instructions", default="",
                    help="In edit mode, extra instructions appended after failures/context")

    # Parameters your template expects:
    ap.add_argument("--axle-length-px", type=float, default=30.0)
    ap.add_argument("--sensor-range-px", type=int, default=220)
    ap.add_argument("--sensor-fov-deg", type=float, default=55.0)
    ap.add_argument("--n-rays", type=int, default=18)
    ap.add_argument("--lookahead", type=float, default=35.0)
    ap.add_argument("--stop-dist", type=float, default=35.0)
    ap.add_argument("--slow-dist", type=float, default=95.0)

    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--max-output-tokens", type=int, default=9000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # 1) Ensure inputs exist
    needed = [args.template, args.map, args.params, args.robot, args.robotpy]
    for path in needed:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    # 2) Read robot.py + params.json text, validate JSON
    robot_py_text = read_text(args.robotpy)
    params_json_text = read_text(args.params)
    json.loads(params_json_text)  # raises if invalid

    # 3) Build the base prompt text from your template
    prompt_template = load_prompt_template(args.template)
    variables: Dict[str, Any] = {
        "MAP_IMG": args.map,
        "PARAMS_JSON": args.params,
        "ROBOT_IMG": args.robot,

        "AXLE_LENGTH_PX": float(args.axle_length_px),
        "SENSOR_RANGE_PX": int(args.sensor_range_px),
        "SENSOR_FOV_DEG": float(args.sensor_fov_deg),
        "N_RAYS": int(args.n_rays),
        "LOOKAHEAD": float(args.lookahead),
        "STOP_DIST": float(args.stop_dist),
        "SLOW_DIST": float(args.slow_dist),

        "ROBOT_PY_TEXT": robot_py_text,
        "PARAMS_JSON_TEXT": params_json_text,
    }
    prompt = prompt_template.safe_substitute(**variables)

    # If editing, append controller + failures context, but DO NOT touch controller_template.py
    if args.mode == "edit":
        controller_in = args.controller_in or args.out
        if not os.path.exists(controller_in):
            raise FileNotFoundError(f"Edit mode requires an existing file: {controller_in}")
        controller_existing_text = read_text(controller_in)
        failures_payload = _read_failures_payload(args.failures_json, args.failures_text)
        prompt = _append_edit_block(prompt, controller_existing_text, failures_payload, args.edit_instructions)

    # 4) Create client and get file_ids for images (with caching)
    client = OpenAI()
    cache_path = ".openai_cache/vision_file_ids.json"
    cache = load_json(cache_path)

    map_file_id = get_image_file_id(client, args.map, cache)
    robot_file_id = get_image_file_id(client, args.robot, cache)
    save_json(cache_path, cache)

    # 5) Send prompt + images to the model
    content = [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "file_id": map_file_id},
        {"type": "input_image", "file_id": robot_file_id},
    ]

    if args.debug:
        print(f"[debug] mode={args.mode} model={args.model}")
        print(f"[debug] prompt_chars={len(prompt)}")
        print(f"[debug] map_file_id={map_file_id}")
        print(f"[debug] robot_file_id={robot_file_id}")

    resp = client.responses.create(
        model=args.model,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "file_id": map_file_id},
                    {"type": "input_image", "file_id": robot_file_id},
                ],
            }
        ],
        max_output_tokens=args.max_output_tokens,
    )


    # 6) Extract and validate output, then write controller.py
    controller_text = strip_code_fences(getattr(resp, "output_text", "") or "")
    sanity_check_controller(controller_text)

    ensure_parent_dir(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(controller_text)

    print(f"[code_agent] mode={args.mode} wrote: {args.out}")


if __name__ == "__main__":
    main()
