#!/usr/bin/env python3
"""
code_agent.py

Always generate controller.py using:
- controller_template.py (PROMPT = Template(r))
- robot.py (text)
- params.json (text, validated)
- occupancy map PNG (image)
- robot PNG (image)

Usage:
  python code_agent.py \
    --template controller_template.py \
    --map occupancy.png \
    --params params.json \
    --robot DDR.png \
    --robotpy robot.py \
    --out controller.py
"""
import argparse
import base64
import importlib.util
import json
import os
import re
from typing import Any, Dict, List

from openai import OpenAI


# -------------------------
# File helpers
# -------------------------
def read_text(path: str) -> str:
    """Read UTF-8 text from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_bytes(path: str) -> bytes:
    """Read raw bytes from a file (e.g., PNG)."""
    with open(path, "rb") as f:
        return f.read()


def png_bytes_to_data_url(png_bytes: bytes) -> str:
    """Convert PNG bytes to data URL for Responses API input_image."""
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# -------------------------
# Template loader (needed for controller_template.py)
# -------------------------
def load_prompt_from_py(template_py_path: str):
    """
    Load controller_template.py and return PROMPT.

    Why needed:
    - controller_template.py is Python, not plain text.
    - PROMPT is defined inside it as a string.Template.
    """
    spec = importlib.util.spec_from_file_location("controller_template_mod", template_py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import template module: {template_py_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "PROMPT"):
        raise RuntimeError(f"{template_py_path} must define PROMPT")

    prompt_obj = mod.PROMPT
    if not hasattr(prompt_obj, "safe_substitute"):
        raise RuntimeError("PROMPT must be a string.Template (support .safe_substitute).")

    return prompt_obj


# -------------------------
# Output helpers
# -------------------------
def strip_code_fences(text: str) -> str:
    """Remove ```...``` if the model returns fenced code."""
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    if t and not t.endswith("\n"):
        t += "\n"
    return t


def sanity_check_controller(py_text: str) -> None:
    """Fail fast if controller.py is clearly malformed."""
    if "```" in py_text:
        raise RuntimeError("Output contains markdown fences (```), forbidden.")
    if "def main" not in py_text:
        raise RuntimeError("Missing `def main` in controller.py.")
    if '__name__ == "__main__"' not in py_text:
        raise RuntimeError("Missing __main__ guard in controller.py.")
    if "from robot import" not in py_text:
        raise RuntimeError("Missing `from robot import ...` import in controller.py.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="controller_template.py (defines PROMPT)")
    ap.add_argument("--map", required=True, help="Map image (occupancy.png)")
    ap.add_argument("--params", required=True, help="params.json")
    ap.add_argument("--robot", default="DDR.png", help="Robot image (DDR.png)")
    ap.add_argument("--robotpy", default="robot.py", help="robot.py")
    ap.add_argument("--out", default="controller.py", help="Output controller.py path")

    # These are REQUIRED because your controller_template.py references them.
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

    # Ensure all required files exist (we ALWAYS include everything)
    for label, path in [
        ("template", args.template),
        ("map image", args.map),
        ("params.json", args.params),
        ("robot image", args.robot),
        ("robot.py", args.robotpy),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    # Read text context
    robot_py_text = read_text(args.robotpy)
    params_json_text = read_text(args.params)
    json.loads(params_json_text)  # validate JSON

    # Read image context
    map_png = read_bytes(args.map)
    robot_png = read_bytes(args.robot)

    # Load template PROMPT from controller_template.py
    prompt_tmpl = load_prompt_from_py(args.template)

    # Fill placeholders EXACTLY as your template expects
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

    prompt = prompt_tmpl.safe_substitute(**variables)

    if args.debug:
        print(f"[debug] model={args.model}")
        print(f"[debug] prompt_chars={len(prompt)}")
        print(f"[debug] map_bytes={len(map_png)} robot_bytes={len(robot_png)}")

    # Send prompt + images
    content: List[dict] = [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": png_bytes_to_data_url(map_png)},
        {"type": "input_image", "image_url": png_bytes_to_data_url(robot_png)},
    ]

    client = OpenAI()
    resp = client.responses.create(
        model=args.model,
        input=[{"role": "user", "content": content}],
        max_output_tokens=args.max_output_tokens,
    )

    controller_text = strip_code_fences(getattr(resp, "output_text", "") or "")
    sanity_check_controller(controller_text)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(controller_text)

    print(f"[code_agent] Generated: {args.out}")
    print("[code_agent] Run: python controller.py")


if __name__ == "__main__":
    main()
