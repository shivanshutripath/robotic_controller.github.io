#!/usr/bin/env python3
"""
code_agent.py

Generates (or edits) controller.py using OpenAI or Claude models.

Modes:
- generate (default): produce a fresh controller.py from controller_template.py + robot.py + params.json (+ images)
- edit: modify an existing controller.py using pytest failure summaries, WITHOUT changing controller_template.py

Supports both OpenAI and Claude models via unified interface.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI


# -----------------------------
# Model Client Wrapper
# -----------------------------
class ModelClient:
    """Unified client for OpenAI and Claude models."""
    
    # Model name mappings
    MODEL_MAP = {
        "gpt-5.2": "gpt-5.2",
        "gpt-4.1": "gpt-4.1",
        "gpt-4o": "gpt-4o",
        "gpt-4": "gpt-4",
        "claude-opus-4.5": "claude-opus-4-5-20251101",
        "claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
        "claude-haiku-4.5": "claude-haiku-4-5-20251001",
    }
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_claude = model_name.startswith("claude")
        self.api_model = self.MODEL_MAP.get(model_name, model_name)
        
        if self.is_claude:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set for Claude models")
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = OpenAI()
    
    def generate_with_images(
        self,
        prompt: str,
        image_paths: list[str],
        max_tokens: int = 9000
    ) -> str:
        """Generate response with text prompt and images."""
        if self.is_claude:
            return self._generate_claude(prompt, image_paths, max_tokens)
        else:
            return self._generate_openai(prompt, image_paths, max_tokens)
    
    def _generate_openai(self, prompt: str, image_paths: list[str], max_tokens: int) -> str:
        """Generate using OpenAI API with file uploads."""
        # Upload images and get file IDs (same as original)
        cache_path = ".openai_cache/vision_file_ids.json"
        cache = load_json(cache_path)
        
        file_ids = []
        for img_path in image_paths:
            file_ids.append(get_image_file_id(self.client, img_path, cache))
        
        save_json(cache_path, cache)
        
        # Build content
        content = [{"type": "input_text", "text": prompt}]
        for fid in file_ids:
            content.append({"type": "input_image", "file_id": fid})
        
        resp = self.client.responses.create(
            model=self.api_model,
            input=[{"type": "message", "role": "user", "content": content}],
            max_output_tokens=max_tokens,
        )
        
        return getattr(resp, "output_text", "") or ""
    
    def _generate_claude(self, prompt: str, image_paths: list[str], max_tokens: int) -> str:
        """Generate using Claude API with base64 images."""
        # Claude uses base64-encoded images in the message content
        content = []
        
        # Add images first
        for img_path in image_paths:
            with open(img_path, "rb") as f:
                img_data = base64.standard_b64encode(f.read()).decode("utf-8")
            
            # Determine media type
            ext = Path(img_path).suffix.lower()
            media_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }.get(ext, "image/png")
            
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": img_data,
                }
            })
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # Make API call
        response = self.client.messages.create(
            model=self.api_model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": content
            }]
        )
        
        # Extract text from response
        return response.content[0].text


# -----------------------------
# Code Extraction (handles both GPT and Claude)
# -----------------------------
def extract_code(response: str, model_name: str = "") -> str:
    """
    Extract Python code from model response.
    Handles both GPT and Claude formatting.
    """
    import re
    
    original = response.strip()
    extracted = original
    
    # Step 1: Remove markdown code blocks
    if "```python" in extracted:
        match = re.search(r'```python\s*\n(.*?)```', extracted, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
    elif "```" in extracted:
        parts = extracted.split("```")
        if len(parts) >= 3:
            code_block = parts[1].strip()
            code_block = re.sub(r'^[a-zA-Z]+\s*\n', '', code_block)
            extracted = code_block
    
    # Step 2: Remove preamble
    lines = extracted.split('\n')
    code_start = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if (stripped.startswith('def ') or stripped.startswith('class ') or
            stripped.startswith('import ') or stripped.startswith('from ') or
            stripped.startswith('@')):
            code_start = i
            break
    
    if code_start is not None and code_start > 0:
        lines = lines[code_start:]
    
    # Step 3: Remove postamble
    code_end = len(lines)
    explanation_markers = [
        'this function', 'this code', 'this implementation',
        'the function', 'the code', 'note:', 'example:', 'usage:',
        'explanation:', 'here\'s how', 'as you can see'
    ]
    
    for i, line in enumerate(lines):
        if i < 5:
            continue
        stripped = line.strip().lower()
        if any(stripped.startswith(m) for m in explanation_markers):
            code_end = i
            break
    
    lines = lines[:code_end]
    extracted = '\n'.join(lines).strip()
    
    if not extracted:
        return original
    
    return extracted


# -----------------------------
# Original utility functions
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
    """Hash a file so we can cache uploads by content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prompt_template(template_path: str):
    """Load PROMPT template from controller_template.py."""
    spec = importlib.util.spec_from_file_location("controller_template_mod", template_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import template module: {template_path}")
    
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    if not hasattr(mod, "PROMPT"):
        raise RuntimeError(f"{template_path} must define PROMPT")
    
    prompt_obj = mod.PROMPT
    if not hasattr(prompt_obj, "safe_substitute"):
        raise RuntimeError("PROMPT must be a string.Template")
    
    return prompt_obj


def get_image_file_id(client: OpenAI, image_path: str, cache: Dict[str, str]) -> str:
    """Get or create file_id for OpenAI image upload."""
    digest = file_sha256(image_path)
    if digest in cache:
        return cache[digest]
    
    with open(image_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="vision")
    
    cache[digest] = uploaded.id
    return uploaded.id


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences if present."""
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
    """Basic validation of generated controller."""
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
# Main program
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--robot", default="DDR.png")
    ap.add_argument("--robotpy", default="robot.py")
    ap.add_argument("--out", default="controller.py")
    
    ap.add_argument("--mode", choices=["generate", "edit"], default="generate")
    ap.add_argument("--controller-in", default="")
    ap.add_argument("--failures-json", default="")
    ap.add_argument("--failures-text", default="")
    ap.add_argument("--edit-instructions", default="")
    
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
    
    # Validate inputs
    needed = [args.template, args.map, args.params, args.robot, args.robotpy]
    for path in needed:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
    
    # Read context files
    robot_py_text = read_text(args.robotpy)
    params_json_text = read_text(args.params)
    json.loads(params_json_text)  # validate
    
    # Build prompt
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
    
    # Add edit context if needed
    if args.mode == "edit":
        controller_in = args.controller_in or args.out
        if not os.path.exists(controller_in):
            raise FileNotFoundError(f"Edit mode requires: {controller_in}")
        controller_existing = read_text(controller_in)
        failures = _read_failures_payload(args.failures_json, args.failures_text)
        prompt = _append_edit_block(prompt, controller_existing, failures, args.edit_instructions)
    
    if args.debug:
        print(f"[debug] mode={args.mode} model={args.model}")
        print(f"[debug] prompt_chars={len(prompt)}")
    
    # Generate with appropriate client
    client = ModelClient(args.model)
    image_paths = [args.map, args.robot]
    
    raw_response = client.generate_with_images(
        prompt=prompt,
        image_paths=image_paths,
        max_tokens=args.max_output_tokens
    )
    
    # Extract and clean code
    controller_text = extract_code(raw_response, args.model)
    controller_text = strip_code_fences(controller_text)
    
    # Validate and write
    sanity_check_controller(controller_text)
    
    ensure_parent_dir(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(controller_text)
    
    print(f"[code_agent] mode={args.mode} model={args.model} wrote: {args.out}")


if __name__ == "__main__":
    main()