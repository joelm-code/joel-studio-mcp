#!/usr/bin/env python3
"""
video-production image generation wrapper — Google Gemini Nano Banana

Models:
    gemini-3.1-flash-image-preview  — Nano Banana 2 (fast, go-to default)
    gemini-3-pro-image-preview      — Nano Banana Pro (professional, thinking)
    gemini-2.5-flash-image          — Nano Banana (speed/efficiency fallback)

Usage:
    python generate.py generate --prompt "..." --output /path/to/image.png [--aspect 16:9] [--size 2K]
    python generate.py edit --input /path/to/source.png --prompt "make it warmer" --output /path/to/edited.png
    python generate.py compose --inputs img1.png img2.png img3.png --prompt "group photo" --output /path/to/out.png
    python generate.py batch --prompts-file prompts.json --output-dir /path/to/frames/
    python generate.py check

Requires:
    pip install google-genai Pillow
    export GEMINI_API_KEY="your-key-here"
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Central config location for API keys (stored once, used by all projects)
CENTRAL_CONFIG_DIR = Path.home() / ".config" / "video-production"
CENTRAL_ENV_FILE = CENTRAL_CONFIG_DIR / ".env"

# Auto-load .env: check CWD first, then walk up, then central config
def _load_dotenv():
    # Priority 1: Project-local .env (CWD and parent directories)
    search_roots = [
        Path.cwd(),                      # project directory (where user runs from)
        Path(__file__).resolve().parent,  # script directory (fallback)
    ]
    for start in search_roots:
        d = start
        for _ in range(10):
            env_file = d / ".env"
            if env_file.exists():
                _parse_env_file(env_file)
                return  # found and loaded, stop searching
            d = d.parent

    # Priority 2: Central config (~/.config/video-production/.env)
    if CENTRAL_ENV_FILE.exists():
        _parse_env_file(CENTRAL_ENV_FILE)


def _parse_env_file(env_file):
    """Parse a .env file and set environment variables (setdefault — won't overwrite)."""
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()

DEFAULT_MODEL = "gemini-3.1-flash-image-preview"
VALID_ASPECTS = ["1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9"]
VALID_SIZES = ["512px", "1K", "2K", "4K"]

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False


def get_client():
    """Initialize Gemini client with API key from environment."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(json.dumps({
            "status": "error",
            "error": "GEMINI_API_KEY environment variable not set",
            "fix": "export GEMINI_API_KEY='your-key-here'",
            "get_key": "https://aistudio.google.com/apikey"
        }))
        sys.exit(1)
    return genai.Client(api_key=api_key)


def _build_config(aspect_ratio: str = "16:9", image_size: str = "1K",
                  web_search: bool = False, image_search: bool = False,
                  thinking: str = None,
                  image_only: bool = False,
                  model: str = None) -> "types.GenerateContentConfig":
    """Build the GenerateContentConfig with proper ImageConfig.

    Search grounding options:
        web_search: Enable Google Web Search for real-time data (all models)
        image_search: Enable Google Image Search for visual references (3.1 Flash only)
    """

    modalities = ["IMAGE"] if image_only else ["TEXT", "IMAGE"]

    image_config = types.ImageConfig(
        aspect_ratio=aspect_ratio,
    )
    # image_size is supported by Gemini 3+ models only (not gemini-2.5-flash-image)
    supports_image_size = model and "gemini-3" in model
    if image_size and image_size != "1K" and supports_image_size:
        try:
            image_config = types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=image_size,
            )
        except Exception:
            pass  # model doesn't support image_size, use default

    config_kwargs = {
        "response_modalities": modalities,
        "image_config": image_config,
    }

    # Google Search grounding — web and/or image search
    if web_search or image_search:
        search_types_kwargs = {}
        if web_search:
            search_types_kwargs["web_search"] = types.WebSearch()
        if image_search:
            search_types_kwargs["image_search"] = types.ImageSearch()

        if search_types_kwargs:
            config_kwargs["tools"] = [
                types.Tool(google_search=types.GoogleSearch(
                    search_types=types.SearchTypes(**search_types_kwargs)
                ))
            ]
        else:
            config_kwargs["tools"] = [{"google_search": {}}]

    # Thinking level (Gemini 3+ only)
    if thinking:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking,
            include_thoughts=False,
        )

    return types.GenerateContentConfig(**config_kwargs)


def _extract_result(response, output_path: str) -> tuple:
    """Extract image and text from API response. Returns (image_saved, text_response, img)."""
    image_saved = False
    text_response = ""
    img = None

    for part in response.parts:
        # Skip thinking parts
        if hasattr(part, 'thought') and part.thought:
            continue

        if part.inline_data is not None:
            import io
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            img = Image.open(io.BytesIO(part.inline_data.data))
            img.save(str(output))
            image_saved = True
        elif part.text is not None:
            text_response += part.text

    return image_saved, text_response, img


def generate_image(prompt: str, output_path: str, aspect_ratio: str = "16:9",
                   image_size: str = "1K", model: str = DEFAULT_MODEL,
                   web_search: bool = False, image_search: bool = False,
                   thinking: str = None, image_only: bool = False,
                   # Legacy compat
                   search_grounding: bool = False) -> dict:
    """Generate an image from a text prompt."""
    # Legacy: search_grounding=True maps to web_search=True
    if search_grounding and not web_search:
        web_search = True

    client = get_client()

    config = _build_config(
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        web_search=web_search,
        image_search=image_search,
        thinking=thinking,
        image_only=image_only,
        model=model,
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    image_saved, text_response, img = _extract_result(response, output_path)

    result = {
        "status": "success" if image_saved else "no_image",
        "path": str(output_path) if image_saved else None,
        "model": model,
        "aspect_ratio": aspect_ratio,
        "image_size": image_size,
        "prompt": prompt,
        "text_response": text_response if text_response else None,
    }

    if image_saved and img:
        result["width"] = img.width
        result["height"] = img.height

    # Include grounding metadata if search was used
    if (web_search or image_search) and response.candidates:
        gm = getattr(response.candidates[0], 'grounding_metadata', None)
        if gm:
            grounding = {}
            if hasattr(gm, 'grounding_chunks') and gm.grounding_chunks:
                grounding["sources"] = []
                for chunk in gm.grounding_chunks:
                    source = {}
                    if hasattr(chunk, 'web') and chunk.web:
                        source["uri"] = getattr(chunk.web, 'uri', None)
                        source["title"] = getattr(chunk.web, 'title', None)
                    if hasattr(chunk, 'image_uri'):
                        source["image_uri"] = chunk.image_uri
                    if source:
                        grounding["sources"].append(source)
            if hasattr(gm, 'image_search_queries') and gm.image_search_queries:
                grounding["image_search_queries"] = list(gm.image_search_queries)
            if grounding:
                result["grounding"] = grounding

    return result


def edit_image(input_path: str, prompt: str, output_path: str,
               aspect_ratio: str = None, image_size: str = "1K",
               model: str = DEFAULT_MODEL) -> dict:
    """Edit an existing image with a text prompt."""
    # Validate input file exists
    if not Path(input_path).exists():
        return {
            "status": "error",
            "error": f"Source image not found: {input_path}",
            "fix": "Check the file path — the source image may have been moved or deleted"
        }

    client = get_client()

    source_img = Image.open(input_path)

    # If no aspect ratio specified, infer from source image
    if not aspect_ratio:
        w, h = source_img.size
        ratio = w / h
        if ratio > 1.7:
            aspect_ratio = "16:9"
        elif ratio > 1.4:
            aspect_ratio = "3:2"
        elif ratio > 1.1:
            aspect_ratio = "4:3"
        elif ratio > 0.9:
            aspect_ratio = "1:1"
        elif ratio > 0.7:
            aspect_ratio = "3:4"
        elif ratio > 0.55:
            aspect_ratio = "9:16"
        else:
            aspect_ratio = "9:16"

    config_kwargs = {
        "image_size": image_size,
        "aspect_ratio": aspect_ratio,
        "model": model,
    }

    config = _build_config(**config_kwargs)

    response = client.models.generate_content(
        model=model,
        contents=[prompt, source_img],
        config=config,
    )

    image_saved, text_response, img = _extract_result(response, output_path)

    result = {
        "status": "success" if image_saved else "no_image",
        "path": str(output_path) if image_saved else None,
        "source": str(input_path),
        "model": model,
        "prompt": prompt,
        "text_response": text_response if text_response else None,
    }

    if image_saved and img:
        result["width"] = img.width
        result["height"] = img.height

    return result


def compose_images(input_paths: list, prompt: str, output_path: str,
                   aspect_ratio: str = "16:9", image_size: str = "1K",
                   model: str = DEFAULT_MODEL) -> dict:
    """Compose a new image from multiple reference images (up to 14).

    Supports:
    - Up to 10 object reference images (high-fidelity) on 3.1 Flash
    - Up to 4 character reference images on 3.1 Flash
    - Up to 6 object + 5 character on Pro
    """
    client = get_client()

    if len(input_paths) > 14:
        return {"status": "error", "error": f"Maximum 14 input images, got {len(input_paths)}"}

    # Validate all input files exist
    for path in input_paths:
        if not Path(path).exists():
            return {
                "status": "error",
                "error": f"Reference image not found: {path}",
                "fix": "Check the file path in prompt_template.characters — the reference image may have been moved or deleted"
            }

    # Build contents: prompt + all images
    contents = [prompt]
    for path in input_paths:
        contents.append(Image.open(path))

    config = _build_config(
        aspect_ratio=aspect_ratio,
        image_size=image_size,
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    image_saved, text_response, img = _extract_result(response, output_path)

    result = {
        "status": "success" if image_saved else "no_image",
        "path": str(output_path) if image_saved else None,
        "sources": [str(p) for p in input_paths],
        "source_count": len(input_paths),
        "model": model,
        "prompt": prompt,
        "text_response": text_response if text_response else None,
    }

    if image_saved and img:
        result["width"] = img.width
        result["height"] = img.height

    return result


def batch_generate(prompts_file: str, output_dir: str, aspect_ratio: str = "16:9",
                   image_size: str = "1K", model: str = DEFAULT_MODEL,
                   search_grounding: bool = False) -> dict:
    """Generate multiple images from a JSON prompts file.

    Prompts file format:
    [
        {"id": "frame-01", "prompt": "..."},
        {"id": "frame-02", "prompt": "...", "aspect": "9:16", "size": "2K"},
        ...
    ]
    """
    with open(prompts_file) as f:
        prompts = json.load(f)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    results = []
    for i, item in enumerate(prompts):
        frame_id = item.get("id", f"frame-{i+1:02d}")
        prompt = item["prompt"]
        # Per-frame overrides
        frame_aspect = item.get("aspect", aspect_ratio)
        frame_size = item.get("size", image_size)
        out_path = output / f"{frame_id}.png"

        print(f"[{i+1}/{len(prompts)}] Generating {frame_id}...", file=sys.stderr)

        try:
            result = generate_image(
                prompt=prompt,
                output_path=str(out_path),
                aspect_ratio=frame_aspect,
                image_size=frame_size,
                model=model,
                search_grounding=search_grounding,
            )
            result["id"] = frame_id
            results.append(result)
        except Exception as e:
            results.append({
                "id": frame_id,
                "status": "error",
                "error": str(e),
                "prompt": prompt,
            })

    summary = {
        "total": len(prompts),
        "success": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] != "success"),
        "output_dir": str(output_dir),
        "results": results,
    }
    return summary


# ---------------------------------------------------------------------------
# Video generation (Veo)
# ---------------------------------------------------------------------------
DEFAULT_VIDEO_MODEL = "veo-3.1-generate-preview"
VALID_VIDEO_MODELS = [
    "veo-3.1-generate-preview",
    "veo-3.1-fast-generate-preview",
    "veo-2.0-generate-001",
]
VALID_VIDEO_ASPECTS = ["16:9", "9:16"]
VALID_VIDEO_RESOLUTIONS = ["720p", "1080p", "4k"]
VALID_VIDEO_DURATIONS = [4, 6, 8]


def _poll_video_operation(client, operation, poll_interval: int = 10,
                          max_wait: int = 600) -> object:
    """Poll a video generation operation until complete or timeout."""
    import time
    elapsed = 0
    while not operation.done:
        if elapsed >= max_wait:
            raise TimeoutError(
                f"Video generation timed out after {max_wait}s. "
                f"Operation: {operation.name}"
            )
        time.sleep(poll_interval)
        elapsed += poll_interval
        operation = client.operations.get(operation)
    return operation


def generate_video(prompt: str, output_path: str,
                   model: str = DEFAULT_VIDEO_MODEL,
                   aspect_ratio: str = "16:9",
                   resolution: str = "720p",
                   duration: int = 8,
                   person_generation: str = "allow_all",
                   image_path: str = None,
                   last_frame_path: str = None,
                   reference_image_paths: list = None) -> dict:
    """Generate a video from text, optionally with image inputs.

    Supports:
        - Text-to-video (prompt only)
        - Image-to-video (prompt + image_path as first frame)
        - Interpolation (prompt + image_path + last_frame_path)
        - Reference images (prompt + reference_image_paths, up to 3)
    """
    client = get_client()
    Image = __import__('PIL.Image', fromlist=['Image']).Image if (image_path or last_frame_path or reference_image_paths) else None

    # Build config
    config_kwargs = {
        "aspect_ratio": aspect_ratio,
        "person_generation": person_generation,
    }

    if resolution != "720p":
        config_kwargs["resolution"] = resolution

    if duration != 8:
        config_kwargs["duration_seconds"] = duration

    # Handle reference images (Veo 3.1 only, up to 3)
    if reference_image_paths:
        ref_images = []
        for ref_path in reference_image_paths[:3]:
            if not Path(ref_path).exists():
                return {"status": "error", "error": f"Reference image not found: {ref_path}"}
            ref_images.append(types.VideoGenerationReferenceImage(
                image=Image.open(ref_path),
                reference_type="asset",
            ))
        config_kwargs["reference_images"] = ref_images

    # Handle last frame (interpolation)
    if last_frame_path:
        if not Path(last_frame_path).exists():
            return {"status": "error", "error": f"Last frame image not found: {last_frame_path}"}
        config_kwargs["last_frame"] = Image.open(last_frame_path)

    config = types.GenerateVideosConfig(**config_kwargs)

    # Handle first frame image
    image_input = None
    if image_path:
        if not Path(image_path).exists():
            return {"status": "error", "error": f"First frame image not found: {image_path}"}
        image_input = Image.open(image_path)

    # Start generation
    try:
        operation = client.models.generate_videos(
            model=model,
            prompt=prompt,
            image=image_input,
            config=config,
        )
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Poll until done
    try:
        operation = _poll_video_operation(client, operation)
    except TimeoutError as e:
        return {"status": "timeout", "error": str(e), "operation_name": operation.name}

    # Download and save
    try:
        video = operation.response.generated_videos[0]
        client.files.download(file=video.video)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        video.video.save(str(output))
    except Exception as e:
        return {"status": "error", "error": f"Download failed: {e}"}

    return {
        "status": "success",
        "path": str(output_path),
        "model": model,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "duration_seconds": duration,
        "prompt": prompt,
    }


def extend_video(video_path: str, prompt: str, output_path: str,
                 model: str = DEFAULT_VIDEO_MODEL) -> dict:
    """Extend a previously Veo-generated video by ~7 seconds.

    The input video must be a Veo-generated video, 720p, <=141s, and generated
    within the last 2 days.
    """
    client = get_client()

    if not Path(video_path).exists():
        return {"status": "error", "error": f"Video not found: {video_path}"}

    try:
        # Read video bytes for the API
        video_bytes = Path(video_path).read_bytes()
        video_input = types.Video(video_bytes=video_bytes)

        operation = client.models.generate_videos(
            model=model,
            prompt=prompt,
            video=video_input,
            config=types.GenerateVideosConfig(
                number_of_videos=1,
                resolution="720p",
            ),
        )
    except Exception as e:
        return {"status": "error", "error": str(e)}

    # Poll until done
    try:
        operation = _poll_video_operation(client, operation)
    except TimeoutError as e:
        return {"status": "timeout", "error": str(e), "operation_name": operation.name}

    # Download and save
    try:
        video = operation.response.generated_videos[0]
        client.files.download(file=video.video)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        video.video.save(str(output))
    except Exception as e:
        return {"status": "error", "error": f"Download failed: {e}"}

    return {
        "status": "success",
        "path": str(output_path),
        "source_video": str(video_path),
        "model": model,
        "prompt": prompt,
    }


def check_setup() -> dict:
    """Verify environment: API key, dependencies, and test connectivity."""
    issues = []

    if not HAS_GENAI:
        issues.append({"component": "google-genai", "status": "missing", "fix": "pip install google-genai"})
    else:
        issues.append({"component": "google-genai", "status": "ok"})

    if not HAS_PILLOW:
        issues.append({"component": "Pillow", "status": "missing", "fix": "pip install Pillow"})
    else:
        issues.append({"component": "Pillow", "status": "ok"})

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        issues.append({
            "component": "GEMINI_API_KEY",
            "status": "missing",
            "fix": "export GEMINI_API_KEY='your-key-here'",
            "get_key": "https://aistudio.google.com/apikey"
        })
    else:
        # Mask the key for display
        masked = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "****"
        issues.append({"component": "GEMINI_API_KEY", "status": "ok", "value": masked})

    # Python version
    issues.insert(0, {"component": "python3", "status": "ok", "version": sys.version.split()[0]})

    all_ok = all(i["status"] == "ok" for i in issues)

    result = {
        "status": "ready" if all_ok else "not_ready",
        "checks": issues,
        "default_model": DEFAULT_MODEL,
        "available_models": [
            {"name": "gemini-3.1-flash-image-preview", "alias": "Nano Banana 2", "tier": "fast, go-to"},
            {"name": "gemini-3-pro-image-preview", "alias": "Nano Banana Pro", "tier": "professional, thinking"},
            {"name": "gemini-2.5-flash-image", "alias": "Nano Banana", "tier": "speed/efficiency"},
        ],
        "supported_aspects": VALID_ASPECTS,
        "supported_sizes": VALID_SIZES,
        "available_video_models": [
            {"name": "veo-3.1-generate-preview", "alias": "Veo 3.1", "tier": "best quality, audio, 720p-4k"},
            {"name": "veo-3.1-fast-generate-preview", "alias": "Veo 3.1 Fast", "tier": "fast, good quality"},
            {"name": "veo-2.0-generate-001", "alias": "Veo 2", "tier": "stable, 720p, no audio"},
        ],
    }

    # Test API connectivity with a real API call
    if all_ok:
        try:
            client = genai.Client(api_key=api_key)
            # Actually hit the API to verify the key works
            client.models.get(model=DEFAULT_MODEL)
            result["connectivity"] = "ok"
        except Exception as e:
            error_msg = str(e)
            if "API_KEY_INVALID" in error_msg or "API key not valid" in error_msg:
                result["connectivity"] = "error: API key is invalid"
            else:
                result["connectivity"] = f"error: {error_msg}"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="video-production image generation — Gemini Nano Banana API wrapper"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- generate ---
    gen = subparsers.add_parser("generate", help="Generate an image from a prompt")
    gen.add_argument("--prompt", required=True, help="Text prompt for image generation")
    gen.add_argument("--output", required=True, help="Output file path (.png, .jpg, .webp)")
    gen.add_argument("--aspect", default="16:9", help=f"Aspect ratio (default: 16:9). Options: {', '.join(VALID_ASPECTS)}")
    gen.add_argument("--size", default="1K", help=f"Image size (default: 1K). Options: {', '.join(VALID_SIZES)}")
    gen.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model (default: {DEFAULT_MODEL})")
    gen.add_argument("--web-search", action="store_true", help="Enable Google Web Search grounding for real-time data")
    gen.add_argument("--image-search", action="store_true", help="Enable Google Image Search grounding (3.1 Flash only)")
    gen.add_argument("--search", action="store_true", help="(Legacy) Alias for --web-search")
    gen.add_argument("--image-only", action="store_true", help="Return only the image, no text")
    gen.add_argument("--thinking", choices=["minimal", "high"], default=None, help="Thinking level (Gemini 3+ only)")

    # --- edit ---
    edit = subparsers.add_parser("edit", help="Edit an existing image with a prompt")
    edit.add_argument("--input", required=True, help="Source image path")
    edit.add_argument("--prompt", required=True, help="Edit instructions")
    edit.add_argument("--output", required=True, help="Output file path")
    edit.add_argument("--aspect", default=None, help="Override aspect ratio (default: match input)")
    edit.add_argument("--size", default="1K", help=f"Image size (default: 1K)")
    edit.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model (default: {DEFAULT_MODEL})")

    # --- compose ---
    comp = subparsers.add_parser("compose", help="Compose from multiple reference images (up to 14)")
    comp.add_argument("--inputs", nargs="+", required=True, help="Input image paths (up to 14)")
    comp.add_argument("--prompt", required=True, help="Composition instructions")
    comp.add_argument("--output", required=True, help="Output file path")
    comp.add_argument("--aspect", default="16:9", help="Aspect ratio (default: 16:9)")
    comp.add_argument("--size", default="1K", help="Image size (default: 1K)")
    comp.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model (default: {DEFAULT_MODEL})")

    # --- batch ---
    batch = subparsers.add_parser("batch", help="Batch generate from a prompts JSON file")
    batch.add_argument("--prompts-file", required=True, help="JSON file with prompts")
    batch.add_argument("--output-dir", required=True, help="Output directory")
    batch.add_argument("--aspect", default="16:9", help="Default aspect ratio (default: 16:9)")
    batch.add_argument("--size", default="1K", help="Default image size (default: 1K)")
    batch.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model (default: {DEFAULT_MODEL})")
    batch.add_argument("--search", action="store_true", help="Enable Google Search grounding")

    # --- check ---
    subparsers.add_parser("check", help="Verify environment setup and connectivity")

    args = parser.parse_args()

    if args.command == "generate":
        result = generate_image(
            prompt=args.prompt,
            output_path=args.output,
            aspect_ratio=args.aspect,
            image_size=args.size,
            model=args.model,
            web_search=args.web_search or args.search,
            image_search=args.image_search,
            thinking=args.thinking,
            image_only=args.image_only,
        )
    elif args.command == "edit":
        result = edit_image(
            input_path=args.input,
            prompt=args.prompt,
            output_path=args.output,
            aspect_ratio=args.aspect,
            image_size=args.size,
            model=args.model,
        )
    elif args.command == "compose":
        result = compose_images(
            input_paths=args.inputs,
            prompt=args.prompt,
            output_path=args.output,
            aspect_ratio=args.aspect,
            image_size=args.size,
            model=args.model,
        )
    elif args.command == "batch":
        result = batch_generate(
            prompts_file=args.prompts_file,
            output_dir=args.output_dir,
            aspect_ratio=args.aspect,
            image_size=args.size,
            model=args.model,
            search_grounding=args.search,
        )
    elif args.command == "check":
        result = check_setup()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()