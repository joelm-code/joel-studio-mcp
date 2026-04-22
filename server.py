#!/usr/bin/env python3
"""
joel-studio-mcp — FastMCP server wrapping Nano Banana image/video generation.

Tools:
    generate_image   — text → image
    edit_image       — text + image → edited image
    compose_images   — up to 14 reference images → new image
    batch_generate   — JSON prompts file → multiple images
    generate_video   — text/image → video (Veo)
    extend_video     — extend a Veo-generated video
    check_setup      — verify API key, deps, connectivity

Resources:
    models://image   — available image models
    models://video   — available video models
    config://options — valid aspects, sizes, durations

Prompts:
    image_prompt_template  — scaffolds a rich photorealistic prompt
    batch_prompts_template — scaffolds a valid prompts.json for batch generation
"""

import json
import os
import sys
from pathlib import Path

# Ensure scirpt.py (same directory) is importable
sys.path.insert(0, str(Path(__file__).parent))

from fastmcp import FastMCP, Context
from fastmcp.server.middleware.logging import LoggingMiddleware
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware

import scirpt as studio

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "joel-studio-mcp",
    instructions=(
        "joel-studio-mcp gives you full access to Nano Banana image and video generation "
        "via the Gemini API. Use generate_image for text-to-image, edit_image to modify an "
        "existing image, compose_images to blend up to 14 references, batch_generate for "
        "bulk frame creation, and generate_video / extend_video for Veo video generation. "
        "Read models://image and config://options to see available models and settings "
        "before calling tools."
    ),
)

mcp.add_middleware(ErrorHandlingMiddleware(include_traceback=True))
mcp.add_middleware(LoggingMiddleware())


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def generate_image(
    prompt: str,
    output_path: str,
    aspect_ratio: str = "16:9",
    image_size: str = "1K",
    model: str = studio.DEFAULT_MODEL,
    web_search: bool = False,
    image_search: bool = False,
    thinking: str = None,
    image_only: bool = False,
    ctx: Context = None,
) -> dict:
    """Generate an image from a text prompt using Nano Banana (Gemini image models).

    Args:
        prompt: Descriptive text prompt. Narrative descriptions work better than keyword lists.
        output_path: Where to save the image (.png, .jpg, .webp). Parent dirs created automatically.
        aspect_ratio: One of 1:1, 3:2, 16:9, 9:16, 4:3, 2:3, 21:9, etc. Default 16:9.
        image_size: 512px, 1K, 2K, or 4K. 2K/4K require gemini-3.x models. Default 1K.
        model: gemini-3.1-flash-image-preview (fast), gemini-3-pro-image-preview (pro/thinking),
               or gemini-2.5-flash-image (speed fallback).
        web_search: Ground generation with live Google Web Search data.
        image_search: Ground with Google Image Search (3.1 Flash only).
        thinking: None (default), "minimal", or "high" — controls reasoning depth (3.x only).
        image_only: If True, suppress text response and return image only.
    """
    if ctx:
        await ctx.info(f"Generating {image_size} {aspect_ratio} image with {model}...")

    result = studio.generate_image(
        prompt=prompt,
        output_path=output_path,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        model=model,
        web_search=web_search,
        image_search=image_search,
        thinking=thinking,
        image_only=image_only,
    )

    if ctx:
        if result.get("status") == "success":
            await ctx.info(f"Saved {result.get('width')}x{result.get('height')} → {output_path}")
        else:
            await ctx.warning(f"Generation result: {result.get('status')}")

    return result


@mcp.tool()
async def edit_image(
    input_path: str,
    prompt: str,
    output_path: str,
    aspect_ratio: str = None,
    image_size: str = "1K",
    model: str = studio.DEFAULT_MODEL,
    ctx: Context = None,
) -> dict:
    """Edit an existing image using a text prompt.

    Provide an image and describe what to change — add/remove elements, restyle,
    adjust color grading, change lighting, etc. The model infers the aspect ratio
    from the source image unless you override it.

    Args:
        input_path: Path to the source image to edit.
        prompt: Edit instructions (e.g. "make it warmer", "remove the background").
        output_path: Where to save the edited image.
        aspect_ratio: Override aspect ratio. Leave None to match source image.
        image_size: 512px, 1K, 2K, or 4K. Default 1K.
        model: Gemini image model to use.
    """
    if ctx:
        await ctx.info(f"Editing {input_path} with prompt: {prompt[:60]}...")

    result = studio.edit_image(
        input_path=input_path,
        prompt=prompt,
        output_path=output_path,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        model=model,
    )

    if ctx:
        if result.get("status") == "success":
            await ctx.info(f"Saved edited image → {output_path}")
        else:
            await ctx.warning(f"Edit result: {result.get('status')} — {result.get('error', '')}")

    return result


@mcp.tool()
async def compose_images(
    input_paths: list,
    prompt: str,
    output_path: str,
    aspect_ratio: str = "16:9",
    image_size: str = "1K",
    model: str = studio.DEFAULT_MODEL,
    ctx: Context = None,
) -> dict:
    """Compose a new image from multiple reference images (up to 14).

    Mix object references and character references to produce a new image that
    incorporates elements from all inputs.

    Flash (3.1): up to 10 object + 4 character references.
    Pro: up to 6 object + 5 character references.

    Args:
        input_paths: List of paths to reference images (up to 14).
        prompt: Instructions describing how to combine the references.
        output_path: Where to save the composed image.
        aspect_ratio: Output aspect ratio. Default 16:9.
        image_size: Output size. Default 1K.
        model: Gemini image model to use.
    """
    if ctx:
        await ctx.info(f"Composing {len(input_paths)} reference images...")

    result = studio.compose_images(
        input_paths=input_paths,
        prompt=prompt,
        output_path=output_path,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        model=model,
    )

    if ctx:
        if result.get("status") == "success":
            await ctx.info(f"Composed image saved → {output_path}")
        else:
            await ctx.warning(f"Compose result: {result.get('status')} — {result.get('error', '')}")

    return result


@mcp.tool()
async def batch_generate(
    prompts_file: str,
    output_dir: str,
    aspect_ratio: str = "16:9",
    image_size: str = "1K",
    model: str = studio.DEFAULT_MODEL,
    web_search: bool = False,
    ctx: Context = None,
) -> dict:
    """Generate multiple images from a JSON prompts file.

    The prompts file must be a JSON array. Each item needs a "prompt" field
    and optionally "id", "aspect", and "size" for per-frame overrides.

    Example prompts.json:
        [
            {"id": "frame-01", "prompt": "sunrise over mountains"},
            {"id": "frame-02", "prompt": "midday forest", "aspect": "9:16", "size": "2K"}
        ]

    Args:
        prompts_file: Path to the JSON prompts file.
        output_dir: Directory where generated images will be saved.
        aspect_ratio: Default aspect ratio (can be overridden per frame). Default 16:9.
        image_size: Default image size (can be overridden per frame). Default 1K.
        model: Gemini image model to use for all frames.
        web_search: Enable Google Web Search grounding for all frames.
    """
    if ctx:
        try:
            import json as _json
            prompts = _json.loads(Path(prompts_file).read_text())
            await ctx.info(f"Starting batch: {len(prompts)} images → {output_dir}")
        except Exception:
            await ctx.info(f"Starting batch generation → {output_dir}")

    result = studio.batch_generate(
        prompts_file=prompts_file,
        output_dir=output_dir,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        model=model,
        search_grounding=web_search,
    )

    if ctx:
        await ctx.info(
            f"Batch complete: {result.get('success')}/{result.get('total')} succeeded"
        )

    return result


@mcp.tool()
async def generate_video(
    prompt: str,
    output_path: str,
    model: str = studio.DEFAULT_VIDEO_MODEL,
    aspect_ratio: str = "16:9",
    resolution: str = "720p",
    duration: int = 8,
    image_path: str = None,
    last_frame_path: str = None,
    ctx: Context = None,
) -> dict:
    """Generate a video using Veo (Google's video generation model).

    Supports three modes:
    - Text-to-video: prompt only
    - Image-to-video: prompt + image_path (first frame)
    - Interpolation: prompt + image_path (first frame) + last_frame_path (last frame)

    Note: Video generation takes 1–5 minutes. The tool polls until complete.

    Args:
        prompt: Description of the video to generate.
        output_path: Where to save the video (.mp4).
        model: veo-3.1-generate-preview (best, has audio), veo-3.1-fast-generate-preview,
               or veo-2.0-generate-001 (stable, no audio).
        aspect_ratio: 16:9 (landscape) or 9:16 (portrait/vertical).
        resolution: 720p, 1080p, or 4k. Default 720p.
        duration: Video length in seconds — 4, 6, or 8. Default 8.
        image_path: Optional first frame image for image-to-video.
        last_frame_path: Optional last frame for interpolation (requires image_path too).
    """
    if ctx:
        await ctx.info(
            f"Starting video generation: {duration}s {resolution} {aspect_ratio} with {model}..."
        )
        await ctx.info("This will take 1–5 minutes, polling until complete.")

    result = studio.generate_video(
        prompt=prompt,
        output_path=output_path,
        model=model,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        duration=duration,
        image_path=image_path,
        last_frame_path=last_frame_path,
    )

    if ctx:
        if result.get("status") == "success":
            await ctx.info(f"Video saved → {output_path}")
        else:
            await ctx.warning(f"Video result: {result.get('status')} — {result.get('error', '')}")

    return result


@mcp.tool()
async def extend_video(
    video_path: str,
    prompt: str,
    output_path: str,
    model: str = studio.DEFAULT_VIDEO_MODEL,
    ctx: Context = None,
) -> dict:
    """Extend a previously Veo-generated video by ~7 seconds.

    The source video must be Veo-generated, 720p, under 141 seconds long,
    and generated within the last 2 days.

    Args:
        video_path: Path to the source Veo-generated video.
        prompt: Description of how to continue the video.
        output_path: Where to save the extended video.
        model: Veo model to use (should match the original generation model).
    """
    if ctx:
        await ctx.info(f"Extending video {video_path} by ~7 seconds...")

    result = studio.extend_video(
        video_path=video_path,
        prompt=prompt,
        output_path=output_path,
        model=model,
    )

    if ctx:
        if result.get("status") == "success":
            await ctx.info(f"Extended video saved → {output_path}")
        else:
            await ctx.warning(f"Extend result: {result.get('status')} — {result.get('error', '')}")

    return result


@mcp.tool()
async def check_setup(ctx: Context = None) -> dict:
    """Verify the environment: API key, installed packages, and Gemini API connectivity.

    Returns status of each component and whether the server is ready to generate.
    Run this first if you encounter errors.
    """
    if ctx:
        await ctx.info("Checking setup...")

    result = studio.check_setup()

    if ctx:
        status = result.get("status")
        connectivity = result.get("connectivity", "not tested")
        await ctx.info(f"Setup status: {status} | connectivity: {connectivity}")

    return result


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@mcp.resource("models://image")
def image_models() -> str:
    """Available Nano Banana image generation models with aliases and capabilities."""
    models = [
        {
            "name": "gemini-3.1-flash-image-preview",
            "alias": "Nano Banana 2",
            "tier": "fast, go-to default",
            "features": ["up to 4K", "web search", "image search grounding", "up to 14 refs", "thinking minimal/high"],
        },
        {
            "name": "gemini-3-pro-image-preview",
            "alias": "Nano Banana Pro",
            "tier": "professional, best quality",
            "features": ["up to 4K", "web search grounding", "up to 14 refs", "advanced thinking"],
        },
        {
            "name": "gemini-2.5-flash-image",
            "alias": "Nano Banana",
            "tier": "speed/efficiency fallback",
            "features": ["1K only", "fast, low-latency"],
        },
    ]
    return json.dumps({"default": studio.DEFAULT_MODEL, "models": models}, indent=2)


@mcp.resource("models://video")
def video_models() -> str:
    """Available Veo video generation models with aliases and capabilities."""
    models = [
        {
            "name": "veo-3.1-generate-preview",
            "alias": "Veo 3.1",
            "tier": "best quality",
            "features": ["audio generation", "720p–4K", "8s max", "image-to-video", "interpolation"],
        },
        {
            "name": "veo-3.1-fast-generate-preview",
            "alias": "Veo 3.1 Fast",
            "tier": "fast, good quality",
            "features": ["720p–4K", "faster generation"],
        },
        {
            "name": "veo-2.0-generate-001",
            "alias": "Veo 2",
            "tier": "stable",
            "features": ["720p only", "no audio", "most stable"],
        },
    ]
    return json.dumps({"default": studio.DEFAULT_VIDEO_MODEL, "models": models}, indent=2)


@mcp.resource("config://options")
def config_options() -> str:
    """All valid configuration options: aspect ratios, image sizes, video resolutions, durations."""
    return json.dumps({
        "image": {
            "aspect_ratios": studio.VALID_ASPECTS,
            "sizes": studio.VALID_SIZES,
            "note": "2K/4K require gemini-3.x models. 512px only on gemini-3.1-flash.",
        },
        "video": {
            "aspect_ratios": studio.VALID_VIDEO_ASPECTS,
            "resolutions": studio.VALID_VIDEO_RESOLUTIONS,
            "durations_seconds": studio.VALID_VIDEO_DURATIONS,
            "note": "4K resolution requires veo-3.1 models.",
        },
    }, indent=2)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

@mcp.prompt()
def image_prompt_template(
    subject: str,
    style: str = "photorealistic",
    aspect_ratio: str = "16:9",
    size: str = "2K",
) -> str:
    """Scaffold a rich, detailed image generation prompt.

    Args:
        subject: What to generate (e.g. "elderly ceramicist in a workshop").
        style: Visual style — photorealistic, illustration, isometric, sketch, sticker, etc.
        aspect_ratio: Target aspect ratio (used as a hint in the prompt).
        size: Target size — 1K, 2K, or 4K.
    """
    return f"""Generate a {style} image of: {subject}

Suggested parameters:
- aspect_ratio: {aspect_ratio}
- image_size: {size}
- model: gemini-3.1-flash-image-preview

Prompt writing tips for best results:
- Describe the scene narratively, not as a keyword list
- Include: lighting conditions, camera/lens details, mood, textures
- For photorealistic: mention shot type (close-up, aerial, wide), lens (85mm, 24mm), lighting (golden hour, studio, overcast)
- For illustrations: specify art style, line style, color palette, background color
- For stickers/icons: request white background (transparent not supported)

Example rich prompt for "{subject}":
A {style} [shot type] of {subject}, [action or expression], set in [environment].
The scene is illuminated by [lighting], creating a [mood] atmosphere.
[Camera/lens details]. The image should be {aspect_ratio} orientation."""


@mcp.prompt()
def batch_prompts_template(
    frames: int = 5,
    theme: str = "cinematic landscape sequence",
    aspect_ratio: str = "16:9",
    size: str = "1K",
) -> str:
    """Scaffold a prompts.json file for batch image generation.

    Args:
        frames: Number of frames to scaffold.
        theme: Overall theme or narrative for the sequence.
        aspect_ratio: Default aspect ratio for all frames.
        size: Default image size for all frames.
    """
    example_frames = []
    for i in range(1, frames + 1):
        example_frames.append({
            "id": f"frame-{i:02d}",
            "prompt": f"[Frame {i} of {frames}] {theme} — describe this specific frame here",
            "aspect": aspect_ratio,
            "size": size,
        })

    return f"""Create a prompts.json file for a {frames}-frame batch generation with the theme: "{theme}"

Save this as prompts.json, fill in each frame's prompt, then call batch_generate:
- prompts_file: "/path/to/prompts.json"
- output_dir: "/path/to/output/frames/"
- aspect_ratio: "{aspect_ratio}" (default, can override per frame)
- image_size: "{size}" (default, can override per frame)

Template:
{json.dumps(example_frames, indent=2)}

Tips:
- Each frame prompt should describe a specific moment, not just the overall theme
- Use consistent style keywords across frames for visual coherence
- Override "aspect" or "size" per frame for mixed-format sequences"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # HTTP mode when running on Railway (or any cloud), stdio for local Claude Desktop
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "http":
        import uvicorn
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request
        from starlette.responses import JSONResponse

        port = int(os.getenv("PORT", 8000))

        # Intercept /health before it reaches FastMCP
        class HealthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                if request.url.path == "/health":
                    return JSONResponse({"status": "ok", "server": "joel-studio-mcp"})
                return await call_next(request)

        app = mcp.http_app(transport="streamable-http")
        app.add_middleware(HealthMiddleware)

        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        mcp.run()
