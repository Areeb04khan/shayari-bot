# -*- coding: utf-8 -*-
"""
Instagram Shayari Bot v6.1 (Multi-Tier Failover Engine)
- AI Chain: Gemini -> OpenRouter -> Groq -> NVIDIA NIM
- TTS Chain: ElevenLabs -> OpenRouter TTS -> Edge-TTS (Urdu)
- Media Host Chain: tmpfiles.org -> tempfile.org
"""

import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = getattr(PIL.Image, 'Resampling', PIL.Image).LANCZOS

from google import genai
from google.genai import types
from openai import OpenAI
import requests
import json
import os
import sys
import time
import textwrap
import random
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(line_buffering=True)

# ============================================================
# CONFIGURATION & ENVIRONMENT
# ============================================================
GEMINI_API_KEY         = os.environ.get("GEMINI_API_KEY", "")
OPENROUTER_API_KEY     = os.environ.get("OPENROUTER_API_KEY", "")
GROQ_API_KEY           = os.environ.get("GROQ_API_KEY", "")
NVIDIA_API_KEY         = os.environ.get("NVIDIA_API_KEY", "")
ELEVENLABS_API_KEY     = os.environ.get("ELEVENLABS_API_KEY", "")

INSTAGRAM_ACCESS_TOKEN = os.environ.get("INSTAGRAM_ACCESS_TOKEN", "")
INSTAGRAM_USER_ID      = os.environ.get("INSTAGRAM_USER_ID", "")
PEXELS_API_KEY         = os.environ.get("PEXELS_API_KEY", "")
UNSPLASH_ACCESS_KEY    = os.environ.get("UNSPLASH_ACCESS_KEY", "")
POST_TYPE              = os.environ.get("POST_TYPE", "photo").lower()
IG_HANDLE              = "@ak_apak"

# --- NEW: free, reliable media host (replaces envs.sh / 0x0.st) ---------
# GITHUB_TOKEN: the auto-generated Actions token (NOT a secret you create
#   yourself). It costs nothing and needs no sign-up -- we just have to
#   pass it into this step's `env:` block in main.yml (done below).
# GITHUB_REPOSITORY: set automatically by GitHub Actions on every run as
#   "owner/repo", e.g. "Areeb04khan/shayari-bot". Nothing to configure.
GITHUB_TOKEN            = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPOSITORY       = os.environ.get("GITHUB_REPOSITORY", "")
MEDIA_RELEASE_TAG        = "media-cache"  # fixed tag we reuse every run purely as free file storage

REQUIRED_ENV_VARS = ["INSTAGRAM_ACCESS_TOKEN", "INSTAGRAM_USER_ID", "GITHUB_TOKEN"]

def validate_environment():
    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        print(f"❌ FATAL: Missing required secret(s): {', '.join(missing)}")
        sys.exit(1)
    if not any([GEMINI_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY, NVIDIA_API_KEY]):
        print("❌ FATAL: At least one AI API key must be provided!")
        sys.exit(1)

FONT_SERIF  = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
FONT_ITALIC = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf"
FONT_SANS   = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

POET_SCHEDULE = [
    {"name": "Mirza Ghalib", "era": "1797-1869"},
    {"name": "Mir Taqi Mir", "era": "1723-1810"},
    {"name": "Jaun Elia", "era": "1931-2002"},
    {"name": "Faiz Ahmed Faiz", "era": "1911-1984"},
    {"name": "Ahmad Faraz", "era": "1931-2008"},
    {"name": "Parveen Shakir", "era": "1952-1994"},
    {"name": "Rahat Indori", "era": "1950-2020"},
    {"name": "Gulzar", "era": "1934-"}
]

EMOTION_PALETTES = {
    "ishq":     {"bg":"#1a0010","text":"#f5c6d0","accent":"#e8587a","sub":"#b03060"},
    "dard":     {"bg":"#0a0a1a","text":"#c8d4e8","accent":"#7090d0","sub":"#405080"},
    "tanhai":   {"bg":"#060d0d","text":"#b8d8d8","accent":"#40a0a0","sub":"#206060"}
}
DEFAULT_PALETTE = EMOTION_PALETTES["dard"]

VIRAL_HOOKS = [
    "Read this twice if you're missing someone silently...",
    "When {poet_name} said this, it hit differently...",
    "For the nights when words fail you...",
    "Send this to someone you can't text anymore."
]

# ============================================================
# MULTI-TIER AI CONTENT GENERATOR (Failover Chain)
# ============================================================
def generate_content(poet: dict) -> dict:
    hook = random.choice(VIRAL_HOOKS).format(poet_name=poet['name'])
    prompt = (
        f"You run a high-engagement Instagram Shayari page.\n"
        f"Poet: {poet['name']} ({poet['era']})\n\n"
        "RULES:\n"
        f"1. Give ONE famous 2-line couplet (sher) strictly by {poet['name']}.\n"
        "2. Roman Urdu transliteration (sher_roman) - MAX 2 lines.\n"
        "3. Exact Urdu script (sher_urdu) for audio synthesis.\n"
        "4. Poetic English translation (english_translation) - MAX 1 line.\n"
        "5. Search query (search_query) for dark aesthetic background imagery (e.g., 'dark rain night', 'misty road').\n"
        "6. Short caption story.\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        f'  "hook": "{hook}",\n'
        '  "sher_roman": "Line 1\\nLine 2",\n'
        '  "sher_urdu": "...",\n'
        '  "english_translation": "...",\n'
        '  "emotion": "dard",\n'
        '  "search_query": "dark rain night",\n'
        '  "caption": "..."\n'
        "}"
    )

    if GEMINI_API_KEY:
        try:
            print(f"🧠 [1/4] Querying Gemini AI for poet: {poet['name']}...")
            client = genai.Client(
                api_key=GEMINI_API_KEY,
                http_options=types.HttpOptions(
                    retry_options=types.HttpRetryOptions(attempts=1)
                )
            )
            response = client.models.generate_content(
                model="gemini-3.5-flash", 
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.7)
            )
            raw = response.text.strip().replace("```json","").replace("```","").strip()
            data = json.loads(raw)
            print("✅ Generated Sher successfully via Gemini!")
            return data
        except Exception as e:
            print(f"⚠️ Gemini failed ({e}). Moving to Fallback Chain...")

    fallbacks = [
        {"name": "OpenRouter", "api_key": OPENROUTER_API_KEY, "base_url": "https://openrouter.ai/api/v1", "model": "openrouter/free"},
        {"name": "Groq", "api_key": GROQ_API_KEY, "base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
        {"name": "NVIDIA NIM", "api_key": NVIDIA_API_KEY, "base_url": "https://integrate.api.nvidia.com/v1", "model": "meta/llama-3.1-70b-instruct"}
    ]

    for index, provider in enumerate(fallbacks, start=2):
        if not provider["api_key"]:
            continue
        try:
            print(f"🔄 [{index}/4] Trying {provider['name']} Fallback...")
            client = OpenAI(base_url=provider["base_url"], api_key=provider["api_key"])
            response = client.chat.completions.create(
                model=provider["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            raw = response.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
            data = json.loads(raw)
            print(f"✅ Generated Sher successfully via {provider['name']}!")
            return data
        except Exception as err:
            print(f"⚠️ {provider['name']} failed: {err}")

    print("❌ FATAL: All AI providers failed.")
    sys.exit(1)

# ============================================================
# MEDIA ENGINE (Unsplash + Pexels)
# ============================================================
def fetch_unsplash_photo(query: str) -> str:
    if not UNSPLASH_ACCESS_KEY: return None
    try:
        headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
        url = f"https://api.unsplash.com/photos/random?query={query}&orientation=squarish"
        res = requests.get(url, headers=headers, timeout=10)
        if res.ok:
            img_url = res.json().get("urls", {}).get("regular")
            if img_url:
                p_path = f"output/unsplash_{int(time.time())}.jpg"
                with open(p_path, "wb") as f:
                    f.write(requests.get(img_url, timeout=30).content)
                return p_path
    except Exception:
        pass
    return None

def fetch_pexels_photo(query: str) -> str:
    if not PEXELS_API_KEY: return None
    try:
        headers = {"Authorization": PEXELS_API_KEY}
        url = f"https://api.pexels.com/v1/search?query={query}&orientation=square&per_page=5"
        res = requests.get(url, headers=headers, timeout=10)
        if res.ok:
            photos = res.json().get("photos", [])
            if photos:
                img_url = random.choice(photos).get("src", {}).get("large2x")
                if img_url:
                    p_path = f"output/pexels_img_{int(time.time())}.jpg"
                    with open(p_path, "wb") as f:
                        f.write(requests.get(img_url, timeout=30).content)
                    return p_path
    except Exception:
        pass
    return None

def get_photo_background(query: str) -> str:
    os.makedirs("output", exist_ok=True)
    for fetch_func in [fetch_unsplash_photo, fetch_pexels_photo]:
        path = fetch_func(query)
        if path and os.path.exists(path): return path
    return None

def fetch_pexels_video(query: str) -> str:
    if not PEXELS_API_KEY: return None
    try:
        headers = {"Authorization": PEXELS_API_KEY}
        url = f"https://api.pexels.com/videos/search?query={query}&orientation=portrait&per_page=5"
        res = requests.get(url, headers=headers, timeout=10)
        if res.ok:
            videos = res.json().get("videos", [])
            if videos:
                video = random.choice(videos)
                for vf in video.get("video_files", []):
                    if vf.get("file_type") == "video/mp4" and vf.get("width", 0) >= 720:
                        v_path = f"output/pexels_vid_{int(time.time())}.mp4"
                        with open(v_path, "wb") as f:
                            f.write(requests.get(vf["link"], timeout=30).content)
                        return v_path
    except Exception:
        pass
    return None

def fetch_unsplash_video_equivalent(query: str) -> str:
    if not UNSPLASH_ACCESS_KEY: return None
    try:
        headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
        url = f"https://api.unsplash.com/photos/random?query={query}&orientation=portrait"
        res = requests.get(url, headers=headers, timeout=10)
        if res.ok:
            img_url = res.json().get("urls", {}).get("regular")
            if img_url:
                p_path = f"output/unsplash_portrait_{int(time.time())}.jpg"
                with open(p_path, "wb") as f:
                    f.write(requests.get(img_url, timeout=30).content)
                return p_path
    except Exception:
        pass
    return None

def get_reel_background(query: str) -> tuple:
    os.makedirs("output", exist_ok=True)
    v_path = fetch_pexels_video(query)
    if v_path: return (v_path, True)
    u_path = fetch_unsplash_video_equivalent(query)
    if u_path: return (u_path, False)
    return (None, False)

# ============================================================
# PHOTO COMPOSITOR
# ============================================================
def create_photo_image(data: dict, poet: dict) -> str:
    print("🎨 Rendering 1080x1080 Photo Image...")
    W, H = 1080, 1080
    palette = EMOTION_PALETTES.get(data.get("emotion","dard"), DEFAULT_PALETTE)
    bg_photo_path = get_photo_background(data.get("search_query", "dark rain"))

    if bg_photo_path and os.path.exists(bg_photo_path):
        base_img = Image.open(bg_photo_path).convert("RGBA").resize((W, H), Image.Resampling.LANCZOS)
        dark_overlay = Image.new("RGBA", (W, H), (10, 10, 20, 160))
        img = Image.alpha_composite(base_img, dark_overlay).convert("RGB")
    else:
        img = Image.new("RGB", (W, H), color=palette["bg"])

    draw = ImageDraw.Draw(img)
    try:
        font_poet  = ImageFont.truetype(FONT_SERIF, 32)
        font_sher  = ImageFont.truetype(FONT_SERIF, 44)
        font_trans = ImageFont.truetype(FONT_ITALIC, 22)
        font_brand = ImageFont.truetype(FONT_SANS, 18)
    except:
        font_poet = font_sher = font_trans = font_brand = ImageFont.load_default()

    def center(text, y, font, color):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((W - tw) / 2, y), text, font=font, fill=color)

    center(f"-- {poet['name']} --", 120, font_poet, "#E0C080")
    lines = data["sher_roman"].strip().split("\n")
    y_pos = 380
    for line in lines:
        for wline in textwrap.wrap(line, width=32):
            center(wline, y_pos, font_sher, "#FFFFFF")
            y_pos += 65

    y_pos += 50
    for tline in textwrap.wrap(f'"{data["english_translation"]}"', width=48):
        center(tline, y_pos, font_trans, "#D0D0D0")
        y_pos += 35

    center(IG_HANDLE, 960, font_brand, "#AAAAAA")
    fname = f"output/photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    img.save(fname, "JPEG", quality=95)
    return fname

# ============================================================
# MULTI-TIER TTS FAILOVER ENGINE (Urdu)
# ============================================================
def generate_tts(data: dict) -> list:
    lines = [line.strip() for line in data["sher_urdu"].split("\n") if line.strip()]
    if not lines: lines = [data["sher_urdu"]]
    output_paths = []

    if ELEVENLABS_API_KEY:
        try:
            print("🎙️ [TTS 1/3] Trying ElevenLabs...")
            from elevenlabs.client import ElevenLabs
            client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
            for i, line in enumerate(lines):
                out_path = f"output/tts_line_{i}_{int(time.time())}.mp3"
                audio_stream = client.text_to_speech.convert(
                    text=line, voice_id="N2lVS1w4EtoT3dr4eOWO",
                    model_id="eleven_multilingual_v2", output_format="mp3_44100_128"
                )
                with open(out_path, "wb") as f:
                    for chunk in audio_stream:
                        if chunk: f.write(chunk)
                output_paths.append(out_path)
            print("✅ ElevenLabs Audio generated successfully!")
            return output_paths
        except Exception as e:
            print(f"⚠️ ElevenLabs failed ({e}). Moving to OpenRouter TTS...")

    if OPENROUTER_API_KEY:
        try:
            print("🎙️ [TTS 2/3] Trying OpenRouter TTS...")
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
            for i, line in enumerate(lines):
                out_path = f"output/tts_line_{i}_{int(time.time())}.mp3"
                response = client.audio.speech.create(
                    model="fish-audio/s2.1-pro-free:free",
                    voice="alloy",
                    input=line
                )
                response.stream_to_file(out_path)
                output_paths.append(out_path)
            print("✅ OpenRouter TTS Audio generated successfully!")
            return output_paths
        except Exception as e:
            print(f"⚠️ OpenRouter TTS failed ({e}). Moving to Edge-TTS...")

    try:
        print("🎙️ [TTS 3/3] Generating fallback via Edge-TTS...")
        import asyncio
        import edge_tts
        async def _speak():
            for i, line in enumerate(lines):
                out_path = f"output/tts_line_{i}_{int(time.time())}.mp3"
                communicate = edge_tts.Communicate(line, "ur-PK-AsadNeural", rate="-15%", pitch="-6Hz")
                await communicate.save(out_path)
                output_paths.append(out_path)
        asyncio.run(_speak())
        print("✅ Edge-TTS Audio generated successfully!")
        return output_paths
    except Exception as e:
        print(f"❌ FATAL: All TTS providers failed: {e}")
        return []

# ============================================================
# REEL COMPOSITOR
# ============================================================
def create_reel_video(data: dict, poet: dict, tts_paths: list) -> str:
    print("🎬 Compositing 1080x1920 Reel Video with MoviePy...")
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, ImageClip, concatenate_audioclips
        import numpy as np

        audio_clips = []
        base_clip = AudioFileClip(tts_paths[0])
        silence = base_clip.subclip(0, min(0.1, base_clip.duration)).volumex(0)
        silence = concatenate_audioclips([silence] * 10)

        for i, path in enumerate(tts_paths):
            audio_clips.append(AudioFileClip(path))
            if i < len(tts_paths) - 1: audio_clips.append(silence)

        tts_audio = concatenate_audioclips(audio_clips)
        duration = min(tts_audio.duration + 3, 30)

        bg_path, is_video = get_reel_background(data.get("search_query", "dark rain"))
        if bg_path and is_video:
            bg_clip = VideoFileClip(bg_path).subclip(0, duration).resize(height=1920)
            if bg_clip.w < 1080: bg_clip = bg_clip.resize(width=1080)
            bg_clip = bg_clip.crop(x_center=bg_clip.w/2, y_center=bg_clip.h/2, width=1080, height=1920)
            bg_clip = bg_clip.fl_image(lambda image: (image * 0.4).astype(np.uint8))
        elif bg_path and not is_video:
            bg_img = Image.open(bg_path).convert("RGBA").resize((1080, 1920), Image.Resampling.LANCZOS)
            dark_overlay = Image.new("RGBA", (1080, 1920), (10, 10, 20, 160))
            bg_img = Image.alpha_composite(bg_img, dark_overlay).convert("RGB")
            bg_img_path = f"output/reel_bg_img_{int(time.time())}.jpg"
            bg_img.save(bg_img_path)
            bg_clip = ImageClip(bg_img_path, duration=duration)
        else:
            clean_bg = Image.new("RGB", (1080, 1920), color="#0a0a14")
            clean_bg_path = f"output/clean_bg_{int(time.time())}.jpg"
            clean_bg.save(clean_bg_path)
            bg_clip = ImageClip(clean_bg_path, duration=duration)

        overlay_img = Image.new("RGBA", (1080, 1920), (0,0,0,0))
        draw = ImageDraw.Draw(overlay_img)
        try:
            font_hook = ImageFont.truetype(FONT_ITALIC, 32)
            font_sher = ImageFont.truetype(FONT_SERIF, 48)
            font_poet = ImageFont.truetype(FONT_SERIF, 32)
        except:
            font_hook = font_sher = font_poet = ImageFont.load_default()

        hook_text = textwrap.fill(data.get("hook", ""), width=32)
        draw.text((540, 380), hook_text, font=font_hook, fill="#E0E0E0", anchor="mm", align="center")

        sher_lines = data["sher_roman"].strip().split("\n")
        wrapped_lines = []
        for line in sher_lines: wrapped_lines.extend(textwrap.wrap(line, width=30))
        final_sher_text = "\n".join(wrapped_lines)
        draw.text((540, 960), final_sher_text, font=font_sher, fill="#FFFFFF", anchor="mm", align="center", spacing=24)
        draw.text((540, 1400), f"-- {poet['name']} --", font=font_poet, fill="#C0A060", anchor="mm")
        draw.text((540, 1750), IG_HANDLE, font=font_poet, fill="#888888", anchor="mm")

        overlay_fname = f"output/overlay_{int(time.time())}.png"
        overlay_img.save(overlay_fname)
        txt_clip = ImageClip(overlay_fname, duration=duration)

        final_video = CompositeVideoClip([bg_clip, txt_clip]).set_audio(tts_audio)
        reel_path = f"output/reel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        final_video.write_videofile(reel_path, fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        return reel_path
    except Exception as e:
        print(f"❌ Video render failure: {e}")
        return None

# ============================================================
# MEDIA UPLOADER  (GitHub Releases — free, no sign-up, no new secrets)
# ============================================================
# WHY THIS CHANGED:
#   envs.sh started failing DNS lookups, and 0x0.st has PERMANENTLY turned
#   off anonymous uploads (it told us so in the error: "uploads disabled
#   because it's been almost nothing but AI botnet spam"). Both were free
#   anonymous hosts with no guarantee they'd keep working -- exactly what
#   just happened. Instead, we now upload the photo/reel as an asset on a
#   GitHub "Release" in this SAME repo. That gives us a real, stable public
#   URL (a github.com / githubusercontent.com link) that Instagram's
#   servers can fetch -- and it's 100% free with your existing GitHub
#   account, since it's your own repo. No new account, no API key, no cost.
#
#   We reuse one fixed release (tagged "media-cache") as scratch space, and
#   delete each file right after Instagram has fetched it (see
#   delete_public_media below), so this never piles up storage over time.

def _gh_headers() -> dict:
    """Standard headers GitHub's API expects on every request."""
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

def get_or_create_media_release() -> dict:
    """
    Finds our dedicated 'media-cache' Release. If it doesn't exist yet
    (e.g. first time this fix runs), creates it automatically -- nothing
    for you to set up by hand on github.com.
    """
    api_base = f"https://api.github.com/repos/{GITHUB_REPOSITORY}"

    # Step 1: does it already exist? (After the very first run, it will.)
    res = requests.get(f"{api_base}/releases/tags/{MEDIA_RELEASE_TAG}", headers=_gh_headers(), timeout=15)
    if res.status_code == 200:
        return res.json()

    # Step 2: create it. "prerelease: True" + "make_latest: false" keep it
    # from ever showing up as your repo's "Latest release" badge -- it's
    # just storage, not a real software release.
    print("ℹ️ First run: creating one-time 'media-cache' Release for hosting...")
    res = requests.post(
        f"{api_base}/releases",
        headers=_gh_headers(),
        timeout=15,
        json={
            "tag_name": MEDIA_RELEASE_TAG,
            "name": "Media Cache (auto-managed, safe to ignore)",
            "body": "Used by poster.py as free temporary file storage so Instagram can fetch photos/reels. Files here are deleted right after each post.",
            "prerelease": True,
            "make_latest": "false",
        },
    )
    res.raise_for_status()  # if this fails, the error below tells us exactly why
    return res.json()

def upload_public_media(path: str) -> tuple:
    """
    Uploads one file to the media-cache Release.
    Returns (public_url, asset_id) -- asset_id is kept so we can delete
    the file again once Instagram is done with it.
    """
    release = get_or_create_media_release()
    filename = os.path.basename(path)
    content_type = "video/mp4" if filename.endswith(".mp4") else "image/jpeg"

    print(f"☁️ Uploading {filename} to GitHub Release (free host)...")
    upload_url = f"https://uploads.github.com/repos/{GITHUB_REPOSITORY}/releases/{release['id']}/assets"
    with open(path, "rb") as f:
        res = requests.post(
            upload_url,
            headers={**_gh_headers(), "Content-Type": content_type},
            params={"name": filename},   # asset filename; using our unique timestamped name avoids clashes
            data=f,                      # raw file bytes, not multipart -- this endpoint expects that
            timeout=120,
        )
    res.raise_for_status()
    asset = res.json()
    url = asset["browser_download_url"]
    print(f"✅ Hosted at: {url}")
    return url, asset["id"]

def delete_public_media(asset_id: int) -> None:
    """
    Best-effort cleanup: removes the file from the Release now that
    Instagram has already fetched it, so the repo doesn't slowly fill up
    with old reels/photos. If this ever fails, we just print a warning --
    it's not worth failing an otherwise-successful post over housekeeping.
    """
    try:
        requests.delete(
            f"https://api.github.com/repos/{GITHUB_REPOSITORY}/releases/assets/{asset_id}",
            headers=_gh_headers(),
            timeout=15,
        )
    except Exception as e:
        print(f"⚠️ Cleanup warning: couldn't delete temp media (asset {asset_id}): {e}")
    
# ============================================================
# INSTAGRAM PUBLISHER
# ============================================================
def post_to_instagram(media_path: str, caption: str, is_video: bool = False) -> bool:
    asset_id = None  # tracks the uploaded file so we can delete it in `finally` below
    try:
        media_url, asset_id = upload_public_media(media_path)
        payload = {"access_token": INSTAGRAM_ACCESS_TOKEN, "caption": caption}
        if is_video:
            payload["media_type"] = "REELS"
            payload["video_url"] = media_url
        else:
            payload["image_url"] = media_url

        print(f"📡 Creating Instagram Media Container...")
        c_res = requests.post(f"https://graph.instagram.com/v21.0/{INSTAGRAM_USER_ID}/media", data=payload).json()
        container_id = c_res.get("id")
        if not container_id: 
            print(f"❌ Instagram Container Error: {c_res}")
            return False

        if is_video:
            print(f"⏳ Waiting for Instagram to process Reel (ID: {container_id})...")
            for attempt in range(1, 21):
                time.sleep(10)
                status_res = requests.get(f"https://graph.instagram.com/v21.0/{container_id}?fields=status_code,status&access_token={INSTAGRAM_ACCESS_TOKEN}").json()
                status_code = status_res.get("status_code")
                print(f"   ↳ Attempt {attempt}: {status_code}")
                
                if status_code == "FINISHED": 
                    break
                elif status_code == "ERROR": 
                    print(f"❌ Instagram Processing Error: {status_res}")
                    return False
        else:
            time.sleep(15)

        print(f"🚀 Publishing Container to Instagram...")
        p_res = requests.post(f"https://graph.instagram.com/v21.0/{INSTAGRAM_USER_ID}/media_publish", data={"creation_id": container_id, "access_token": INSTAGRAM_ACCESS_TOKEN}).json()
        
        if "id" in p_res:
            print(f"✅ Published successfully! Post ID: {p_res['id']}")
            return True
        else:
            print(f"❌ Instagram Publish Error: {p_res}")
            return False
        
    except Exception as e:
        print(f"❌ Instagram Graph API / Upload Failure: {e}")
        return False
    finally:
        # Runs no matter what -- success, failure, or early "return False"
        # above. Instagram only needs the URL for a short window while it
        # fetches the file, so it's safe to remove right after this
        # function is done trying.
        if asset_id:
            delete_public_media(asset_id)

def load_progress() -> dict:
    if os.path.exists("progress.json"):
        with open("progress.json") as f: return json.load(f)
    return {"poet_index": 0, "total_posts": 0}

def save_progress(p: dict):
    with open("progress.json", "w") as f: json.dump(p, f, indent=2)

def run():
    validate_environment()
    p = load_progress()
    poet = POET_SCHEDULE[p["poet_index"] % len(POET_SCHEDULE)]

    print(f"\n🚀 STARTING WORKFLOW: [{POST_TYPE.upper()}] for {poet['name']}\n")
    data = generate_content(poet)
    caption = f"{data.get('hook','')}\n\n{data['sher_roman']}\n\n-- {poet['name']}\n\n{data.get('caption','')}\n\n#urdushayari #hindishayari #poetry"

    success = False
    if POST_TYPE == "photo":
        img_path = create_photo_image(data, poet)
        success = post_to_instagram(img_path, caption, is_video=False)
    elif POST_TYPE == "reel":
        os.makedirs("output", exist_ok=True)
        tts_paths = generate_tts(data)
        if not tts_paths: sys.exit(1)
        reel_path = create_reel_video(data, poet, tts_paths)
        if reel_path: success = post_to_instagram(reel_path, caption, is_video=True)
        else: sys.exit(1)

    if success:
        p["total_posts"] += 1
        p["poet_index"] += 1  # 👈 This ensures every single post generates brand new content!
        save_progress(p)
        print("\n✅ WORKFLOW COMPLETED SUCCESSFULLY!")
    else:
        sys.exit(1)

if __name__ == "__main__":
    run()
