# -*- coding: utf-8 -*-
"""
Instagram Shayari Bot -- CURRENT VERSION: v7.5

HOW TO READ THIS: every version bump below is one round of fixes, newest
at the bottom. Each line says what changed and why. run() prints
BOT_VERSION as the very first line of every job log (see bottom of this
file) -- so to check whether a given run used the latest code, just look
at that first line instead of re-comparing files by hand.

v6.1  - Original baseline (pre-fix). Its old header claimed "Media Host
        Chain: tmpfiles.org -> tempfile.org" but the actual code used
        envs.sh/0x0.st -- that header was already stale/inaccurate even
        before any of the changes below.
v7.0  - envs.sh (DNS failing) + 0x0.st (permanently disabled uploads,
        confirmed via its own error message) replaced with GitHub
        Releases as the media host -- free, self-owned, no new signup.
v7.1  - POET_SCHEDULE expanded 8 -> 40 poets (repeats were too frequent
        with only 8 in rotation).
v7.2  - POET_SCHEDULE expanded 40 -> 52 (added verified contemporary
        shayars + more classical names). Caption now marks the AI's own
        commentary with 💭, separate from the poet's actual quoted lines.
v7.3  - progress.json now remembers each poet's last 60 posted couplets
        (sher_history) and the prompt is told to avoid repeating them,
        with a difflib-based similarity check as a backup and up to 3
        retries. TTS speed moderated + real pause punctuation added
        before synthesis on all 3 TTS tiers.
v7.4  - Fixed a real bug in the OpenRouter TTS tier: `.create().stream_
        to_file()` doesn't actually write valid audio (a documented
        openai-python bug) -- switched to `.with_streaming_response.
        create()`. Added a validation check after every TTS tier so a
        bad file is caught immediately instead of crashing MoviePy
        several steps later. Fixed a second bug found in the process:
        all 3 TTS tiers were sharing one file list, so a fallback could
        silently mix audio from 2 different voices in one video.
v7.5  - THIS VERSION. Removed dead code: MEDIA_HOST, MAX_RETRIES, and
        RETRY_DELAY_SECONDS were being passed in from main.yml but never
        actually read anywhere in this file -- removed from both files.
        Added BOT_VERSION as a real constant, printed as the first line
        of every run -- so the job log itself now proves which version
        actually executed (multiple recent runs kept hitting the v7.4
        bug because the fixed file hadn't reached GitHub yet; this makes
        that gap visible immediately instead of needing another failed
        run + a manual raw-file fetch to notice).

AI Chain:    Gemini -> OpenRouter -> Groq -> NVIDIA NIM
TTS Chain:   ElevenLabs -> OpenRouter TTS -> Edge-TTS (Urdu)
Media Host:  GitHub Releases (this repo, free -- see v7.0 above)
"""
BOT_VERSION = "v7.5"

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
import re
import difflib
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

# Every poet has written far more than 60 couplets in their lifetime, so
# remembering their last 60 posted ones is plenty to avoid repeats without
# progress.json growing forever. At the ~13-day cycle each poet now gets
# (see POET_SCHEDULE below), 60 entries covers roughly 2 years per poet.
MAX_SHER_HISTORY_PER_POET = 60

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

# ------------------------------------------------------------------
# POET SCHEDULE -- chronological, oldest to newest (~1253 to present)
# ------------------------------------------------------------------
# WHY THIS GREW AGAIN (40 -> 52): you asked for current/Instagram-era
# shayars too. I was deliberately more selective adding THOSE than the
# classical names below, for a specific reason: contemporary poets have
# thin, inconsistent documentation online (no Wikipedia page for either
# new name here -- sources disagree even on their exact birth dates).
# Gemini has far less real material to draw on for them, so the risk of
# it inventing a couplet and crediting it to a real, living person is
# genuinely higher than for a classical poet with a century of archived
# work. So: only 2 new living contemporaries were added, both verified
# as authentically famous across multiple independent sources (not
# obscure), while the bulk of this expansion is more classical/20th
# century names -- those are safe to add generously since Gemini has
# far more real material to draw on and far less room to invent.
#
# Every entry's birth/death year was checked against Wikipedia (or,
# for the 2 contemporary names, cross-checked across several
# independent bios) -- not pulled from memory. "1934-" (no end year)
# means the poet is still living.
#
# NOTE: progress.json just keeps counting from wherever it left off --
# reordering/growing this list doesn't break anything, it only changes
# which poet comes up next.
POET_SCHEDULE = [
    # --- Founding voices (13th-17th century) ---
    {"name": "Amir Khusro",            "era": "1253-1325"},
    {"name": "Wali Deccani",           "era": "1667-1707"},

    # --- Mughal-era Delhi School (18th century) ---
    {"name": "Mirza Rafi Sauda",       "era": "1713-1780"},
    {"name": "Khwaja Mir Dard",        "era": "1721-1785"},
    {"name": "Mir Taqi Mir",           "era": "1723-1810"},
    {"name": "Nazeer Akbarabadi",      "era": "1740-1830"},
    {"name": "Insha Allah Khan Insha", "era": "1756-1817"},

    # --- 19th century greats ---
    {"name": "Bahadur Shah Zafar",     "era": "1775-1862"},
    {"name": "Imam Baksh Nasikh",      "era": "1776-1838"},
    {"name": "Khwaja Haidar Ali Atish","era": "1778-1846"},
    {"name": "Sheikh Ibrahim Zauq",    "era": "1789-1854"},
    {"name": "Mirza Ghalib",           "era": "1797-1869"},
    {"name": "Momin Khan Momin",       "era": "1800-1852"},
    {"name": "Mir Anis",               "era": "1803-1874"},
    {"name": "Mirza Salaamat Ali Dabeer","era": "1803-1875"},  # Anis's great mushaira rival -- fitting pair
    {"name": "Amir Meenai",            "era": "1828-1900"},
    {"name": "Dagh Dehlvi",            "era": "1831-1905"},
    {"name": "Altaf Hussain Hali",     "era": "1837-1914"},
    {"name": "Akbar Allahabadi",       "era": "1846-1921"},
    {"name": "Hasrat Mohani",          "era": "1875-1951"},

    # --- Early 20th century ---
    {"name": "Allama Iqbal",           "era": "1877-1938"},
    {"name": "Jigar Moradabadi",       "era": "1890-1961"},
    {"name": "Firaq Gorakhpuri",       "era": "1896-1982"},
    {"name": "Josh Malihabadi",        "era": "1898-1982"},
    {"name": "Faiz Ahmed Faiz",        "era": "1911-1984"},
    {"name": "Majaz Lucknawi",         "era": "1911-1955"},

    # --- Progressive Writers' era / golden age ---
    {"name": "Ali Sardar Jafri",       "era": "1913-2000"},
    {"name": "Ahmad Nadeem Qasmi",     "era": "1916-2006"},
    {"name": "Kaifi Azmi",             "era": "1919-2002"},
    {"name": "Majrooh Sultanpuri",     "era": "1919-2000"},
    {"name": "Qateel Shifai",          "era": "1919-2001"},
    {"name": "Sahir Ludhianvi",        "era": "1921-1980"},
    {"name": "Ada Jafri",              "era": "1924-2015"},    # first major female Urdu poet to publish

    # --- Later 20th century ---
    {"name": "Nasir Kazmi",            "era": "1925-1972"},
    {"name": "Ibn-e-Insha",            "era": "1927-1978"},
    {"name": "Habib Jalib",            "era": "1928-1993"},
    {"name": "Munir Niazi",            "era": "1928-2006"},
    {"name": "Jaun Elia",              "era": "1931-2002"},
    {"name": "Ahmad Faraz",            "era": "1931-2008"},
    {"name": "Gulzar",                 "era": "1934-"},        # still living
    {"name": "Zehra Nigah",            "era": "1935-"},        # still living
    {"name": "Bashir Badr",            "era": "1935-2026"},
    {"name": "Nida Fazli",             "era": "1938-2016"},
    {"name": "Kishwar Naheed",         "era": "1940-"},        # still living
    {"name": "Wasim Barelvi",          "era": "1940-"},        # still living
    {"name": "Javed Akhtar",           "era": "1945-"},        # still living
    {"name": "Fahmida Riaz",           "era": "1946-2018"},
    {"name": "Rahat Indori",           "era": "1950-2020"},
    {"name": "Parveen Shakir",         "era": "1952-1994"},
    {"name": "Munawwar Rana",          "era": "1952-2024"},

    # --- Current / Instagram-era shayars (still living) ---
    {"name": "Ali Zaryoun",            "era": "1983-"},
    {"name": "Tehzeeb Hafi",           "era": "1989-"},
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
def generate_content(poet: dict, used_shers: list = None) -> dict:
    used_shers = used_shers or []
    hook = random.choice(VIRAL_HOOKS).format(poet_name=poet['name'])

    # NEW: tell the AI exactly which couplets we've already posted for this
    # poet, so it reaches for a different one instead of always defaulting
    # to the same "greatest hit." This is the core of the anti-repeat fix --
    # every major poet genuinely has hundreds of documented couplets, so
    # there's no real shortage, just a tendency for the model to default to
    # its single most statistically likely answer unless told otherwise.
    avoid_block = ""
    if used_shers:
        bullet_list = "\n".join(f"- {s}" for s in used_shers)
        avoid_block = (
            f"\n{poet['name']} wrote hundreds of couplets across their lifetime, "
            f"so there is no shortage of real options here. The following have "
            f"ALREADY been posted on this page -- you MUST pick a different, "
            f"genuinely real couplet by {poet['name']}, not a reworded version "
            f"of any of these:\n{bullet_list}\n"
        )

    prompt = (
        f"You run a high-engagement Instagram Shayari page.\n"
        f"Poet: {poet['name']} ({poet['era']})\n"
        f"{avoid_block}\n"
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
# REPEAT DETECTION  (backs up the avoid-list above with an actual check)
# ============================================================
# WHY THIS EXISTS: the avoid-list in the prompt above is an instruction,
# not a guarantee -- the AI could still ignore it. This is a real safety
# net that checks the couplet Gemini actually returned against what's
# already been posted, using Python's built-in difflib (free, no new
# dependency) rather than trusting the AI to have followed instructions.

def _normalize_sher(text: str) -> str:
    """Lowercase + strip punctuation/extra spacing, so two couplets that
    differ only in capitalization or spelling of a word ('khushbu' vs
    'khushboo') still compare as the same thing."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _is_repeat(new_sher: str, used_shers: list, threshold: float = 0.85) -> bool:
    """True if new_sher is an exact or near-identical (>=85% similar)
    match to something already in used_shers."""
    new_norm = _normalize_sher(new_sher)
    for old in used_shers:
        old_norm = _normalize_sher(old)
        if new_norm == old_norm:
            return True
        if difflib.SequenceMatcher(None, new_norm, old_norm).ratio() >= threshold:
            return True
    return False

def generate_unique_content(poet: dict, used_shers: list, max_attempts: int = 3) -> dict:
    """
    Calls generate_content() up to max_attempts times, re-rolling if the
    result repeats something already posted for this poet. If every
    attempt still comes back a repeat (only realistic for a poet with very
    few widely-documented couplets), we post the last attempt anyway
    rather than fail the whole run over it -- but we print a clear warning
    so it's visible in the GitHub Actions log if it ever happens.
    """
    data = None
    for attempt in range(1, max_attempts + 1):
        data = generate_content(poet, used_shers)
        if not _is_repeat(data.get("sher_roman", ""), used_shers):
            return data
        print(f"⚠️ Attempt {attempt}/{max_attempts}: got a couplet too similar to one already posted for {poet['name']}. Retrying for a different one...")
    print(f"⚠️ Still repeating after {max_attempts} attempts for {poet['name']} -- posting it anyway rather than failing the run.")
    return data

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
def _add_natural_pacing(line: str) -> str:
    """
    Ensures a line ends with real punctuation before we hand it to ANY of
    the 3 TTS engines. Every TTS system (ElevenLabs, OpenAI-style, Edge-TTS)
    reads punctuation as a cue to breathe/pause -- a line with no ending
    punctuation gets read in one flat rush with no breath at the end. Since
    this edits the TEXT itself (not an engine-specific setting), it's the
    one pacing fix that keeps working no matter which of the 3 tiers
    actually ends up firing on a given run.
    """
    line = line.strip()
    if not line:
        return line
    if line[-1] not in "۔؟!.?,،":  # Urdu full stop ۔, Urdu comma ،, plus common Latin punctuation
        line += "۔"  # Urdu-script full stop -- correct pause marker since sher_urdu is real Urdu script
    return line

def _looks_like_valid_audio(path: str, min_bytes: int = 2000) -> bool:
    """
    WHY THIS EXISTS (from today's crash): OpenRouter's call raised NO
    exception, so the old code printed "✅ ... successfully!" and handed
    the file straight to MoviePy -- which failed several steps later with
    a confusing ffmpeg wall of text ("Header missing" x82, "Could not find
    codec parameters"). The real problem was upstream: the .mp3 on disk
    was never valid audio in the first place (root-caused below, in the
    OpenRouter tier -- a documented bug in how the openai-python library
    writes streamed responses). This check catches that immediately,
    right after the file is written, instead of several steps later.
    A cheap, real check, no new dependency needed:
      1) file must exist and be a sane minimum size (2000 bytes -- a
         genuinely corrupt/empty response is usually near 0 bytes; even a
         short 2-3 second spoken line is several KB of real audio)
      2) file must start with a real MP3 signature: either an ID3 tag
         ("ID3...") or an MPEG frame sync byte (0xFF followed by a byte
         with its top 3 bits set) -- garbage/error text won't match either.
    """
    if not os.path.exists(path) or os.path.getsize(path) < min_bytes:
        return False
    with open(path, "rb") as f:
        header = f.read(3)
    return header[:3] == b"ID3" or (len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0)

def generate_tts(data: dict) -> list:
    lines = [line.strip() for line in data["sher_urdu"].split("\n") if line.strip()]
    if not lines: lines = [data["sher_urdu"]]
    lines = [_add_natural_pacing(l) for l in lines]  # gives every engine a breath/pause cue per line

    if ELEVENLABS_API_KEY:
        output_paths = []  # fresh per tier -- see note above generate_tts about why this moved
        try:
            print("🎙️ [TTS 1/3] Trying ElevenLabs...")
            from elevenlabs.client import ElevenLabs
            from elevenlabs import VoiceSettings
            client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
            # NEW voice_settings (previously this call passed none at all,
            # so it silently used ElevenLabs' bare defaults):
            #   stability=0.65 -> nudged UP from the 0.5 default. Lower
            #     stability = more emotional variety but "may occasionally
            #     sound unstable" per ElevenLabs' own docs -- which is
            #     exactly what recurring mispronunciation looks like. 0.65
            #     trades a little expressiveness for real consistency.
            #   speed=0.88 -> a real, moderate slowdown from 1.0 (not the
            #     extreme 0.7 floor, which risks quality artifacts per
            #     ElevenLabs' docs -- you asked for "moderate", not crawl-
            #     pace). This is what directly answers "speaking speed is
            #     very fast."
            #   similarity_boost/use_speaker_boost left at ElevenLabs'
            #     own recommended defaults -- no complaint pointed at these.
            voice_settings = VoiceSettings(
                stability=0.65, similarity_boost=0.75, style=0.0,
                use_speaker_boost=True, speed=0.88,
            )
            for i, line in enumerate(lines):
                out_path = f"output/tts_line_{i}_{int(time.time())}.mp3"
                audio_stream = client.text_to_speech.convert(
                    text=line, voice_id="N2lVS1w4EtoT3dr4eOWO",
                    model_id="eleven_multilingual_v2", output_format="mp3_44100_128",
                    voice_settings=voice_settings,
                )
                with open(out_path, "wb") as f:
                    for chunk in audio_stream:
                        if chunk: f.write(chunk)
                if not _looks_like_valid_audio(out_path):
                    raise RuntimeError(f"ElevenLabs wrote an unreadable file for line {i+1}/{len(lines)}")
                output_paths.append(out_path)
            print("✅ ElevenLabs Audio generated successfully!")
            return output_paths
        except Exception as e:
            print(f"⚠️ ElevenLabs failed ({e}). Moving to OpenRouter TTS...")

    if OPENROUTER_API_KEY:
        output_paths = []  # fresh per tier -- same reason as above
        try:
            print("🎙️ [TTS 2/3] Trying OpenRouter TTS...")
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
            for i, line in enumerate(lines):
                out_path = f"output/tts_line_{i}_{int(time.time())}.mp3"
                # ROOT CAUSE OF TODAY'S CRASH: the old code called
                # client.audio.speech.create(...).stream_to_file(out_path)
                # directly. The job log's own DeprecationWarning already
                # named the fix -- openai-python's docs confirm .create()
                # used this way doesn't actually drain the streamed bytes
                # correctly; you have to open it through
                # with_streaming_response as a context manager instead.
                # That's a client-library mechanic, unrelated to which
                # backend URL is hit, so it applies the same via OpenRouter.
                with client.audio.speech.with_streaming_response.create(
                    model="fish-audio/s2.1-pro-free:free",
                    voice="alloy",
                    input=line,
                    speed=0.9,  # moderate slowdown, same intent as ElevenLabs above.
                    # Honest caveat: free/community-routed model behind OpenRouter,
                    # so there's no guarantee it honors `speed` -- if ignored, it
                    # just falls back to that model's normal pace, no error either way.
                ) as response:
                    response.stream_to_file(out_path)
                if not _looks_like_valid_audio(out_path):
                    raise RuntimeError(f"OpenRouter TTS wrote an unreadable file for line {i+1}/{len(lines)}")
                output_paths.append(out_path)
            print("✅ OpenRouter TTS Audio generated successfully!")
            return output_paths
        except Exception as e:
            print(f"⚠️ OpenRouter TTS failed ({e}). Moving to Edge-TTS...")

    output_paths = []  # fresh per tier -- same reason as above
    try:
        print("🎙️ [TTS 3/3] Generating fallback via Edge-TTS...")
        import asyncio
        import edge_tts
        async def _speak():
            for i, line in enumerate(lines):
                out_path = f"output/tts_line_{i}_{int(time.time())}.mp3"
                # rate moderated further: -15% -> -25% (a bigger, real slowdown,
                # answering "very fast"), pitch left as-is since that wasn't the complaint.
                communicate = edge_tts.Communicate(line, "ur-PK-AsadNeural", rate="-25%", pitch="-6Hz")
                await communicate.save(out_path)
                if not _looks_like_valid_audio(out_path):
                    raise RuntimeError(f"Edge-TTS wrote an unreadable file for line {i+1}/{len(lines)}")
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
    """
    progress.json now tracks three things:
      - poet_index / total_posts: unchanged, cycles POET_SCHEDULE
      - sher_history: {poet_name: [sher_roman, ...]} -- every couplet
        already posted per poet, used to tell the AI "don't repeat these"
        the next time that poet comes up (see generate_unique_content).
    setdefault() below means an OLDER progress.json (saved before this
    field existed) still loads fine -- it just starts sher_history empty.
    """
    if os.path.exists("progress.json"):
        with open("progress.json") as f:
            data = json.load(f)
        data.setdefault("sher_history", {})
        return data
    return {"poet_index": 0, "total_posts": 0, "sher_history": {}}

def save_progress(p: dict):
    # ensure_ascii=False keeps Urdu/Roman text human-readable if you ever
    # open progress.json directly, instead of escaped as \uXXXX codes.
    with open("progress.json", "w") as f:
        json.dump(p, f, indent=2, ensure_ascii=False)

def run():
    # Prints first, before anything else can fail -- so ANY job log
    # (success or crash) tells you, at a glance, exactly which code
    # actually ran. Compare this against the version number at the top
    # of the file: if a log ever shows an OLDER version than what you
    # most recently committed, that's proof by itself that the commit
    # didn't reach GitHub -- no need to re-fetch/compare files by hand.
    print(f"📦 Bot version: {BOT_VERSION}")
    validate_environment()
    p = load_progress()
    poet = POET_SCHEDULE[p["poet_index"] % len(POET_SCHEDULE)]
    used_shers = p["sher_history"].get(poet["name"], [])  # this poet's already-posted couplets

    print(f"\n🚀 STARTING WORKFLOW: [{POST_TYPE.upper()}] for {poet['name']} ({len(used_shers)} couplet(s) already posted for them)\n")
    data = generate_unique_content(poet, used_shers)
    # NOTE ON CAPTION STRUCTURE: the "-- {poet['name']}" line signs off the
    # couplet ONLY. Everything after it (data['caption']) is Gemini's own
    # generated reflection, not the poet's words -- the 💭 marks that
    # clearly so it doesn't read as an extension of the quote. This is
    # good practice regardless of Instagram: it's a genuine accuracy fix
    # for readers, not just a workaround for a classifier.
    caption = f"{data.get('hook','')}\n\n{data['sher_roman']}\n\n-- {poet['name']}\n\n💭 {data.get('caption','')}\n\n#urdushayari #hindishayari #poetry"

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

        # Record this couplet so it's never picked again for this poet.
        # Only done on a CONFIRMED successful post -- if Instagram had
        # rejected it, we wouldn't want to "burn" it from the pool for a
        # couplet that was never actually shown publicly.
        sher_roman = data.get("sher_roman", "")
        if sher_roman:
            history = p["sher_history"].setdefault(poet["name"], [])
            history.append(sher_roman)
            del history[:-MAX_SHER_HISTORY_PER_POET]  # keep only the most recent N (see constant above)

        save_progress(p)
        print("\n✅ WORKFLOW COMPLETED SUCCESSFULLY!")
    else:
        sys.exit(1)

if __name__ == "__main__":
    run()
