import os
import re
import sys
import logging
import random
import time
import threading
import uuid
import hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import gradio as gr
import torch
from typing import Optional, Tuple
from funasr import AutoModel
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import voxcpm
from voxcpm.model.utils import resolve_runtime_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BATCH_JOB_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voxcpm-batch")
BATCH_JOBS: dict[str, dict] = {}
BATCH_JOBS_LOCK = threading.Lock()
MODEL_RUNTIME_LOCK = threading.RLock()

MAX_BATCH_CHUNK_CHARS = 220
MAX_BATCH_TOTAL_CHARS = 8000
BATCH_CHUNK_PAUSE_SECONDS = 0.12
BATCH_CHUNK_FADE_SECONDS = 0.035
BATCH_CHUNK_MAX_EDGE_TRIM_SECONDS = 0.16
BATCH_CHUNK_KEEP_EDGE_SECONDS = 0.025


def _split_text_for_tts(text: str, max_chars: int = MAX_BATCH_CHUNK_CHARS) -> list[str]:
    """Split long TXT content into model-friendly chunks while keeping sentence order."""
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if not clean:
        return []

    def split_long_piece(piece: str) -> list[str]:
        piece = piece.strip()
        if len(piece) <= max_chars:
            return [piece] if piece else []

        # Prefer natural sentence boundaries, then clause boundaries, and only
        # hard-split as the final fallback for very long sentences.
        for pattern in (r"(?<=[。！？!?])\s*", r"(?<=[；;])\s*", r"(?<=[，,、])\s*"):
            parts = [part.strip() for part in re.split(pattern, piece) if part.strip()]
            if len(parts) > 1 and max(len(part) for part in parts) < len(piece):
                merged: list[str] = []
                current = ""
                for part in parts:
                    candidate = f"{current}{part}" if current else part
                    if current and len(candidate) > max_chars:
                        merged.extend(split_long_piece(current))
                        current = part
                    else:
                        current = candidate
                if current:
                    merged.extend(split_long_piece(current))
                return merged

        return [piece[start : start + max_chars].strip() for start in range(0, len(piece), max_chars) if piece[start : start + max_chars].strip()]

    sentences = []
    for paragraph in re.split(r"\n\s*\n+", (text or "").strip()):
        sentences.extend(split_long_piece(paragraph))

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current}{sentence}" if current else sentence
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks


def _trim_chunk_edge_silence(wav: np.ndarray, sr: int) -> np.ndarray:
    """Trim only excessive edge silence so generated chunks join more naturally."""
    audio = np.asarray(wav, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio

    max_trim = int(sr * BATCH_CHUNK_MAX_EDGE_TRIM_SECONDS)
    keep = int(sr * BATCH_CHUNK_KEEP_EDGE_SECONDS)
    threshold = max(0.006, float(np.max(np.abs(audio))) * 0.02)
    active = np.flatnonzero(np.abs(audio) > threshold)
    if active.size == 0:
        return audio

    start = max(0, min(int(active[0]), max_trim) - keep)
    end_trim = min(audio.size - int(active[-1]) - 1, max_trim)
    end = audio.size - max(0, end_trim - keep)
    if start >= end:
        return audio
    return audio[start:end]


def _apply_chunk_boundary_fades(wav: np.ndarray, sr: int, *, fade_in: bool, fade_out: bool) -> np.ndarray:
    audio = np.asarray(wav, dtype=np.float32).reshape(-1).copy()
    fade_len = min(int(sr * BATCH_CHUNK_FADE_SECONDS), audio.size // 4)
    if fade_len <= 1:
        return audio
    if fade_in:
        audio[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    if fade_out:
        audio[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    return audio


def _merge_tts_chunks_smoothly(chunk_wavs: list[np.ndarray], sr: int) -> np.ndarray:
    """Join generated chunks with a short breath and fades instead of a hard splice."""
    if not chunk_wavs:
        return np.zeros(0, dtype=np.float32)

    if len(chunk_wavs) == 1:
        return np.asarray(chunk_wavs[0], dtype=np.float32).reshape(-1)

    pause = np.zeros(int(sr * BATCH_CHUNK_PAUSE_SECONDS), dtype=np.float32)
    merged_parts = []
    last_index = len(chunk_wavs) - 1
    for index, chunk_wav in enumerate(chunk_wavs):
        chunk = _trim_chunk_edge_silence(chunk_wav, sr)
        chunk = _apply_chunk_boundary_fades(
            chunk,
            sr,
            fade_in=index > 0,
            fade_out=index < last_index,
        )
        merged_parts.append(chunk)
        if index < last_index:
            merged_parts.append(pause)
    return np.concatenate(merged_parts).astype(np.float32)

# ---------- Inline i18n (en + zh-CN only) ----------

_USAGE_INSTRUCTIONS_EN = (
    "**VoxCPM2 — Three Modes of Speech Generation:**\n\n"
    "🎨 **Voice Design** — Create a brand-new voice  \n"
    "No reference audio required. Describe the desired voice characteristics "
    "(gender, age, tone, emotion, pace …) in **Control Instruction**, and VoxCPM2 "
    "will craft a unique voice from your description alone.\n\n"
    "🎛️ **Controllable Cloning** — Clone a voice with optional style guidance  \n"
    "Upload a reference audio clip, then use **Control Instruction** to steer "
    "emotion, speaking pace, and overall style while preserving the original timbre.\n\n"
    "🎙️ **Ultimate Cloning** — Reproduce every vocal nuance through audio continuation  \n"
    "Turn on **Ultimate Cloning Mode** and provide (or auto-transcribe) the reference audio's transcript. "
    "The model treats the reference clip as a spoken prefix and seamlessly **continues** from it, faithfully preserving every vocal detail."
    "Note: This mode will disable Control Instruction."
)

_EXAMPLES_FOOTER_EN = (
    "---\n"
    "**💡 Voice Description Examples:**  \n"
    "Try the following Control Instructions to explore different voices:  \n\n"
    "**Example 1 — Gentle & Melancholic Girl**  \n"
    '`Control Instruction`: *"A young girl with a soft, sweet voice. '
    'Speaks slowly with a melancholic, slightly tsundere tone."*  \n'
    "`Target Text`: *\"I never asked you to stay… It's not like I care or anything. "
    "But… why does it still hurt so much now that you're gone?\"*  \n\n"
    "**Example 2 — Laid-Back Surfer Dude**  \n"
    '`Control Instruction`: *"Relaxed young male voice, slightly nasal, '
    'lazy drawl, very casual and chill."*  \n'
    '`Target Text`: *"Dude, did you see that set? The waves out there are totally gnarly today. '
    "Just catching barrels all morning — it's like, totally righteous, you know what I mean?\"*"
)

_USAGE_INSTRUCTIONS_ZH = (
    "**VoxCPM2 — 三种语音生成方式：**\n\n"
    "🎨 **声音设计（Voice Design）**  \n"
    "无需参考音频。在 **Control Instruction** 中描述目标音色特征"
    "（性别、年龄、语气、情绪、语速等），VoxCPM2 即可为你从零创造独一无二的声音。\n\n"
    "🎛️ **可控克隆（Controllable Cloning）**  \n"
    "上传参考音频，同时可选地使用 **Control Instruction** 来指定情绪、语速、风格等表达方式，"
    "在保留原始音色的基础上灵活控制说话风格。\n\n"
    "🎙️ **极致克隆（Ultimate Cloning）**  \n"
    "开启 **极致克隆模式** 并提供参考音频的文字内容（可自动识别）。"
    "模型会将参考音频视为已说出的前文，以**音频续写**的方式完整还原参考音频中的所有声音细节。"
    "注意：该模式与可控克隆模式互斥，将禁用Control Instruction。\n\n"
)

_EXAMPLES_FOOTER_ZH = (
    "---\n"
    "**💡 声音描述示例（中英文均可）：**  \n\n"
    "**示例 1 — 深宫太后**  \n"
    '`Control Instruction`: *"中老年女性，声音低沉阴冷，语速缓慢而有力，'
    '字字深思熟虑，带有深不可测的城府与威慑感。"*  \n'
    '`Target Text`: *"哀家在这深宫待了四十年，什么风浪没见过？你以为瞒得过哀家？"*  \n\n'
    "**示例 2 — 暴躁驾校教练**  \n"
    '`Control Instruction`: *"暴躁的中年男声，语速快，充满无奈和愤怒"*  \n'
    '`Target Text`: *"踩离合！踩刹车啊！你往哪儿开呢？前面是树你看不见吗？'
    '我教了你八百遍了，打死方向盘！你是不是想把车给我开到沟里去？"*  \n\n'
    "---\n"
    "**🗣️ 方言生成指南：**  \n"
    "要生成地道的方言语音，请在 **Target Text** 中直接使用方言词汇和句式，"
    "并在 **Control Instruction** 中描述方言特征。  \n\n"
    "**示例 — 广东话**  \n"
    '`Control Instruction`: *"粤语，中年男性，语气平淡"*  \n'
    '✅ 正确（粤语表达）：*"伙計，唔該一個A餐，凍奶茶少甜！"*  \n'
    '❌ 错误（普通话原文）：*"伙计，麻烦来一个A餐，冻奶茶少甜！"*  \n\n'
    "**示例 — 河南话**  \n"
    '`Control Instruction`: *"河南话，接地气的大叔"*  \n'
    '✅ 正确（河南话表达）：*"恁这是弄啥嘞？晌午吃啥饭？"*  \n'
    '❌ 错误（普通话原文）：*"你这是在干什么呢？中午吃什么饭？"*  \n\n'
    "🤖 **小技巧：** 不知道方言怎么写？可以用豆包、DeepSeek、Kimi 等 AI 助手"
    "将普通话翻译为方言文本，再粘贴到 Target Text 中即可。  \n\n"
)

_I18N_TRANSLATIONS = {
    "en": {
        "reference_audio_label": "🎤 Reference Audio (optional — upload for cloning)",
        "show_prompt_text_label": "🎙️ Ultimate Cloning Mode (transcript-guided cloning)",
        "show_prompt_text_info": "Auto-transcribes reference audio for every vocal nuance reproduced. Control Instruction will be disabled when active.",
        "prompt_text_label": "Transcript of Reference Audio (auto-filled via ASR, editable)",
        "prompt_text_placeholder": "The transcript of your reference audio will appear here …",
        "control_label": "🎛️ Control Instruction (optional — supports Chinese & English)",
        "control_placeholder": "e.g. A warm young woman / 年轻女性，温柔甜美 / Excited and fast-paced",
        "target_text_label": "✍️ Target Text — the content to speak",
        "generate_btn": "🔊 Generate Speech",
        "generated_audio_label": "Generated Audio",
        "advanced_settings_title": "⚙️ Advanced Settings",
        "ref_denoise_label": "Reference audio enhancement",
        "ref_denoise_info": "Apply ZipEnhancer denoising to the reference audio before cloning",
        "normalize_label": "Text normalization",
        "normalize_info": "Normalize numbers, dates, and abbreviations via wetext",
        "cfg_label": "CFG (guidance scale)",
        "cfg_info": "Higher → closer to the prompt / reference; lower → more creative variation",
        "dit_steps_label": "LocDiT flow-matching steps",
        "dit_steps_info": "LocDiT flow-matching steps — more steps → maybe better audio quality, but slower",
        "seed_label": "Seed",
        "seed_info": "Seed used for reproducible generation. Updated with the actual successful seed after generation.",
        "random_seed_label": "Random Seed",
        "random_seed_info": "Generate a new seed before each inference run.",
        "usage_instructions": _USAGE_INSTRUCTIONS_EN,
        "examples_footer": _EXAMPLES_FOOTER_EN,
    },
    "zh-CN": {
        "reference_audio_label": "🎤 参考音频（可选 — 上传后用于克隆）",
        "show_prompt_text_label": "🎙️ 极致克隆模式（基于文本引导的极致克隆）",
        "show_prompt_text_info": "自动识别参考音频文本，完整还原音色、节奏、情感等全部声音细节。开启后 Control Instruction 将暂时禁用",
        "prompt_text_label": "参考音频内容文本（ASR 自动填充，可手动编辑）",
        "prompt_text_placeholder": "参考音频的文字内容将自动识别并显示在此处 …",
        "control_label": "🎛️ Control Instruction（可选 — 支持中英文描述）",
        "control_placeholder": "如：年轻女性，温柔甜美 / A warm young woman / 暴躁老哥，语速飞快",
        "target_text_label": "✍️ Target Text — 要合成的目标文本",
        "generate_btn": "🔊 开始生成",
        "generated_audio_label": "生成结果",
        "advanced_settings_title": "⚙️ 高级设置",
        "ref_denoise_label": "参考音频降噪增强",
        "ref_denoise_info": "克隆前使用 ZipEnhancer 对参考音频进行降噪处理",
        "normalize_label": "文本规范化",
        "normalize_info": "自动规范化数字、日期及缩写（基于 wetext）",
        "cfg_label": "CFG（引导强度）",
        "cfg_info": "数值越高 → 越贴合提示/参考音色；数值越低 → 生成风格更自由",
        "dit_steps_label": "LocDiT 流匹配迭代步数",
        "dit_steps_info": "LocDiT 流匹配生成迭代步数 — 步数越多 → 可能生成更好的音频质量，但速度变慢",
        "usage_instructions": _USAGE_INSTRUCTIONS_ZH,
        "examples_footer": _EXAMPLES_FOOTER_ZH,
    },
    "zh-Hans": None,  # alias, filled below
    "zh": None,  # alias, filled below
}
_I18N_TRANSLATIONS["zh-Hans"] = _I18N_TRANSLATIONS["zh-CN"]
_I18N_TRANSLATIONS["zh"] = _I18N_TRANSLATIONS["zh-CN"]

for _d in _I18N_TRANSLATIONS.values():
    if _d is not None:
        for _k, _v in _I18N_TRANSLATIONS["en"].items():
            _d.setdefault(_k, _v)

I18N = gr.I18n(**_I18N_TRANSLATIONS)

DEFAULT_TARGET_TEXT = ""

_CUSTOM_CSS = """
.logo-container {
    text-align: center;
    margin: 0.5rem 0 1rem 0;
}
.logo-container img {
    height: 80px;
    width: auto;
    max-width: 200px;
    display: inline-block;
}

/* Toggle switch style */
.switch-toggle {
    padding: 8px 12px;
    border-radius: 8px;
    background: transparent;
}
.switch-toggle input[type="checkbox"] {
    appearance: none;
    -webkit-appearance: none;
    width: 44px;
    height: 24px;
    background: #ccc;
    border-radius: 12px;
    position: relative;
    cursor: pointer;
    transition: background 0.3s ease;
    flex-shrink: 0;
}
.switch-toggle input[type="checkbox"]::after {
    content: "";
    position: absolute;
    top: 2px;
    left: 2px;
    width: 20px;
    height: 20px;
    background: white;
    border-radius: 50%;
    transition: transform 0.3s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.switch-toggle input[type="checkbox"]:checked {
    background: var(--color-accent);
}
.switch-toggle input[type="checkbox"]:checked::after {
    transform: translateX(20px);
}

/* TXT 批量上传文件列表限高，最多显示3个，超出滚动 */
.txt-upload-limited .file-preview,
#txt-batch-upload .file-preview,
#txt-batch-upload [data-testid="file-preview"],
#txt-batch-upload ul,
#txt-batch-upload .wrap {
    max-height: 118px !important;
    overflow-y: auto;
}
.txt-upload-limited ul {
    max-height: 118px !important;
    overflow-y: auto !important;
}
.settings-panel {
    border: 0;
    box-shadow: none;
    padding: 0;
    margin-bottom: 8px;
    background: transparent;
}
.info-switch-row button {
    min-width: 0;
    font-size: 12px !important;
    padding: 6px 8px !important;
}
.compact-result-row {
    align-items: end;
    margin: 4px 0 8px 0;
}
.compact-result-row .wrap {
    min-height: 0 !important;
}
#batch-audio-selector [role="listbox"],
#batch-audio-selector .options,
#batch-audio-selector ul {
    max-height: 220px !important;
    overflow-y: auto !important;
}
#generation-status-box textarea {
    max-height: 92px !important;
    overflow-y: auto !important;
    resize: vertical;
}
"""

_APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)

# ---------- Voice Library ----------

import json
import shutil

VOICE_LIB_PATH = Path(__file__).parent / "voice_library.json"
VOICES_DIR = Path(__file__).parent / "saved_voices"
VOICES_DIR.mkdir(exist_ok=True)
VOICE_CACHE_DIR = VOICES_DIR / "_cache"
VOICE_CACHE_DIR.mkdir(exist_ok=True)
PREPROCESSED_AUDIO_CACHE_DIR = VOICES_DIR / "_preprocessed"
PREPROCESSED_AUDIO_CACHE_DIR.mkdir(exist_ok=True)
VOICE_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def _load_voice_lib():
    if VOICE_LIB_PATH.exists():
        try:
            data = json.loads(VOICE_LIB_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.warning(f"Failed to load voice library: {exc}")
            return []
    return []


def _save_voice_lib(data):
    VOICE_LIB_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_voice_filename(name: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", (name or "").strip())
    return safe or "voice"


def _voice_cache_path(name: str) -> Path:
    return VOICE_CACHE_DIR / f"{_safe_voice_filename(name)}.prompt_cache.pt"


def _preprocessed_audio_cache_path(audio_path: str) -> Path:
    path = Path(audio_path)
    stat = path.stat()
    digest = hashlib.sha1(
        f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8", errors="ignore")
    ).hexdigest()[:20]
    return PREPROCESSED_AUDIO_CACHE_DIR / f"{path.stem}.{digest}.denoised.wav"


def _get_or_create_denoised_audio(audio_path: Optional[str], demo: "VoxCPMDemo") -> Optional[str]:
    if not audio_path:
        return audio_path
    source = Path(audio_path)
    if not source.exists():
        return audio_path
    cache_path = _preprocessed_audio_cache_path(str(source))
    if cache_path.exists() and cache_path.stat().st_size > 0:
        logger.info("Using cached denoised audio: %s", cache_path)
        return str(cache_path)

    with MODEL_RUNTIME_LOCK:
        model = demo.get_or_load_voxcpm()
        if model.denoiser is None:
            return audio_path
        logger.info("Creating denoised audio cache: %s", cache_path)
        model.denoiser.enhance(str(source), output_path=str(cache_path))
    return str(cache_path)


def _load_prompt_cache_file(cache_path: str | Path):
    path = Path(cache_path)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _save_prompt_cache_file(cache, cache_path: str | Path):
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)


def _find_voice_cache_for_audio(audio_path: Optional[str], prompt_text_value: str = "") -> Optional[Path]:
    entry = _find_voice_entry_for_audio(audio_path, prompt_text_value)
    if not entry:
        return None
    cache_path = entry.get("prompt_cache")
    if cache_path and Path(cache_path).exists():
        return Path(cache_path)
    return None


def _find_voice_entry_for_audio(audio_path: Optional[str], prompt_text_value: str = "") -> Optional[dict]:
    if not audio_path or (prompt_text_value or "").strip():
        return None
    try:
        target = Path(audio_path).resolve()
    except OSError:
        return None

    for entry in _get_voice_entries():
        if entry.get("type") != "clone":
            continue
        entry_audio = entry.get("audio")
        if not entry_audio:
            continue
        try:
            if Path(entry_audio).resolve() == target:
                return entry
        except OSError:
            continue
    return None


def _remember_voice_cache(audio_path: Optional[str], cache_path: Path):
    if not audio_path:
        return
    try:
        target = Path(audio_path).resolve()
    except OSError:
        return

    lib = _load_voice_lib()
    changed = False
    for entry in lib:
        if entry.get("type") != "clone" or not entry.get("audio"):
            continue
        try:
            if Path(entry["audio"]).resolve() == target:
                entry["prompt_cache"] = str(cache_path)
                changed = True
        except OSError:
            continue
    if changed:
        _save_voice_lib(lib)


def _get_voice_entries():
    entries = list(_load_voice_lib())
    seen = {
        (str(v.get("type", "")), str(v.get("name", "")))
        for v in entries
        if isinstance(v, dict)
    }
    for audio_path in sorted(VOICES_DIR.iterdir()):
        if not audio_path.is_file() or audio_path.suffix.lower() not in VOICE_AUDIO_EXTS:
            continue
        name = audio_path.stem
        key = ("clone", name)
        if key in seen:
            continue
        entries.append({
            "name": name,
            "type": "clone",
            "audio": str(audio_path),
            "asr_text": "",
            "auto_scanned": True,
        })
        seen.add(key)
    return entries


def _get_voice_choices():
    entries = _get_voice_entries()
    if not entries:
        return ["(空)"]
    return [f"[{'克隆' if v['type']=='clone' else '设计'}] {v['name']}" for v in entries]


# ---------- Model ----------


REQUIRED_LOCAL_MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "model.safetensors",
    "audiovae.pth",
)


def validate_local_model_dir(model_id: str) -> None:
    model_path = Path(model_id)
    looks_like_local_path = model_path.is_absolute() or os.sep in model_id or (os.altsep and os.altsep in model_id)
    if not model_path.exists():
        if looks_like_local_path:
            raise FileNotFoundError(f"Local model directory does not exist: {model_path}")
        return
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model path is not a directory: {model_path}")
    if not looks_like_local_path:
        return

    missing = [name for name in REQUIRED_LOCAL_MODEL_FILES if not (model_path / name).is_file()]
    if missing:
        missing_list = "\n".join(f"- {model_path / name}" for name in missing)
        raise FileNotFoundError(f"Local model is incomplete. Missing files:\n{missing_list}")


def friendly_runtime_error(exc: Exception) -> RuntimeError:
    message = str(exc)
    lower_message = message.lower()
    if "out of memory" in lower_message or "cuda error" in lower_message:
        return RuntimeError(
            "CUDA memory is not enough for this request. Close other GPU-heavy apps and try again. "
            f"Original error: {message}"
        )
    if isinstance(exc, FileNotFoundError):
        return RuntimeError(f"Required file is missing. {message}")
    return RuntimeError(f"VoxCPM2 failed to run. Original error: {message}")


class VoxCPMDemo:
    def __init__(self, model_id: str = "openbmb/VoxCPM2", device: str = "auto", optimize: bool = False) -> None:
        self.device = resolve_runtime_device(device, "cuda")
        logger.info(f"Running VoxCPM on device: {self.device}")
        self.optimize = bool(optimize and self.device.startswith("cuda"))
        if not self.optimize:
            logger.info("Model compile optimization is disabled for faster interactive startup.")

        self.asr_model_id = "iic/SenseVoiceSmall"
        self.asr_device = "cuda:0" if self.device.startswith("cuda") else "cpu"
        self.asr_model: Optional[AutoModel] = None

        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self._model_id = model_id
        self._model_init_lock = threading.Lock()

    def get_or_load_voxcpm(self) -> voxcpm.VoxCPM:
        if self.voxcpm_model is not None:
            return self.voxcpm_model
        with self._model_init_lock:
            if self.voxcpm_model is not None:
                return self.voxcpm_model
            logger.info(f"Loading model: {self._model_id}")
            try:
                validate_local_model_dir(self._model_id)
                self.voxcpm_model = voxcpm.VoxCPM.from_pretrained(
                    self._model_id,
                    optimize=self.optimize,
                    device=self.device,
                )
            except Exception as exc:
                logger.exception("Failed to load VoxCPM model.")
                raise friendly_runtime_error(exc) from exc
            logger.info("Model loaded successfully.")
        return self.voxcpm_model

    def get_or_load_asr_model(self) -> AutoModel:
        if self.asr_model is not None:
            return self.asr_model
        logger.info(f"Loading ASR model: {self.asr_model_id} on device: {self.asr_device}")
        self.asr_model = AutoModel(
            model=self.asr_model_id,
            disable_update=True,
            log_level="DEBUG",
            device=self.asr_device,
        )
        logger.info("ASR model loaded successfully.")
        return self.asr_model

    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        if prompt_wav is None:
            return ""
        res = self.get_or_load_asr_model().generate(
            input=prompt_wav,
            language="auto",
            use_itn=True,
        )
        return res[0]["text"].split("|>")[-1]

    def _build_generate_kwargs(
        self,
        *,
        final_text: str,
        audio_path: Optional[str],
        prompt_text_clean: Optional[str],
        cfg_value_input: float,
        do_normalize: bool,
        denoise: bool,
        inference_timesteps: int = 10,
        seed: Optional[int] = None,
        retry_badcase: bool = True,
    ) -> dict:
        generate_kwargs = dict(
            text=final_text,
            reference_wav_path=audio_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=inference_timesteps,
            normalize=do_normalize,
            denoise=denoise,
            seed=seed,
            retry_badcase=retry_badcase,
        )
        if prompt_text_clean and audio_path:
            generate_kwargs["prompt_wav_path"] = audio_path
            generate_kwargs["prompt_text"] = prompt_text_clean
        return generate_kwargs

    def generate_tts_audio(
        self,
        text_input: str,
        control_instruction: str = "",
        reference_wav_path_input: Optional[str] = None,
        prompt_text: str = "",
        cfg_value_input: float = 2.0,
        do_normalize: bool = True,
        denoise: bool = True,
        inference_timesteps: int = 10,
        seed: Optional[int] = None,
        retry_badcase: bool = True,
    ) -> Tuple[int, np.ndarray, Optional[int]]:
        current_model = self.get_or_load_voxcpm()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        control = (control_instruction or "").strip()
        # Strip any parentheses (half-width/full-width) from control text to avoid
        # breaking the "(control)text" prompt format expected by the model.
        control = re.sub(r"[()（）]", "", control).strip()
        final_text = f"({control}){text}" if control else text

        audio_path = reference_wav_path_input if reference_wav_path_input else None
        prompt_text_clean = (prompt_text or "").strip() or None
        if denoise and audio_path:
            audio_path = _get_or_create_denoised_audio(audio_path, self)
            denoise = False

        if audio_path and prompt_text_clean:
            logger.info(f"[Voice Cloning] prompt_wav + prompt_text + reference_wav")
        elif audio_path:
            logger.info(f"[Voice Control] reference_wav only")
        else:
            logger.info(f"[Voice Design] control: {control[:50] if control else 'None'}...")

        start_time = time.perf_counter()
        logger.info(
            "Generating audio: chars=%s, steps=%s, retry_badcase=%s, text='%s...'",
            len(text),
            inference_timesteps,
            retry_badcase,
            final_text[:80],
        )
        generate_kwargs = self._build_generate_kwargs(
            final_text=final_text,
            audio_path=audio_path,
            prompt_text_clean=prompt_text_clean,
            cfg_value_input=cfg_value_input,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=inference_timesteps,
            seed=seed,
            retry_badcase=retry_badcase,
        )
        try:
            with MODEL_RUNTIME_LOCK:
                wav = current_model.generate(**generate_kwargs)
        except Exception as exc:
            logger.exception("VoxCPM generation failed.")
            raise friendly_runtime_error(exc) from exc
        logger.info("Generation finished in %.1fs for %s chars.", time.perf_counter() - start_time, len(text))
        last_successful_seed = getattr(current_model.tts_model, "last_successful_seed", seed)
        return (current_model.tts_model.sample_rate, wav, last_successful_seed)

    def generate_tts_audio_with_prompt_cache(
        self,
        text_input: str,
        control_instruction: str,
        prompt_cache,
        cfg_value_input: float = 2.0,
        do_normalize: bool = True,
        inference_timesteps: int = 10,
        seed: Optional[int] = None,
        retry_badcase: bool = True,
    ) -> Tuple[int, np.ndarray, Optional[int]]:
        current_model = self.get_or_load_voxcpm()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        control = (control_instruction or "").strip()
        control = re.sub(r"[()（）]", "", control).strip()
        final_text = f"({control}){text}" if control else text

        start_time = time.perf_counter()
        logger.info(
            "Generating cached audio: chars=%s, steps=%s, retry_badcase=%s, text='%s...'",
            len(text),
            inference_timesteps,
            retry_badcase,
            final_text[:80],
        )
        try:
            with MODEL_RUNTIME_LOCK:
                wav = current_model.generate_with_prompt_cache(
                    text=final_text,
                    prompt_cache=prompt_cache,
                    cfg_value=float(cfg_value_input),
                    inference_timesteps=inference_timesteps,
                    normalize=do_normalize,
                    retry_badcase=retry_badcase,
                    seed=seed,
                )
        except Exception as exc:
            logger.exception("VoxCPM cached generation failed.")
            raise friendly_runtime_error(exc) from exc
        logger.info("Cached generation finished in %.1fs for %s chars.", time.perf_counter() - start_time, len(text))
        last_successful_seed = getattr(current_model.tts_model, "last_successful_seed", seed)
        return (current_model.tts_model.sample_rate, wav, last_successful_seed)


# ---------- UI ----------


def create_demo_interface(demo: VoxCPMDemo):
    gr.set_static_paths(paths=[Path.cwd().absolute() / "assets"])

    def _coerce_seed(seed_value) -> Optional[int]:
        if seed_value is None or seed_value == "":
            return None
        return int(seed_value)

    def _prepare_seed(use_random_seed: bool, seed_value):
        if use_random_seed:
            return random.randint(0, 2**32 - 1)
        return _coerce_seed(seed_value)

    def _on_random_seed_toggle(checked):
        return gr.update(interactive=not checked)

    def _generate(
        text: str,
        control_instruction: str,
        ref_wav: Optional[str],
        use_prompt_text: bool,
        prompt_text_value: str,
        cfg_value: float,
        do_normalize: bool,
        denoise: bool,
        dit_steps: int,
        seed_value,
    ):
        actual_prompt_text = prompt_text_value.strip() if use_prompt_text else ""
        actual_control = "" if use_prompt_text else control_instruction
        seed = _coerce_seed(seed_value)
        sr, wav_np, last_successful_seed = demo.generate_tts_audio(
            text_input=text,
            control_instruction=actual_control,
            reference_wav_path_input=ref_wav,
            prompt_text=actual_prompt_text,
            cfg_value_input=cfg_value,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=int(dit_steps),
            seed=seed,
        )
        return (sr, wav_np), last_successful_seed

    def _on_toggle_instant(checked):
        """Instant UI toggle — no ASR, no blocking."""
        if checked:
            return (
                gr.update(visible=True, value="", placeholder="Recognizing reference audio..."),
                gr.update(visible=False),
            )
        return (
            gr.update(visible=False),
            gr.update(visible=True, interactive=True),
        )

    def _run_asr_if_needed(checked, audio_path):
        """Run ASR after the UI has updated. Only when toggled ON."""
        if not checked or not audio_path:
            return gr.update()
        try:
            logger.info("Running ASR on reference audio...")
            asr_text = demo.prompt_wav_recognition(audio_path)
            logger.info(f"ASR result: {asr_text[:60]}...")
            if not asr_text.strip():
                return gr.update(value="（识别失败，请重试或手动输入）")
            return gr.update(value=asr_text)
        except Exception as e:
            logger.warning(f"ASR recognition failed: {e}")
            return gr.update(value=f"（识别出错: {e}）")

    with gr.Blocks() as interface:
        gr.HTML(
            '<div class="logo-container">'
            '<img src="/gradio_api/file=assets/voxcpm_logo.png" alt="VoxCPM Logo">'
            "</div>"
        )

        with gr.Row():
            with gr.Column():
                reference_wav = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label=I18N("reference_audio_label"),
                )
                control_instruction = gr.Textbox(
                    value="",
                    label=I18N("control_label"),
                    placeholder=I18N("control_placeholder"),
                    lines=2,
                )
                text = gr.Textbox(
                    value=DEFAULT_TARGET_TEXT,
                    label=I18N("target_text_label"),
                    placeholder="请输入要合成的文本；上传 TXT 批量生成时这里可以留空",
                    lines=3,
                )

                # TXT 批量上传
                txt_upload = gr.File(
                    label="📄 上传 TXT 批量生成（多选，每个文件生成一条音频）",
                    file_types=[".txt"],
                    file_count="multiple",
                    height=136,
                    elem_id="txt-batch-upload",
                    elem_classes=["txt-upload-limited"],
                )
                with gr.Group(visible=False) as txt_preview_group:
                    txt_preview_dropdown = gr.Dropdown(
                        label="选择预览 TXT",
                        choices=[],
                        value=None,
                        interactive=True,
                    )
                    with gr.Row():
                        preview_txt_btn = gr.Button("预览TXT内容", size="sm")
                        hide_txt_preview_btn = gr.Button("收起预览", size="sm")
                txt_status = gr.Textbox(
                    label="TXT 读取状态",
                    value="",
                    visible=False,
                    interactive=False,
                    lines=3,
                )

                run_btn = gr.Button(I18N("generate_btn"), variant="primary", size="lg")

            with gr.Column():
                audio_output = gr.Audio(label=I18N("generated_audio_label"))

                # 批量生成结果
                with gr.Group(visible=False) as batch_result_group:
                    with gr.Row(elem_classes=["compact-result-row"]):
                        batch_preview_dropdown = gr.Dropdown(
                            label="",
                            show_label=False,
                            choices=[],
                            value=None,
                            visible=True,
                            interactive=True,
                            scale=2,
                            elem_id="batch-audio-selector",
                        )
                        batch_output = gr.DownloadButton(label="下载全部 ZIP", visible=True, size="sm", scale=1)
                batch_audio_map = gr.State({})
                batch_job_id = gr.State("")
                manual_audio_selection = gr.State(False)
                batch_job_timer = gr.Timer(2.0, active=True)
                generation_status = gr.Textbox(
                    label="生成状态",
                    value="",
                    visible=False,
                    interactive=False,
                    lines=3,
                    elem_id="generation-status-box",
                )

                # 保存音色
                with gr.Accordion("保存音色", open=False):
                    voice_lib_dropdown = gr.Dropdown(
                        choices=_get_voice_choices(),
                        value=None,
                        label="📂 本地音色库（选择后自动加载）",
                        interactive=True,
                    )
                    refresh_voice_lib_btn = gr.Button("刷新音色库", size="sm")
                    voice_name_input = gr.Textbox(
                        label="音色名称", placeholder="输入名称...", lines=1)
                    with gr.Row():
                        save_clone_btn = gr.Button("保存 [克隆]", size="sm")
                        save_design_btn = gr.Button("保存 [设计]", size="sm")
                    save_status = gr.Textbox(label="", interactive=False, lines=1)

                with gr.Row(elem_classes=["info-switch-row"]):
                    examples_info_btn = gr.Button("使用示例 / 方言提示", size="sm")
                    modes_info_btn = gr.Button("VoxCPM2 三种语音生成模式", size="sm")
                    settings_info_btn = gr.Button("高级设置", size="sm")
                examples_info_panel = gr.Markdown(I18N("examples_footer"), visible=False)
                modes_info_panel = gr.Markdown(I18N("usage_instructions"), visible=False)
                info_panel_state = gr.State("")
                with gr.Group(visible=False, elem_classes=["settings-panel"]) as settings_panel:
                    show_prompt_text = gr.Checkbox(
                        value=False,
                        label=I18N("show_prompt_text_label"),
                        info=I18N("show_prompt_text_info"),
                        elem_classes=["switch-toggle"],
                    )
                    prompt_text = gr.Textbox(
                        value="",
                        label=I18N("prompt_text_label"),
                        placeholder=I18N("prompt_text_placeholder"),
                        lines=2,
                        visible=False,
                    )
                    DoDenoisePromptAudio = gr.Checkbox(
                        value=False,
                        label=I18N("ref_denoise_label"),
                        elem_classes=["switch-toggle"],
                        info=I18N("ref_denoise_info"),
                    )
                    DoNormalizeText = gr.Checkbox(
                        value=False,
                        label=I18N("normalize_label"),
                        elem_classes=["switch-toggle"],
                        info=I18N("normalize_info"),
                    )
                    cfg_value = gr.Slider(
                        minimum=1.0,
                        maximum=3.0,
                        value=2.0,
                        step=0.1,
                        label=I18N("cfg_label"),
                        info=I18N("cfg_info"),
                    )
                    dit_steps = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        label=I18N("dit_steps_label"),
                        info=I18N("dit_steps_info"),
                    )
                    with gr.Row():
                        seed_value = gr.Number(
                            value=random.randint(0, 2**32 - 1),
                            precision=0,
                            label=I18N("seed_label"),
                            info=I18N("seed_info"),
                            interactive=False,
                        )
                        random_seed = gr.Checkbox(
                            value=True,
                            label=I18N("random_seed_label"),
                            elem_classes=["switch-toggle"],
                            info=I18N("random_seed_info"),
                        )

        def _toggle_info_panel(active_panel, target_panel):
            next_panel = "" if active_panel == target_panel else target_panel
            return (
                next_panel,
                gr.update(visible=next_panel == "examples"),
                gr.update(visible=next_panel == "modes"),
                gr.update(visible=next_panel == "settings"),
            )

        examples_info_btn.click(
            fn=lambda active_panel: _toggle_info_panel(active_panel, "examples"),
            inputs=[info_panel_state],
            outputs=[info_panel_state, examples_info_panel, modes_info_panel, settings_panel],
            show_progress=False,
        )

        modes_info_btn.click(
            fn=lambda active_panel: _toggle_info_panel(active_panel, "modes"),
            inputs=[info_panel_state],
            outputs=[info_panel_state, examples_info_panel, modes_info_panel, settings_panel],
            show_progress=False,
        )

        settings_info_btn.click(
            fn=lambda active_panel: _toggle_info_panel(active_panel, "settings"),
            inputs=[info_panel_state],
            outputs=[info_panel_state, examples_info_panel, modes_info_panel, settings_panel],
            show_progress=False,
        )

        show_prompt_text.change(
            fn=_on_toggle_instant,
            inputs=[show_prompt_text],
            outputs=[prompt_text, control_instruction],
        ).then(
            fn=_run_asr_if_needed,
            inputs=[show_prompt_text, reference_wav],
            outputs=[prompt_text],
        )

        # Upload audio when toggle is ON → auto-trigger ASR
        reference_wav.change(
            fn=_run_asr_if_needed,
            inputs=[show_prompt_text, reference_wav],
            outputs=[prompt_text],
        )

        random_seed.change(
            fn=_on_random_seed_toggle,
            inputs=[random_seed],
            outputs=[seed_value],
        )

        # ─── 批量 TXT 生成 ───
        def _read_txt_file(fpath):
            path = Path(fpath)
            last_error = None
            for encoding in ("utf-8-sig", "utf-8", "gb18030"):
                try:
                    content = path.read_text(encoding=encoding).strip()
                    return content, encoding
                except UnicodeDecodeError as exc:
                    last_error = exc
            raise UnicodeDecodeError(
                "txt",
                b"",
                0,
                1,
                f"无法读取文本编码：{last_error}",
            )

        def _preview_txt_files(txt_files):
            if not txt_files:
                return (
                    gr.update(value="", visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(choices=[], value=None, visible=False),
                    gr.update(choices=[], value=None),
                    gr.update(value="", visible=False),
                    {},
                )

            choices = []
            for f in txt_files:
                fpath = f.name if hasattr(f, "name") else f
                choices.append((Path(fpath).name, fpath))
            first_value = choices[0][1] if choices else None
            return (
                gr.update(value="", visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(choices=choices, value=first_value, label=f"选择预览 TXT（共 {len(txt_files)} 个）"),
                gr.update(choices=[], value=None, visible=False),
                gr.update(value="", visible=False),
                {},
            )

        def _show_txt_preview(fpath):
            if not fpath:
                return gr.update(value="请先选择一个 TXT 文件。", visible=True, lines=3)
            try:
                content, encoding = _read_txt_file(fpath)
            except Exception as e:
                return gr.update(value=f"读取失败：{e}", visible=True, lines=3)

            max_preview_chars = 3000
            preview = content[:max_preview_chars]
            if len(content) > max_preview_chars:
                preview += f"\n\n... 已截断预览，全文共 {len(content)} 字。"
            value = f"{Path(fpath).name}\n编码：{encoding}\n字数：{len(content)}\n\n{preview}"
            return gr.update(value=value, visible=True, lines=10)

        def _prepare_generation_feedback(use_random_seed: bool, seed_value, txt_files):
            seed = _prepare_seed(use_random_seed, seed_value)
            count = len(txt_files or [])
            if count > 0:
                status = f"正在生成 {count} 个 TXT 对应的音频，请稍等。生成完成后会提供 ZIP 下载和逐条试听。"
                return (
                    gr.update(value=None),
                    seed,
                    gr.update(value=status, visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(choices=[], value=None, visible=False),
                    gr.update(value=f"生成中 0/{count}", interactive=False),
                    {},
                    "",
                    False,
                )
            return (
                gr.update(value=None),
                seed,
                gr.update(value="正在生成单条音频，请稍等。", visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(choices=[], value=None, visible=False),
                gr.update(value="生成中...", interactive=False),
                {},
                "",
                False,
            )

        def _batch_generate(
            txt_files,
            control_instruction_val,
            ref_wav,
            use_prompt_text,
            prompt_text_val,
            cfg_val,
            do_normalize,
            denoise,
            dit_steps_val,
            seed_val,
            progress=gr.Progress(track_tqdm=False),
            job_id=None,
        ):
            if not txt_files:
                return (
                    gr.update(visible=False),
                    gr.update(value="请先上传 TXT 文件。", visible=True),
                    gr.update(visible=False),
                    gr.update(choices=[], value=None, visible=False),
                    gr.update(value=None),
                    {},
                )
            import tempfile, zipfile
            out_dir = Path(tempfile.mkdtemp(prefix="voxcpm_batch_"))
            status_lines = []
            generated_files = []
            success_preview = []
            hidden_success_count = 0
            generated_count = 0
            batch_start_time = time.perf_counter()
            actual_prompt = prompt_text_val.strip() if use_prompt_text else ""
            actual_ctrl = "" if use_prompt_text else control_instruction_val
            prompt_cache = None
            if ref_wav:
                progress(0, desc="正在准备参考音频缓存")
                try:
                    cache_ref_wav = _get_or_create_denoised_audio(ref_wav, demo) if denoise else ref_wav
                    persistent_cache_path = None if denoise else _find_voice_cache_for_audio(ref_wav, actual_prompt)
                    if persistent_cache_path:
                        prompt_cache = _load_prompt_cache_file(persistent_cache_path)
                        logger.info("Loaded persistent voice prompt cache: %s", persistent_cache_path)
                    if prompt_cache is None:
                        with MODEL_RUNTIME_LOCK:
                            prompt_cache = demo.get_or_load_voxcpm().prepare_prompt_cache(
                                prompt_wav_path=cache_ref_wav if actual_prompt else None,
                                prompt_text=actual_prompt or None,
                                reference_wav_path=cache_ref_wav,
                                denoise=False,
                            )
                        if not actual_prompt and not denoise:
                            voice_entry = _find_voice_entry_for_audio(ref_wav, actual_prompt)
                            if voice_entry and voice_entry.get("name"):
                                cache_path = _voice_cache_path(str(voice_entry["name"]))
                                _save_prompt_cache_file(prompt_cache, cache_path)
                                _remember_voice_cache(ref_wav, cache_path)
                                logger.info("Saved persistent voice prompt cache: %s", cache_path)
                except Exception as exc:
                    logger.exception("Failed to prepare prompt cache.")
                    raise friendly_runtime_error(exc) from exc

            for index, f in enumerate(txt_files, 1):
                fpath = f.name if hasattr(f, 'name') else f
                try:
                    content, encoding = _read_txt_file(fpath)
                except Exception as e:
                    status_lines.append(f"读取失败：{Path(fpath).name}（{e}）")
                    continue

                char_count = len(content)
                if not content:
                    status_lines.append(f"跳过空文件：{Path(fpath).name}")
                    continue
                if char_count > MAX_BATCH_TOTAL_CHARS:
                    status_lines.append(
                        f"TXT too long: {Path(fpath).name} ({char_count} chars). Split it into smaller TXT files first."
                    )
                    continue
                chunks = _split_text_for_tts(content)
                if not chunks:
                    status_lines.append(f"Empty TXT: {Path(fpath).name}")
                    continue
                seed = _prepare_seed(True, seed_val)
                try:
                    logger.info(
                        "Batch item %s/%s: %s, chars=%s, chunks=%s",
                        index,
                        len(txt_files),
                        Path(fpath).name,
                        char_count,
                        len(chunks),
                    )
                    chunk_wavs = []
                    sr = None
                    for chunk_index, chunk in enumerate(chunks, 1):
                        progress(
                            ((index - 1) + (chunk_index - 1) / max(1, len(chunks))) / max(1, len(txt_files)),
                            desc=f"正在生成 {Path(fpath).name}：{chunk_index}/{len(chunks)}",
                        )
                        chunk_sr, chunk_wav, _ = demo.generate_tts_audio_with_prompt_cache(
                            text_input=chunk,
                            control_instruction=actual_ctrl,
                            prompt_cache=prompt_cache,
                            cfg_value_input=cfg_val,
                            do_normalize=do_normalize,
                            inference_timesteps=int(dit_steps_val),
                            seed=seed,
                            retry_badcase=True,
                        )
                        sr = chunk_sr
                        chunk_wavs.append(chunk_wav)

                    wav_np = _merge_tts_chunks_smoothly(chunk_wavs, sr)
                    import soundfile as sf
                    safe_stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(fpath).stem).strip() or "txt"
                    out_name = f"{index:03d}_{safe_stem}.wav"
                    out_path = out_dir / out_name
                    sf.write(str(out_path), wav_np, sr)
                    generated_files.append(out_path)
                    generated_count += 1
                    if len(success_preview) < 3:
                        success_preview.append(f"{out_name}（{Path(fpath).name}，{char_count} 字，{encoding}）")
                    else:
                        hidden_success_count += 1
                    if job_id:
                        partial_audio_map = {wav_file.name: str(wav_file) for wav_file in generated_files}
                        partial_choices = list(partial_audio_map.keys())
                        partial_first = partial_choices[0] if partial_choices else None
                        with BATCH_JOBS_LOCK:
                            job = BATCH_JOBS.get(job_id)
                            if job:
                                job["partial_audio_map"] = partial_audio_map
                                job["partial_choices"] = partial_choices
                                job["partial_first_audio"] = partial_audio_map.get(partial_first) if partial_first else None
                                job["partial_latest_audio"] = str(out_path)
                                job["partial_latest_choice"] = out_name
                                job["completed"] = generated_count
                                job["message"] = f"后台生成中：已完成 {generated_count}/{len(txt_files)} 个 TXT。"
                except Exception as e:
                    logger.error(f"Batch gen failed for {fpath}: {e}")
                    status_lines.append(f"生成失败：{Path(fpath).name}（{e}）")

            progress(1, desc="批量生成完成")

            if generated_count == 0:
                status = "\n".join(status_lines) if status_lines else "没有可生成的 TXT 内容。"
                return (
                    gr.update(visible=False),
                    gr.update(value=status, visible=True),
                    gr.update(visible=False),
                    gr.update(choices=[], value=None, visible=False),
                    gr.update(value=None),
                    {},
                )

            zip_path = out_dir.parent / f"{out_dir.name}.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                for wav_file in sorted(out_dir.glob("*.wav")):
                    zf.write(wav_file, wav_file.name)
            summary_lines = [f"完成：生成 {generated_count} 条音频。"]
            if success_preview:
                summary_lines.append("可试听：")
                summary_lines.extend(success_preview)
            if hidden_success_count:
                summary_lines.append(f"... 另外 {hidden_success_count} 条成功结果已收起，可在试听下拉框中选择。")
            summary_lines.append(f"Time used: {time.perf_counter() - batch_start_time:.1f}s")
            status_lines = summary_lines + status_lines
            audio_map = {wav_file.name: str(wav_file) for wav_file in generated_files}
            choices = list(audio_map.keys())
            first_choice = choices[0] if choices else None
            first_audio = audio_map.get(first_choice) if first_choice else None
            return (
                gr.update(value=str(zip_path), visible=True),
                gr.update(value="\n".join(status_lines), visible=True),
                gr.update(visible=True),
                gr.update(choices=choices, value=first_choice, visible=True),
                gr.update(value=first_audio),
                audio_map,
            )

        def _noop_progress(*args, **kwargs):
            return None

        def _run_batch_job(job_id, batch_args):
            with BATCH_JOBS_LOCK:
                job = BATCH_JOBS.get(job_id)
                if job:
                    job["status"] = "running"
                    job["message"] = "后台生成中..."
            try:
                result = _batch_generate(*batch_args, progress=_noop_progress, job_id=job_id)
                with BATCH_JOBS_LOCK:
                    job = BATCH_JOBS.get(job_id)
                    if job:
                        job["status"] = "done"
                        job["result"] = result
                        job["message"] = "后台生成完成。"
            except Exception as exc:
                logger.exception("Background batch job failed.")
                with BATCH_JOBS_LOCK:
                    job = BATCH_JOBS.get(job_id)
                    if job:
                        job["status"] = "failed"
                        job["message"] = f"后台生成失败：{friendly_runtime_error(exc)}"

        def _start_batch_job(batch_args, txt_count):
            job_id = uuid.uuid4().hex
            with BATCH_JOBS_LOCK:
                BATCH_JOBS[job_id] = {
                    "status": "queued",
                    "message": f"已加入后台队列：{txt_count} 个 TXT。页面可继续停留，完成后会自动更新结果。",
                    "total": txt_count,
                    "completed": 0,
                    "created_at": time.time(),
                }
            BATCH_JOB_EXECUTOR.submit(_run_batch_job, job_id, batch_args)
            return job_id

        def _poll_batch_job(job_id, manual_selected):
            if not job_id:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    "",
                    manual_selected,
                )
            with BATCH_JOBS_LOCK:
                job = dict(BATCH_JOBS.get(job_id) or {})
            status = job.get("status")
            total = int(job.get("total") or 0)
            completed = int(job.get("completed") or 0)
            if status in {"queued", "running"}:
                elapsed = time.time() - float(job.get("created_at", time.time()))
                partial_audio_map = job.get("partial_audio_map") or {}
                partial_choices = job.get("partial_choices") or list(partial_audio_map.keys())
                partial_latest = job.get("partial_latest_choice") or (partial_choices[-1] if partial_choices else None)
                button_text = f"生成中 {completed}/{total}" if total else "生成中..."
                if partial_audio_map:
                    return (
                        gr.update() if manual_selected else gr.update(value=job.get("partial_latest_audio")),
                        gr.update(visible=False),
                        gr.update(value=f"{job.get('message', '后台生成中...')}\n已用时：{elapsed:.1f} 秒\n已生成的音频可以先试听。", visible=True),
                        gr.update(visible=True),
                        gr.update(choices=partial_choices, visible=True) if manual_selected else gr.update(choices=partial_choices, value=partial_latest, visible=True),
                        partial_audio_map,
                        gr.update(value=button_text, interactive=True),
                        job_id,
                        manual_selected,
                    )
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(value=f"{job.get('message', '后台生成中...')}\n已用时：{elapsed:.1f} 秒", visible=True),
                    gr.update(visible=False),
                    gr.update(choices=[], value=None, visible=False),
                    gr.update(),
                    gr.update(value=button_text, interactive=True),
                    job_id,
                    manual_selected,
                )
            if status == "done":
                batch_file, status_update, batch_result_visible, preview_choices, first_audio, audio_map = job.get("result")
                with BATCH_JOBS_LOCK:
                    BATCH_JOBS.pop(job_id, None)
                final_choices = list((audio_map or {}).keys())
                final_latest = final_choices[-1] if final_choices else None
                final_audio = (audio_map or {}).get(final_latest) if final_latest else first_audio
                return (
                    gr.update() if manual_selected else gr.update(value=final_audio),
                    batch_file,
                    status_update,
                    batch_result_visible,
                    gr.update(choices=final_choices, visible=True) if manual_selected else gr.update(choices=final_choices, value=final_latest, visible=True),
                    audio_map,
                    gr.update(value="开始生成", interactive=True),
                    "",
                    manual_selected,
                )
            if status == "failed":
                with BATCH_JOBS_LOCK:
                    BATCH_JOBS.pop(job_id, None)
                return (
                    gr.update(value=None),
                    gr.update(visible=False),
                    gr.update(value=job.get("message", "后台生成失败。"), visible=True),
                    gr.update(visible=False),
                    gr.update(choices=[], value=None, visible=False),
                    {},
                    gr.update(value="开始生成", interactive=True),
                    "",
                    manual_selected,
                )
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                "",
                manual_selected,
            )

        def _generate_or_batch(
            text_value,
            control_instruction_val,
            ref_wav,
            use_prompt_text,
            prompt_text_val,
            cfg_val,
            do_normalize,
            denoise,
            dit_steps_val,
            seed_val,
            txt_files,
        ):
            try:
                if txt_files:
                    batch_args = (
                        txt_files,
                        control_instruction_val,
                        ref_wav,
                        use_prompt_text,
                        prompt_text_val,
                        cfg_val,
                        do_normalize,
                        denoise,
                        dit_steps_val,
                        seed_val,
                    )
                    job_id = _start_batch_job(batch_args, len(txt_files or []))
                    return (
                        gr.update(value=None),
                        seed_val,
                        gr.update(visible=False),
                        gr.update(value=f"已进入后台生成队列：{len(txt_files or [])} 个 TXT。完成后会自动显示 ZIP 和试听下拉框。", visible=True),
                        gr.update(visible=False),
                        gr.update(choices=[], value=None, visible=False),
                        gr.update(value="开始生成", interactive=True),
                        {},
                        job_id,
                        False,
                    )

                audio, last_successful_seed = _generate(
                    text_value,
                    control_instruction_val,
                    ref_wav,
                    use_prompt_text,
                    prompt_text_val,
                    cfg_val,
                    do_normalize,
                    denoise,
                    dit_steps_val,
                    seed_val,
                )
                return (
                    audio,
                    last_successful_seed,
                    gr.update(visible=False),
                    gr.update(value="完成：单条音频已生成。", visible=True),
                    gr.update(visible=False),
                    gr.update(choices=[], value=None, visible=False),
                    gr.update(value="开始生成", interactive=True),
                    {},
                    "",
                    False,
                )
            except Exception as exc:
                logger.exception("Generation failed.")
                return (
                    gr.update(value=None),
                    seed_val,
                    gr.update(visible=False),
                    gr.update(value=f"生成失败：{exc}", visible=True),
                    gr.update(visible=False),
                    gr.update(choices=[], value=None, visible=False),
                    gr.update(value="开始生成", interactive=True),
                    {},
                    "",
                    False,
                )

        txt_upload.change(
            fn=_preview_txt_files,
            inputs=[txt_upload],
            outputs=[
                txt_status,
                batch_output,
                batch_result_group,
                txt_preview_group,
                txt_preview_dropdown,
                batch_preview_dropdown,
                generation_status,
                batch_audio_map,
            ],
            show_progress=False,
        )

        preview_txt_btn.click(
            fn=_show_txt_preview,
            inputs=[txt_preview_dropdown],
            outputs=[txt_status],
            show_progress=False,
        )

        hide_txt_preview_btn.click(
            fn=lambda: gr.update(value="", visible=False),
            outputs=[txt_status],
            show_progress=False,
        )

        run_btn.click(
            fn=_prepare_generation_feedback,
            inputs=[random_seed, seed_value, txt_upload],
            outputs=[audio_output, seed_value, generation_status, batch_output, batch_result_group, batch_preview_dropdown, run_btn, batch_audio_map, batch_job_id, manual_audio_selection],
            show_progress=False,
        ).then(
            fn=_generate_or_batch,
            inputs=[
                text,
                control_instruction,
                reference_wav,
                show_prompt_text, prompt_text,
                cfg_value, DoNormalizeText, DoDenoisePromptAudio,
                dit_steps, seed_value, txt_upload,
            ],
            outputs=[audio_output, seed_value, batch_output, generation_status, batch_result_group, batch_preview_dropdown, run_btn, batch_audio_map, batch_job_id, manual_audio_selection],
            show_progress=True,
            api_name="generate",
        )

        batch_job_timer.tick(
            fn=_poll_batch_job,
            inputs=[batch_job_id, manual_audio_selection],
            outputs=[audio_output, batch_output, generation_status, batch_result_group, batch_preview_dropdown, batch_audio_map, run_btn, batch_job_id, manual_audio_selection],
            show_progress=False,
        )

        batch_preview_dropdown.input(
            fn=lambda choice, audio_map: (gr.update(value=(audio_map or {}).get(choice)), True),
            inputs=[batch_preview_dropdown, batch_audio_map],
            outputs=[audio_output, manual_audio_selection],
        )

        # ─── 保存音色 [克隆] ───
        def _save_voice_clone(name, ref_audio, asr_text):
            if not name or not name.strip():
                return "请输入音色名称"
            if not ref_audio:
                return "请先上传参考音频"
            voice_name = name.strip()
            lib = _load_voice_lib()
            audio_copy = VOICES_DIR / f"{_safe_voice_filename(voice_name)}.wav"
            shutil.copy2(ref_audio, audio_copy)
            cache_path = _voice_cache_path(voice_name)
            cache_status = ""
            try:
                with MODEL_RUNTIME_LOCK:
                    cache = demo.get_or_load_voxcpm().prepare_prompt_cache(
                        reference_wav_path=str(audio_copy),
                        denoise=False,
                    )
                _save_prompt_cache_file(cache, cache_path)
                cache_status = "，缓存已生成"
            except Exception as exc:
                logger.exception("Failed to save voice prompt cache.")
                cache_path = None
                cache_status = f"，但缓存生成失败：{friendly_runtime_error(exc)}"
            lib.append({
                "name": voice_name,
                "type": "clone",
                "audio": str(audio_copy),
                "asr_text": asr_text or "",
                "prompt_cache": str(cache_path) if cache_path else "",
            })
            _save_voice_lib(lib)
            return f"✓ 已保存: [克隆] {voice_name}{cache_status}"

        save_clone_btn.click(
            fn=_save_voice_clone,
            inputs=[voice_name_input, reference_wav, prompt_text],
            outputs=[save_status],
        ).then(
            fn=lambda: gr.update(choices=_get_voice_choices()),
            outputs=[voice_lib_dropdown],
        )

        # ─── 保存音色 [设计] ───
        def _save_voice_design(name, ctrl, cfg_val, steps_val, seed_val):
            if not name or not name.strip():
                return "请输入音色名称"
            lib = _load_voice_lib()
            lib.append({
                "name": name.strip(),
                "type": "design",
                "ctrl": ctrl or "",
                "cfg": float(cfg_val),
                "steps": int(steps_val),
                "seed": int(seed_val) if seed_val else 0,
            })
            _save_voice_lib(lib)
            return f"✓ 已保存: [设计] {name.strip()}"

        save_design_btn.click(
            fn=_save_voice_design,
            inputs=[voice_name_input, control_instruction, cfg_value, dit_steps, seed_value],
            outputs=[save_status],
        ).then(
            fn=lambda: gr.update(choices=_get_voice_choices()),
            outputs=[voice_lib_dropdown],
        )

        refresh_voice_lib_btn.click(
            fn=lambda: gr.update(choices=_get_voice_choices(), value=None),
            outputs=[voice_lib_dropdown],
        )

        # ─── 加载音色 ───
        def _load_voice(choice):
            if not choice or choice == "(空)":
                return [gr.update()] * 4
            lib = _get_voice_entries()
            tag = "克隆" if "[克隆]" in choice else "设计"
            name = choice.split("] ")[1] if "] " in choice else choice
            for v in lib:
                v_tag = "克隆" if v["type"] == "clone" else "设计"
                if v["name"] == name and v_tag == tag:
                    if v["type"] == "clone":
                        return [
                            gr.update(value=v.get("audio")),
                            gr.update(value=v.get("asr_text", "")),
                            gr.update(),
                            gr.update(),
                        ]
                    else:
                        return [
                            gr.update(),
                            gr.update(),
                            gr.update(value=v.get("ctrl", "")),
                            gr.update(value=v.get("cfg", 2.0)),
                        ]
            return [gr.update()] * 4

        voice_lib_dropdown.change(
            fn=_load_voice,
            inputs=[voice_lib_dropdown],
            outputs=[reference_wav, prompt_text, control_instruction, cfg_value],
        )

    return interface


def _start_background_warmup(demo: VoxCPMDemo, delay_seconds: float = 4.0):
    def _warmup_worker():
        time.sleep(delay_seconds)
        try:
            logger.info("Starting VoxCPM warmup.")
            demo.generate_tts_audio(
                text_input="热身。",
                control_instruction="",
                reference_wav_path_input=None,
                prompt_text="",
                cfg_value_input=2.0,
                do_normalize=False,
                denoise=False,
                inference_timesteps=10,
                seed=1,
                retry_badcase=True,
            )
            logger.info("VoxCPM warmup finished.")
        except Exception as exc:
            logger.warning("VoxCPM warmup skipped or failed: %s", exc)

    thread = threading.Thread(target=_warmup_worker, name="voxcpm-warmup", daemon=True)
    thread.start()


def run_demo(
    server_name: str = "127.0.0.1",
    server_port: int = 8808,
    show_error: bool = True,
    model_id: str = "openbmb/VoxCPM2",
    device: str = "auto",
    optimize: bool = False,
    warmup: bool = True,
):
    demo = VoxCPMDemo(model_id=model_id, device=device, optimize=optimize)
    interface = create_demo_interface(demo)
    if warmup:
        _start_background_warmup(demo)
    interface.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name=server_name,
        server_port=server_port,
        show_error=show_error,
        i18n=I18N,
        theme=_APP_THEME,
        css=_CUSTOM_CSS,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="openbmb/VoxCPM2",
        help="Local path or HuggingFace repo ID (default: openbmb/VoxCPM2)",
    )
    parser.add_argument("--port", type=int, default=8808, help="Server port")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind address. Use 127.0.0.1 to restrict access to the local machine; "
             "use 0.0.0.0 only when you intentionally want LAN access (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Runtime device: auto, cpu, mps, cuda, or cuda:N (default: auto)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable torch compile optimization. This can improve repeated inference speed but makes first use slower.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable background model warmup after server startup.",
    )
    args = parser.parse_args()
    run_demo(
        model_id=args.model_id,
        server_name=args.host,
        server_port=args.port,
        device=args.device,
        optimize=args.optimize,
        warmup=not args.no_warmup,
    )
