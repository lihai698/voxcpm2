import os
import re
import sys
import logging
import random
import numpy as np
import gradio as gr
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

DEFAULT_TARGET_TEXT = (
    "VoxCPM2 is a creative multilingual TTS model from ModelBest, " "designed to generate highly realistic speech."
)

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
.txt-upload-limited .file-preview {
    max-height: 108px;
    overflow-y: auto;
}
.txt-upload-limited ul {
    max-height: 108px;
    overflow-y: auto;
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
    def __init__(self, model_id: str = "openbmb/VoxCPM2", device: str = "auto") -> None:
        self.device = resolve_runtime_device(device, "cuda")
        logger.info(f"Running VoxCPM on device: {self.device}")
        self.optimize = self.device.startswith("cuda")

        self.asr_model_id = "iic/SenseVoiceSmall"
        self.asr_device = "cuda:0" if self.device.startswith("cuda") else "cpu"
        self.asr_model: Optional[AutoModel] = None

        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self._model_id = model_id

    def get_or_load_voxcpm(self) -> voxcpm.VoxCPM:
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
    ) -> dict:
        generate_kwargs = dict(
            text=final_text,
            reference_wav_path=audio_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=inference_timesteps,
            normalize=do_normalize,
            denoise=denoise,
            seed=seed,
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

        if audio_path and prompt_text_clean:
            logger.info(f"[Voice Cloning] prompt_wav + prompt_text + reference_wav")
        elif audio_path:
            logger.info(f"[Voice Control] reference_wav only")
        else:
            logger.info(f"[Voice Design] control: {control[:50] if control else 'None'}...")

        logger.info(f"Generating audio for text: '{final_text[:80]}...'")
        generate_kwargs = self._build_generate_kwargs(
            final_text=final_text,
            audio_path=audio_path,
            prompt_text_clean=prompt_text_clean,
            cfg_value_input=cfg_value_input,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=inference_timesteps,
            seed=seed,
        )
        try:
            wav = current_model.generate(**generate_kwargs)
        except Exception as exc:
            logger.exception("VoxCPM generation failed.")
            raise friendly_runtime_error(exc) from exc
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
                    lines=3,
                )

                # TXT 批量上传
                txt_upload = gr.File(
                    label="📄 上传 TXT 批量生成（多选，每个文件生成一条音频）",
                    file_types=[".txt"],
                    file_count="multiple",
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
                with gr.Row(elem_classes=["compact-result-row"]):
                    batch_preview_dropdown = gr.Dropdown(
                        label="🎧 试听批量结果",
                        choices=[],
                        value=None,
                        visible=False,
                        interactive=True,
                        scale=2,
                    )
                    batch_output = gr.File(label="📦 ZIP 下载", visible=False, scale=1)
                batch_preview_audio = gr.Audio(label="当前试听音频", visible=False)

                # 保存音色
                with gr.Accordion("💾 保存音色", open=False):
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

        examples_info_btn.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)),
            outputs=[examples_info_panel, modes_info_panel, settings_panel],
            show_progress=False,
        )

        modes_info_btn.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)),
            outputs=[examples_info_panel, modes_info_panel, settings_panel],
            show_progress=False,
        )

        settings_info_btn.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)),
            outputs=[examples_info_panel, modes_info_panel, settings_panel],
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
                    gr.update(choices=[], value=None, visible=False),
                    gr.update(choices=[], value=None, visible=False),
                    gr.update(value=None, visible=False),
                )

            choices = []
            for f in txt_files:
                fpath = f.name if hasattr(f, "name") else f
                choices.append((Path(fpath).name, fpath))
            first_value = choices[0][1] if choices else None
            return (
                gr.update(value="", visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(choices=choices, value=first_value, label=f"选择预览 TXT（共 {len(txt_files)} 个）"),
                gr.update(choices=[], value=None, visible=False),
                gr.update(value=None, visible=False),
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
                    seed,
                    gr.update(value=status, visible=True),
                    gr.update(visible=False),
                    gr.update(choices=[], value=None, visible=False),
                    gr.update(value=None, visible=False),
                )
            return (
                seed,
                gr.update(value="", visible=False),
                gr.update(visible=False),
                gr.update(choices=[], value=None, visible=False),
                gr.update(value=None, visible=False),
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
        ):
            if not txt_files:
                return (
                    gr.update(visible=False),
                    gr.update(value="请先上传 TXT 文件。", visible=True),
                    gr.update(choices=[], value=None, visible=False),
                    gr.update(value=None, visible=False),
                )
            import tempfile, zipfile
            out_dir = Path(tempfile.mkdtemp(prefix="voxcpm_batch_"))
            status_lines = []
            generated_files = []
            generated_count = 0
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
                preview = re.sub(r"\s+", " ", content[:80])
                status_lines.append(f"已读取：{Path(fpath).name}，{char_count} 字，编码 {encoding}，预览：{preview}")
                seed = _prepare_seed(True, seed_val)
                actual_prompt = prompt_text_val.strip() if use_prompt_text else ""
                actual_ctrl = "" if use_prompt_text else control_instruction_val
                try:
                    sr, wav_np, _ = demo.generate_tts_audio(
                        text_input=content,
                        control_instruction=actual_ctrl,
                        reference_wav_path_input=ref_wav,
                        prompt_text=actual_prompt,
                        cfg_value_input=cfg_val,
                        do_normalize=do_normalize,
                        denoise=denoise,
                        inference_timesteps=int(dit_steps_val),
                        seed=seed,
                    )
                    import soundfile as sf
                    safe_stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(fpath).stem).strip() or "txt"
                    out_name = f"{index:03d}_{safe_stem}.wav"
                    out_path = out_dir / out_name
                    sf.write(str(out_path), wav_np, sr)
                    generated_files.append(out_path)
                    generated_count += 1
                except Exception as e:
                    logger.error(f"Batch gen failed for {fpath}: {e}")
                    status_lines.append(f"生成失败：{Path(fpath).name}（{e}）")

            if generated_count == 0:
                status = "\n".join(status_lines) if status_lines else "没有可生成的 TXT 内容。"
                return (
                    gr.update(visible=False),
                    gr.update(value=status, visible=True),
                    gr.update(choices=[], value=None, visible=False),
                    gr.update(value=None, visible=False),
                )

            zip_path = out_dir.parent / f"{out_dir.name}.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                for wav_file in sorted(out_dir.glob("*.wav")):
                    zf.write(wav_file, wav_file.name)
            status_lines.append(f"完成：生成 {generated_count} 条音频。")
            choices = [(wav_file.name, str(wav_file)) for wav_file in generated_files]
            first_audio = str(generated_files[0]) if generated_files else None
            return (
                gr.update(value=str(zip_path), visible=True),
                gr.update(value="\n".join(status_lines), visible=True),
                gr.update(choices=choices, value=first_audio, visible=True),
                gr.update(value=first_audio, visible=True),
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
            if txt_files:
                batch_file, status, preview_choices, preview_audio = _batch_generate(
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
                return (
                    gr.update(value=None),
                    seed_val,
                    batch_file,
                    status,
                    preview_choices,
                    preview_audio,
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
                gr.update(value="", visible=False),
                gr.update(choices=[], value=None, visible=False),
                gr.update(value=None, visible=False),
            )

        txt_upload.change(
            fn=_preview_txt_files,
            inputs=[txt_upload],
            outputs=[txt_status, batch_output, txt_preview_group, txt_preview_dropdown, batch_preview_dropdown, batch_preview_audio],
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
            outputs=[seed_value, txt_status, batch_output, batch_preview_dropdown, batch_preview_audio],
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
            outputs=[audio_output, seed_value, batch_output, txt_status, batch_preview_dropdown, batch_preview_audio],
            show_progress=True,
            api_name="generate",
        )

        batch_preview_dropdown.change(
            fn=lambda audio_path: gr.update(value=audio_path, visible=bool(audio_path)),
            inputs=[batch_preview_dropdown],
            outputs=[batch_preview_audio],
        )

        # ─── 保存音色 [克隆] ───
        def _save_voice_clone(name, ref_audio, asr_text):
            if not name or not name.strip():
                return "请输入音色名称"
            if not ref_audio:
                return "请先上传参考音频"
            lib = _load_voice_lib()
            audio_copy = VOICES_DIR / f"{name.strip()}.wav"
            shutil.copy2(ref_audio, audio_copy)
            lib.append({
                "name": name.strip(),
                "type": "clone",
                "audio": str(audio_copy),
                "asr_text": asr_text or "",
            })
            _save_voice_lib(lib)
            return f"✓ 已保存: [克隆] {name.strip()}"

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


def run_demo(
    server_name: str = "127.0.0.1",
    server_port: int = 8808,
    show_error: bool = True,
    model_id: str = "openbmb/VoxCPM2",
    device: str = "auto",
):
    demo = VoxCPMDemo(model_id=model_id, device=device)
    interface = create_demo_interface(demo)
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
    args = parser.parse_args()
    run_demo(
        model_id=args.model_id,
        server_name=args.host,
        server_port=args.port,
        device=args.device,
    )
