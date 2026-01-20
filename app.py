import os
import time
import random
import base64
import textwrap
import io
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st
import yaml

# Optional imports – wrapped in try/except
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    import anthropic
except Exception:
    anthropic = None

import httpx

# Optional doc/OCR libs
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import easyocr
except Exception:
    easyocr = None


# ------------- GLOBAL CONSTANTS ------------------------------------------------

APP_TITLE = "FDA 510(k) Agentic Review WOW Studio"

MODEL_OPTIONS = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "grok-4-fast-reasoning",
    "grok-3-mini",
]

LLM_OCR_MODELS = [
    "gpt-4o-mini",
    "gemini-3-flash-preview",
]

PAINTER_STYLES = {
    "Van Gogh": {"accent": "#FFB300", "bg": "linear-gradient(135deg,#1e3c72,#2a5298)"},
    "Monet": {"accent": "#5EC2F2", "bg": "linear-gradient(135deg,#b2fefa,#0ed2f7)"},
    "Da Vinci": {"accent": "#CDAA7D", "bg": "linear-gradient(135deg,#2c3e50,#bdc3c7)"},
    "Picasso": {"accent": "#FF4B81", "bg": "linear-gradient(135deg,#000000,#434343)"},
    "Matisse": {"accent": "#1ABC9C", "bg": "linear-gradient(135deg,#1abc9c,#16a085)"},
    "Klimt": {"accent": "#F1C40F", "bg": "linear-gradient(135deg,#f1c40f,#e67e22)"},
    "Hokusai": {"accent": "#2980B9", "bg": "linear-gradient(135deg,#2c3e50,#2980b9)"},
    "Frida Kahlo": {"accent": "#E74C3C", "bg": "linear-gradient(135deg,#e74c3c,#8e44ad)"},
    "Rembrandt": {"accent": "#A67C52", "bg": "linear-gradient(135deg,#3c2a21,#8e5a2a)"},
    "Dali": {"accent": "#F39C12", "bg": "linear-gradient(135deg,#f39c12,#d35400)"},
    "Cézanne": {"accent": "#3498DB", "bg": "linear-gradient(135deg,#3498db,#2ecc71)"},
    "Renoir": {"accent": "#E67E22", "bg": "linear-gradient(135deg,#f39c12,#e67e22)"},
    "Turner": {"accent": "#F5B041", "bg": "linear-gradient(135deg,#f5b041,#f7dc6f)"},
    "Goya": {"accent": "#7F8C8D", "bg": "linear-gradient(135deg,#2c3e50,#7f8c8d)"},
    "Basquiat": {"accent": "#F1C40F", "bg": "linear-gradient(135deg,#000000,#f1c40f)"},
    "Pollock": {"accent": "#E74C3C", "bg": "linear-gradient(135deg,#2c3e50,#e74c3c)"},
    "O'Keeffe": {"accent": "#9B59B6", "bg": "linear-gradient(135deg,#9b59b6,#e91e63)"},
    "Chagall": {"accent": "#8E44AD", "bg": "linear-gradient(135deg,#0f2027,#8e44ad)"},
    "Vermeer": {"accent": "#2980B9", "bg": "linear-gradient(135deg,#f1c40f,#2980b9)"},
    "Michelangelo": {"accent": "#D35400", "bg": "linear-gradient(135deg,#2c3e50,#d35400)"},
}


# ------------- INTERNATIONALIZATION -------------------------------------------

def get_i18n_dict() -> Dict[str, Dict[str, str]]:
    return {
        "app_title": {"en": APP_TITLE, "zh-tw": "FDA 510(k) 智慧審查 WOW 工作室"},
        "tab_agents": {"en": "Agent Runner", "zh-tw": "代理人執行器"},
        "tab_dashboard": {"en": "Dashboard", "zh-tw": "儀表板"},
        "tab_notes": {"en": "AI Note Keeper", "zh-tw": "AI 筆記管家"},
        "tab_docs": {"en": "Doc OCR & Summary", "zh-tw": "文件 OCR 與摘要"},
        "tab_agents_yaml": {"en": "Agent YAML Studio", "zh-tw": "代理 YAML 工作室"},
        "tab_skill_md": {"en": "SKILL.md Studio", "zh-tw": "SKILL.md 工作室"},
        "sidebar_language": {"en": "Language", "zh-tw": "語言"},
        "sidebar_theme": {"en": "Theme", "zh-tw": "主題"},
        "sidebar_style": {"en": "Painter Style", "zh-tw": "畫風樣式"},
        "sidebar_jackpot": {"en": "Jackpot Style", "zh-tw": "隨機大獎樣式"},
        "sidebar_api_keys": {"en": "API Keys", "zh-tw": "API 金鑰"},
        "sidebar_llm_settings": {"en": "LLM Settings", "zh-tw": "LLM 設定"},
        "light": {"en": "Light", "zh-tw": "亮色"},
        "dark": {"en": "Dark", "zh-tw": "暗色"},
        "provider_openai": {"en": "OpenAI", "zh-tw": "OpenAI"},
        "provider_gemini": {"en": "Gemini", "zh-tw": "Gemini"},
        "provider_anthropic": {"en": "Anthropic", "zh-tw": "Anthropic"},
        "provider_grok": {"en": "Grok", "zh-tw": "Grok"},
        "api_from_env": {"en": "Using environment key", "zh-tw": "使用環境變數金鑰"},
        "api_enter": {"en": "Enter API key", "zh-tw": "請輸入 API 金鑰"},
        "model": {"en": "Model", "zh-tw": "模型"},
        "max_tokens": {"en": "Max tokens", "zh-tw": "最大 tokens"},
        "temperature": {"en": "Temperature", "zh-tw": "溫度"},
        "agent_select": {"en": "Select agent", "zh-tw": "選擇代理人"},
        "agent_input": {"en": "Agent input (you can edit previous output here)", "zh-tw": "代理人輸入（可在此編輯上一個輸出）"},
        "run_agent": {"en": "Run agent", "zh-tw": "執行代理人"},
        "use_last_output": {"en": "Use last agent output as input", "zh-tw": "使用上一個代理人輸出作為輸入"},
        "save_for_next": {"en": "Save as input for next agent", "zh-tw": "儲存為下一個代理人輸入"},
        "output_markdown": {"en": "Markdown view", "zh-tw": "Markdown 檢視"},
        "output_text": {"en": "Text view (editable)", "zh-tw": "文字檢視（可編輯）"},
        "status_section": {"en": "WOW Status Indicators", "zh-tw": "WOW 狀態指標"},
        "status_api": {"en": "API Connectivity", "zh-tw": "API 連線狀態"},
        "status_docs": {"en": "Documents", "zh-tw": "文件"},
        "status_runs": {"en": "Agent Runs", "zh-tw": "代理人執行次數"},
        "notes_paste": {"en": "Paste your text / markdown", "zh-tw": "貼上文字或 Markdown"},
        "notes_transform": {"en": "Transform to organized markdown", "zh-tw": "轉換為結構化 Markdown"},
        "notes_model": {"en": "Model for note operations", "zh-tw": "筆記處理模型"},
        "notes_view_mode": {"en": "Note view mode", "zh-tw": "筆記檢視模式"},
        "view_markdown": {"en": "Markdown", "zh-tw": "Markdown"},
        "view_text": {"en": "Text", "zh-tw": "文字"},
        "ai_magics": {"en": "AI Magics", "zh-tw": "AI 魔法"},
        "magic_format": {"en": "AI Formatting", "zh-tw": "AI 格式優化"},
        "magic_keywords": {"en": "AI Keywords Highlighter", "zh-tw": "AI 關鍵字標示"},
        "magic_summary": {"en": "AI Summary", "zh-tw": "AI 摘要"},
        "magic_translate": {"en": "AI EN ↔ 繁中 Translate", "zh-tw": "AI 中英互譯"},
        "magic_expand": {"en": "AI Expansion / Elaboration", "zh-tw": "AI 擴寫說明"},
        "magic_actions": {"en": "AI Action Items", "zh-tw": "AI 行動清單"},
        "keywords_input": {"en": "Keywords (comma separated)", "zh-tw": "關鍵字（以逗號分隔）"},
        "keyword_color": {"en": "Keyword color", "zh-tw": "關鍵字顏色"},
    }


def t(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    return get_i18n_dict().get(key, {}).get(lang, key)


# ------------- SESSION STATE INIT & THEME -------------------------------------

def init_session_state():
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"
    if "theme" not in st.session_state:
        st.session_state["theme"] = "light"
    if "painter_style" not in st.session_state:
        st.session_state["painter_style"] = "Van Gogh"
    if "api_keys" not in st.session_state:
        st.session_state["api_keys"] = {}
    if "agents_config" not in st.session_state:
        st.session_state["agents_config"] = load_agents_config()
    if "run_log" not in st.session_state:
        st.session_state["run_log"] = []
    if "last_agent_output" not in st.session_state:
        st.session_state["last_agent_output"] = ""
    if "note_content" not in st.session_state:
        st.session_state["note_content"] = ""
    if "note_view_mode" not in st.session_state:
        st.session_state["note_view_mode"] = "markdown"

    # Doc workspace state
    if "uploaded_docs_raw" not in st.session_state:
        st.session_state["uploaded_docs_raw"] = []
    if "doc_paste_raw" not in st.session_state:
        st.session_state["doc_paste_raw"] = ""
    if "doc_pdf_bytes" not in st.session_state:
        st.session_state["doc_pdf_bytes"] = None
    if "doc_pdf_name" not in st.session_state:
        st.session_state["doc_pdf_name"] = None
    if "doc_pdf_page_count" not in st.session_state:
        st.session_state["doc_pdf_page_count"] = 0
    if "doc_selected_pages" not in st.session_state:
        st.session_state["doc_selected_pages"] = []
    if "doc_ocr_text" not in st.session_state:
        st.session_state["doc_ocr_text"] = ""
    if "doc_summary_md" not in st.session_state:
        st.session_state["doc_summary_md"] = ""
    if "doc_view_mode_ocr" not in st.session_state:
        st.session_state["doc_view_mode_ocr"] = "markdown"
    if "doc_view_mode_summary" not in st.session_state:
        st.session_state["doc_view_mode_summary"] = "markdown"
    if "doc_chat_history" not in st.session_state:
        st.session_state["doc_chat_history"] = []  # list[{role, content}]

    # YAML studio
    if "agents_yaml_text" not in st.session_state:
        st.session_state["agents_yaml_text"] = dump_yaml_safe(st.session_state["agents_config"])
    if "agents_yaml_last_standardized" not in st.session_state:
        st.session_state["agents_yaml_last_standardized"] = ""

    # SKILL.md studio
    if "skill_instructions" not in st.session_state:
        st.session_state["skill_instructions"] = ""
    if "skill_md_text" not in st.session_state:
        st.session_state["skill_md_text"] = ""


def apply_theme():
    theme = st.session_state.get("theme", "light")
    style_name = st.session_state.get("painter_style", "Van Gogh")
    style_cfg = PAINTER_STYLES.get(style_name, PAINTER_STYLES["Van Gogh"])
    accent = style_cfg["accent"]
    bg = style_cfg["bg"]

    base_bg = "#111111" if theme == "dark" else "#FFFFFF"
    base_text = "#F5F5F5" if theme == "dark" else "#111111"

    css = f"""
    <style>
      .stApp {{
        background: {bg};
        background-attachment: fixed;
      }}
      .main-wrapping-container {{
        background: {base_bg}CC;
        padding: 1.5rem;
        border-radius: 1rem;
      }}
      .wow-badge {{
        display:inline-block;
        padding:0.15rem 0.45rem;
        border-radius:999px;
        font-size:0.7rem;
        margin-right:0.15rem;
        background:{accent}22;
        color:{accent};
        border:1px solid {accent}66;
      }}
      .wow-chip-ok {{
        background:#2ecc7122;
        color:#2ecc71;
        border:1px solid #2ecc7166;
      }}
      .wow-chip-warn {{
        background:#f1c40f22;
        color:#f1c40f;
        border:1px solid #f1c40f66;
      }}
      .wow-chip-bad {{
        background:#e74c3c22;
        color:#e74c3c;
        border:1px solid #e74c3c66;
      }}
      h1, h2, h3, h4, h5, h6, p, span, label {{
        color:{base_text};
      }}
      .coral-keyword {{
        color: coral;
        font-weight: 700;
      }}
      .small-muted {{
        opacity: 0.8;
        font-size: 0.9rem;
      }}
      code, pre {{
        white-space: pre-wrap !important;
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ------------- YAML UTILS ------------------------------------------------------

def dump_yaml_safe(obj: Dict[str, Any]) -> str:
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)


def load_yaml_safe(text: str) -> Dict[str, Any]:
    return yaml.safe_load(text) or {}


def standardize_agents_yaml(raw: Any) -> Dict[str, Any]:
    """
    Accepts many shapes and normalizes into:
    {
      defaults: {max_tokens:int, temperature:float},
      agents: {
        agent_id: {
          skill_number:int, name:str, category:str, difficulty:str,
          description:str, dependencies:list, estimated_tokens:int,
          default_model:str, system_prompt:str
        }
      }
    }
    """
    defaults = {"max_tokens": 12000, "temperature": 0.2}
    out = {"defaults": defaults, "agents": {}}

    if raw is None:
        return out

    # If someone uploads a list of agents
    if isinstance(raw, list):
        agents_list = raw
    elif isinstance(raw, dict):
        if "agents" in raw and isinstance(raw["agents"], dict):
            # almost standard already
            out["defaults"] = {**defaults, **(raw.get("defaults") or {})}
            for agent_id, cfg in (raw.get("agents") or {}).items():
                out["agents"][str(agent_id)] = normalize_agent_cfg(cfg, agent_id=str(agent_id))
            # ensure skill_number ordering if missing
            out["agents"] = ensure_skill_numbers(out["agents"])
            return out
        if "agents" in raw and isinstance(raw["agents"], list):
            agents_list = raw["agents"]
        else:
            # heuristic: dict might itself be agent map
            if any(isinstance(v, dict) and ("system_prompt" in v or "prompt" in v) for v in raw.values()):
                agents_list = []
                for k, v in raw.items():
                    cfg = v if isinstance(v, dict) else {"system_prompt": str(v)}
                    cfg["_id_hint"] = str(k)
                    agents_list.append(cfg)
            else:
                # fallback: treat entire dict as one agent
                agents_list = [raw]
    else:
        agents_list = [{"system_prompt": str(raw)}]

    for idx, a in enumerate(agents_list, start=1):
        agent_id = str(a.get("id") or a.get("_id_hint") or a.get("name") or f"agent_{idx}")
        agent_id = to_snake_kebab(agent_id).replace("-", "_")
        out["agents"][agent_id] = normalize_agent_cfg(a, agent_id=agent_id)

    out["agents"] = ensure_skill_numbers(out["agents"])
    return out


def ensure_skill_numbers(agents: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    # assign missing numbers and then re-sort
    used = set()
    for cfg in agents.values():
        n = cfg.get("skill_number")
        if isinstance(n, int):
            used.add(n)

    next_n = 1
    for agent_id, cfg in agents.items():
        if not isinstance(cfg.get("skill_number"), int):
            while next_n in used:
                next_n += 1
            cfg["skill_number"] = next_n
            used.add(next_n)
            next_n += 1

    # return sorted by skill_number, but keep as dict
    items = sorted(agents.items(), key=lambda kv: kv[1].get("skill_number", 999999))
    return {k: v for k, v in items}


def normalize_agent_cfg(cfg: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
    cfg = cfg or {}
    sys_prompt = cfg.get("system_prompt") or cfg.get("prompt") or cfg.get("system") or ""
    if not isinstance(sys_prompt, str):
        sys_prompt = str(sys_prompt)

    default_model = cfg.get("default_model") or cfg.get("model") or "gpt-4o-mini"
    if default_model not in MODEL_OPTIONS:
        # keep unknown model but don't crash; user may add custom
        default_model = str(default_model)

    normalized = {
        "skill_number": cfg.get("skill_number"),
        "name": cfg.get("name") or agent_id.replace("_", " ").title(),
        "category": cfg.get("category") or "General",
        "difficulty": cfg.get("difficulty") or "intermediate",
        "description": cfg.get("description") or "",
        "dependencies": cfg.get("dependencies") or [],
        "estimated_tokens": int(cfg.get("estimated_tokens") or 8000),
        "default_model": default_model,
        "system_prompt": sys_prompt.strip(),
    }
    if not isinstance(normalized["dependencies"], list):
        normalized["dependencies"] = [str(normalized["dependencies"])]
    return normalized


def to_snake_kebab(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s\-]+", "", s)
    s = re.sub(r"[\s]+", "-", s)
    s = re.sub(r"[-]+", "-", s)
    return s.strip("-")


# ------------- LLM PROVIDER MANAGER ------------------------------------------

class LLMProviderManager:
    def __init__(self):
        self.env_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "grok": os.getenv("GROK_API_KEY"),
        }

    def get_effective_key(self, provider: str) -> Optional[str]:
        user_key = st.session_state["api_keys"].get(provider)
        if user_key:
            return user_key
        return self.env_keys.get(provider)

    def identify_provider(self, model: str) -> str:
        if model.startswith("gpt-"):
            return "openai"
        if model.startswith("gemini-"):
            return "gemini"
        if "claude" in model or "anthropic" in model:
            return "anthropic"
        if "grok" in model:
            return "grok"
        return "openai"

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 12000,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        provider = self.identify_provider(model)
        api_key = self.get_effective_key(provider)
        if not api_key:
            raise RuntimeError(f"No API key found for provider: {provider}")

        start = time.time()
        content = ""
        tokens_used = None

        if provider == "openai":
            if OpenAI is None:
                raise RuntimeError("openai package not available")
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = resp.choices[0].message.content
            tokens_used = getattr(resp.usage, "total_tokens", None)

        elif provider == "gemini":
            if genai is None:
                raise RuntimeError("google-generativeai package not available")
            genai.configure(api_key=api_key)
            model_obj = genai.GenerativeModel(model)
            joined = []
            for m in messages:
                prefix = "System: " if m["role"] == "system" else "User: "
                joined.append(prefix + m["content"])
            prompt = "\n".join(joined)
            resp = model_obj.generate_content(prompt)
            content = resp.text

        elif provider == "anthropic":
            if anthropic is None:
                raise RuntimeError("anthropic package not available")
            client = anthropic.Anthropic(api_key=api_key)
            sys_prompt = ""
            user_content = ""
            for m in messages:
                if m["role"] == "system":
                    sys_prompt += m["content"] + "\n"
                else:
                    user_content += m["content"] + "\n"
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_prompt.strip(),
                messages=[{"role": "user", "content": user_content.strip()}],
            )
            content = "".join([b.text for b in resp.content if getattr(b, "type", "") == "text"])
            tokens_used = getattr(resp.usage, "input_tokens", 0) + getattr(resp.usage, "output_tokens", 0)

        elif provider == "grok":
            headers = {"Authorization": f"Bearer {api_key}"}
            json_payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            with httpx.Client(timeout=180) as client:
                r = client.post("https://api.x.ai/v1/chat/completions", headers=headers, json=json_payload)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            tokens_used = data.get("usage", {}).get("total_tokens")

        duration = time.time() - start
        return {"content": content, "tokens_used": tokens_used, "duration": duration, "provider": provider}

    def ocr_images_with_llm(
        self,
        model: str,
        images: List["Image.Image"],
        lang: str = "en",
        max_tokens: int = 4096,
    ) -> str:
        """
        Multimodal OCR via OpenAI or Gemini. Returns markdown.
        """
        provider = self.identify_provider(model)
        api_key = self.get_effective_key(provider)
        if not api_key:
            raise RuntimeError(f"No API key found for provider: {provider}")
        if Image is None:
            raise RuntimeError("Pillow (PIL) is required for LLM OCR")

        sys = (
            "You are an OCR engine. Extract all readable text accurately. "
            "Preserve paragraphs, headings, tables as markdown when possible. "
            "Do not hallucinate. If uncertain, mark as [unclear]. "
        )
        if lang == "zh-tw":
            sys += "The document may contain Traditional Chinese. Keep original language."

        if provider == "openai":
            if OpenAI is None:
                raise RuntimeError("openai package not available")
            client = OpenAI(api_key=api_key)

            content_parts = [{"type": "text", "text": "Perform OCR on these page images and output markdown."}]
            for im in images:
                b64 = pil_to_base64_png(im)
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": content_parts},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return resp.choices[0].message.content

        if provider == "gemini":
            if genai is None:
                raise RuntimeError("google-generativeai package not available")
            genai.configure(api_key=api_key)
            model_obj = genai.GenerativeModel(model)
            parts = [sys + "\n\nPerform OCR on the images and output markdown.\n"]
            parts.extend(images)
            resp = model_obj.generate_content(parts)
            return resp.text

        raise RuntimeError(f"LLM OCR not supported for provider: {provider}")


def pil_to_base64_png(im: "Image.Image") -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ------------- AGENT EXECUTOR -------------------------------------------------

def load_agents_config() -> Dict[str, Any]:
    path = Path("agents.yaml")
    if not path.exists():
        return {"agents": {}, "defaults": {"max_tokens": 12000, "temperature": 0.2}}
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return standardize_agents_yaml(raw)


class AgentExecutor:
    def __init__(self, llm_manager: LLMProviderManager, agents_config: Dict[str, Any]):
        self.llm_manager = llm_manager
        self.config = agents_config or {"agents": {}, "defaults": {}}
        self.defaults = self.config.get("defaults", {})

    def list_agents(self) -> List[Dict[str, Any]]:
        agents = []
        for agent_id, cfg in self.config.get("agents", {}).items():
            agents.append({"id": agent_id, **cfg})
        agents.sort(key=lambda a: a.get("skill_number", 999999))
        return agents

    def execute(
        self,
        agent_id: str,
        user_input: str,
        model_override: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        user_prompt_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        agents = self.config.get("agents", {})
        if agent_id not in agents:
            raise RuntimeError(f"Unknown agent: {agent_id}")
        cfg = agents[agent_id]

        system_prompt = user_prompt_override if user_prompt_override is not None else cfg.get("system_prompt", "")
        model = model_override or cfg.get("default_model") or st.session_state.get("global_model", MODEL_OPTIONS[0])
        max_tokens = max_tokens or self.defaults.get("max_tokens", 12000)
        temperature = temperature if temperature is not None else self.defaults.get("temperature", 0.2)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input.strip() or "Use your configured behavior with the current context."},
        ]

        resp = self.llm_manager.chat_completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return {
            "status": "success",
            "output": resp["content"],
            "model": model,
            "tokens_used": resp.get("tokens_used"),
            "duration_seconds": resp.get("duration"),
        }


# ------------- SIDEBAR --------------------------------------------------------

def render_sidebar(llm_manager: LLMProviderManager):
    with st.sidebar:
        st.markdown("### 🎨 WOW Studio")

        lang = st.radio(
            t("sidebar_language"),
            ["en", "zh-tw"],
            index=0 if st.session_state["lang"] == "en" else 1,
            horizontal=True,
        )
        st.session_state["lang"] = lang

        theme = st.radio(
            t("sidebar_theme"),
            [t("light"), t("dark")],
            index=0 if st.session_state["theme"] == "light" else 1,
            horizontal=True,
        )
        st.session_state["theme"] = "light" if theme == t("light") else "dark"

        style = st.selectbox(
            t("sidebar_style"),
            list(PAINTER_STYLES.keys()),
            index=list(PAINTER_STYLES.keys()).index(st.session_state["painter_style"]),
        )
        st.session_state["painter_style"] = style
        if st.button("🎰 " + t("sidebar_jackpot")):
            st.session_state["painter_style"] = random.choice(list(PAINTER_STYLES.keys()))
            st.rerun()

        st.markdown("---")
        st.markdown(f"### 🔐 {t('sidebar_api_keys')}")

        def key_row(provider_key: str, label_key: str, env_var: str):
            env_val = llm_manager.env_keys.get(provider_key)
            st.caption(f"{t(label_key)} ({env_var})")
            col1, col2 = st.columns([3, 2])
            with col1:
                if env_val:
                    st.success("✅ " + t("api_from_env"))
                else:
                    api_val = st.text_input(
                        t("api_enter"),
                        type="password",
                        key=f"api_{provider_key}",
                    )
                    if api_val:
                        st.session_state["api_keys"][provider_key] = api_val
            with col2:
                effective = llm_manager.get_effective_key(provider_key)
                status_class = "wow-chip-ok" if effective else "wow-chip-bad"
                status_label = "ON" if effective else "OFF"
                st.markdown(f'<span class="wow-badge {status_class}">{status_label}</span>', unsafe_allow_html=True)

        key_row("openai", "provider_openai", "OPENAI_API_KEY")
        key_row("gemini", "provider_gemini", "GEMINI_API_KEY")
        key_row("anthropic", "provider_anthropic", "ANTHROPIC_API_KEY")
        key_row("grok", "provider_grok", "GROK_API_KEY")

        st.markdown("---")
        st.markdown(f"### 🤖 {t('sidebar_llm_settings')}")
        model = st.selectbox(t("model"), MODEL_OPTIONS, index=0)
        st.session_state["global_model"] = model
        max_tokens = st.number_input(t("max_tokens"), min_value=256, max_value=120000, value=12000, step=256)
        st.session_state["global_max_tokens"] = int(max_tokens)
        temperature = st.slider(t("temperature"), min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        st.session_state["global_temperature"] = float(temperature)


# ------------- WOW STATUS + DASHBOARD -----------------------------------------

def render_status_header(llm_manager: LLMProviderManager):
    st.markdown("#### ⚡ " + t("status_section"))
    col1, col2, col3 = st.columns(3)

    with col1:
        any_api = any(llm_manager.get_effective_key(p) for p in ["openai", "gemini", "anthropic", "grok"])
        cls = "wow-chip-ok" if any_api else "wow-chip-bad"
        label = "OK" if any_api else "Missing"
        st.markdown(f'{t("status_api")}<br><span class="wow-badge {cls}">{label}</span>', unsafe_allow_html=True)

    with col2:
        docs_loaded = bool(st.session_state.get("uploaded_docs_raw")) or bool(st.session_state.get("doc_pdf_bytes")) or bool(st.session_state.get("doc_paste_raw"))
        cls = "wow-chip-ok" if docs_loaded else "wow-chip-warn"
        label = "Loaded" if docs_loaded else "None"
        st.markdown(f'{t("status_docs")}<br><span class="wow-badge {cls}">{label}</span>', unsafe_allow_html=True)

    with col3:
        runs = len(st.session_state["run_log"])
        cls = "wow-chip-ok" if runs > 0 else "wow-chip-warn"
        st.markdown(f'{t("status_runs")}<br><span class="wow-badge {cls}">{runs} run(s)</span>', unsafe_allow_html=True)


def pd_timestamp_now():
    import pandas as _pd
    return _pd.Timestamp.utcnow()


def render_dashboard_tab():
    import pandas as pd
    import altair as alt

    st.markdown("### 📊 Interactive Review Dashboard")

    logs = st.session_state.get("run_log", [])
    if not logs:
        st.info("No agent runs yet. Execute an agent run first.")
        return

    df = pd.DataFrame(logs)
    with st.expander("Raw run log"):
        st.dataframe(df, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total runs", len(df))
    with col2:
        st.metric("Unique agents", df["agent_id"].nunique() if "agent_id" in df.columns else 0)
    with col3:
        st.metric("Total tokens (if reported)", int(df.get("tokens_used", pd.Series([0]*len(df))).fillna(0).sum()))

    if "tokens_used" in df.columns and "agent_id" in df.columns:
        chart_data = df.groupby("agent_id")["tokens_used"].sum().reset_index().sort_values("tokens_used", ascending=False)
        st.markdown("#### Tokens by agent")
        chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X("agent_id", sort="-y", title="Agent"),
                y=alt.Y("tokens_used", title="Tokens"),
                tooltip=["agent_id", "tokens_used"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    if "timestamp" in df.columns:
        st.markdown("#### Runs over time")
        chart2 = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x="timestamp:T",
                y="count()",
                tooltip=["timestamp", "agent_id", "model"],
                color="agent_id",
            )
            .properties(height=300)
        )
        st.altair_chart(chart2, use_container_width=True)


# ------------- AGENT RUNNER TAB ----------------------------------------------

def render_agent_output(result: Dict[str, Any], agent_cfg: Dict[str, Any], show_metrics: bool = True):
    if show_metrics:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", result.get("model", ""))
        with col2:
            st.metric("Tokens", result.get("tokens_used", 0) or 0)
        with col3:
            st.metric("Duration (s)", round(result.get("duration_seconds", 0) or 0, 2))

    tab1, tab2 = st.tabs([t("output_markdown"), t("output_text")])
    with tab1:
        st.markdown(result.get("output", ""), unsafe_allow_html=True)
    with tab2:
        edited = st.text_area(
            t("output_text"),
            value=result.get("output", ""),
            height=260,
        )
        if st.button("💾 " + t("save_for_next")):
            st.session_state["last_agent_output"] = edited
            st.success("Saved for next agent input.")


def render_agent_runner_tab(executor: AgentExecutor):
    st.markdown("### 🧠 " + t("tab_agents"))

    with st.expander("📁 Upload or paste submission materials", expanded=False):
        files = st.file_uploader(
            "Upload PDFs / text files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        if files:
            for f in files:
                try:
                    content = f.read()
                    try:
                        text = content.decode("utf-8", errors="ignore")
                    except Exception:
                        text = str(content)
                    st.session_state["uploaded_docs_raw"].append({"name": f.name, "size": f.size, "content": text})
                    st.success(f"Loaded {f.name}")
                except Exception as e:
                    st.error(f"Error reading {f.name}: {e}")

        paste = st.text_area("Paste text", height=150)
        if st.button("Save pasted as document"):
            if paste.strip():
                st.session_state["uploaded_docs_raw"].append(
                    {"name": f"Pasted-{len(st.session_state['uploaded_docs_raw'])+1}", "size": len(paste), "content": paste}
                )
                st.success("Pasted content saved.")
            else:
                st.warning("Nothing to save.")

        docs = st.session_state["uploaded_docs_raw"]
        if docs:
            st.markdown("**Loaded documents:**")
            for d in docs[-5:]:
                st.markdown(f"- {d['name']} ({len(d['content'])} chars)")

    agents = executor.list_agents()
    if not agents:
        st.error("No agents found in agents.yaml")
        return

    agent_labels = [f"[#{a.get('skill_number','?')}] {a.get('name','')} ({a['id']})" for a in agents]
    idx = st.selectbox(t("agent_select"), list(range(len(agents))), format_func=lambda i: agent_labels[i])
    agent_cfg = agents[idx]

    with st.expander("Agent details", expanded=False):
        st.markdown(f"**ID:** `{agent_cfg['id']}`")
        st.markdown(f"**Skill #:** {agent_cfg.get('skill_number','?')}")
        st.markdown(f"**Category:** {agent_cfg.get('category','')}")
        st.markdown(f"**Description:** {agent_cfg.get('description','')}")
        st.markdown("**System prompt (editable override below if needed):**")
        st.code(textwrap.shorten(agent_cfg.get("system_prompt", ""), width=1400, placeholder="..."))

    col_a, col_b = st.columns([2, 1])
    with col_b:
        use_last = st.checkbox(t("use_last_output"), value=False)

    if use_last and st.session_state["last_agent_output"]:
        default_input = st.session_state["last_agent_output"]
    else:
        docs = st.session_state.get("uploaded_docs_raw", [])
        merged = "\n\n".join(d["content"] for d in docs)
        default_input = textwrap.shorten(merged, width=8000, placeholder="\n\n[TRUNCATED DOCUMENT CONTENT]\n")

    with col_a:
        user_input = st.text_area(t("agent_input"), value=default_input, height=220)

    with st.expander("Override prompt / model & parameters (optional)", expanded=False):
        prompt_override = st.text_area(
            "Override system prompt (leave blank to use agent system prompt)",
            value="",
            height=160,
        )
        prompt_override = prompt_override if prompt_override.strip() else None

        model_override = st.selectbox("Model override", ["(use agent default)"] + MODEL_OPTIONS, index=0)
        if model_override == "(use agent default)":
            model_override = None

        max_tokens_override = st.number_input(
            "Max tokens",
            min_value=256,
            max_value=120000,
            value=st.session_state.get("global_max_tokens", 12000),
            step=256,
        )
        temp_override = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("global_temperature", 0.2),
            step=0.05,
        )

    if st.button("🚀 " + t("run_agent"), type="primary"):
        try:
            with st.spinner("Running agent..."):
                res = executor.execute(
                    agent_id=agent_cfg["id"],
                    user_input=user_input,
                    model_override=model_override,
                    max_tokens=int(max_tokens_override),
                    temperature=float(temp_override),
                    user_prompt_override=prompt_override,
                )
            st.session_state["last_agent_output"] = res["output"]
            st.session_state["run_log"].append(
                {
                    "timestamp": pd_timestamp_now(),
                    "agent_id": agent_cfg["id"],
                    "model": res["model"],
                    "tokens_used": res.get("tokens_used"),
                    "duration": res.get("duration_seconds"),
                    "status": res.get("status"),
                }
            )
            st.success("Agent executed.")
            render_agent_output(res, agent_cfg)
        except Exception as e:
            st.error(f"Error during agent execution: {e}")

    if st.session_state.get("last_agent_output"):
        with st.expander("Latest output", expanded=True):
            render_agent_output(
                {"output": st.session_state["last_agent_output"], "model": "(last)", "tokens_used": None, "duration_seconds": None, "status": "success"},
                agent_cfg,
                show_metrics=False,
            )


# ------------- NOTE KEEPER TAB (original + preserved) -------------------------

def format_note_with_coral_keywords(text: str) -> str:
    lines = [l.strip() for l in text.splitlines()]
    bullets = [f"- {l}" for l in lines if l]
    return "\n".join(bullets)


def run_note_llm(
    llm_manager: LLMProviderManager,
    note: str,
    magic: str,
    model: str,
    extra: Dict[str, Any] = None,
) -> str:
    extra = extra or {}
    system_prompt = ""
    if magic == "format":
        system_prompt = (
            "You are an expert technical editor. Clean and reformat this note into well-structured markdown with "
            "clear headings, bullet lists, and a short summary. Do not invent new facts."
        )
    elif magic == "keywords":
        kws = extra.get("keywords", [])
        color = extra.get("color", "coral")
        system_prompt = (
            "You are a highlighting engine. Given the note and a list of keywords, return markdown where each exact "
            f"keyword is wrapped in <span style='color:{color}; font-weight:700;'>keyword</span>. "
            "Do not change other text.\n\n"
            f"Keywords: {', '.join(kws)}"
        )
    elif magic == "summary":
        system_prompt = "Summarize the note into concise bullet points and a short abstract, in markdown."
    elif magic == "translate":
        system_prompt = (
            "Detect if the note is in English or Traditional Chinese and translate it to the OTHER language. "
            "Keep markdown formatting."
        )
    elif magic == "expand":
        system_prompt = (
            "Expand and elaborate the note, adding clarifying explanations and examples, but keep original structure. "
            "Output markdown."
        )
    elif magic == "actions":
        system_prompt = (
            "Extract and list all clear action items from the note. For each action, provide: "
            "1) action statement, 2) owner if mentioned, 3) due date if mentioned, 4) priority if inferable. "
            "Output as a markdown task list."
        )
    else:
        return note

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": note}]
    resp = llm_manager.chat_completion(model=model, messages=messages, max_tokens=4096, temperature=0.1)
    return resp["content"]


def render_note_keeper_tab(llm_manager: LLMProviderManager):
    st.markdown("### 📝 " + t("tab_notes"))

    col1, col2 = st.columns([2, 1])
    with col1:
        raw = st.text_area(t("notes_paste"), height=200)
    with col2:
        if st.button("✨ " + t("notes_transform")):
            if raw.strip():
                st.session_state["note_content"] = format_note_with_coral_keywords(raw)
            else:
                st.warning("Nothing to transform.")

    if not st.session_state["note_content"] and raw.strip():
        st.session_state["note_content"] = format_note_with_coral_keywords(raw)

    st.markdown("---")
    st.markdown("#### 🔧 Note workspace")

    model = st.selectbox(
        t("notes_model"),
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(st.session_state.get("global_model", MODEL_OPTIONS[0])),
        key="notes_model_select",
    )

    view_mode = st.radio(
        t("notes_view_mode"),
        [t("view_markdown"), t("view_text")],
        index=0 if st.session_state["note_view_mode"] == "markdown" else 1,
        horizontal=True,
    )
    st.session_state["note_view_mode"] = "markdown" if view_mode == t("view_markdown") else "text"

    if st.session_state["note_view_mode"] == "markdown":
        edited = st.text_area("Markdown", value=st.session_state["note_content"], height=260)
        st.session_state["note_content"] = edited
        st.markdown("---")
        st.markdown("Preview:")
        st.markdown(edited, unsafe_allow_html=True)
    else:
        text_view = st.text_area("Text", value=st.session_state["note_content"], height=260)
        st.session_state["note_content"] = text_view

    st.markdown("---")
    st.markdown("#### 🪄 " + t("ai_magics"))

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🎯 " + t("magic_format")):
            with st.spinner("Formatting note..."):
                st.session_state["note_content"] = run_note_llm(llm_manager, st.session_state["note_content"], "format", model)
        if st.button("🧠 " + t("magic_summary")):
            with st.spinner("Summarizing note..."):
                st.session_state["note_content"] = run_note_llm(llm_manager, st.session_state["note_content"], "summary", model)
        if st.button("🌐 " + t("magic_translate")):
            with st.spinner("Translating note..."):
                st.session_state["note_content"] = run_note_llm(llm_manager, st.session_state["note_content"], "translate", model)
    with col_b:
        kw_str = st.text_input(t("keywords_input"), "")
        kw_color = st.color_picker(t("keyword_color"), value="#FF7F50")
        if st.button("🔎 " + t("magic_keywords")):
            kws = [k.strip() for k in kw_str.split(",") if k.strip()]
            if not kws:
                st.warning("No keywords entered.")
            else:
                with st.spinner("Highlighting keywords..."):
                    st.session_state["note_content"] = run_note_llm(
                        llm_manager,
                        st.session_state["note_content"],
                        "keywords",
                        model,
                        extra={"keywords": kws, "color": kw_color},
                    )
        if st.button("📈 " + t("magic_expand")):
            with st.spinner("Expanding note..."):
                st.session_state["note_content"] = run_note_llm(llm_manager, st.session_state["note_content"], "expand", model)
        if st.button("✅ " + t("magic_actions")):
            with st.spinner("Extracting action items..."):
                st.session_state["note_content"] = run_note_llm(llm_manager, st.session_state["note_content"], "actions", model)


# ------------- DOC OCR & SUMMARY TAB ------------------------------------------

def render_doc_tab(llm_manager: LLMProviderManager, executor: AgentExecutor):
    st.markdown("### 📄 " + t("tab_docs"))

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### 1) Provide document content")
        st.session_state["doc_paste_raw"] = st.text_area(
            "Paste text/markdown (optional)",
            value=st.session_state.get("doc_paste_raw", ""),
            height=160,
        )

        pdf = st.file_uploader("Upload PDF (optional)", type=["pdf"], accept_multiple_files=False)
        if pdf is not None:
            st.session_state["doc_pdf_bytes"] = pdf.read()
            st.session_state["doc_pdf_name"] = pdf.name
            st.session_state["doc_pdf_page_count"] = get_pdf_page_count(st.session_state["doc_pdf_bytes"])
            st.success(f"Loaded PDF: {pdf.name} ({st.session_state['doc_pdf_page_count']} pages)")

    with right:
        st.markdown("#### 2) Preview + page selection")
        if st.session_state.get("doc_pdf_bytes"):
            page_count = st.session_state.get("doc_pdf_page_count", 0)
            if page_count > 0:
                pages = list(range(1, page_count + 1))
                default_sel = st.session_state.get("doc_selected_pages") or (pages[: min(3, page_count)])
                sel = st.multiselect("Select pages for OCR", options=pages, default=default_sel)
                st.session_state["doc_selected_pages"] = sel

                with st.expander("PDF page preview (selected pages)", expanded=False):
                    preview_pdf_pages(st.session_state["doc_pdf_bytes"], st.session_state["doc_selected_pages"])
            else:
                st.warning("Unable to detect PDF page count.")
        else:
            st.info("Upload a PDF to enable preview and page selection.")

    st.markdown("---")
    st.markdown("#### 3) OCR options")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        ocr_mode = st.radio("OCR mode", ["No OCR (use pasted text)", "Python OCR", "LLM OCR"], horizontal=False)
    with col2:
        ocr_lang = st.selectbox("OCR language", ["en", "zh-tw"], index=0)
    with col3:
        st.caption("Tip: If Tesseract isn't installed on HF, use LLM OCR.")

    if ocr_mode == "Python OCR":
        engine = st.selectbox("Python OCR engine", ["pytesseract", "easyocr"], index=0)
        if st.button("🔍 Run Python OCR", type="primary"):
            try:
                if not st.session_state.get("doc_pdf_bytes"):
                    st.error("Upload a PDF first.")
                else:
                    pages = st.session_state.get("doc_selected_pages") or []
                    if not pages:
                        st.error("Select at least one page.")
                    else:
                        with st.spinner("Running Python OCR..."):
                            images = pdf_to_pil_images(st.session_state["doc_pdf_bytes"], pages)
                            text = run_python_ocr(images, engine=engine, lang=ocr_lang)
                            st.session_state["doc_ocr_text"] = text
                            st.success("Python OCR completed.")
            except Exception as e:
                st.error(f"Python OCR failed: {e}")

    elif ocr_mode == "LLM OCR":
        model = st.selectbox("LLM OCR model", LLM_OCR_MODELS, index=0)
        if st.button("🧠 Run LLM OCR", type="primary"):
            try:
                if not st.session_state.get("doc_pdf_bytes"):
                    st.error("Upload a PDF first.")
                else:
                    pages = st.session_state.get("doc_selected_pages") or []
                    if not pages:
                        st.error("Select at least one page.")
                    else:
                        with st.spinner("Running LLM OCR (multimodal)..."):
                            images = pdf_to_pil_images(st.session_state["doc_pdf_bytes"], pages)
                            ocr_md = llm_manager.ocr_images_with_llm(model=model, images=images, lang=ocr_lang)
                            st.session_state["doc_ocr_text"] = ocr_md
                            st.success("LLM OCR completed.")
            except Exception as e:
                st.error(f"LLM OCR failed: {e}")

    st.markdown("---")
    st.markdown("#### 4) OCR result editor")

    # Default doc base text:
    base_text = ""
    if ocr_mode == "No OCR (use pasted text)":
        base_text = st.session_state.get("doc_paste_raw", "")
        if not st.session_state.get("doc_ocr_text") and base_text.strip():
            st.session_state["doc_ocr_text"] = base_text

    view = st.radio("OCR view", ["Markdown", "Text"], horizontal=True, index=0 if st.session_state["doc_view_mode_ocr"] == "markdown" else 1)
    st.session_state["doc_view_mode_ocr"] = "markdown" if view == "Markdown" else "text"
    st.session_state["doc_ocr_text"] = st.text_area(
        "OCR / Document text (editable)",
        value=st.session_state.get("doc_ocr_text", ""),
        height=260,
    )

    st.markdown("---")
    st.markdown("#### 5) Generate comprehensive summary (2000–3000 words)")

    sum_model = st.selectbox("Summary model", MODEL_OPTIONS, index=MODEL_OPTIONS.index(st.session_state.get("global_model", MODEL_OPTIONS[0])))
    if st.button("✨ Generate Summary", type="primary"):
        try:
            if not st.session_state.get("doc_ocr_text", "").strip():
                st.error("No document text available. Paste text or run OCR first.")
            else:
                with st.spinner("Generating summary (this may take a while)..."):
                    st.session_state["doc_summary_md"] = generate_long_doc_summary(
                        llm_manager=llm_manager,
                        model=sum_model,
                        doc_text=st.session_state["doc_ocr_text"],
                    )
                st.success("Summary generated.")
        except Exception as e:
            st.error(f"Summary generation failed: {e}")

    st.markdown("#### 6) Summary editor")
    sum_view = st.radio("Summary view", ["Markdown", "Text"], horizontal=True, index=0 if st.session_state["doc_view_mode_summary"] == "markdown" else 1)
    st.session_state["doc_view_mode_summary"] = "markdown" if sum_view == "Markdown" else "text"
    st.session_state["doc_summary_md"] = st.text_area(
        "Summary markdown (editable)",
        value=st.session_state.get("doc_summary_md", ""),
        height=320,
    )
    if st.session_state["doc_view_mode_summary"] == "markdown" and st.session_state["doc_summary_md"].strip():
        with st.expander("Summary preview", expanded=False):
            st.markdown(st.session_state["doc_summary_md"], unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 7) Use the doc/summary with prompts or agents")

    doc_context_choice = st.radio("Context source", ["OCR/Doc text", "Summary"], horizontal=True)
    context = st.session_state["doc_ocr_text"] if doc_context_choice == "OCR/Doc text" else st.session_state["doc_summary_md"]

    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown("##### 💬 Doc Prompt (chat)")
        chat_model = st.selectbox("Chat model", MODEL_OPTIONS, index=MODEL_OPTIONS.index(st.session_state.get("global_model", MODEL_OPTIONS[0])), key="doc_chat_model")
        user_q = st.text_area("Ask anything about the document", height=100)
        if st.button("Send", key="doc_chat_send"):
            if not user_q.strip():
                st.warning("Enter a question.")
            elif not context.strip():
                st.warning("No context available.")
            else:
                with st.spinner("Thinking..."):
                    answer = doc_chat(llm_manager, model=chat_model, context=context, question=user_q)
                st.session_state["doc_chat_history"].append({"role": "user", "content": user_q})
                st.session_state["doc_chat_history"].append({"role": "assistant", "content": answer})

        if st.session_state["doc_chat_history"]:
            with st.expander("Chat history", expanded=True):
                for m in st.session_state["doc_chat_history"][-12:]:
                    st.markdown(f"**{m['role'].title()}**")
                    st.markdown(m["content"], unsafe_allow_html=True)

    with colB:
        st.markdown("##### 🤖 Run an agent on the doc")
        agents = executor.list_agents()
        if not agents:
            st.warning("No agents loaded.")
        else:
            agent_labels = [f"[#{a.get('skill_number','?')}] {a.get('name','')} ({a['id']})" for a in agents]
            idx = st.selectbox("Agent", list(range(len(agents))), format_func=lambda i: agent_labels[i], key="doc_agent_select")
            agent_cfg = agents[idx]

            model_override = st.selectbox("Model override", ["(use agent default)"] + MODEL_OPTIONS, index=0, key="doc_agent_model_ovr")
            model_override = None if model_override == "(use agent default)" else model_override
            max_tokens_override = st.number_input("Max tokens", min_value=256, max_value=120000, value=st.session_state.get("global_max_tokens", 12000), step=256, key="doc_agent_max_tokens")
            temp_override = st.slider("Temperature", 0.0, 1.0, st.session_state.get("global_temperature", 0.2), 0.05, key="doc_agent_temp")

            user_prompt_override = st.text_area("Override system prompt (optional)", value="", height=120, key="doc_agent_prompt_ovr")
            user_prompt_override = user_prompt_override if user_prompt_override.strip() else None

            if st.button("Run agent on context", key="doc_run_agent", type="primary"):
                if not context.strip():
                    st.error("No context available.")
                else:
                    with st.spinner("Running agent..."):
                        res = executor.execute(
                            agent_id=agent_cfg["id"],
                            user_input=context,
                            model_override=model_override,
                            max_tokens=int(max_tokens_override),
                            temperature=float(temp_override),
                            user_prompt_override=user_prompt_override,
                        )
                    st.session_state["last_agent_output"] = res["output"]
                    st.session_state["run_log"].append(
                        {
                            "timestamp": pd_timestamp_now(),
                            "agent_id": agent_cfg["id"],
                            "model": res["model"],
                            "tokens_used": res.get("tokens_used"),
                            "duration": res.get("duration_seconds"),
                            "status": res.get("status"),
                        }
                    )
                    st.success("Agent executed on doc context.")
                    render_agent_output(res, agent_cfg)


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    if fitz is None:
        return 0
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return int(doc.page_count)


def pdf_to_pil_images(pdf_bytes: bytes, pages_1based: List[int], zoom: float = 2.0) -> List["Image.Image"]:
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not installed. Add pymupdf to requirements.")
    if Image is None:
        raise RuntimeError("Pillow not installed. Add pillow to requirements.")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = []
    mat = fitz.Matrix(zoom, zoom)
    for p in pages_1based:
        page = doc.load_page(int(p) - 1)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        out.append(img)
    return out


def preview_pdf_pages(pdf_bytes: bytes, pages_1based: List[int]):
    if not pages_1based:
        st.info("No pages selected.")
        return
    try:
        imgs = pdf_to_pil_images(pdf_bytes, pages_1based, zoom=1.5)
        for i, im in enumerate(imgs):
            st.caption(f"Page {pages_1based[i]}")
            st.image(im, use_container_width=True)
    except Exception as e:
        st.error(f"Preview failed: {e}")


def run_python_ocr(images: List["Image.Image"], engine: str, lang: str) -> str:
    if Image is None:
        raise RuntimeError("Pillow not installed")
    if not images:
        return ""

    if engine == "pytesseract":
        if pytesseract is None:
            raise RuntimeError("pytesseract not installed")
        tess_lang = "eng" if lang == "en" else "chi_tra"
        chunks = []
        for idx, im in enumerate(images, start=1):
            txt = pytesseract.image_to_string(im, lang=tess_lang)
            chunks.append(f"\n\n## Page {idx}\n\n{txt}".strip())
        return "\n\n".join(chunks)

    if engine == "easyocr":
        if easyocr is None:
            raise RuntimeError("easyocr not installed")
        langs = ["en"] if lang == "en" else ["ch_tra", "en"]
        reader = easyocr.Reader(langs, gpu=False)
        chunks = []
        for idx, im in enumerate(images, start=1):
            results = reader.readtext(np_from_pil(im), detail=0, paragraph=True)
            txt = "\n".join(results)
            chunks.append(f"\n\n## Page {idx}\n\n{txt}".strip())
        return "\n\n".join(chunks)

    raise RuntimeError(f"Unknown OCR engine: {engine}")


def np_from_pil(im: "Image.Image"):
    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError("numpy required for easyocr") from e
    return np.array(im)


def generate_long_doc_summary(llm_manager: LLMProviderManager, model: str, doc_text: str) -> str:
    # Hard constraints in prompt:
    # - 2000~3000 words
    # - coral keywords spans
    # - a Mermaid wordgraph
    # - exactly 3 tables
    sys = """
You are a senior technical medical documentation analyst and writer.
Your job is to produce a comprehensive, structured markdown summary of a document.

STRICT OUTPUT REQUIREMENTS:
1) Output MUST be valid Markdown.
2) Word count target: 2000–3000 words (English words; for Chinese, approximate same length by detail).
3) Highlight important keywords and regulatory terms by wrapping the keyword with:
   <span style="color:coral; font-weight:700;">KEYWORD</span>
   Use it naturally throughout (not spam).
4) Include ONE "Wordgraph" section with a Mermaid diagram:
   - Use a fenced code block ```mermaid
   - Use graph TD or mindmap
   - Nodes should be key concepts and relationships from the document
5) Include EXACTLY THREE tables in markdown.
   - Each table must have a clear title immediately above it.
   - Tables should be useful and distinct (e.g., Key Facts, Risks & Mitigations, Evidence/Claims Map).
6) Do NOT invent facts. If missing, mark as [Not specified].

STRUCTURE (use headings):
# Title
## Executive Abstract
## Key Sections & Findings
## Wordgraph
## Tables
### Table 1: ...
(table)
### Table 2: ...
(table)
### Table 3: ...
(table)
## Detailed Summary (deep, multi-section)
## Open Questions / Missing Info
## Suggested Next Actions
"""
    user = f"""DOCUMENT TEXT (may include OCR noise):
{doc_text[:200000]}
"""
    resp = llm_manager.chat_completion(
        model=model,
        messages=[{"role": "system", "content": sys.strip()}, {"role": "user", "content": user}],
        max_tokens=min(12000, int(st.session_state.get("global_max_tokens", 12000))),
        temperature=0.2,
    )
    return resp["content"]


def doc_chat(llm_manager: LLMProviderManager, model: str, context: str, question: str) -> str:
    sys = (
        "You are a helpful analyst. Answer the user's question using ONLY the provided context. "
        "If the answer is not in the context, say so clearly.\n"
        "Return markdown."
    )
    user = f"""CONTEXT:
{context[:120000]}

QUESTION:
{question}
"""
    resp = llm_manager.chat_completion(
        model=model,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        max_tokens=4096,
        temperature=0.2,
    )
    return resp["content"]


# ------------- AGENT YAML STUDIO TAB ------------------------------------------

def render_agents_yaml_studio_tab():
    st.markdown("### 🧩 " + t("tab_agents_yaml"))
    st.caption("Upload/edit agents.yaml. Non-standard files will be auto-standardized to the app schema.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Current agents.yaml (editable)")
        st.session_state["agents_yaml_text"] = st.text_area(
            "agents.yaml content",
            value=st.session_state.get("agents_yaml_text", ""),
            height=420,
        )

        cA, cB, cC = st.columns([1, 1, 1])
        with cA:
            if st.button("✅ Validate + Standardize"):
                try:
                    raw = load_yaml_safe(st.session_state["agents_yaml_text"])
                    standardized = standardize_agents_yaml(raw)
                    st.session_state["agents_yaml_last_standardized"] = dump_yaml_safe(standardized)
                    st.success("Standardized successfully.")
                except Exception as e:
                    st.error(f"YAML parse/standardize failed: {e}")
        with cB:
            if st.button("💾 Apply to app"):
                try:
                    raw = load_yaml_safe(st.session_state["agents_yaml_text"])
                    standardized = standardize_agents_yaml(raw)
                    st.session_state["agents_config"] = standardized
                    st.success("Applied to app (in-memory).")
                except Exception as e:
                    st.error(f"Apply failed: {e}")
        with cC:
            if st.button("📝 Save to agents.yaml"):
                try:
                    raw = load_yaml_safe(st.session_state["agents_yaml_text"])
                    standardized = standardize_agents_yaml(raw)
                    Path("agents.yaml").write_text(dump_yaml_safe(standardized), encoding="utf-8")
                    st.success("Saved to agents.yaml")
                except Exception as e:
                    st.error(f"Save failed: {e}")

        st.download_button(
            "⬇️ Download current (as typed)",
            data=st.session_state["agents_yaml_text"].encode("utf-8"),
            file_name="agents.yaml",
            mime="text/yaml",
        )

    with col2:
        st.markdown("#### Upload agents.yaml")
        up = st.file_uploader("Upload YAML", type=["yaml", "yml"], accept_multiple_files=False, key="upload_agents_yaml")
        if up is not None:
            try:
                text = up.read().decode("utf-8", errors="ignore")
                raw = load_yaml_safe(text)
                standardized = standardize_agents_yaml(raw)
                st.session_state["agents_yaml_text"] = dump_yaml_safe(standardized)
                st.session_state["agents_yaml_last_standardized"] = st.session_state["agents_yaml_text"]
                st.success("Uploaded and standardized into editor.")
            except Exception as e:
                st.error(f"Upload failed: {e}")

        if st.session_state.get("agents_yaml_last_standardized"):
            st.markdown("#### Last standardized output (read-only)")
            st.code(st.session_state["agents_yaml_last_standardized"], language="yaml")

            st.download_button(
                "⬇️ Download standardized agents.yaml",
                data=st.session_state["agents_yaml_last_standardized"].encode("utf-8"),
                file_name="agents.standardized.yaml",
                mime="text/yaml",
            )


# ------------- SKILL.MD STUDIO TAB --------------------------------------------

def render_skill_md_studio_tab(llm_manager: LLMProviderManager):
    st.markdown("### 🧠 " + t("tab_skill_md"))
    st.caption("Paste instructions; the system will generate SKILL.md with EXACT required format and 30 skills.")

    model = st.selectbox("Model", MODEL_OPTIONS, index=MODEL_OPTIONS.index(st.session_state.get("global_model", MODEL_OPTIONS[0])), key="skill_model")

    st.session_state["skill_instructions"] = st.text_area(
        "Instructions (text/markdown)",
        value=st.session_state.get("skill_instructions", ""),
        height=200,
    )

    if st.button("⚙️ Generate SKILL.md", type="primary"):
        if not st.session_state["skill_instructions"].strip():
            st.warning("Please paste instructions first.")
        else:
            with st.spinner("Generating SKILL.md..."):
                st.session_state["skill_md_text"] = generate_skill_md(
                    llm_manager=llm_manager,
                    model=model,
                    instructions=st.session_state["skill_instructions"],
                )

    st.markdown("#### SKILL.md (editable)")
    st.session_state["skill_md_text"] = st.text_area(
        "SKILL.md content",
        value=st.session_state.get("skill_md_text", ""),
        height=420,
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.download_button(
            "⬇️ Download SKILL.md",
            data=st.session_state["skill_md_text"].encode("utf-8"),
            file_name="SKILL.md",
            mime="text/markdown",
        )
    with col2:
        up = st.file_uploader("Upload SKILL.md", type=["md"], accept_multiple_files=False, key="upload_skill_md")
        if up is not None:
            st.session_state["skill_md_text"] = up.read().decode("utf-8", errors="ignore")
            st.success("SKILL.md uploaded.")
    with col3:
        if st.button("🧾 Preview SKILL.md"):
            with st.expander("Preview", expanded=True):
                st.markdown(st.session_state["skill_md_text"], unsafe_allow_html=True)


def generate_skill_md(llm_manager: LLMProviderManager, model: str, instructions: str) -> str:
    # IMPORTANT: We preserve the user's strict output protocol by embedding it verbatim.
    sys = """You are a documentation generator.

You MUST follow the user's required output protocol exactly:
- First write a <scratchpad> plan.
- Then write the complete SKILL.md inside <skill_md> tags.
- Your output MUST contain ONLY the scratchpad and the final SKILL.md file.
- Generate EXACTLY 30 distinct skills, each as a level-2 heading (##).
"""

    user = f"""{instructions}

Before generating the final SKILL.md file, use the scratchpad to plan your approach:

<scratchpad>
- Analyze what the code does and what capabilities it provides
- Review the user's instructions to understand the intended purpose
- Brainstorm 30 distinct skills that would be useful for this code/functionality
- Organize these skills in a logical progression
- Determine an appropriate name and description for the frontmatter
</scratchpad>

After your planning, write the complete SKILL.md file inside <skill_md> tags. Your output should contain only the scratchpad and the final SKILL.md file - do not include additional commentary outside these sections.

The final SKILL.md file should be complete, well-formatted markdown that could be saved directly as a .md file.
"""
    resp = llm_manager.chat_completion(
        model=model,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        max_tokens=12000,
        temperature=0.2,
    )
    return resp["content"]


# ------------- MAIN -----------------------------------------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_session_state()
    apply_theme()

    llm_manager = LLMProviderManager()
    executor = AgentExecutor(llm_manager=llm_manager, agents_config=st.session_state["agents_config"])

    st.markdown('<div class="main-wrapping-container">', unsafe_allow_html=True)

    render_sidebar(llm_manager)
    st.title(t("app_title"))
    render_status_header(llm_manager)

    tabs = st.tabs([
        t("tab_agents"),
        t("tab_docs"),
        t("tab_dashboard"),
        t("tab_notes"),
        t("tab_agents_yaml"),
        t("tab_skill_md"),
    ])

    with tabs[0]:
        render_agent_runner_tab(executor)
    with tabs[1]:
        render_doc_tab(llm_manager, executor)
    with tabs[2]:
        render_dashboard_tab()
    with tabs[3]:
        render_note_keeper_tab(llm_manager)
    with tabs[4]:
        render_agents_yaml_studio_tab()
    with tabs[5]:
        render_skill_md_studio_tab(llm_manager)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
