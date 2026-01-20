import os
import time
import random
import textwrap
from pathlib import Path
from typing import Dict, Any, List, Optional

import streamlit as st
import yaml

# Optional imports – wrapped in try/except in LLM manager
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
        "app_title": {
            "en": "FDA 510(k) Agentic Review WOW Studio",
            "zh-tw": "FDA 510(k) 智慧審查 WOW 工作室",
        },
        "tab_agents": {"en": "Agent Runner", "zh-tw": "代理人執行器"},
        "tab_dashboard": {"en": "Dashboard", "zh-tw": "儀表板"},
        "tab_notes": {"en": "AI Note Keeper", "zh-tw": "AI 筆記管家"},
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
        font-weight: 600;
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


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
            # join all messages into single prompt with simple roles
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
            # Use OpenAI-compatible HTTP endpoint for xAI Grok
            headers = {"Authorization": f"Bearer {api_key}"}
            json = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            with httpx.Client(timeout=120) as client:
                r = client.post("https://api.x.ai/v1/chat/completions", headers=headers, json=json)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            tokens_used = data.get("usage", {}).get("total_tokens")

        duration = time.time() - start
        return {"content": content, "tokens_used": tokens_used, "duration": duration, "provider": provider}


# ------------- AGENT EXECUTOR -------------------------------------------------

def load_agents_config() -> Dict[str, Any]:
    path = Path("agents.yaml")
    if not path.exists():
        return {"agents": {}, "defaults": {"max_tokens": 12000, "temperature": 0.2}}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class AgentExecutor:
    def __init__(self, llm_manager: LLMProviderManager, agents_config: Dict[str, Any]):
        self.llm_manager = llm_manager
        self.config = agents_config or {"agents": {}, "defaults": {}}
        self.defaults = self.config.get("defaults", {})

    def list_agents(self) -> List[Dict[str, Any]]:
        agents = []
        for agent_id, cfg in self.config.get("agents", {}).items():
            agents.append({"id": agent_id, **cfg})
        # sort by skill_number if present
        agents.sort(key=lambda a: a.get("skill_number", 999))
        return agents

    def execute(
        self,
        agent_id: str,
        user_input: str,
        model_override: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        agents = self.config.get("agents", {})
        if agent_id not in agents:
            raise RuntimeError(f"Unknown agent: {agent_id}")
        cfg = agents[agent_id]
        system_prompt = cfg.get("system_prompt", "")
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


# ------------- SIDEBAR (THEME, LANGUAGE, API, MODEL) -------------------------

def render_sidebar(llm_manager: LLMProviderManager):
    with st.sidebar:
        st.markdown("### 🎨 WOW Studio")
        lang = st.radio(t("sidebar_language"), ["en", "zh-tw"], index=0 if st.session_state["lang"] == "en" else 1,
                        horizontal=True)
        st.session_state["lang"] = lang

        theme = st.radio(
            t("sidebar_theme"),
            [t("light"), t("dark")],
            index=0 if st.session_state["theme"] == "light" else 1,
            horizontal=True,
        )
        st.session_state["theme"] = "light" if theme == t("light") else "dark"

        style = st.selectbox(t("sidebar_style"), list(PAINTER_STYLES.keys()), index=list(PAINTER_STYLES.keys()).index(
            st.session_state["painter_style"]))
        st.session_state["painter_style"] = style
        if st.button("🎰 " + t("sidebar_jackpot")):
            st.session_state["painter_style"] = random.choice(list(PAINTER_STYLES.keys()))
            st.experimental_rerun()

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
                st.markdown(f'<span class="wow-badge {status_class}">{status_label}</span>',
                            unsafe_allow_html=True)

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


# ------------- WOW STATUS INDICATORS & DASHBOARD -----------------------------

def render_status_header(llm_manager: LLMProviderManager):
    st.markdown("#### ⚡ " + t("status_section"))
    col1, col2, col3 = st.columns(3)

    with col1:
        any_api = any(llm_manager.get_effective_key(p) for p in ["openai", "gemini", "anthropic", "grok"])
        cls = "wow-chip-ok" if any_api else "wow-chip-bad"
        label = "OK" if any_api else "Missing"
        st.markdown(
            f'{t("status_api")}<br><span class="wow-badge {cls}">{label}</span>',
            unsafe_allow_html=True,
        )
    with col2:
        docs_loaded = bool(st.session_state.get("uploaded_docs_raw"))
        cls = "wow-chip-ok" if docs_loaded else "wow-chip-warn"
        label = "Loaded" if docs_loaded else "None"
        st.markdown(
            f'{t("status_docs")}<br><span class="wow-badge {cls}">{label}</span>',
            unsafe_allow_html=True,
        )
    with col3:
        runs = len(st.session_state["run_log"])
        cls = "wow-chip-ok" if runs > 0 else "wow-chip-warn"
        st.markdown(
            f'{t("status_runs")}<br><span class="wow-badge {cls}">{runs} run(s)</span>',
            unsafe_allow_html=True,
        )


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
        st.metric("Unique agents", df["agent_id"].nunique())
    with col3:
        st.metric("Total tokens (if reported)", int(df["tokens_used"].fillna(0).sum()))

    # Tokens by agent
    if "tokens_used" in df.columns:
        chart_data = (
            df.groupby("agent_id")["tokens_used"]
            .sum()
            .reset_index()
            .sort_values("tokens_used", ascending=False)
        )
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

    # Runs over time
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

def render_agent_runner_tab(executor: AgentExecutor):
    st.markdown("### 🧠 " + t("tab_agents"))

    # Document upload / paste
    if "uploaded_docs_raw" not in st.session_state:
        st.session_state["uploaded_docs_raw"] = []

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
                    except AttributeError:
                        text = content
                    st.session_state["uploaded_docs_raw"].append(
                        {"name": f.name, "size": f.size, "content": text}
                    )
                    st.success(f"Loaded {f.name}")
                except Exception as e:
                    st.error(f"Error reading {f.name}: {e}")

        paste = st.text_area("Paste text", height=150)
        if st.button("Save pasted as document"):
            if paste.strip():
                st.session_state["uploaded_docs_raw"].append(
                    {"name": f"Pasted-{len(st.session_state['uploaded_docs_raw'])+1}", "size": len(paste),
                     "content": paste}
                )
                st.success("Pasted content saved.")
            else:
                st.warning("Nothing to save.")

        docs = st.session_state["uploaded_docs_raw"]
        if docs:
            st.markdown("**Loaded documents:**")
            for d in docs[-5:]:
                st.markdown(f"- {d['name']} ({len(d['content'])} chars)")

    # Agent selection
    agents = executor.list_agents()
    if not agents:
        st.error("No agents found in agents.yaml")
        return

    agent_labels = [
        f"[#{a.get('skill_number','?')}] {a.get('name','')} ({a['id']})" for a in agents
    ]
    idx = st.selectbox(
        t("agent_select"),
        list(range(len(agents))),
        format_func=lambda i: agent_labels[i],
    )
    agent_cfg = agents[idx]

    with st.expander("Agent details", expanded=False):
        st.markdown(f"**ID:** `{agent_cfg['id']}`")
        st.markdown(f"**Skill #:** {agent_cfg.get('skill_number','?')}")
        st.markdown(f"**Category:** {agent_cfg.get('category','')}")
        st.markdown(f"**Description:** {agent_cfg.get('description','')}")
        st.markdown("**System prompt (truncated):**")
        sys_prompt = agent_cfg.get("system_prompt", "")
        st.code(textwrap.shorten(sys_prompt, width=1200, placeholder="..."))

    # Input source
    col_a, col_b = st.columns([2, 1])
    with col_b:
        use_last = st.checkbox(t("use_last_output"), value=False)
    default_input = ""
    if use_last and st.session_state["last_agent_output"]:
        default_input = st.session_state["last_agent_output"]
    else:
        # Default: concat docs (trimmed)
        docs = st.session_state.get("uploaded_docs_raw", [])
        merged = "\n\n".join(d["content"] for d in docs)
        default_input = textwrap.shorten(merged, width=8000, placeholder="\n\n[TRUNCATED DOCUMENT CONTENT]\n")

    with col_a:
        user_input = st.text_area(
            t("agent_input"),
            value=default_input,
            height=220,
        )

    # Override model / params
    with st.expander("Override model & parameters (optional)", expanded=False):
        model_override = st.selectbox(
            "Model override",
            ["(use agent default)"] + MODEL_OPTIONS,
            index=0,
            key="agent_model_override",
        )
        if model_override == "(use agent default)":
            model_override = None

        max_tokens_override = st.number_input(
            "Max tokens (blank = global default)",
            min_value=256,
            max_value=120000,
            value=st.session_state.get("global_max_tokens", 12000),
            step=256,
        )
        temp_override = st.slider(
            "Temperature (blank = global default)",
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
                )
            st.session_state["last_agent_output"] = res["output"]

            # Log
            st.session_state["run_log"].append(
                {
                    "timestamp": pd_timestamp_now(),  # defined below
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

    # Always show latest output if exists
    if st.session_state.get("last_agent_output"):
        with st.expander("Latest output", expanded=True):
            render_agent_output(
                {
                    "output": st.session_state["last_agent_output"],
                    "model": "(last)",
                    "tokens_used": None,
                    "duration_seconds": None,
                    "status": "success",
                },
                agent_cfg,
                show_metrics=False,
            )


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


# ------------- AI NOTE KEEPER TAB --------------------------------------------

def format_note_with_coral_keywords(text: str) -> str:
    # Basic structure; LLM will improve later if user wants.
    lines = [l.strip() for l in text.splitlines()]
    bullets = [f"- {l}" for l in lines if l]
    md = "\n".join(bullets)
    return md


def highlight_keywords(note: str, keywords: List[str], color: str) -> str:
    # naive keyword replacement
    out = note
    for kw in sorted(set(k.strip() for k in keywords if k.strip()), key=len, reverse=True):
        out = out.replace(
            kw,
            f'<span style="color:{color}; font-weight:600;">{kw}</span>',
        )
    return out


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
            f"keyword is wrapped in <span style='color:{color}; font-weight:600;'>keyword</span>. Do not change other text."
            f"\n\nKeywords: {', '.join(kws)}"
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
            "1) action statement, 2) owner if mentioned, 3) due date if mentioned, 4) priority if inferable."
            "Output as a markdown task list."
        )
    else:
        return note

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": note},
    ]
    resp = llm_manager.chat_completion(
        model=model,
        messages=messages,
        max_tokens=4096,
        temperature=0.1,
    )
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
        # initialize if user typed but didn't click
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
        edited = st.text_area(
            "Markdown",
            value=st.session_state["note_content"],
            height=260,
        )
        st.session_state["note_content"] = edited
        st.markdown("---")
        st.markdown("Preview:")
        st.markdown(edited, unsafe_allow_html=True)
    else:
        text_view = st.text_area(
            "Text",
            value=st.session_state["note_content"],
            height=260,
        )
        st.session_state["note_content"] = text_view

    st.markdown("---")
    st.markdown("#### 🪄 " + t("ai_magics"))

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🎯 " + t("magic_format")):
            with st.spinner("Formatting note..."):
                st.session_state["note_content"] = run_note_llm(
                    llm_manager,
                    st.session_state["note_content"],
                    "format",
                    model,
                )
        if st.button("🧠 " + t("magic_summary")):
            with st.spinner("Summarizing note..."):
                st.session_state["note_content"] = run_note_llm(
                    llm_manager,
                    st.session_state["note_content"],
                    "summary",
                    model,
                )
        if st.button("🌐 " + t("magic_translate")):
            with st.spinner("Translating note..."):
                st.session_state["note_content"] = run_note_llm(
                    llm_manager,
                    st.session_state["note_content"],
                    "translate",
                    model,
                )
    with col_b:
        kw_str = st.text_input(t("keywords_input"), "")
        kw_color = st.color_picker(t("keyword_color"), value="#FF7F50")  # coral default
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
                st.session_state["note_content"] = run_note_llm(
                    llm_manager,
                    st.session_state["note_content"],
                    "expand",
                    model,
                )
        if st.button("✅ " + t("magic_actions")):
            with st.spinner("Extracting action items..."):
                st.session_state["note_content"] = run_note_llm(
                    llm_manager,
                    st.session_state["note_content"],
                    "actions",
                    model,
                )


# ------------- UTIL ----------------------------------------------------------

def pd_timestamp_now():
    # tiny helper without importing pandas at top-level
    import pandas as _pd
    return _pd.Timestamp.utcnow()


# ------------- MAIN ----------------------------------------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_session_state()
    apply_theme()

    llm_manager = LLMProviderManager()
    executor = AgentExecutor(llm_manager=llm_manager, agents_config=st.session_state["agents_config"])

    # Wrap main content for background CSS
    st.markdown('<div class="main-wrapping-container">', unsafe_allow_html=True)

    render_sidebar(llm_manager)
    st.title(t("app_title"))
    render_status_header(llm_manager)

    tabs = st.tabs([t("tab_agents"), t("tab_dashboard"), t("tab_notes")])

    with tabs[0]:
        render_agent_runner_tab(executor)
    with tabs[1]:
        render_dashboard_tab()
    with tabs[2]:
        render_note_keeper_tab(llm_manager)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
