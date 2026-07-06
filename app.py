import streamlit as st
import joblib
import re
import fitz  # PyMuPDF
import nltk
import pandas as pd
import plotly.graph_objects as go
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ------------------------------------------------------------------
# PAGE CONFIG (must be first Streamlit call)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="WEB-SCAN | Resume Screening AI",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# NLTK SETUP
# ------------------------------------------------------------------
@st.cache_resource
def setup_nltk():
    for pkg in ["punkt", "stopwords", "wordnet", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
        except Exception:
            nltk.download(pkg, quiet=True)

setup_nltk()

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def preprocess(text: str) -> str:
    """Exact same cleaning pipeline used during training."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [
        LEMMATIZER.lemmatize(word)
        for word in tokens
        if word.isalpha() and word not in STOP_WORDS
    ]
    return " ".join(tokens)


def extract_text(uploaded_file) -> str:
    """Extract raw text from an uploaded PDF or TXT file."""
    if uploaded_file.name.lower().endswith(".pdf"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    elif uploaded_file.name.lower().endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    return ""


# ------------------------------------------------------------------
# LOAD MODEL + VECTORIZER
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("resume_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


model_load_error = None
try:
    model, vectorizer = load_artifacts()
except Exception as e:
    model_load_error = str(e)

# ------------------------------------------------------------------
# SPIDER-MAN THEME — CSS
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bangers&family=Rajdhani:wght@400;500;600;700&display=swap');

    :root{
        --spider-red: #E62429;
        --spider-red-dark: #9B0F13;
        --spider-blue: #0B3D91;
        --spider-blue-light: #1E5FCC;
        --web-black: #0A0A0F;
        --panel-black: #121218;
    }

    /* Overall app background: black with a faint web pattern */
    .stApp {
        background:
            radial-gradient(circle at 50% -10%, rgba(230,36,41,0.12), transparent 40%),
            repeating-conic-gradient(from 0deg, rgba(255,255,255,0.015) 0deg 1deg, transparent 1deg 12deg),
            linear-gradient(180deg, #05050a 0%, #0A0A0F 40%, #0d0d16 100%);
        color: #EAEAF0;
    }

    /* Spiderweb corner decorations */
    .stApp::before{
        content:"";
        position:fixed; top:-120px; left:-120px; width:340px; height:340px;
        background: repeating-radial-gradient(circle, transparent 0 18px, rgba(230,36,41,0.10) 19px 20px),
                    repeating-conic-gradient(from 0deg, rgba(230,36,41,0.12) 0deg 1deg, transparent 1deg 15deg);
        border-radius:50%;
        pointer-events:none;
        z-index:0;
    }
    .stApp::after{
        content:"";
        position:fixed; bottom:-140px; right:-140px; width:380px; height:380px;
        background: repeating-radial-gradient(circle, transparent 0 18px, rgba(11,61,145,0.14) 19px 20px),
                    repeating-conic-gradient(from 0deg, rgba(11,61,145,0.16) 0deg 1deg, transparent 1deg 15deg);
        border-radius:50%;
        pointer-events:none;
        z-index:0;
    }

    /* Hide default Streamlit chrome */
    #MainMenu, header, footer {visibility:hidden;}

    /* Hero title */
    .hero-title{
        font-family:'Bangers', cursive;
        font-size:64px;
        letter-spacing:3px;
        text-align:center;
        background: linear-gradient(180deg, #ff4d4f 0%, var(--spider-red) 55%, var(--spider-red-dark) 100%);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        text-shadow: 0 0 30px rgba(230,36,41,0.35);
        margin-bottom:0;
        line-height:1.05;
    }
    .hero-sub{
        font-family:'Rajdhani', sans-serif;
        text-align:center;
        color:#8FB4FF;
        letter-spacing:6px;
        font-weight:600;
        font-size:15px;
        text-transform:uppercase;
        margin-top:-6px;
        margin-bottom:28px;
    }

    /* Section headers */
    h2, h3 {
        font-family:'Rajdhani', sans-serif !important;
        color:#F2F2F7 !important;
        font-weight:700 !important;
        border-left: 5px solid var(--spider-red);
        padding-left:12px;
    }

    p, li, span, label, div { font-family:'Rajdhani', sans-serif; }

    /* Card container */
    .web-card{
        background: linear-gradient(145deg, rgba(18,18,24,0.9), rgba(11,11,16,0.9));
        border: 1px solid rgba(230,36,41,0.35);
        border-radius:16px;
        padding:26px 28px;
        box-shadow: 0 0 25px rgba(11,61,145,0.15), inset 0 0 40px rgba(230,36,41,0.03);
        margin-bottom:22px;
        position:relative;
        z-index:1;
    }

    /* Result banner */
    .result-banner{
        background: linear-gradient(135deg, var(--spider-red-dark) 0%, var(--spider-red) 45%, var(--spider-blue) 130%);
        border-radius:18px;
        padding:30px;
        text-align:center;
        box-shadow: 0 8px 40px rgba(230,36,41,0.4);
        border: 2px solid rgba(255,255,255,0.15);
        margin: 20px 0;
    }
    .result-banner .role{
        font-family:'Bangers', cursive;
        font-size:42px;
        letter-spacing:2px;
        color:#fff;
        text-shadow: 0 0 20px rgba(0,0,0,0.6);
        margin:0;
    }
    .result-banner .label{
        font-family:'Rajdhani', sans-serif;
        text-transform:uppercase;
        letter-spacing:4px;
        color:#DDE7FF;
        font-size:13px;
        font-weight:600;
    }

    /* Buttons */
    .stButton>button{
        background: linear-gradient(135deg, var(--spider-red) 0%, var(--spider-red-dark) 100%);
        color:white;
        border:none;
        border-radius:10px;
        padding:12px 28px;
        font-family:'Rajdhani', sans-serif;
        font-weight:700;
        font-size:16px;
        letter-spacing:1.5px;
        text-transform:uppercase;
        box-shadow: 0 4px 20px rgba(230,36,41,0.45);
        transition: all 0.2s ease;
        width:100%;
    }
    .stButton>button:hover{
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 6px 28px rgba(11,61,145,0.55);
        border: 1px solid var(--spider-blue-light);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"]{ gap: 6px; }
    .stTabs [data-baseweb="tab"]{
        background: rgba(255,255,255,0.03);
        border-radius:10px 10px 0 0;
        padding:10px 20px;
        font-family:'Rajdhani', sans-serif;
        font-weight:600;
        color:#AAB4CC;
        border: 1px solid rgba(230,36,41,0.2);
        border-bottom:none;
    }
    .stTabs [aria-selected="true"]{
        background: linear-gradient(135deg, rgba(230,36,41,0.25), rgba(11,61,145,0.25));
        color:#fff !important;
    }

    /* File uploader */
    [data-testid="stFileUploaderDropzone"]{
        background: rgba(11,61,145,0.08);
        border: 2px dashed rgba(230,36,41,0.5) !important;
        border-radius:14px;
    }

    /* Text area */
    .stTextArea textarea{
        background: rgba(255,255,255,0.03);
        color:#EAEAF0;
        border: 1px solid rgba(11,61,145,0.4);
        border-radius:10px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"]{
        background: linear-gradient(180deg, #0d0d16 0%, #0A0A0F 100%);
        border-right: 1px solid rgba(230,36,41,0.25);
    }

    /* Metric-like small chips */
    .chip{
        display:inline-block;
        background: rgba(11,61,145,0.18);
        border: 1px solid rgba(30,95,204,0.5);
        color:#CFE0FF;
        padding:4px 14px;
        border-radius:20px;
        font-size:13px;
        font-weight:600;
        margin:3px;
    }

    hr{ border-color: rgba(230,36,41,0.25); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.markdown('<div class="hero-title">WEB-SCAN</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">AI-Powered Resume &amp; Job Role Screening System</div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🕸️ MISSION BRIEF")
    st.markdown(
        """
        This system scans a resume and predicts the
        **most suitable job role** using a trained
        **TF-IDF + Logistic Regression** pipeline.
        """
    )
    st.markdown("---")
    st.markdown("### ⚙️ HOW IT WORKS")
    st.markdown(
        """
        1. Upload a resume (**PDF / TXT**) or paste text
        2. Text is cleaned & lemmatized
        3. TF-IDF vectorizer transforms it
        4. Model predicts the job role
        """
    )
    st.markdown("---")
    if model_load_error:
        st.error("Model files not found in this folder.")
    else:
        st.success("Model & Vectorizer: LOADED ✅")
        try:
            roles = list(model.classes_)
            st.markdown("**Roles the model recognizes:**")
            chips = "".join(f'<span class="chip">{r}</span>' for r in roles)
            st.markdown(chips, unsafe_allow_html=True)
        except Exception:
            pass

# ------------------------------------------------------------------
# STOP IF MODEL FAILED TO LOAD
# ------------------------------------------------------------------
if model_load_error:
    st.error(
        f"""
        **Could not load model files.**

        Make sure these two files are in the **same folder** as `app.py`:
        - `resume_model.pkl`
        - `tfidf_vectorizer.pkl`

        Error detail: `{model_load_error}`
        """
    )
    st.stop()

# ------------------------------------------------------------------
# MAIN INPUT AREA
# ------------------------------------------------------------------
st.markdown("## 🕷️ Deploy a Resume for Scanning")

tab1, tab2 = st.tabs(["📄 Upload File", "📝 Paste Text"])

resume_text = ""

with tab1:
    st.markdown('<div class="web-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop a resume file here (PDF or TXT)",
        type=["pdf", "txt"],
        help="The app extracts text automatically from PDFs using PyMuPDF.",
    )
    if uploaded_file is not None:
        with st.spinner("Extracting text from file..."):
            resume_text = extract_text(uploaded_file)
        if resume_text.strip():
            st.success(f"Extracted {len(resume_text.split())} words from **{uploaded_file.name}**")
            with st.expander("Preview extracted text"):
                st.text(resume_text[:1500] + ("..." if len(resume_text) > 1500 else ""))
        else:
            st.warning("No text could be extracted from this file.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="web-card">', unsafe_allow_html=True)
    pasted_text = st.text_area(
        "Paste resume content here",
        height=250,
        placeholder="Paste the full resume text here...",
    )
    if pasted_text.strip():
        resume_text = pasted_text
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col_a, col_b, col_c = st.columns([1, 1, 1])
with col_b:
    scan_clicked = st.button("🕸️ SCAN RESUME")

# ------------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------------
if scan_clicked:
    if not resume_text or not resume_text.strip():
        st.warning("Upload a file or paste resume text before scanning.")
    else:
        with st.spinner("Web-slinging through the resume..."):
            cleaned = preprocess(resume_text)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]

            proba_available = hasattr(model, "predict_proba")
            proba_df = None
            if proba_available:
                probs = model.predict_proba(vectorized)[0]
                proba_df = (
                    pd.DataFrame({"Role": model.classes_, "Confidence": probs})
                    .sort_values("Confidence", ascending=False)
                    .reset_index(drop=True)
                )

        st.markdown(
            f"""
            <div class="result-banner">
                <div class="label">Predicted Job Role</div>
                <p class="role">{prediction}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if proba_df is not None:
            st.markdown("### 🎯 Confidence Breakdown")
            top = proba_df.head(6)

            fig = go.Figure(
                go.Bar(
                    x=top["Confidence"] * 100,
                    y=top["Role"],
                    orientation="h",
                    marker=dict(
                        color=top["Confidence"],
                        colorscale=[[0, "#0B3D91"], [1, "#E62429"]],
                        line=dict(color="rgba(255,255,255,0.2)", width=1),
                    ),
                    text=[f"{v:.1f}%" for v in top["Confidence"] * 100],
                    textposition="outside",
                    textfont=dict(color="#EAEAF0"),
                )
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#EAEAF0", family="Rajdhani"),
                xaxis=dict(title="Confidence (%)", gridcolor="rgba(255,255,255,0.08)", range=[0, 100]),
                yaxis=dict(autorange="reversed"),
                height=380,
                margin=dict(l=10, r=30, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("🔍 See cleaned / preprocessed text sent to the model"):
            st.code(cleaned[:2000], language="text")

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color:#6b7280; font-size:13px;'>"
    "WEB-SCAN · Built with Streamlit · Your Friendly Neighborhood Resume Screener 🕷️"
    "</div>",
    unsafe_allow_html=True,
)
