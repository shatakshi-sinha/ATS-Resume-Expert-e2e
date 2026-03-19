from dotenv import load_dotenv
load_dotenv()
import base64
import streamlit as st
import os
import io
import json
import time
import urllib.error
import urllib.request
from PIL import Image 
import pdf2image
import plotly.graph_objects as go
from typing import Dict, List, Optional

# If NIM key is provided, prefer NVIDIA NIM for LLM calls.
# Otherwise, fall back to Google Gemini via `google.generativeai`.
NIM_API_KEY = (os.getenv("NIM_API_KEY") or "").strip()
NIM_BASE_URL = (os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1") or "").strip()

if not NIM_API_KEY:
    import google.generativeai as genai

    # Configure Gemini API
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input_text: str, pdf_content: List[Dict], prompt: str, 
                       model_name: str = "gemini-1.5-flash") -> str:
    """
    Get response from Gemini LLM
    """
    # NVIDIA NIM mode (text-only, ignores `pdf_content` for now).
    if NIM_API_KEY:
        url = NIM_BASE_URL.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {NIM_API_KEY}",
            "Content-Type": "application/json",
        }

        # OpenAI-compatible payload. For now we only send text.
        user_message = f"{input_text}\n\n{prompt}"

        models_to_try: List[str] = []
        for m in [
            model_name,
            "meta/llama-3.1-8b-instruct",
            "meta/llama-3.1-70b-instruct",
            "meta/llama-3.3-70b-instruct",
        ]:
            if m and m not in models_to_try:
                models_to_try.append(m)

        last_error: Optional[str] = None
        for candidate in models_to_try:
            payload = {
                "model": candidate,
                "messages": [{"role": "user", "content": user_message}],
                "temperature": 0.2,
                "max_tokens": 2048,
            }
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw)
                return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    body = ""
                body_preview = (body[:800] + "...") if len(body) > 800 else body
                last_error = f"NIM HTTP {e.code} ({e.reason}). Model: {candidate}. Body: {body_preview}"
                if e.code == 404:
                    continue
                return f"Error: {last_error}"
            except Exception as e:
                last_error = f"NIM request failed for model {candidate}. Detail: {str(e)}"
                # For non-HTTP errors, no point in retrying other models.
                break

        return f"Error: {last_error if last_error else 'Unknown NIM error'}"

    # Google Gemini mode
    # Some model aliases (like "*-latest") may not be available depending on
    # your API project/account permissions. Try the requested model first,
    # then fall back to commonly-available alternatives.
    models_to_try: List[str] = []
    for m in [
        model_name,
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-pro",
    ]:
        if m and m not in models_to_try:
            models_to_try.append(m)

    last_error: Optional[Exception] = None

    for candidate in models_to_try:
        try:
            model = genai.GenerativeModel(candidate)
            if pdf_content:
                # Pass all pages as images
                content_parts = [input_text, prompt]
                for page in pdf_content:
                    content_parts.append({
                        "mime_type": page["mime_type"],
                        "data": page["data"],
                    })
                response = model.generate_content(content_parts)
            else:
                response = model.generate_content([input_text, prompt])
            return response.text
        except Exception as e:
            last_error = e
            err_lower = str(e).lower()

            # If the model isn't found / isn't supported for generateContent,
            # keep trying the next candidate. Otherwise, stop early.
            if any(token in err_lower for token in ["404", "not found", "is not supported"]):
                continue
            break

    return f"Error: {str(last_error) if last_error else 'Unknown error'}"


@st.cache_data(ttl=3600, show_spinner=False)
def get_available_nim_models() -> List[str]:
    """
    Fetch available NVIDIA NIM model IDs via the OpenAI-compatible `/models` endpoint.
    Falls back to a small list if the call fails.
    """
    fallback = [
        "meta/llama-3.1-8b-instruct",
        "meta/llama-3.1-70b-instruct",
        "meta/llama-3.3-70b-instruct",
    ]
    if not NIM_API_KEY:
        return []

    try:
        url = NIM_BASE_URL.rstrip("/") + "/models"
        headers = {"Authorization": f"Bearer {NIM_API_KEY}"}
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)

        ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
        ids = [i for i in ids if i]

        # Keep results stable for UI.
        return ids[:20] if ids else fallback
    except Exception:
        return fallback

def nim_get_embeddings(texts: List[str], model: str) -> List[List[float]]:
    raise RuntimeError(
        "Embeddings mode is disabled in this version. "
        "Use prompt-based text generation via POST /v1/chat/completions."
    )


def cosine_similarity(a: List[float], b: List[float]) -> float:
    raise RuntimeError("Cosine similarity helper is unused (prompt-based mode).")


def extract_job_keywords(job_description: str, max_keywords: int = 30) -> List[str]:
    raise RuntimeError("Heuristic keyword extraction is disabled (prompt-based mode).")


def build_summary_from_heuristics(metrics: Dict) -> str:
    raise RuntimeError("Heuristic summary builder is disabled (prompt-based mode).")


def build_match_score_text_from_metrics(metrics: Dict) -> str:
    raise RuntimeError("Heuristic match score text is disabled (prompt-based mode).")


def input_pdf_setup(uploaded_file) -> List[Dict]:
    """Process uploaded PDF file and extract content"""
    if uploaded_file is not None:
        try:
            # Save the uploaded file to bytes first
            uploaded_file.seek(0)
            pdf_bytes = uploaded_file.read()
            
            # Convert the PDF to images
            images = pdf2image.convert_from_bytes(pdf_bytes)
            
            # Process all pages for analysis context
            pdf_parts = []
            for i, image in enumerate(images):
                # Convert to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG', quality=85)
                img_byte_arr = img_byte_arr.getvalue()
                
                pdf_parts.append({
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(img_byte_arr).decode()
                })
            
            # Reset file pointer for other uses
            uploaded_file.seek(0)
            return pdf_parts
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            raise FileNotFoundError(f"PDF processing failed: {str(e)}")
    else:
        raise FileNotFoundError("No file uploaded")

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from PDF for text-based analysis"""
    try:
        import PyPDF2
        uploaded_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        uploaded_file.seek(0)
        return text
    except Exception as e:
        st.warning(f"Text extraction limited: {str(e)}")
        return ""

def visualize_match_percentage(percentage: float):
    """Create visualization for match percentage"""
    fig = go.Figure()
    
    # Define color based on percentage
    if percentage >= 80:
        bar_color = "green"
    elif percentage >= 60:
        bar_color = "orange"
    else:
        bar_color = "red"
    
    # Create gauge chart
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "ATS Match Score", 'font': {'size': 24}},
        delta={'reference': 80, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': bar_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'lightcoral'},
                {'range': [50, 80], 'color': 'lightyellow'},
                {'range': [80, 100], 'color': 'lightgreen'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_skill_radar_chart(metrics: Dict):
    """Create radar chart for skills"""
    categories = ['Skill Match', 'Experience Match', 'Education Match', 'Keyword Match']
    values = [
        metrics.get('skill_match', 0),
        metrics.get('experience_match', 0),
        metrics.get('education_match', 0),
        metrics.get('keyword_match', 0)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Match Percentage',
        fillcolor='rgba(59, 130, 246, 0.5)',
        line_color='rgb(59, 130, 246)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        height=300
    )
    
    return fig

def generate_resume_suggestions(
    resume_text: str,
    job_description: str,
    pdf_content: List[Dict],
    model_name: str,
) -> Dict:
    """Generate structured suggestions for resume improvement"""
    
    suggestion_prompt = f"""
    Analyze this resume against the job description and provide structured suggestions in JSON format.
    
    RESUME CONTEXT:
    {resume_text}
    
    JOB DESCRIPTION:
    {job_description}
    
    IMPORTANT: Return ONLY valid JSON with this exact structure:
    {{
        "missing_keywords": ["keyword1", "keyword2", "keyword3"],
        "skill_gaps": ["skill1", "skill2"],
        "strengths": ["strength1", "strength2"],
        "weaknesses": ["weakness1", "weakness2"],
        "formatting_issues": ["issue1", "issue2"],
        "actionable_suggestions": ["suggestion1", "suggestion2", "suggestion3"],
        "ats_optimization_tips": ["tip1", "tip2"]
    }}
    
    Rules:
    1. Provide specific, actionable suggestions based on the actual resume and job description
    2. missing_keywords: List important technical and soft skills from the job description that are missing in the resume
    3. skill_gaps: List specific skills that need improvement or are not demonstrated sufficiently
    4. strengths: Highlight what's working well in the resume
    5. weaknesses: Identify specific areas that need improvement
    6. actionable_suggestions: Provide concrete, implementable recommendations
    7. Do NOT include placeholder or generic data - be specific to this resume and job
    
    Make the response completely based on the provided resume and job description.
    """
    
    try:
        response = get_gemini_response(job_description, pdf_content, suggestion_prompt, model_name)
        
        # Extract JSON from response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = response[start_idx:end_idx]
            suggestions = json.loads(json_str)
            
            # Validate structure and ensure all keys exist
            required_keys = ["missing_keywords", "skill_gaps", "strengths", 
                           "weaknesses", "formatting_issues", 
                           "actionable_suggestions", "ats_optimization_tips"]
            
            for key in required_keys:
                if key not in suggestions:
                    suggestions[key] = []
            
            return suggestions
        else:
            # If no JSON found, return empty structure
            return {
                "missing_keywords": [],
                "skill_gaps": [],
                "strengths": [],
                "weaknesses": [],
                "formatting_issues": [],
                "actionable_suggestions": [],
                "ats_optimization_tips": []
            }
    except Exception as e:
        st.warning(f"Could not generate structured suggestions: {str(e)}")
        # Return empty structure
        return {
            "missing_keywords": [],
            "skill_gaps": [],
            "strengths": [],
            "weaknesses": [],
            "formatting_issues": [],
            "actionable_suggestions": [],
            "ats_optimization_tips": []
        }

def calculate_match_metrics(
    resume_text: str,
    job_description: str,
    pdf_content: List[Dict],
    model_name: str,
) -> Dict:
    """Calculate various match metrics using Gemini"""
    
    metrics_prompt = f"""
    Analyze the resume against the job description and calculate precise match metrics.
    
    Resume Summary: {resume_text}
    Job Description: {job_description}
    
    IMPORTANT: Return ONLY valid JSON with this exact structure:
    {{
        "overall_match": 85,
        "skill_match": 80,
        "experience_match": 90,
        "education_match": 75,
        "keyword_match": 85,
        "missing_keywords_count": 5,
        "matching_keywords": ["keyword1", "keyword2", "keyword3"],
        "missing_keywords": ["keyword4", "keyword5", "keyword6"],
        "predicted_interview_chance": "High/Medium/Low",
        "confidence_score": 88
    }}
    
    Rules for calculation:
    1. overall_match: Calculate based on weighted average of skill_match, experience_match, education_match, and keyword_match
    2. skill_match: Percentage match between skills in resume and required skills in job description
    3. experience_match: How well the experience aligns with job requirements
    4. education_match: Match between educational qualifications and job requirements
    5. keyword_match: Percentage of important keywords from job description found in resume
    6. matching_keywords: List specific keywords from job description that ARE found in the resume
    7. missing_keywords: List specific important keywords from job description that are NOT found in the resume
    8. predicted_interview_chance: "High" if overall_match >= 80, "Medium" if >= 60, "Low" if < 60
    9. confidence_score: Your confidence in this analysis (0-100)
    
    Base ALL calculations ONLY on the provided resume and job description.
    Do NOT use any predetermined or generic data.
    Be specific and realistic in your analysis.
    """
    
    try:
        response = get_gemini_response(job_description, pdf_content, metrics_prompt, model_name)
        
        # Extract JSON from response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = response[start_idx:end_idx]
            metrics = json.loads(json_str)
            
            # Validate structure
            required_keys = ["overall_match", "skill_match", "experience_match", 
                           "education_match", "keyword_match", "missing_keywords_count",
                           "matching_keywords", "missing_keywords", "predicted_interview_chance",
                           "confidence_score"]
            
            for key in required_keys:
                if key not in metrics:
                    # If key is missing, use empty/default values
                    if key == "matching_keywords" or key == "missing_keywords":
                        metrics[key] = []
                    elif key == "predicted_interview_chance":
                        metrics[key] = "Medium"
                    elif key.endswith("_match") or key == "confidence_score":
                        metrics[key] = 0
                    elif key == "missing_keywords_count":
                        metrics[key] = 0
            
            # Ensure missing_keywords_count matches actual count
            if "missing_keywords" in metrics:
                metrics["missing_keywords_count"] = len(metrics["missing_keywords"])
            
            return metrics
        else:
            # If no JSON found, return empty structure
            return {
                "overall_match": 0,
                "skill_match": 0,
                "experience_match": 0,
                "education_match": 0,
                "keyword_match": 0,
                "missing_keywords_count": 0,
                "matching_keywords": [],
                "missing_keywords": [],
                "predicted_interview_chance": "N/A",
                "confidence_score": 0
            }
    except Exception as e:
        st.warning(f"Could not calculate match metrics: {str(e)}")
        # Return empty structure
        return {
            "overall_match": 0,
            "skill_match": 0,
            "experience_match": 0,
            "education_match": 0,
            "keyword_match": 0,
            "missing_keywords_count": 0,
            "matching_keywords": [],
            "missing_keywords": [],
            "predicted_interview_chance": "N/A",
            "confidence_score": 0
        }

# Streamlit App Configuration
st.set_page_config(
    page_title="ATS Resume Expert Pro",
    page_icon="ATS",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #374151;
        margin-top: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3B82F6;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .suggestion-box {
        background-color: #EFF6FF;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #93C5FD;
        box-shadow: 0 2px 4px rgba(147, 197, 253, 0.3);
    }
    .highlight {
        background-color: #FEF3C7;
        padding: 0.3rem 0.6rem;
        border-radius: 5px;
        font-weight: 500;
        margin: 0.2rem;
        display: inline-block;
    }
    .success-box {
        background-color: #D1FAE5;
        border: 2px solid #10B981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        border: 2px solid #F59E0B;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border: none;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .tag {
        display: inline-block;
        background: #3B82F6;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    .missing-tag {
        display: inline-block;
        background: #EF4444;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/resume.png", width=80)
    st.title("ATS Resume Expert")
    
    st.markdown("---")
    st.subheader("Configuration")

    dark_mode = st.toggle("Dark mode", value=False, help="Toggle the app theme")
    
    if NIM_API_KEY:
        # NVIDIA NIM model selection for text generation (OpenAI-compatible).
        nim_models = get_available_nim_models()
        llm_model = st.selectbox(
            "NVIDIA NIM Model",
            nim_models if nim_models else ["meta/llama-3.1-8b-instruct"],
            index=0,
            help="Choose the NIM model for analysis"
        )
    else:
        # Gemini model selection
        llm_model = st.selectbox(
            "Gemini Model",
            [
                "gemini-1.5-flash",
                "gemini-1.5-pro",
                "gemini-pro",
                "gemini-1.5-flash-latest",
                "gemini-1.5-pro-latest",
            ],
            index=0,
            help="Choose the Gemini model for analysis"
        )
    
    st.markdown("---")
    st.subheader("Analysis Options")
    
    analyze_keywords = st.checkbox("Keyword Analysis", value=True)
    show_visualizations = st.checkbox("Show Visualizations", value=True)
    
    st.markdown("---")
    st.subheader("About")
    st.info("""
    **ATS Resume Expert** analyzes your resume against job descriptions using the selected LLM.
    
    Features:
    - ATS compatibility scoring
    - Keyword gap analysis
    - Resume improvement suggestions
    
    *Powered by the configured LLM provider*
    """)

# Dark mode overrides
if dark_mode:
    st.markdown(
        """
        <style>
            body, .stApp { background-color: #0b1220 !important; color: #e6edf3 !important; }
            .main-header { color: #e6edf3 !important; background: linear-gradient(135deg, #1d4ed8 0%, #4c1d95 100%) !important; }
            .sub-header { color: #e6edf3 !important; border-bottom-color: #60a5fa !important; }
            .metric-card { background: #0f172a !important; border-left-color: #60a5fa !important; box-shadow: 0 2px 4px rgba(0,0,0,0.4) !important; }
            .suggestion-box { background-color: #0b2239 !important; border-color: #1d4ed8 !important; }
            .success-box { background-color: #052e1a !important; border-color: #10b981 !important; }
            .warning-box { background-color: #2a1f0b !important; border-color: #f59e0b !important; }
            .tag { background: #2563eb !important; }
            .missing-tag { background: #ef4444 !important; }
            input, textarea { background: #0b1220 !important; color: #e6edf3 !important; border-color: #334155 !important; }
            .stSelectbox > div, .stTextInput > div { background: #0b1220 !important; color: #e6edf3 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Main Content
llm_provider_label = "NVIDIA NIM" if NIM_API_KEY else "Google Gemini"
st.markdown(f'<h1 class="main-header">ATS Resume Expert - {llm_provider_label}</h1>', unsafe_allow_html=True)

# Two-column layout for input
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Job Details")
    input_text = st.text_area(
        "**Job Description**",
        height=200,
        placeholder="Paste the complete job description here...",
        key="input",
        help="Copy and paste the full job description for accurate analysis"
    )
    
    company_name = st.text_input(
        "**Company Name** (optional)",
        placeholder="e.g., Google, Amazon, Microsoft"
    )
    
    job_title = st.text_input(
        "**Job Title** (optional)",
        placeholder="e.g., Data Scientist, Software Engineer"
    )

with col2:
    st.markdown("### Resume Upload")
    
    uploaded_file = st.file_uploader(
        "**Upload your resume (PDF only)**",
        type=["pdf"],
        help="Upload your resume in PDF format for analysis"
    )
    
    if uploaded_file is not None:
        st.success("Resume uploaded successfully!")
        
        # Quick preview
        with st.expander("Resume Preview", expanded=False):
            try:
                # Read the file once and reset
                uploaded_file.seek(0)
                images = pdf2image.convert_from_bytes(uploaded_file.read())
                uploaded_file.seek(0)
                if images:
                    # `use_column_width` was deprecated in Streamlit; use the container/column width instead.
                    st.image(images[0], caption="First Page Preview", use_container_width=True)
                    
                    # File info
                    file_size = uploaded_file.size / (1024 * 1024)  # MB
                    st.caption(f"File: {uploaded_file.name} | Size: {file_size:.2f} MB | Pages: {len(images)}")
            except Exception as e:
                st.warning(f"Preview not available: {str(e)}")

# Initialize session state
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ""
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'current_file' not in st.session_state:
    st.session_state.current_file = None

# Analysis Buttons
st.markdown("---")
st.markdown("### Select Analysis Type")

button_col1, button_col2, button_col3 = st.columns(3)

with button_col1:
    submit_comprehensive = st.button("Comprehensive Analysis", use_container_width=True)

with button_col2:
    submit_match = st.button("Match Score", use_container_width=True)

with button_col3:
    submit_suggestions = st.button("Improvement Tips", use_container_width=True)

# Main Processing Logic
if uploaded_file is not None and input_text.strip():
    
    # Store uploaded file in session state to avoid reprocessing
    if st.session_state.current_file != uploaded_file.name:
        with st.spinner("Processing resume..."):
            # Extract resume text
            uploaded_file.seek(0)
            resume_text = extract_text_from_pdf(uploaded_file)
            uploaded_file.seek(0)
            
            # Process PDF for image analysis
            pdf_content = input_pdf_setup(uploaded_file)
            
            st.session_state.current_file = uploaded_file.name
            st.session_state.resume_text = resume_text
            st.session_state.pdf_content = pdf_content
    
    # Use stored values
    resume_text = st.session_state.resume_text
    pdf_content = st.session_state.pdf_content
    
    # Comprehensive Analysis
    if submit_comprehensive:
        with st.spinner("Performing comprehensive analysis..."):
            st.markdown("---")
            st.markdown('<h2 class="sub-header">Comprehensive Analysis Results</h2>', unsafe_allow_html=True)
            
            # Calculate metrics
            metrics = calculate_match_metrics(resume_text, input_text, pdf_content, llm_model)
            st.session_state.metrics = metrics
            
            # Display metrics in cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                match_value = metrics.get('overall_match', 0)
                match_color = '#10B981' if match_value >= 80 else '#F59E0B' if match_value >= 60 else '#EF4444'
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #3B82F6; margin: 0;">Overall Match</h3>
                    <h1 style="color: {match_color}; margin: 0.5rem 0;">
                        {match_value}%
                    </h1>
                    <p style="color: #6B7280; margin: 0;">ATS Compatibility Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                interview_chance = metrics.get('predicted_interview_chance', 'N/A')
                chance_color = {
                    'High': '#10B981',
                    'Medium': '#F59E0B',
                    'Low': '#EF4444'
                }.get(interview_chance, '#6B7280')
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #3B82F6; margin: 0;">Interview Chance</h3>
                    <h1 style="color: {chance_color}; margin: 0.5rem 0;">
                        {interview_chance}
                    </h1>
                    <p style="color: #6B7280; margin: 0;">Predicted likelihood</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                missing_count = metrics.get('missing_keywords_count', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #3B82F6; margin: 0;">Missing Keywords</h3>
                    <h1 style="color: #EF4444; margin: 0.5rem 0;">
                        {missing_count}
                    </h1>
                    <p style="color: #6B7280; margin: 0;">From job description</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                confidence = metrics.get('confidence_score', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #3B82F6; margin: 0;">Confidence</h3>
                    <h1 style="color: #8B5CF6; margin: 0.5rem 0;">
                        {confidence}%
                    </h1>
                    <p style="color: #6B7280; margin: 0;">Analysis confidence</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualizations
            if show_visualizations and metrics.get('overall_match', 0) > 0:
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    st.plotly_chart(visualize_match_percentage(metrics['overall_match']), 
                                  use_container_width=True)
                
                with viz_col2:
                    st.plotly_chart(create_skill_radar_chart(metrics), 
                                  use_container_width=True)
            
            # Summary Analysis (prompt-based, not detailed analysis function)
            st.markdown("### Analysis Summary")
            
            summary_prompt = f"""
            Provide a concise, professional summary analysis of how well the resume matches the job description.
            
            RESUME CONTEXT:
            {resume_text}
            
            Based on your analysis, provide:
            1. Overall fit assessment (2-3 sentences)
            2. Key strengths in alignment with the job
            3. Main areas that need improvement
            4. Top 2-3 actionable recommendations
            
            Be specific to this resume and job description. Do not use generic statements.
            Focus on concrete observations from the resume content.
            """
            
            summary_analysis = get_gemini_response(input_text, pdf_content, summary_prompt, llm_model)
            st.write(summary_analysis)
            
            # Keyword Analysis
            if analyze_keywords:
                st.markdown("### Keyword Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    matching_keywords = metrics.get('matching_keywords', [])
                    if matching_keywords:
                        st.markdown("#### Matching Keywords")
                        for keyword in matching_keywords[:10]:
                            st.markdown(f'<span class="tag">{keyword}</span>', unsafe_allow_html=True)
                    else:
                        st.info("No matching keywords identified")
                
                with col2:
                    missing_keywords = metrics.get('missing_keywords', [])
                    if missing_keywords:
                        st.markdown("#### Missing Keywords")
                        for keyword in missing_keywords[:10]:
                            st.markdown(f'<span class="missing-tag">{keyword}</span>', unsafe_allow_html=True)
                    else:
                        st.info("No missing keywords identified")
            
            # Generate suggestions
            st.markdown("### Structured Suggestions")
            suggestions = generate_resume_suggestions(resume_text, input_text, pdf_content, llm_model)
            
            # Check if suggestions is not None before accessing it
            if suggestions is not None:
                tabs = st.tabs(["Improvements", "Strengths", "Weaknesses", "ATS Tips"])
                
                with tabs[0]:
                    actionable_suggestions = suggestions.get('actionable_suggestions', [])
                    if actionable_suggestions:
                        for suggestion in actionable_suggestions[:5]:
                            st.markdown(f"- {suggestion}")
                    else:
                        st.info("No improvement suggestions generated")
                
                with tabs[1]:
                    strengths = suggestions.get('strengths', [])
                    if strengths:
                        for strength in strengths[:5]:
                            st.markdown(f"{strength}")
                    else:
                        st.info("No strengths identified")
                
                with tabs[2]:
                    weaknesses = suggestions.get('weaknesses', [])
                    if weaknesses:
                        for weakness in weaknesses[:5]:
                            st.markdown(f"{weakness}")
                    else:
                        st.info("No weaknesses identified")
                
                with tabs[3]:
                    ats_tips = suggestions.get('ats_optimization_tips', [])
                    if ats_tips:
                        for tip in ats_tips[:5]:
                            st.markdown(f"{tip}")
                    else:
                        st.info("No ATS optimization tips generated")
            else:
                st.warning("Could not generate suggestions. Please try again.")
    
    # Match Score Analysis
    elif submit_match:
        with st.spinner("Calculating match score..."):
            st.markdown("---")
            st.markdown('<h2 class="sub-header">Match Score Analysis</h2>', unsafe_allow_html=True)
            
            match_prompt = f"""
            Analyze the match between the resume and job description and provide a detailed match score analysis.
            
            RESUME CONTEXT:
            {resume_text}
            
            Provide:
            1. Overall Match Percentage (0-100%)
            2. Breakdown by Category with percentages:
               - Skills match
               - Experience match  
               - Education match
               - Keyword match
            3. Key findings about what matches well
            4. Specific areas that are missing or need improvement
            5. Quick recommendations to improve the match score
            
            Be specific and base all analysis ONLY on the provided resume and job description.
            Do not use generic statements or predetermined data.
            """
            
            response = get_gemini_response(input_text, pdf_content, match_prompt, llm_model)
            st.write(response)
    
    # Improvement Suggestions
    elif submit_suggestions:
        with st.spinner("Generating improvement suggestions..."):
            st.markdown("---")
            st.markdown('<h2 class="sub-header">Resume Improvement Suggestions</h2>', unsafe_allow_html=True)
            
            suggestions = generate_resume_suggestions(resume_text, input_text, pdf_content, llm_model)
            
            # Check if suggestions is not None before accessing it
            if suggestions is not None:
                # Display in organized sections
                sections = [
                    ("Missing Keywords", suggestions.get('missing_keywords', []), "Keywords from job description not found in resume"),
                    ("Skill Gaps", suggestions.get('skill_gaps', []), "Skills mentioned in job description but missing in resume"),
                    ("Strengths", suggestions.get('strengths', []), "What's working well in your resume"),
                    ("Weaknesses", suggestions.get('weaknesses', []), "Areas that need improvement"),
                    ("Actionable Suggestions", suggestions.get('actionable_suggestions', []), "Specific actions to improve your resume"),
                    ("ATS Optimization", suggestions.get('ats_optimization_tips', []), "Tips to optimize for Applicant Tracking Systems")
                ]
                
                for title, items, description in sections:
                    if items:
                        with st.expander(f"{title} ({len(items)})", expanded=True if title == "Actionable Suggestions" else False):
                            st.caption(description)
                            for item in items:
                                if title == "Missing Keywords":
                                    st.markdown(f'<span class="missing-tag">{item}</span>', unsafe_allow_html=True)
                                elif title == "Skill Gaps":
                                    st.markdown(f"- **{item}** - Consider adding relevant experience or training")
                                else:
                                    st.markdown(f"- {item}")
                    else:
                        with st.expander(f"{title} (0)", expanded=False):
                            st.info(f"No {title.lower()} identified from the analysis.")
            else:
                st.warning("Could not generate suggestions. Please try again.")
            
else:
    if not input_text.strip() and not uploaded_file:
        st.warning("""
        Please provide both:
        1. **Job Description** - Paste the job description in the left panel
        2. **Resume** - Upload your PDF resume in the right panel
        """)
    elif not input_text.strip():
        st.warning("Please enter a job description in the left panel.")
    elif not uploaded_file:
        st.warning("Please upload your resume in PDF format.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 10px;'>
    <h3 style='color: #1E3A8A;'>Pro Tips for Better Results</h3>
    <div style='display: flex; justify-content: center; flex-wrap: wrap; gap: 1rem; margin-top: 1rem;'>
        <div style='flex: 1; min-width: 200px; padding: 1rem; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <strong>Use Detailed Job Descriptions</strong>
            <p style='font-size: 0.9rem; color: #6B7280; margin: 0.5rem 0 0 0;'>
                The more detailed the job description, the more accurate the analysis
            </p>
        </div>
        <div style='flex: 1; min-width: 200px; padding: 1rem; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <strong>Include Relevant Keywords</strong>
            <p style='font-size: 0.9rem; color: #6B7280; margin: 0.5rem 0 0 0;'>
                Add keywords from the job description to your resume
            </p>
        </div>
        <div style='flex: 1; min-width: 200px; padding: 1rem; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <strong>Quantify Achievements</strong>
            <p style='font-size: 0.9rem; color: #6B7280; margin: 0.5rem 0 0 0;'>
                Use numbers and metrics to showcase your impact
            </p>
        </div>
        <div style='flex: 1; min-width: 200px; padding: 1rem; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <strong>ATS-Friendly Format</strong>
            <p style='font-size: 0.9rem; color: #6B7280; margin: 0.5rem 0 0 0;'>
                Use standard fonts and avoid complex formatting
            </p>
        </div>
    </div>
    <p style='margin-top: 2rem; color: #4B5563;'>
        Powered by <strong>{llm_provider_label}</strong> - For best results, ensure your resume is up-to-date and in PDF format
    </p>
</div>
""", unsafe_allow_html=True)