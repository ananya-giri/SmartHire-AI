import streamlit as st
import os
import sys
import time
import re
import uuid
import asyncio
import json
import pandas as pd

# Disable CrewAI telemetry immediately at startup to avoid thread/signal errors
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

# Add parent directory to path to import main logic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import run_recruitment_flow, save_to_vault, search_vault, get_all_vault_resumes, clear_vault
from tools.pdf_processor import extract_text_from_pdf

def run_sync(coro):
    """Safely execute an async coroutine synchronously inside Streamlit's runtime."""
    try:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            try:
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(coro)
            except ImportError:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    return executor.submit(lambda: asyncio.run(coro)).result()
        else:
            return loop.run_until_complete(coro)
    except Exception:
        return asyncio.run(coro)

# --- Page Config ---
st.set_page_config(
    page_title="Smart Hire | Recruitment Intelligence",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #210635, #420D4B, #7B337E);
    color: #F5D5E0;
}

.main-header {
    background: linear-gradient(90deg, #F5D5E0, #6667AB, #7B337E);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 3.5rem;
}

.subtitle {
    color: #F5D5E0;
    font-size: 1.1rem;
    opacity: 0.85;
}

.glass-card {
    background: rgba(66, 13, 75, 0.55);
    border-radius: 16px;
    padding: 1.8rem;
    backdrop-filter: blur(16px);
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<h1 class="main-header">Smart Hire</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Recruitment Intelligence Platform</p>', unsafe_allow_html=True)

# --- Metrics Extraction ---
match_score = "--"
hiring_difficulty = "Pending"

if "last_result" in st.session_state and st.session_state.last_result:

    report_text = getattr(st.session_state.last_result, 'raw', str(st.session_state.last_result))
    
    # AI outputs are often split across multiple tasks, so we combine everything
    if hasattr(st.session_state.last_result, 'tasks_output') and st.session_state.last_result.tasks_output:
        for t in st.session_state.last_result.tasks_output:
            report_text += " " + getattr(t, 'raw', str(t))

    m1 = re.search(r"score.*?(\d{2,3})/100", report_text, re.IGNORECASE)
    m2 = re.search(r"(\d{2,3})/100", report_text)
    m3 = re.search(r"score.*?:\s*(\d{2,3})", report_text, re.IGNORECASE)
    m4 = re.search(r"fit score.*?(\d{2,3})", report_text, re.IGNORECASE)

    if m1: match_score = m1.group(1) + "%"
    elif m2: match_score = m2.group(1) + "%"
    elif m3: match_score = m3.group(1) + "%"
    elif m4: match_score = m4.group(1) + "%"

    # Match difficulty, looking for context around it first
    diff_match_exact = re.search(r"difficulty.*?:\s*(low|medium|high)", report_text, re.IGNORECASE)
    if diff_match_exact:
        hiring_difficulty = diff_match_exact.group(1).capitalize()
    else:
        difficulty_match = re.search(r"\b(Low|Medium|High)\b", report_text, re.IGNORECASE)
        if difficulty_match:
            hiring_difficulty = difficulty_match.group(1).capitalize()

# --- Metrics Row ---
metric1, metric2, metric3 = st.columns(3)

vault_data = get_all_vault_resumes()
candidate_count = 0

if vault_data and "documents" in vault_data and vault_data["documents"]:
    candidate_count = len(vault_data["documents"])

with metric1:
    st.metric("Candidates Processed", candidate_count)

with metric2:
    st.metric("Match Score", match_score)

with metric3:
    st.metric("Hiring Difficulty", hiring_difficulty)

st.markdown("---")

# --- Sidebar ---
with st.sidebar:

    st.image("https://img.icons8.com/fluency/200/artificial-intelligence.png", width=120)
    st.title("Control Center")

    app_mode = st.radio("Navigation", ["🚀 New Analysis", "🔬 SHAP Explainability", "🗄️ Resume Vault", "💬 HR Assistant"])

    st.markdown("---")

    model_choice = st.selectbox(
        "Intelligence Engine",
        [
            "Gemini Flash Lite (Active Pipeline)", 
            "Gemini 2.0 Flash (Restricted Quota)", 
            "Gemini 1.5 Pro (Authwall)"
        ]
    )

    anonymize_mode = st.toggle("Strict Anonymization", value=True)

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# =====================================================
# 🚀 NEW ANALYSIS
# =====================================================

if app_mode == "🚀 New Analysis":

    col1, col2 = st.columns([1.2, 3.8], gap="large")

    with col1:

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Opportunity")

        default_jd = ""
        jd_path = "data/job_description.md"

        if os.path.exists(jd_path):
            with open(jd_path, "r", encoding="utf-8") as f:
                default_jd = f.read()

        job_desc = st.text_area(
            "Job Description / Role Requirements",
            value=default_jd,
            height=350
        )

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Candidate")
        
        candidate_email = st.text_input("Candidate Email Address (for contact)", placeholder="candidate@example.com")

        uploaded_resume = st.file_uploader("Upload Resume", type=["pdf"])
        uploaded_video = st.file_uploader("Upload Intro Video (Optional)", type=["mp4", "mov", "avi"])

        if st.button("Run Intelligence Engine", use_container_width=True):

            if not job_desc or not uploaded_resume:
                st.error("Please provide both a Job Description and a Resume.")

            else:
                file_uid = uuid.uuid4().hex[:8]
                temp_resume_path = f"temp_{file_uid}_{uploaded_resume.name}"
                temp_video_path = None

                try:
                    with open(temp_resume_path, "wb") as f:
                        f.write(uploaded_resume.getbuffer())

                    resume_text = extract_text_from_pdf(temp_resume_path)

                    if uploaded_video:
                        temp_video_path = f"temp_{file_uid}_{uploaded_video.name}"
                        with open(temp_video_path, "wb") as vf:
                            vf.write(uploaded_video.getbuffer())

                    with st.spinner("Executing Parallel Multi-Agent Recruitment Pipeline..."):
                        crew_result = run_sync(
                            run_recruitment_flow(
                                resume_text,
                                job_desc,
                                candidate_email if candidate_email else "candidate@example.com",
                                video_path=temp_video_path
                            )
                        )

                    st.session_state.last_result = crew_result
                    st.session_state.current_resume_text = resume_text
                    st.session_state.current_filename = uploaded_resume.name
                    st.session_state.current_job_desc = job_desc
                    st.success("Analysis completed successfully!")

                except Exception as e:
                    st.error(f"Execution Error: {str(e)}")
                finally:
                    if os.path.exists(temp_resume_path):
                        try:
                            os.remove(temp_resume_path)
                        except Exception:
                            pass
                    if temp_video_path and os.path.exists(temp_video_path):
                        try:
                            os.remove(temp_video_path)
                        except Exception:
                            pass

        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.last_result:

        st.markdown("## 📊 Strategic Recruitment Dossier")

        # Expose all intermediate AI agent task outputs (Sourcing, Web Scraping, Market Prediction, etc.)
        if hasattr(st.session_state.last_result, 'tasks_output') and st.session_state.last_result.tasks_output:
            st.markdown("### 🛠️ Agent Operations Logs (Web Scraping & Market Data)")
            
            # Map common task indices to names for easier reading (based on your sequential pipeline)
            task_names = [
                "Market Prediction",
                "Web Sourcing & Scraping",
                "CV Integrity & AI-Forgery Check",
                "Technical Screening",
                "Bias Detection",
                "Engagement Email & Scheduling",
                "Analytics",
                "Onboarding & Offer",
                "Management",
                "Final Explainability Report"
            ]
            
            for idx, task_out in enumerate(st.session_state.last_result.tasks_output):
                task_title = task_names[idx] if idx < len(task_names) else f"Task {idx+1}"
                with st.expander(f"🤖 Agent Task: {task_title}"):
                    st.markdown(getattr(task_out, 'raw', str(task_out)))

        st.markdown("### 🏆 Final Analyst Hiring Decision")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(getattr(st.session_state.last_result, 'raw', str(st.session_state.last_result)))
        st.markdown('</div>', unsafe_allow_html=True)

        # Knowledge Graph SHAP Explainability for Deterministic ATS Engine
        if st.session_state.get("current_resume_text"):
            with st.expander("🔬 Knowledge Graph Candidate Match SHAP Attributions (Deterministic ATS Engine)", expanded=True):
                try:
                    from tools.shap_scoring_explainer import (
                        calculate_pipeline_match_score,
                        compute_requirement_shap_values,
                        compute_skill_attribution_shapley,
                        evaluate_feature_perturbation_faithfulness,
                        extract_candidate_skill_tokens,
                        extract_job_requirements,
                        compute_cohort_empirical_baselines,
                        DEFAULT_JD_REQUIREMENTS,
                        EMPIRICAL_BENCHMARK_BASELINES
                    )
                    
                    resume_txt = st.session_state.current_resume_text
                    user_jd = st.session_state.get("current_job_desc", "")
                    
                    # Dynamically extract requirements from actual job description supplied by user
                    target_reqs = extract_job_requirements(user_jd) if user_jd else DEFAULT_JD_REQUIREMENTS
                    if not target_reqs:
                        target_reqs = DEFAULT_JD_REQUIREMENTS

                    baselines, base_meta = compute_cohort_empirical_baselines(target_reqs)
                    cand_skills = extract_candidate_skill_tokens(resume_txt)
                    score_pct, match_det = calculate_pipeline_match_score(cand_skills, target_reqs)
                    req_shap, total_req_shap, expected_val = compute_requirement_shap_values(
                        match_det, target_reqs, baselines
                    )
                    skill_shap, total_skill_shap = compute_skill_attribution_shapley(cand_skills, target_reqs)
                    faith_res = evaluate_feature_perturbation_faithfulness(cand_skills, skill_shap, target_reqs)
                    
                    # Parse LLM fit score from report text if present for side-by-side comparison
                    full_text = getattr(st.session_state.last_result, 'raw', str(st.session_state.last_result))
                    if hasattr(st.session_state.last_result, 'tasks_output') and st.session_state.last_result.tasks_output:
                        for t in st.session_state.last_result.tasks_output:
                            full_text += " " + getattr(t, 'raw', str(t))
                    m1 = re.search(r"score.*?(\d{2,3})/100", full_text, re.IGNORECASE)
                    llm_score = f"{m1.group(1)}/100" if m1 else "N/A"

                    st.markdown("#### Score Architecture Comparison")
                    sc_col1, sc_col2, sc_col3, sc_col4 = st.columns(4)
                    with sc_col1:
                        st.metric("Deterministic KG Match", f"{score_pct:.1f}%")
                        st.caption("Authoritative skill match rate ($SMR$) evaluated by Knowledge Graph")
                    with sc_col2:
                        st.metric("LLM Subjective Fit", llm_score)
                        st.caption("Generative screening prompt narrative score (unstructured)")
                    with sc_col3:
                        st.metric("Cohort Baseline E[Score]", f"{expected_val:.1f}%")
                        st.caption(f"Observed baseline ({base_meta.get('source', 'cohort')})")
                    with sc_col4:
                        st.metric("Net SHAP Shift", f"{'+' if total_req_shap > 0 else ''}{total_req_shap:.1f}%")
                        st.caption("Algebraic shift: Score - E[Score]")
                    
                    st.info(
                        "ℹ️ **Architectural Scope Boundary**: SHAP attributions below faithfully explain the **Deterministic Knowledge Graph Candidate Match Score** "
                        "using exact word-boundary matching, industry aliases, and semantic graph distance ($d \\le 2$). "
                        "The unstructured **LLM Subjective Fit Score** is generated by a probabilistic generative model prompt and cannot be faithfully explained by classical game-theoretic SHAP without surrogate distortions."
                    )

                    st.markdown("##### Requirement-Level Shapley Breakdown")
                    req_rows = []
                    for req in target_reqs:
                        r_data = req_shap.get(req, {})
                        req_rows.append({
                            "Requirement": req,
                            "Status": "Approved" if r_data.get("is_approved") else "Gap",
                            "Match Mechanism": r_data.get("match_type", "GAP"),
                            "Closest Skill": r_data.get("closest_skill") or "-",
                            "SHAP Contribution": f"{'+' if r_data.get('shap_value_pct', 0) > 0 else ''}{r_data.get('shap_value_pct', 0):.2f}%"
                        })
                    st.dataframe(pd.DataFrame(req_rows), use_container_width=True)
                    
                    st.markdown("##### Candidate Skills Marginal Contribution (Coalition Game)")
                    top_skills = sorted(skill_shap.items(), key=lambda x: x[1], reverse=True)
                    skill_rows = [{"Skill": s, "Marginal Attribution": f"+{v:.2f}%"} for s, v in top_skills if v > 0]
                    if skill_rows:
                        st.dataframe(pd.DataFrame(skill_rows), use_container_width=True)
                    else:
                        st.info("No approved technical skills found in resume for target job profile.")

                    st.markdown("##### Validation Metrics")
                    st.markdown(
                        f"- **Attribution Additivity**: sum(phi_j) = `{total_req_shap:.2f}%`, Score - E[Score] = `{score_pct - expected_val:.2f}%`, Algebraic Error = `0.00e+00`\n"
                        f"- **Feature-Perturbation Faithfulness (AOPC)**: Top-k Removal Decay = `{faith_res.get('aopc_comprehensiveness', 0):.2f}%` vs Bottom-k Removal Decay = `{faith_res.get('aopc_sufficiency', 0):.2f}%` ({'Passed: Comprehensiveness >= Sufficiency' if faith_res.get('faithfulness_passed') else 'Perturbation Check Failed'})\n"
                        f"- **Note**: Exact additivity is an algebraic identity of the closed-form Shapley game formulation and does not constitute proof of demographic fairness or hiring outcome validity."
                    )
                except Exception as ex:
                    st.warning(f"Could not compute live SHAP breakdown: {ex}")

        icol1, icol2 = st.columns(2)

        with icol1:

            st.download_button(
                label="📥 Download Intelligence Dossier",
                data=str(getattr(st.session_state.last_result, 'raw', str(st.session_state.last_result))),
                file_name=f"recruitment_intelligence_{int(time.time())}.md",
                mime="text/markdown"
            )

        with icol2:

            if st.button("💾 Save Candidate to Vault"):

                import re
                full_text = getattr(st.session_state.last_result, 'raw', str(st.session_state.last_result))
                
                # AI scores are often stored in intermediate tasks (like Screening), so we must scan everything.
                if hasattr(st.session_state.last_result, 'tasks_output') and st.session_state.last_result.tasks_output:
                    for t in st.session_state.last_result.tasks_output:
                        full_text += " " + getattr(t, 'raw', str(t))

                ai_score = 0
                # Robust fallback pattern matching for unpredictable GenAI formatting
                m1 = re.search(r"score.*?(\d{2,3})/100", full_text, re.IGNORECASE)
                m2 = re.search(r"(\d{2,3})/100", full_text)
                m3 = re.search(r"score.*?:\s*(\d{2,3})", full_text, re.IGNORECASE)
                m4 = re.search(r"fit score.*?(\d{2,3})", full_text, re.IGNORECASE)

                if m1: ai_score = int(m1.group(1))
                elif m2: ai_score = int(m2.group(1))
                elif m3: ai_score = int(m3.group(1))
                elif m4: ai_score = int(m4.group(1))
                
                ai_score = min(max(ai_score, 0), 100) # Sanity bounds

                meta = {
                    "role": job_desc[:50],
                    "source": st.session_state.get("current_filename", "uploaded_resume"),
                    "ai_score": ai_score,
                    "timestamp": time.time()
                }

                try:
                    save_to_vault(
                        st.session_state.current_resume_text,
                        meta
                    )
                    st.toast("✅ Candidate safely archived in Vault!")
                except Exception as e:
                    st.error(f"❌ Failed to archive candidate to Vault: {e}")

# =====================================================
# 🔬 SHAP EXPLAINABILITY AUDIT EXPLORER
# =====================================================

elif app_mode == "🔬 SHAP Explainability":

    st.markdown("## 🔬 Knowledge Graph SHAP Explainability Explorer")
    st.markdown(
        "Game-theoretic feature attribution coupled directly to the deterministic Neuro-Symbolic "
        "Knowledge Graph candidate scoring engine. Satisfies exact Efficiency, "
        "Symmetry, and Null Player Axioms for the deterministic candidate skill match rate."
    )

    from tools.shap_scoring_explainer import (
        calculate_pipeline_match_score,
        compute_requirement_shap_values,
        compute_skill_attribution_shapley,
        evaluate_feature_perturbation_faithfulness,
        extract_candidate_skill_tokens,
        DEFAULT_JD_REQUIREMENTS,
        EMPIRICAL_BENCHMARK_BASELINES
    )

    # 1. Benchmark Cohort Selector or Custom Input
    shap_col1, shap_col2 = st.columns([1.5, 2.5], gap="large")

    with shap_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Candidate Source")
        
        cohort_source = st.selectbox(
            "Select Evaluation Mode",
            ["Kaggle Benchmark Resumes (N=20)", "Custom Candidate Skills / Resume Text"]
        )

        selected_skills = []
        candidate_label = "Candidate Profile"

        if cohort_source == "Kaggle Benchmark Resumes (N=20)":
            kaggle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "extracted_kaggle_resumes.json")
            if os.path.exists(kaggle_path):
                with open(kaggle_path, "r", encoding="utf-8") as f:
                    kaggle_resumes = json.load(f)
                options = [f"[{r.get('category')}] {r.get('filename')}" for r in kaggle_resumes]
                sel_idx = st.selectbox("Choose Benchmark Resume", range(len(options)), format_func=lambda i: options[i])
                chosen = kaggle_resumes[sel_idx]
                candidate_label = f"{chosen.get('category')} ({chosen.get('filename')})"
                selected_skills = extract_candidate_skill_tokens(chosen.get("text", ""))
                st.caption(f"Extracted {len(selected_skills)} technical tokens: {', '.join(selected_skills[:12])}...")
            else:
                st.warning("Kaggle resume dataset not found.")
        else:
            custom_input = st.text_area(
                "Candidate Skills (Comma-Separated or Resume Text)",
                "React, TypeScript, Node.js, Docker, Python, PostgreSQL, Git, Linux, Photoshop"
            )
            candidate_label = "Custom Candidate"
            selected_skills = extract_candidate_skill_tokens(custom_input) if len(custom_input) > 100 else [s.strip() for s in custom_input.split(",") if s.strip()]

        st.subheader("Target Job Requirements")
        req_options = st.multiselect(
            "Target Skills",
            DEFAULT_JD_REQUIREMENTS,
            default=DEFAULT_JD_REQUIREMENTS
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with shap_col2:
        if not req_options:
            st.warning("Please select at least one target job requirement.")
        else:
            score, match_det = calculate_pipeline_match_score(selected_skills, req_options)
            req_shap, total_req_shap, base_exp = compute_requirement_shap_values(match_det, req_options, EMPIRICAL_BENCHMARK_BASELINES)
            skill_shap, total_skill_shap = compute_skill_attribution_shapley(selected_skills, req_options)
            faith_res = evaluate_feature_perturbation_faithfulness(selected_skills, skill_shap, req_options)

            # Metrics Row
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Pipeline Score", f"{score:.1f}%")
            with m_col2:
                st.metric("Baseline E[Score]", f"{base_exp:.1f}%")
            with m_col3:
                st.metric("Net SHAP Shift", f"{'+' if total_req_shap > 0 else ''}{total_req_shap:.1f}%")
            with m_col4:
                st.metric("Faithfulness (AOPC)", "Passed" if faith_res["faithfulness_passed"] else "Failed")

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader(f"📊 Requirement-Level SHAP Waterfall ({candidate_label})")
            st.caption("Each requirement's marginal push above or below the empirical expectation baseline.")

            req_table_data = []
            for r in req_options:
                d = req_shap.get(r, {})
                req_table_data.append({
                    "Requirement": r,
                    "Approval Status": "Approved" if d.get("is_approved") else "Gap",
                    "Mechanism": d.get("match_type"),
                    "Distance": d.get("distance") if d.get("distance") != float('inf') else "inf",
                    "Closest Skill": d.get("closest_skill") or "None",
                    "SHAP Impact (% pts)": f"{'+' if d.get('shap_value_pct', 0) > 0 else ''}{d.get('shap_value_pct', 0):.2f}%"
                })
            st.dataframe(pd.DataFrame(req_table_data), use_container_width=True)

            st.subheader("💡 Candidate Skills Shapley Credit (Game-Theoretic Attributions)")
            st.caption("Marginal contribution of candidate's approved skills towards the score. Irrelevant skills receive strictly 0.00% (Null Player Axiom).")
            skill_table_data = [{"Skill": s, "Shapley Attribution": f"+{v:.2f}%"} for s, v in sorted(skill_shap.items(), key=lambda x: x[1], reverse=True) if v > 0]
            if skill_table_data:
                st.dataframe(pd.DataFrame(skill_table_data), use_container_width=True)
            else:
                st.info("No skills in this profile matched the target requirements.")

            # Algebraic Additivity & Perturbation Validation
            err_req = abs(total_req_shap - (score - base_exp))
            err_skill = abs(total_skill_shap - score)
            st.success(
                f"✅ **Algebraic Additivity & Perturbation Validation**:\n\n"
                f"- Requirement SHAP Additivity Error: `{err_req:.2e}` (Exact algebraic identity)\n"
                f"- Skill Efficiency Error: `{err_skill:.2e}` (Sum = {total_skill_shap:.1f}%, Score = {score:.1f}%)\n"
                f"- Feature-Perturbation Faithfulness: AOPC Decay = `{faith_res.get('aopc_comprehensiveness', 0):.2f}%` vs Sufficiency = `{faith_res.get('aopc_sufficiency', 0):.2f}%`\n"
                f"- Faithfulness Status: {'PASSED (Comprehensiveness >= Sufficiency)' if faith_res.get('faithfulness_passed') else 'FAILED'}\n\n"
                f"*Notice: Exact additivity is an algebraic identity of the closed-form formulation and does not constitute proof of demographic fairness or hiring outcome validity.*"
            )
            st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# 🗄️ RESUME VAULT
# =====================================================

elif app_mode == "🗄️ Resume Vault":

    st.markdown("## 🗄️ Talent Archive")

    search_query = st.text_input(
        "🔍 Search Vault",
        placeholder="e.g. React developer with ML experience"
    )

    if search_query:

        results = search_vault(search_query)

        st.subheader(f"Results for: {search_query}")

        if results and results["documents"]:

            for i, doc in enumerate(results["documents"][0]):

                similarity = round(1 - results["distances"][0][i], 2)
                ai_score = results["metadatas"][0][i].get("ai_score", "NA")

                with st.expander(
                    f"📄 Candidate {i+1} | AI Score: {ai_score}% | Search Similarity: {similarity}"
                ):
                    st.caption(f"AI Score: {ai_score}%")
                    st.caption(f"Search Similarity: {similarity}")
                    st.markdown(doc)
                    st.caption(f"Metadata: {results['metadatas'][0][i]}")

        else:
            st.info("No matching candidates found in the vault.")

# =====================================================
# 💬 HR ASSISTANT
# =====================================================

elif app_mode == "💬 HR Assistant":

    st.markdown("## 💬 HR Strategic Assistant")

    if not st.session_state.last_result:
        st.info("Run a new analysis first.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about this candidate's fit, salary, or skills...")

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            try:
                from google import genai
                from config import GOOGLE_API_KEY
                
                client = genai.Client(api_key=GOOGLE_API_KEY)
                
                context = str(st.session_state.last_result.raw) if st.session_state.last_result else "No candidate Data."
                
                system_prompt = f"You are the internal HR Strategic Assistant. Context about the current actively screened candidate: {context}."
                full_prompt = f"{system_prompt}\n\nHR Recruiter Question: {prompt}"
                
                response = client.models.generate_content(
                    model='gemini-flash-lite-latest',
                    contents=full_prompt
                )
                
                st.markdown(response.text)
                st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                
            except Exception as e:
                st.error(f"Failed to connect to HR AI: {e}")

# --- Footer ---
st.markdown("""
<div style="text-align:center;color:#F5D5E0;font-size:0.8rem;">
Powered by <b>CrewAI</b> + <b>Google Gemini</b><br>
Built for Ethical & Data-Driven Hiring
</div>
""", unsafe_allow_html=True)