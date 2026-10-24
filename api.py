# pyrefly: ignore [missing-import]
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import shutil
import uuid
import time
import asyncio
from main import run_recruitment_flow, save_to_vault, search_vault, get_all_vault_resumes
from tools.pdf_processor import extract_text_from_pdf
from security_guardrails import AIHiringGuardrails

app = FastAPI(
    title="Smart Hire Enterprise API",
    description="High-Throughput Parallel Multi-Agent AI Recruitment Platform",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-Memory Job Store for Async Scalability & Task Tracking
JOBS_STORE: Dict[str, Dict[str, Any]] = {}

class ChatMessage(BaseModel):
    message: str
    context: str

class VaultDocument(BaseModel):
    content: str
    email: str
    role: str

@app.get("/health")
def health_check():
    """System Design Health & Performance Monitoring Endpoint"""
    return {
        "status": "online",
        "active_jobs_count": sum(1 for j in JOBS_STORE.values() if j.get("status") == "processing"),
        "completed_jobs_count": sum(1 for j in JOBS_STORE.values() if j.get("status") == "completed"),
        "timestamp": time.time()
    }

async def _process_recruitment_job(
    job_id: str,
    safe_resume_text: str,
    job_desc_text: str,
    candidate_email: str,
    video_path: Optional[str] = None
):
    """Background Worker function to execute parallel multi-agent DAG"""
    start_time = time.time()
    try:
        JOBS_STORE[job_id]["status"] = "processing"
        JOBS_STORE[job_id]["stage"] = "Executing Parallel Multi-Agent Crew (4-Tier DAG)"

        crew_result = await run_recruitment_flow(
            safe_resume_text,
            job_desc_text,
            candidate_email,
            video_path=video_path
        )

        result_raw = crew_result.raw if hasattr(crew_result, 'raw') else str(crew_result)
        elapsed = round(time.time() - start_time, 2)

        JOBS_STORE[job_id]["status"] = "completed"
        JOBS_STORE[job_id]["result"] = result_raw
        JOBS_STORE[job_id]["elapsed_seconds"] = elapsed
        JOBS_STORE[job_id]["completed_at"] = time.time()
    except Exception as e:
        JOBS_STORE[job_id]["status"] = "failed"
        JOBS_STORE[job_id]["error"] = str(e)
        JOBS_STORE[job_id]["elapsed_seconds"] = round(time.time() - start_time, 2)
    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass

@app.post("/analyze")
async def analyze_candidate(
    background_tasks: BackgroundTasks,
    job_desc: UploadFile = File(...),
    email: str = Form(...),
    resume: UploadFile = File(...),
    video: Optional[UploadFile] = File(None),
    async_mode: Optional[bool] = Form(False)
):
    file_uid = uuid.uuid4().hex[:8]
    temp_resume_path = f"temp_{file_uid}_{resume.filename}"
    temp_jd_path = f"temp_{file_uid}_{job_desc.filename}"
    temp_video_path = f"temp_{file_uid}_{video.filename}" if video and video.filename else None

    try:
        with open(temp_resume_path, "wb") as buffer:
            shutil.copyfileobj(resume.file, buffer)
            
        if temp_resume_path.lower().endswith('.pdf'):
            resume_text = extract_text_from_pdf(temp_resume_path)
            if not resume_text or resume_text.startswith("Error:"):
                with open(temp_resume_path, "r", encoding="utf-8", errors="replace") as f:
                    resume_text = f.read()
        else:
            with open(temp_resume_path, "r", encoding="utf-8", errors="replace") as f:
                resume_text = f.read()
        
        with open(temp_jd_path, "wb") as buffer:
            shutil.copyfileobj(job_desc.file, buffer)
            
        if temp_jd_path.lower().endswith('.pdf'):
            job_desc_text = extract_text_from_pdf(temp_jd_path)
            if not job_desc_text or job_desc_text.startswith("Error:"):
                with open(temp_jd_path, "r", encoding="utf-8", errors="replace") as f:
                    job_desc_text = f.read()
        else:
            with open(temp_jd_path, "r", encoding="utf-8", errors="replace") as f:
                job_desc_text = f.read()

        if video and temp_video_path:
            with open(temp_video_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)
        
        # --- INPUT GUARDRAILS (Gap-2 fix: full production scan, all layers) ---
        try:
            for _label, _text in (("resume", resume_text), ("job_description", job_desc_text)):
                _v = AIHiringGuardrails.scan_document(_text)
                if _v["blocked"]:
                    _fired = [k for k, v in _v["layers"].items() if v]
                    return {"status": "error",
                            "message": f"Security Guardrail: prompt injection detected in {_label}. Layers fired: {_fired}"}
                if _label == "resume":
                    resume_text = _v["sanitized_text"]
            AIHiringGuardrails.llama_guard_check(resume_text)
            AIHiringGuardrails.llama_guard_check(job_desc_text)
        except Exception as e:
            detail = getattr(e, 'detail', str(e))
            return {"status": "error", "message": f"Security Guardrail: {detail}"}
        
        # Privacy Guardrail: Redact PII before sending to agents
        safe_resume_text = AIHiringGuardrails.redact_pii(resume_text)
        # -------------------------
        
        # Save current candidate to RAG Resume Vault
        cand_email = email if email else "candidate@example.com"
        candidate_metadata = {
            "source": resume.filename,
            "email": cand_email,
            "timestamp": time.time(),
            "role": job_desc_text[:50]
        }
        try:
            save_to_vault(resume_text, candidate_metadata)
        except Exception as e:
            print(f"[VAULT WARNING] Could not auto-archive to vault: {e}")

        # Async Job Creation
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        JOBS_STORE[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": time.time(),
            "email": cand_email
        }

        if async_mode:
            # Non-blocking async response for frontend polling / WebSockets
            background_tasks.add_task(
                _process_recruitment_job,
                job_id,
                safe_resume_text,
                job_desc_text,
                cand_email,
                temp_video_path
            )
            return {
                "status": "queued",
                "job_id": job_id,
                "message": "Candidate analysis job queued successfully. Poll /analyze/status/{job_id} for updates."
            }
        else:
            # Synchronous direct wait mode (runs parallel DAG and returns result directly)
            await _process_recruitment_job(
                job_id,
                safe_resume_text,
                job_desc_text,
                cand_email,
                temp_video_path
            )
            job_data = JOBS_STORE[job_id]
            if job_data.get("status") == "completed":
                return {
                    "status": "success",
                    "result": job_data["result"],
                    "elapsed_seconds": job_data.get("elapsed_seconds")
                }
            else:
                return {
                    "status": "error",
                    "message": job_data.get("error", "Processing failed.")
                }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        # Clean up uploaded temp files
        if os.path.exists(temp_resume_path):
            try:
                os.remove(temp_resume_path)
            except Exception:
                pass
        if os.path.exists(temp_jd_path):
            try:
                os.remove(temp_jd_path)
            except Exception:
                pass
        if not async_mode and temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass

@app.get("/analyze/status/{job_id}")
def get_job_status(job_id: str):
    """Poll status for async background jobs"""
    if job_id not in JOBS_STORE:
        return {"status": "error", "message": "Job ID not found."}
    return JOBS_STORE[job_id]

@app.get("/guardrail/status")
def guardrail_status():
    """Gap-2 verification: which guardrail layers are ACTIVE in this deployment."""
    return AIHiringGuardrails.guardrail_status()

class ScoreRequest(BaseModel):
    resume_text: str
    job_description: str

@app.post("/score")
def deterministic_score(req: ScoreRequest):
    """Gap-1 verification: deterministic Graph_ATS score + Shapley explanation
    for a real candidate, without invoking any LLM. Blocked documents are
    rejected by the same production guardrail as /analyze."""
    _v = AIHiringGuardrails.scan_document(req.resume_text)
    if _v["blocked"]:
        return {"status": "error", "message": "Security Guardrail: prompt injection detected in resume."}
    from tools.shap_scoring_explainer import (
        extract_job_requirements, extract_candidate_skill_tokens,
        calculate_pipeline_match_score, generate_pipeline_shap_report,
        compute_requirement_shap_values, compute_cohort_empirical_baselines)
    reqs = extract_job_requirements(req.job_description)
    if not reqs:
        return {"status": "error", "message": "No recognised technical requirements extracted from job description."}
    skills = extract_candidate_skill_tokens(_v["sanitized_text"])
    score, details = calculate_pipeline_match_score(skills, reqs)
    baselines, meta = compute_cohort_empirical_baselines(reqs)
    req_shap, total_shap, expected = compute_requirement_shap_values(details, reqs, baselines)
    return {"status": "success", "deterministic_match_score_pct": score,
            "requirements": reqs, "candidate_skills_extracted": skills,
            "match_details": {k: {"is_approved": v["is_approved"], "match_type": v["match_type"],
                                  "distance": (None if v["distance"] == float("inf") else v["distance"]),
                                  "closest_skill": v["closest_skill"]} for k, v in details.items()},
            "shapley_requirement_attributions": req_shap,
            "expected_cohort_score_pct": expected, "baseline_metadata": meta,
            "report": generate_pipeline_shap_report(skills, reqs)}

@app.post("/vault")
def add_document_to_vault(doc: VaultDocument):
    try:
        metadata = {
            "source": "Screening Dossier",
            "email": doc.email,
            "timestamp": time.time(),
            "role": doc.role
        }
        cand_id = save_to_vault(doc.content, metadata)
        return {"status": "success", "candidate_id": cand_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/vault")
async def get_vault():
    try:
        res = get_all_vault_resumes()
        if not res:
            return {"ids": [], "documents": [], "metadatas": []}
        return res
    except Exception as e:
        return {"ids": [], "documents": [], "metadatas": [], "error": str(e)}

@app.post("/chat")
async def chat_with_hr_assistant(chat_msg: ChatMessage):
    try:
        # --- OUTPUT/TOPIC GUARDRAIL ---
        AIHiringGuardrails.enforce_hr_topic(chat_msg.message)
        AIHiringGuardrails.llama_guard_check(chat_msg.message)
        # ------------------------------
        
        # pyrefly: ignore [missing-import]
        from google import genai
        from config import GOOGLE_API_KEY
        
        if not GOOGLE_API_KEY or "PASTE_YOUR_KEY_HERE" in GOOGLE_API_KEY:
            return {"response": f"Strategic HR Assistant Note: Candidate context received. Please set a valid GOOGLE_API_KEY in .env for live Gemini interactive chat responses."}

        client = genai.Client(api_key=GOOGLE_API_KEY)
        system_prompt = f"You are the internal HR Strategic Assistant. Context about the current actively screened candidate: {chat_msg.context}."
        full_prompt = f"{system_prompt}\n\nHR Recruiter Question: {chat_msg.message}"
        
        response = client.models.generate_content(
            model='gemini-flash-lite-latest',
            contents=full_prompt
        )
        return {"response": response.text}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
