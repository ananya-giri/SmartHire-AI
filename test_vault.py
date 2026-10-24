import os
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

from tools.rag_processor import vault

try:
    print("Attempting to save to vault...")
    cand_id = vault.add_resume("Candidate has 5 years React, TypeScript, and AWS experience.", {"role": "Frontend Engineer", "source": "test_resume.pdf", "ai_score": 88, "timestamp": 1234.5, "email": "engineer@example.com"})
    print(f"Save completed with ID: {cand_id}")
    all_res = vault.get_all_resumes()
    count = len(all_res.get('documents', [])) if all_res else 0
    print("Current Vault contents Count:", count)
    
    search_res = vault.search_similar_resumes("React developer")
    print("Search returned results successfully!")
except Exception as e:
    import traceback
    traceback.print_exc()
