import os
import chromadb
import hashlib
import numpy as np
from typing import List
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from config import GOOGLE_API_KEY

_EMBEDDING_CACHE = {}

def _deterministic_fallback_embedding(text: str, dim: int = 768) -> List[float]:
    """Fallback deterministic unit embedding when API keys or network are unavailable."""
    seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()

class GoogleGenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key=None, model_name="models/text-embedding-004"):
        self.api_key = api_key or GOOGLE_API_KEY
        self.model_name = model_name
        self.client = None
        if self.api_key and not any(x in str(self.api_key) for x in ["your_gemini_api_key_here", "PASTE_YOUR_KEY_HERE"]):
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[RAG WARNING] Failed to initialize Google GenAI Client: {e}")
                self.client = None

    def __call__(self, input: Documents) -> Embeddings:
        results = []
        missing_texts = []
        missing_indices = []

        for idx, text in enumerate(input):
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            if text_hash in _EMBEDDING_CACHE:
                results.append((idx, _EMBEDDING_CACHE[text_hash]))
            else:
                missing_texts.append(text)
                missing_indices.append((idx, text_hash))

        if missing_texts:
            if self.client:
                try:
                    response = self.client.models.embed_content(
                        model=self.model_name,
                        contents=missing_texts
                    )
                    for (orig_idx, t_hash), emb in zip(missing_indices, response.embeddings):
                        _EMBEDDING_CACHE[t_hash] = emb.values
                        results.append((orig_idx, emb.values))
                except Exception as e:
                    # Try fallback model or fallback embedding
                    try:
                        response = self.client.models.embed_content(
                            model="models/gemini-embedding-001",
                            contents=missing_texts
                        )
                        for (orig_idx, t_hash), emb in zip(missing_indices, response.embeddings):
                            _EMBEDDING_CACHE[t_hash] = emb.values
                            results.append((orig_idx, emb.values))
                    except Exception as inner_e:
                        print(f"[RAG EMBEDDING FALLBACK] Remote embedding API failed ({inner_e}). Using local deterministic vectors.")
                        for orig_idx, t_hash in missing_indices:
                            emb = _deterministic_fallback_embedding(input[orig_idx])
                            _EMBEDDING_CACHE[t_hash] = emb
                            results.append((orig_idx, emb))
            else:
                for orig_idx, t_hash in missing_indices:
                    emb = _deterministic_fallback_embedding(input[orig_idx])
                    _EMBEDDING_CACHE[t_hash] = emb
                    results.append((orig_idx, emb))

        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]


class ResumeVault:
    def __init__(self, persist_directory="./db/resume_vault"):
        os.makedirs(persist_directory, exist_ok=True)

        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)

        self.embedding_function = GoogleGenAIEmbeddingFunction(
            api_key=GOOGLE_API_KEY
        )

        self.collection = self.client.get_or_create_collection(
            name="resumes",
            embedding_function=self.embedding_function
        )

    def add_resume(self, resume_text, metadata):
        """Add resume and store embedding with metadata."""
        resume_id = metadata.get("id", os.urandom(4).hex())
        # Truncate text to avoid token limit exceptions
        safe_resume_text = resume_text[:8000] if resume_text else "No content"

        self.collection.add(
            documents=[safe_resume_text],
            metadatas=[metadata],
            ids=[resume_id]
        )
        return resume_id

    def search_similar_resumes(self, query_text, n_results=3):
        try:
            count = self.collection.count()
            if count == 0:
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            
            actual_n = min(n_results, count)
            results = self.collection.query(
                query_texts=[query_text],
                n_results=actual_n
            )
            return results
        except Exception as e:
            print(f"[RAG SEARCH WARNING] {e}")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def get_all_resumes(self):
        try:
            return self.collection.get()
        except Exception as e:
            print(f"[RAG GET ALL WARNING] {e}")
            return {"ids": [], "documents": [], "metadatas": []}

    def clear_vault(self):
        try:
            self.client.delete_collection(name="resumes")
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name="resumes",
            embedding_function=self.embedding_function
        )
        return True


# Global instance
vault = ResumeVault()


from crewai.tools import tool

@tool("search_resume_vault")
def search_resume_vault(query: str) -> str:
    """Useful to search the internal database (Resume Vault) for existing resumes and candidate profiles
    matching a job description or set of skills. Returns top matching candidates and their details."""
    try:
        results = vault.search_similar_resumes(query, n_results=3)
        if not results or not results.get('documents') or not results['documents'][0]:
            return "No candidates found in the Resume Vault matching this query."
        
        output = "--- TOP MATCHING CANDIDATES FROM RESUME VAULT ---\n"
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            meta = results['metadatas'][0][i] if results.get('metadatas') else {}
            cand_id = results['ids'][0][i] if results.get('ids') else f"cand_{i+1}"
            
            output += f"Candidate ID: {cand_id}\n"
            output += f"Name/Email: {meta.get('email', 'Unknown')}\n"
            output += f"Resume Content:\n{doc}\n"
            output += "-" * 40 + "\n"
        return output
    except Exception as e:
        return f"Error searching the Resume Vault: {str(e)}"