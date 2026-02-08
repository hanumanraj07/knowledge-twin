import os
import math
import json
import requests
from typing import List, Dict, Any
from pypdf import PdfReader

# Simple Document Class
class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class SimpleRAG:
    def __init__(self):
        self.documents: List[Document] = []
        self.embeddings: List[List[float]] = []
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            print("WARNING: GEMINI_API_KEY is not set.")
        else:
            print(f"DEBUG: Loaded API Key ending in ...{self.api_key[-5:]}")

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from Gemini API."""
        if not self.api_key:
            return [0.0] * 768  # Gemini embedding-001 is 768 dims
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "models/embedding-001",
            "content": {
                "parts": [{"text": text}]
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result['embedding']['values']
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0.0] * 768

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)

    def ingest_file(self, file_path: str) -> int:
        """Ingest a file and store its chunks and embeddings."""
        text = ""
        if file_path.lower().endswith('.pdf'):
            try:
                reader = PdfReader(file_path)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            except Exception as e:
                print(f"pypdf failed: {e}")
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

        # Clean text
        text = text.replace('\x00', '')

        print(f"Extracted {len(text)} characters from {file_path}")
        if not text.strip():
            print("WARNING: No text extracted. likely image-based PDF.")
            # We return 0 chunks, but we should probably handle this better in a real app
            return 0

        # Simple splitting
        chunk_size = 1000
        overlap = 100
        chunks = []
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) < 10: continue 
            chunks.append(chunk)

        print(f"Ingesting {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks):
            embedding = self._get_embedding(chunk)
            self.documents.append(Document(page_content=chunk, metadata={"source": file_path}))
            self.embeddings.append(embedding)
            
        return len(chunks)

    def query(self, question: str, history: list = []) -> str:
        """Retrieve relevant docs and answer question with emojis and personality. 🤖✨"""
        import time
        
        # 1. GENERATOR SETUP
        if not self.api_key:
            return "GEMINI_API_KEY not found. 🔑"

        # Verified Working Models for this Key
        models_to_try = [
            "gemini-2.0-flash",
            "gemini-flash-latest",
            "gemini-2.0-flash-lite",
            "gemini-pro-latest"
        ]
        headers = {"Content-Type": "application/json"}
        
        # 2. CONTEXT RETRIEVAL
        context = ""
        if self.documents:
            q_embedding = self._get_embedding(question)
            scores = []
            for i, doc_embedding in enumerate(self.embeddings):
                score = self._cosine_similarity(q_embedding, doc_embedding)
                scores.append((score, i))
                
            scores.sort(key=lambda x: x[0], reverse=True)
            
            # Threshold (0.15) for ignoring document for greetings/unrelated chat
            best_score = scores[0][0] if scores else 0
            if best_score > 0.15:
                top_k_indices = [idx for _, idx in scores[:3]]
                context = "\n\n".join([self.documents[idx].page_content for idx in top_k_indices])

        # 3. CHAT HISTORY PROCESSING (Last 10 messages)
        history_str = ""
        if history:
            clean_history = history[-10:] # Keep it short and relevant
            for msg in clean_history:
                role = "User" if msg.get('role') == 'user' else "Twin"
                content = msg.get('content', '')
                if msg.get('type') == 'file': content = f"[Uploaded File: {content}]"
                history_str += f"{role}: {content}\n"

        # 4. PROMPT ENGINEERING (STATEFUL & WORLD-CLASS)
        history_block = f"\n**Recent Conversation History:**\n{history_str}\n" if history_str else ""
        
        if context:
            prompt = f"""You are a World-Class Knowledge Expert, similar to ChatGPT. 🧠✨
Answer this question with PRECISE DEPTH using the Context below. 🔍

Context:
{context}
{history_block}
Current Question: {question}

**Instructions:**
1. **World-Class Quality:** Provide a thorough, deep, and structured explanation. 📖
2. **Zero Filler:** Every sentence must add value. Eliminate all fluff. 🎯
3. **Context Focus:** Build your answer directly from the document provided. 📄
4. **Natural Follow-up:** End with one relevant follow-up question.
5. **Tone:** Professional, intelligent, and friendly with relevant emojis. 👋✨

Answer:"""
        else:
            prompt = f"""You are a World-Class Knowledge Expert, similar to ChatGPT. 🧠✨
{history_block}
Current Question: {question}

**Instructions:**
1. Provide a high-quality, detailed, and precise answer. 🎯
2. Eliminate all useless filler. ⏱️
3. End with a natural suggestion for the next logical question. 😊
4. Use emojis tastefully. ✨"""

        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        last_error = ""
        # 4. GENTLE PAUSE (Reduce RPM Pressure)
        time.sleep(1)
        
        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
            print(f"DEBUG: Trying model {model_name}...")
            
            try:
                response = requests.post(url, headers=headers, json=data)
                
                if response.status_code == 429:
                    print(f"WARNING: Rate limit (429) hit for {model_name}. Waiting 10s...")
                    last_error = f"Rate limit reached on {model_name}. ⏱️"
                    time.sleep(10)
                    continue
                    
                if response.status_code != 200:
                    error_text = response.text
                    print(f"WARNING: Model {model_name} failed with status {response.status_code}: {error_text}")
                    last_error = f"Status {response.status_code}: {error_text[:100]}"
                    continue

                result = response.json()
                if 'candidates' in result and result['candidates']:
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    print(f"WARNING: No candidates in response for {model_name}: {result}")
                    last_error = "Empty response from API"
            except Exception as e:
                print(f"CRITICAL: Model {model_name} search failed: {e}")
                last_error = str(e)
        
        return f"Error: All my brain modules are busy! 🤯 Please try again in a moment. (Details: {last_error})"

    def summarize_document(self) -> str:
        """Simple and direct summary."""
        if not self.documents:
            return "No document found. 📄"
            
        context = "\n\n".join([doc.page_content for doc in self.documents[:10]])
        
        prompt = f"""Summarize this document in 3-5 short bullet points. 📑
**SKIP INTRO.** Just direct highlights with emojis! ✨

{context}"""

        # Verified Working Models
        models_to_try = [
            "gemini-2.0-flash",
            "gemini-flash-latest",
            "gemini-2.0-flash-lite",
            "gemini-pro-latest"
        ]
        
        headers = {"Content-Type": "application/json"}
        last_error = ""

        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
            print(f"DEBUG: Generating summary with model {model_name}...")
            
            try:
                response = requests.post(url, headers=headers, json=data)
                
                if response.status_code == 429:
                    print(f"WARNING: Rate limit (429) hit. Waiting 10s...")
                    import time
                    time.sleep(10)
                    continue

                response.raise_for_status()
                result = response.json()
                return result['candidates'][0]['content']['parts'][0]['text']
            except Exception as e:
                error_body = response.text if 'response' in locals() and hasattr(response, 'text') else str(e)
                print(f"WARNING: Summary failed on {model_name}: {e}")
                last_error = f"{e} - {error_body[:100]}"
        
        return f"Could not generate summary. All brains busy! 💤 (Details: {last_error})"

    def clear_database(self):
        self.documents = []
        self.embeddings = []
