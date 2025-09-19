import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
import traceback

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

# Load components
embedder = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="products")

# Load fine-tuned model
model_name = "google/flan-t5-base"
try:
    tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, "./fine_tuned_model")
    print("Loaded model from fine_tuned_model directory")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

def parse_intent(query: str) -> dict:
    try:
        input_text = f"Parse this search intent to JSON: {query}"
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=128, do_sample=False)
        parsed_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Model output: '{parsed_text}'")
        
        # If empty or invalid, fallback to simple parsing
        if not parsed_text.strip():
            print("Empty model output, using fallback")
            return {"keywords": query}
            
        # Try to parse JSON
        try:
            return json.loads(parsed_text)
        except json.JSONDecodeError:
            print(f"Invalid JSON from model: '{parsed_text}', using fallback")
            return {"keywords": query}
            
    except Exception as e:
        print(f"Error parsing intent: {e}")
        return {"keywords": query}

@app.get("/test-db")
def test_db():
    try:
        count = collection.count()
        return {"collection_count": count, "status": "ok"}
    except Exception as e:
        return {"error": str(e), "status": "error"}

@app.post("/search")
def search(request: QueryRequest):
    try:
        query = request.query.strip().lower()
        print(f"Processing search query: {query}")
        
        # Parse intent with improved fallback
        parsed = parse_intent(query)
        print(f"Parsed intent: {parsed}")
        
        # Generate embedding
        keywords = parsed.get('keywords', query)
        query_embedding = embedder.encode(keywords).tolist()
        
        # Build metadata filter
        metadata_filter = {"$and": []}
        
        if 'brand' in parsed and parsed['brand']:
            metadata_filter["$and"].append({"brand": {"$eq": parsed['brand']}})
            
        if 'max_price' in parsed and parsed['max_price']:
            metadata_filter["$and"].append({"price": {"$lte": parsed['max_price']}})
            
        if 'category_contains' in parsed and parsed['category_contains']:
            metadata_filter["$and"].append({"category": {"$contains": parsed['category_contains']}})
        
        # Query with or without filters
        where_clause = metadata_filter if metadata_filter["$and"] else None
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=50,
            where=where_clause
        )
        
        print(f"Query returned {len(results['metadatas'][0])} results")
        
        # Format results with safe field access
        matches = []
        for i in range(min(10, len(results['metadatas'][0]))):
            meta = results['metadatas'][0][i]
            matches.append({
                'title': meta.get('title', 'N/A'),
                'description': meta.get('description', 'N/A'),
                'category': meta.get('category', 'N/A'),
                'price': meta.get('price', None),
                'brand': meta.get('brand', 'N/A'),
                'asin': meta.get('asin', 'N/A')
            })
        
        response = {
            "results": matches, 
            "message": f"{len(matches)} matches found",
            "parsed_intent": parsed
        }
        
        return response
    
    except Exception as e:
        print(f"Error in search: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
def ingest():
    try:
        # Run the ingestion process
        exec(open('phase2_ingest.py').read())
        return {"message": "Data ingested successfully"}
    except Exception as e:
        print(f"Error in ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)