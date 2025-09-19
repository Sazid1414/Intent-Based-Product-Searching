import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load cleaned (remove /content/ prefix)
df = pd.read_csv('cleaned_products.csv')

# Embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Embed descriptions
descriptions = df['description'].astype(str).tolist()
embeddings = embedder.encode(descriptions, batch_size=32, show_progress_bar=True)

# Chroma setup
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="products")

# Upsert
for i in range(len(df)):
    row = df.iloc[i]
    metadata = {
        'title': str(row['title']),
        'description': str(row['description']),
        'category': str(row['category']),
        'price': float(row['price']) if pd.notna(row['price']) else None,
        'brand': str(row['brand']),
        'asin': str(row['asin'])
    }
    collection.upsert(
        ids=[str(row['asin'])],
        embeddings=[embeddings[i].tolist()],
        metadatas=[metadata]
    )

print(f"Ingested {collection.count()} products")