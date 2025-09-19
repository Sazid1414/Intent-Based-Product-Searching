# Intent-Based Product Search System

A sophisticated e-commerce search system that understands natural language queries and provides intelligent product recommendations using advanced NLP and vector similarity search.

## 🎯 Project Overview

This project implements an intelligent product search engine that goes beyond traditional keyword matching. It understands user intent, extracts meaningful information from natural language queries, and returns semantically relevant products with smart filtering capabilities.

## 🚀 Key Features

- **🧠 Intent Understanding**: Fine-tuned T5 transformer model for query interpretation
- **🔍 Semantic Search**: Vector-based similarity matching using SentenceTransformers
- **🎛️ Smart Filtering**: Automatic price, brand, and category filtering based on extracted intent
- **⚡ Fast API**: RESTful web service with sub-second response times
- **📊 Vector Database**: Efficient similarity search with ChromaDB
- **🔧 Robust Architecture**: Error handling and fallback mechanisms

## 🏗️ System Architecture

```
Natural Language Query
         ↓
Intent Parser (Fine-tuned T5)
         ↓
Structured Intent (JSON)
         ↓
Vector Embedding Generation
         ↓
ChromaDB Similarity Search
         ↓
Metadata Filtering & Ranking
         ↓
Top-10 Product Results
```

## 📁 Project Structure

```
Intent-Based-Product-Searching/
├── 📄 phase2_ingest.py          # Data preprocessing and vector embedding
├── 🚀 phase5_api.py             # FastAPI web service implementation
├── 🤖 fine_tuned_model/         # Custom T5 model for intent parsing
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── checkpoint-*/            # Training checkpoints
├── 🗄️ chroma_db/               # Vector database storage
│   └── chroma.sqlite3
├── 📊 cleaned_products.csv      # Product dataset (992 items)
└── 📖 README.md                 # Project documentation
```

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Web Framework** | FastAPI | REST API development |
| **Vector Database** | ChromaDB | Similarity search storage |
| **Embeddings** | SentenceTransformers | Text-to-vector conversion |
| **Intent Parsing** | Transformers (T5) | Natural language understanding |
| **Fine-tuning** | PEFT (LoRA) | Parameter-efficient training |
| **Data Processing** | Pandas | Dataset manipulation |

## 📈 What This System Does

### 🔄 Phase 1: Data Ingestion (`phase2_ingest.py`)
- Loads product catalog from CSV (992 products)
- Generates 384-dimensional embeddings for product descriptions
- Stores structured data with metadata in ChromaDB
- Creates persistent vector index for fast retrieval

### 🧠 Phase 2: Intent Understanding
- **Input**: Natural language queries like "affordable laptops under $800"
- **Processing**: Fine-tuned T5 model extracts structured intent
- **Output**: JSON with keywords, price limits, brand preferences, categories

### 🔍 Phase 3: Smart Search (`phase5_api.py`)
- Converts intent keywords to vector embeddings
- Performs semantic similarity search in vector space
- Applies metadata filters (price, brand, category)
- Returns ranked, relevant product recommendations

## 🚀 Quick Start Guide

### Prerequisites
```bash
pip install fastapi uvicorn chromadb sentence-transformers transformers peft pandas
```

### Running the System

1. **Start the API Server**:
```bash
python phase5_api.py
```

2. **Verify System Health**:
```bash
curl http://localhost:8000/health
```

3. **Check Database**:
```bash
curl http://localhost:8000/test-db
```

4. **Test Search**:
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "wireless bluetooth headphones under 100"}'
```

### 📚 API Documentation
Access interactive docs at: `http://localhost:8000/docs`

## 🔍 Supported Query Types

The system intelligently handles various query patterns:

| Query Type | Example | Extracted Intent |
|------------|---------|------------------|
| **Price-constrained** | "laptops under $500" | `{"keywords": "laptops", "max_price": 500}` |
| **Brand-specific** | "Samsung smartphones" | `{"keywords": "smartphones", "brand": "Samsung"}` |
| **Category-focused** | "gaming headphones" | `{"keywords": "gaming headphones", "category_contains": "gaming"}` |
| **Complex intent** | "affordable iPhone under 800" | `{"keywords": "iPhone", "brand": "Apple", "max_price": 800}` |
| **Natural language** | "I need a good camera for photography" | `{"keywords": "camera photography"}` |

## 📊 API Response Format

```json
{
  "results": [
    {
      "title": "Sony WH-CH720N Wireless Noise Canceling Headphones",
      "description": "Industry-leading noise canceling with Dual Noise Sensor technology",
      "category": "Electronics > Audio > Headphones",
      "price": 149.99,
      "brand": "Sony",
      "asin": "B09ZMCJ1Q7"
    }
  ],
  "message": "10 matches found",
  "parsed_intent": {
    "keywords": "wireless bluetooth headphones",
    "max_price": 100
  }
}
```

## 🎯 Core Capabilities

### 🔍 Semantic Understanding
- **Vector Similarity**: 384-dimensional embeddings capture semantic meaning
- **Context Awareness**: Understands synonyms and related concepts
- **Relevance Scoring**: Cosine similarity for accurate matching

### 🎛️ Intelligent Filtering
- **Price Constraints**: Automatic budget filtering
- **Brand Recognition**: Identifies and filters by brand preferences
- **Category Matching**: Contextual category classification
- **Multi-criteria**: Combines multiple filters seamlessly

### ⚡ Performance Metrics
- **Search Speed**: < 100 Millisecond
- **Accuracy**: High relevance through semantic + metadata matching
- **Scalability**: Efficient vector operations with ChromaDB



## 🎯 Business Applications

- **🛒 E-commerce**: Enhanced product discovery and conversion
- **🤖 Chatbots**: Natural language product assistance
- **📱 Voice Commerce**: "Hey Google, find me..." queries
- **📊 Analytics**: Understanding customer search patterns
- **🎁 Recommendations**: Intent-aware product suggestions

## 📊 Dataset Information

- **Total Products**: 992 items across multiple categories
- **Data Fields**: title, description, category, price, brand, ASIN
- **Categories**: Electronics, Home & Garden, Sports, Fashion, etc.
- **Price Range**: Covers various budget segments

## 🚀 Future Roadmap

- [ ] **Multi-modal Search**: Add image and voice query support
- [ ] **Personalization**: User preference learning and history
- [ ] **Real-time Updates**: Live product catalog synchronization
- [ ] **Advanced Analytics**: Search pattern insights and reporting
- [ ] **A/B Testing**: Intent parsing model comparison
- [ ] **Caching Layer**: Redis for frequent query optimization

## 🏆 Technical Achievements

This project demonstrates:
- **Advanced NLP**: Fine-tuned transformer models for domain-specific tasks
- **Vector Search**: Efficient similarity search at scale
- **Intent Recognition**: Structured information extraction from natural language
- **Production-Ready API**: Robust web service with proper error handling
- **Modern ML Stack**: Integration of cutting-edge AI/ML technologies

