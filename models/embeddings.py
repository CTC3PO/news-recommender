"""
Generate embeddings for news articles using Sentence Transformers
Build FAISS index for fast similarity search
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import pickle
from tqdm import tqdm
import time


class EmbeddingEngine:
    """
    Handles embedding generation and similarity search
    Uses sentence-transformers for semantic embeddings
    Uses FAISS for efficient similarity search
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize embedding model

        Args:
            model_name: HuggingFace model name
                - all-MiniLM-L6-v2: Fast, 384 dimensions (RECOMMENDED)
                - all-mpnet-base-v2: Better quality, 768 dimensions (slower)
        """
        print(f"\n{'='*60}")
        print(f"Loading Embedding Model: {model_name}")
        print(f"{'='*60}")

        start = time.time()
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.news_ids = []
        self.embeddings = None

        print(f"- Model loaded in {time.time() - start:.2f}s")
        print(f"  Embedding dimension: {self.dimension}")

    def create_embeddings(self, news_df, text_column='text', batch_size=32):
        """
        Generate embeddings for all articles

        Args:
            news_df: DataFrame with articles
            text_column: Column containing text to embed
            batch_size: Batch size for encoding
        """
        print(f"\n{'='*60}")
        print(f"Generating Embeddings")
        print(f"{'='*60}")

        texts = news_df[text_column].tolist()
        self.news_ids = news_df['news_id'].tolist()

        print(f"Encoding {len(texts):,} articles...")
        print(f"Batch size: {batch_size}")

        start = time.time()

        # Generate embeddings with progress bar
        self.embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )

        elapsed = time.time() - start
        print(f"\n- Embeddings created in {elapsed:.2f}s")
        print(f"  Speed: {len(texts) / elapsed:.1f} articles/second")
        print(f"  Shape: {self.embeddings.shape}")

        return self.embeddings

    def build_index(self, embeddings=None):
        """
        Build FAISS index for fast similarity search

        Args:
            embeddings: Optional embeddings array (uses self.embeddings if None)
        """
        if embeddings is None:
            embeddings = self.embeddings

        if embeddings is None:
            raise ValueError(
                "No embeddings available. Run create_embeddings first.")

        print(f"\n{'='*60}")
        print(f"Building FAISS Index")
        print(f"{'='*60}")

        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        # This is fast enough for <1M vectors and gives exact results
        self.index = faiss.IndexFlatIP(self.dimension)

        # Add vectors to index
        self.index.add(embeddings.astype('float32'))

        print(f"- Index built successfully")
        print(f"  Total vectors: {self.index.ntotal:,}")
        print(f"  Index type: Flat (exact search)")

    def search(self, query, k=10):
        """
        Search for similar articles

        Args:
            query: Query text (str) or embedding (np.array)
            k: Number of results to return

        Returns:
            List of dicts with news_id and similarity score
        """
        if self.index is None:
            raise ValueError("Index not built. Run build_index first.")

        # Handle text query
        if isinstance(query, str):
            query_embedding = self.model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        else:
            query_embedding = query
            # Normalize if not already
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm

        # Reshape for FAISS
        query_embedding = query_embedding.reshape(1, -1).astype('float32')

        # Search
        scores, indices = self.index.search(query_embedding, k)

        # Format results
        results = [
            {
                'news_id': self.news_ids[idx],
                'score': float(score)
            }
            for idx, score in zip(indices[0], scores[0])
        ]

        return results

    def get_embedding(self, news_id):
        """Get embedding for a specific article"""
        try:
            idx = self.news_ids.index(news_id)
            return self.embeddings[idx]
        except ValueError:
            return None

    def save(self, output_dir="data/processed"):
        """Save embeddings and index to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Saving Embeddings & Index")
        print(f"{'='*60}")

        # Save FAISS index
        index_path = output_path / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        print(f"- FAISS index: {index_path}")
        print(f"  Size: {index_path.stat().st_size / 1024 / 1024:.2f} MB")

        # Save news IDs mapping
        ids_path = output_path / "news_ids.pkl"
        with open(ids_path, 'wb') as f:
            pickle.dump(self.news_ids, f)
        print(f"- News IDs: {ids_path}")

        # Save raw embeddings (optional, for later use)
        emb_path = output_path / "embeddings.npy"
        np.save(emb_path, self.embeddings)
        print(f"- Embeddings: {emb_path}")
        print(f"  Size: {emb_path.stat().st_size / 1024 / 1024:.2f} MB")

    def load(self, output_dir="data/processed"):
        """Load saved index and embeddings"""
        output_path = Path(output_dir)

        print(f"\n{'='*60}")
        print(f"Loading Saved Embeddings")
        print(f"{'='*60}")

        # Load FAISS index
        index_path = output_path / "faiss_index.bin"
        self.index = faiss.read_index(str(index_path))
        print(f"- Loaded FAISS index: {self.index.ntotal:,} vectors")

        # Load news IDs
        ids_path = output_path / "news_ids.pkl"
        with open(ids_path, 'rb') as f:
            self.news_ids = pickle.load(f)
        print(f"- Loaded {len(self.news_ids):,} news IDs")

        # Load embeddings
        emb_path = output_path / "embeddings.npy"
        if emb_path.exists():
            self.embeddings = np.load(emb_path)
            print(f"- Loaded embeddings: {self.embeddings.shape}")


def test_search(engine, news_df, queries):
    """Test search with sample queries"""
    print(f"\n{'='*60}")
    print(f"TESTING SEARCH")
    print(f"{'='*60}")

    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 60)

        results = engine.search(query, k=5)

        for i, result in enumerate(results, 1):
            article = news_df[news_df['news_id'] == result['news_id']].iloc[0]
            print(f"\n{i}. [Score: {result['score']:.4f}]")
            print(f"   Category: {article['category']}")
            print(f"   Title: {article['title'][:80]}")


def main():
    """Main embedding generation pipeline"""
    print("="*60)
    print("EMBEDDING GENERATION PIPELINE")
    print("="*60)

    # Load processed news data
    print("\n- Loading processed data...")
    news_df = pd.read_parquet("data/processed/news_features.parquet")
    print(f"- Loaded {len(news_df):,} articles")

    # Initialize embedding engine
    engine = EmbeddingEngine(model_name='all-MiniLM-L6-v2')

    # Create embeddings
    embeddings = engine.create_embeddings(news_df, batch_size=64)

    # Build search index
    engine.build_index(embeddings)

    # Save everything
    engine.save()

    # Test with sample queries
    test_queries = [
        "artificial intelligence and machine learning",
        "sports and football news",
        "stock market and economy"
    ]
    test_search(engine, news_df, test_queries)

    print("\n* Embedding pipeline complete!")
    print("\nNext step: Train ranking model")
    print("  → python models/train_ranker.py")


if __name__ == "__main__":
    main()
