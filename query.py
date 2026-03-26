from llama_index.embeddings.ollama import OllamaEmbedding
from vector_db import QdrantStorage

EMBED_MODEL_NAME = "nomic-embed-text"


def main():
    query_text = input("Enter your search query: ").strip()

    print("Loading embedding model...")
    embed_model = OllamaEmbedding(model_name=EMBED_MODEL_NAME)

    print("Embedding query...")
    query_vector = embed_model.get_query_embedding(query_text)

    db = QdrantStorage(
        url="http://localhost:6333",
        collection="ubereats_restaurants",
        dim=len(query_vector),
    )

    print("Searching Qdrant...")
    results = db.search(query_vector, top_k=5)

    print("\nTop results:\n")
    for i, point in enumerate(results, start=1):
        payload = point.payload or {}

        print(f"Result {i}")
        print(f"Name: {payload.get('name', '')}")
        print(f"Category: {payload.get('category', '')}")
        print(f"Price range: {payload.get('price_range', '')}")
        print(f"Address: {payload.get('full_address', '')}")
        print(f"Ratings: {payload.get('ratings', '')}")
        print(f"Score: {payload.get('score', '')}")

        menu_items = payload.get("menu_items", [])
        if menu_items:
            print("Menu items:")
            for item in menu_items[:5]:
                print(f"  - {item}")

        print("-" * 50)


if __name__ == "__main__":
    main()