#read restaurants
import pandas as pd
from llama_index.embeddings.ollama import OllamaEmbedding

from vector_db import QdrantStorage

RESTAURANTS_PATH = "data/restaurants.csv"
MENUS_PATH = "data/restaurant-menus.csv"

CHUNK_SIZE = 200
EMBED_MODEL_NAME = "nomic-embed-text"
MAX_MENU_ITEMS = 10

def load_menu_map(menu_path, max_items_per_restaurant=10):
    df = pd.read_csv(menu_path)
    df = df.fillna("")

    menu_map = {}

    for _, row in df.iterrows():
        restaurant_id = row["restaurant_id"]

        menu_text = f"{row['name']} | {row['category']} | {row['description']} | {row['price']}"

        if restaurant_id not in menu_map:
            menu_map[restaurant_id] = []

        if menu_text not in menu_map[restaurant_id] and len(menu_map[restaurant_id]) < max_items_per_restaurant:
            menu_map[restaurant_id].append(menu_text)

    return menu_map


def row_to_text(row,menu_map):
    restaurant_id = row.get("id", None)
    menu_items = menu_map.get(restaurant_id, [])

    return f"""
    Restaurant: {row.get('name', '')}
    Category: {row.get('category', '')}
    Price range: {row.get('price_range', '')}
    Address: {row.get('full_address', '')}
    Zipcode: {row.get('zip_code', '')}
    Ratings count: {row.get('ratings', '')}
    Score: {row.get('score', '')}
    Latitude: {row.get('lat', '')}
    Longitude: {row.get('lng', '')}
    Menu items: {';'.join(menu_items)}
""".strip()

def main():
    print("Loading menu data...")
    menu_map = load_menu_map(MENUS_PATH, max_items_per_restaurant=MAX_MENU_ITEMS)
    print(f"Loaded menus for {len(menu_map)} restaurants")

    print("Loading embedding model...")
    embed_model = OllamaEmbedding(model_name=EMBED_MODEL_NAME)

    sample_vector = embed_model.get_text_embedding("test")
    vector_size = len(sample_vector)
    print(f"Vector size: {vector_size}")

    db = QdrantStorage(
        url="http://localhost:6333",
        collection="ubereats_restaurants",
        dim=vector_size,
    )

    print("Starting ingestion...")
    point_id = 0

    for chunk_index, df in enumerate(pd.read_csv(RESTAURANTS_PATH, chunksize=CHUNK_SIZE)):
        df = df.fillna("")

        ids = []
        vectors = []
        payloads = []

        for _, row in df.iterrows():
            text = row_to_text(row, menu_map)
            vector = embed_model.get_text_embedding(text)

            payload = {
                "id": row.get("id", ""),
                "name": row.get("name", ""),
                "category": row.get("category", ""),
                "price_range": row.get("price_range", ""),
                "full_address": row.get("full_address", ""),
                "zip_code": row.get("zip_code", ""),
                "ratings": row.get("ratings", ""),
                "score": row.get("score", ""),
                "lat": row.get("lat", ""),
                "lng": row.get("lng", ""),
                "menu_items": menu_map.get(row.get("id", None), []),
                "text": text,
            }

            ids.append(point_id)
            vectors.append(vector)
            payloads.append(payload)
            point_id += 1

        db.upsert(ids=ids, vectors=vectors, payloads=payloads)
        print(f"Chunk {chunk_index + 1} done. Total points: {point_id}")
    print("Ingestion complete")
if __name__ == "__main__":
    main()





