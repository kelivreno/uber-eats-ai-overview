import re
from typing import List, Dict, Any, Optional

import streamlit as st
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from vector_db import QdrantStorage

EMBED_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME = "llama3.2"
COLLECTION_NAME = "ubereats_restaurants"
TOP_K = 8
SUGGESTED_QUESTIONS = [
    "Do they sell milkshakes?",
    "Are they open late?",
    "Is the food halal?",
    "Do they have mac and cheese?",
    "What are some popular items?",
    "Can I order online?",
    "Is it good for groups?",
]


@st.cache_resource
def get_embed_model() -> OllamaEmbedding:
    return OllamaEmbedding(model_name=EMBED_MODEL_NAME)


@st.cache_resource
def get_llm() -> Ollama:
    return Ollama(model=LLM_MODEL_NAME, request_timeout=120.0)


@st.cache_resource
def get_db() -> QdrantStorage:
    return QdrantStorage(
        url="http://localhost:6333",
        collection=COLLECTION_NAME,
        dim=768,
    )



def extract_city_state(address: str) -> tuple[str, str]:
    if not address:
        return "", ""
    parts = [p.strip() for p in address.split(",")]
    city = parts[-3] if len(parts) >= 3 else ""
    state_zip = parts[-2] if len(parts) >= 2 else ""
    state_match = re.search(r"\b([A-Z]{2})\b", state_zip)
    state = state_match.group(1) if state_match else ""
    return city, state



def looks_like_location_query(query: str) -> bool:
    q = query.lower()
    location_words = [
        " in ", " near ", " around ", " los angeles", " la", " chicago", " seattle",
        " new york", " san francisco", " houston", " dallas", " miami", " ca", " tx", " wa", " ny"
    ]
    return any(word in q for word in location_words)



def search_restaurants(query_text: str, top_k: int = TOP_K) -> List[Any]:
    embed_model = get_embed_model()
    db = get_db()
    query_vector = embed_model.get_query_embedding(query_text)
    results = db.search(query_vector, top_k=top_k)
    return results



def result_label(point: Any) -> str:
    payload = point.payload or {}
    name = payload.get("name", "Unknown")
    category = payload.get("category", "")
    price = payload.get("price_range", "")
    address = payload.get("full_address", "")
    return f"{name} • {category} • {price} • {address}"



def menu_lookup_answer(question: str, menu_items: List[str]) -> Optional[str]:
    q = question.lower().strip()
    joined = " ".join(menu_items).lower()

    checks = {
        "milkshake": ["milkshake", "shake"],
        "mac and cheese": ["mac and cheese", "mac & cheese", "mac n cheese"],
        "ramen": ["ramen"],
        "boba": ["boba", "bubble tea"],
        "halal": ["halal"],
    }

    for label, keywords in checks.items():
        if label in q:
            found = [item for item in menu_items if any(k in item.lower() for k in keywords)]
            if found:
                return f"Yes, the menu appears to include {label}. Examples: " + "; ".join(found[:3])
            return f"I do not see {label} clearly listed in the menu items I have for this restaurant."

    if "popular" in q:
        if menu_items:
            return "Some menu items shown in the data are: " + "; ".join(menu_items[:5])

    return None



def build_restaurant_context(payload: Dict[str, Any]) -> str:
    menu_items = payload.get("menu_items", []) or []
    reviews = payload.get("reviews", []) or []

    menu_block = "\n".join(f"- {item}" for item in menu_items[:15]) if menu_items else "- No menu items available"
    review_block = "\n".join(f"- {review}" for review in reviews[:12]) if reviews else "- No reviews available"

    return f"""
Restaurant: {payload.get('name', '')}
Category: {payload.get('category', '')}
Price range: {payload.get('price_range', '')}
Address: {payload.get('full_address', '')}
Ratings count: {payload.get('ratings', '')}
Score: {payload.get('score', '')}

Menu items:
{menu_block}

Reviews:
{review_block}
""".strip()



def answer_restaurant_question(payload: Dict[str, Any], question: str) -> str:
    menu_items = payload.get("menu_items", []) or []

    direct_answer = menu_lookup_answer(question, menu_items)
    if direct_answer:
        return direct_answer

    context = build_restaurant_context(payload)
    prompt = f"""
You are answering questions about a single restaurant.
Use only the provided restaurant information.
If the information is not clearly present, say you are not sure.
Keep the answer short and practical.

Restaurant information:
{context}

User question:
{question}
""".strip()

    llm = get_llm()
    response = llm.complete(prompt)
    return str(response)



def render_restaurant_card(payload: Dict[str, Any]) -> None:
    st.subheader(payload.get("name", "Unknown restaurant"))
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(payload.get("category", ""))
        st.caption(payload.get("full_address", ""))
    with col2:
        st.write(f"Price: {payload.get('price_range', '') or 'N/A'}")
        st.write(f"Score: {payload.get('score', '') or 'N/A'}")

    menu_items = payload.get("menu_items", []) or []
    if menu_items:
        st.markdown("**Menu preview**")
        for item in menu_items[:5]:
            st.write(f"- {item}")



def main() -> None:
    st.set_page_config(page_title="UberEats Restaurant Q&A", page_icon="🍽️", layout="wide")
    st.title("UberEats Restaurant Q&A")
    st.caption("Search restaurants, pick one, then ask questions about that place.")

    if "selected_payload" not in st.session_state:
        st.session_state.selected_payload = None
    if "selected_question" not in st.session_state:
        st.session_state.selected_question = ""

    search_query = st.text_input("Search restaurants", placeholder="cheap sushi in LA")

    if search_query:
        with st.spinner("Searching restaurants..."):
            results = search_restaurants(search_query, top_k=TOP_K)

        if not results:
            st.warning("No restaurants found.")
        else:
            options = {result_label(point): point.payload for point in results}
            selected_label = st.selectbox("Pick a restaurant", list(options.keys()))
            st.session_state.selected_payload = options[selected_label]

    payload = st.session_state.selected_payload

    if payload:
        st.divider()
        render_restaurant_card(payload)
        st.divider()

        st.markdown("### Ask a question")
        chip_cols = st.columns(2)
        for i, question in enumerate(SUGGESTED_QUESTIONS):
            with chip_cols[i % 2]:
                if st.button(question, use_container_width=True):
                    st.session_state.selected_question = question

        user_question = st.text_input(
            "Ask a question about this place",
            value=st.session_state.selected_question,
            placeholder="Do they have mac and cheese?",
        )

        if st.button("Get answer", type="primary") and user_question.strip():
            with st.spinner("Thinking..."):
                answer = answer_restaurant_question(payload, user_question.strip())
            st.markdown("### Answer")
            st.write(answer)

            menu_items = payload.get("menu_items", []) or []
            if menu_items:
                st.markdown("### Supporting menu items")
                for item in menu_items[:8]:
                    st.write(f"- {item}")

            reviews = payload.get("reviews", []) or []
            if reviews:
                st.markdown("### Supporting review snippets")
                for review in reviews[:5]:
                    st.write(f"- {review}")
        elif user_question.strip():
            st.info("Press 'Get answer' to ask about this restaurant.")
    else:
        st.info("Search for a restaurant first, then pick one to open the Q&A view.")


if __name__ == "__main__":
    main()
