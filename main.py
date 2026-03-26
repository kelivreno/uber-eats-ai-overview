import logging

from fastapi import FastAPI
from dotenv import load_dotenv

import inngest
import inngest.fast_api

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


load_dotenv()

# LlamaIndex global settings
Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Inngest client
inngest_client = inngest.Inngest(
    app_id="UE_AI_Overview",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
)

# Inngest function
@inngest_client.create_function(
    fn_id="rag_ingest_csv",
    trigger=inngest.TriggerEvent(event="rag/ingest.csv"),
)
async def rag_ingest_csv(ctx: inngest.Context):
    ctx.logger.info("Received event: %s", ctx.event)
    return {"hello": "world"}


# FastAPI app
app = FastAPI()

# Serve Inngest endpoint
inngest.fast_api.serve(app, inngest_client, [rag_ingest_csv])