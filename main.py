import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
from inngest.experimental import ai
import uuid
import os
import datetime

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="UE_AI_Overview",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest CSV",
    trigger=inngest.TriggerEvent(event="rag/ingest.csv")


)
async def rag_ingest_csv(ctx: inngest.Context):
    return {"hello":"world"}


app = FastAPI()

inngest.fast_api.serve(app,inngest_client, [rag_ingest_csv])
