import base64
import io
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from research_rag import (
    Embedder, Reranker, VectorStore,
    ingest_pdf, search_and_summarize, deep_scan_papers, synthesize_answer,
)
from research_rag.config import (
    APP_HOST, APP_PORT, OLLAMA_MODEL, TOP_K, WEAVIATE_SCRATCH_COLLECTION,
)
from research_rag.config import PAPERS_DIR as PAPERS_DIR_SETTING
from research_rag.pdf_viewer import render_chunk as _render_chunk
from research_rag.synthesize import MAX_SOURCES
from research_rag.vector_store import WeaviateUnavailableError

PAPERS_DIR = Path(PAPERS_DIR_SETTING)
STATIC_DIR = Path("./static")
SCRATCH_DIR = Path("./scratch")
for _d in (PAPERS_DIR, STATIC_DIR, SCRATCH_DIR):
    _d.mkdir(parents=True, exist_ok=True)

print("Loading embedding model...")
embedder = Embedder()
print("Loading reranker model...")
reranker_model = Reranker()
print("Connecting to Weaviate...")
try:
    store = VectorStore()
except WeaviateUnavailableError as exc:
    print(f"\n{exc}\n", file=sys.stderr)
    raise SystemExit(1)
# Ad-hoc uploads go to their own collection, so nothing dropped in to ask a
# single question can surface in a corpus-wide answer, and cleanup is a delete.
scratch_store = VectorStore(collection=WEAVIATE_SCRATCH_COLLECTION)
print("All services ready.\n")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    # Weaviate's client holds an HTTP session and a gRPC channel; close them
    # so shutdown is clean rather than leaving warnings on exit.
    store.close()
    scratch_store.close()


app = FastAPI(title="Research RAG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/api/papers")
def list_papers():
    return {"papers": store.list_sources()}


@app.post("/api/papers")
def upload_papers(files: List[UploadFile] = File(...)):
    results = []
    for f in files:
        name = Path(f.filename or "unnamed.pdf").name
        dest = PAPERS_DIR / name
        with open(dest, "wb") as fh:
            fh.write(f.file.read())
        try:
            n = ingest_pdf(str(dest), embedder, store)
            results.append({
                "name": name,
                "status": "ok",
                "message": f"{n} chunks stored" if n else "already ingested",
            })
        except Exception as e:
            results.append({"name": name, "status": "error", "message": str(e)})
    return {"results": results, "papers": store.list_sources()}


@app.delete("/api/papers/{filename:path}")
def remove_paper(filename: str):
    store.delete_source(filename)
    path = PAPERS_DIR / filename
    if path.exists():
        path.unlink()
    return {"removed": filename, "papers": store.list_sources()}


class Turn(BaseModel):
    role: str            # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    query: str
    deep_scan: bool = False
    use_reranker: bool = True
    top_k: int = TOP_K
    source_filter: str | None = None


class AskRequest(BaseModel):
    """
    One grounded answer with citations, across the corpus or within one paper.

    `source_filter` set is the single-paper conversation; unset searches
    everything. Same endpoint either way - the only difference is scope.
    """

    query: str
    source_filter: str | None = None
    history: List[Turn] = []
    use_reranker: bool = True
    max_sources: int = MAX_SOURCES
    scratch: bool = False       # ask about an ad-hoc upload instead of the corpus


@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        if req.deep_scan:
            result = deep_scan_papers(
                query=req.query, store=store, model=OLLAMA_MODEL,
                batch_size=3, forced_sections=None,
                source_filter=req.source_filter or None,
            )
            result["deep_scan"] = True
        else:
            result = search_and_summarize(
                query=req.query, embedder=embedder, store=store, top_k=req.top_k,
                model=OLLAMA_MODEL,
                reranker=reranker_model if req.use_reranker else None,
                forced_sections=None,
                source_filter=req.source_filter or None,
            )
            result["deep_scan"] = False

        if result.get("papers_found", 0) == 0:
            return {"ok": False, "error": result.get("error", "No results found.")}

        papers_out = []
        for paper in result.get("results", []):
            papers_out.append({
                "source": paper["source"],
                "summary": paper["summary"],
                "chunks_used": paper["chunks_used"],
                "sections_used": paper.get("sections_used", []),
                "map_summaries": paper.get("map_summaries", []),
                "reranked": paper.get("reranked", False),
                "chunks": [
                    {
                        "text": c["text"],
                        "heading": c.get("heading", ""),
                        "section_name": c.get("section_name", ""),
                        "section_type": c.get("section_type", ""),
                        "page_numbers": c.get("page_numbers", ""),
                        "source_path": c.get("source_path", ""),
                        "distance": c.get("distance"),
                    }
                    for c in paper.get("chunks", [])
                ],
            })

        return {
            "ok": True,
            "mode": "deep_scan" if result["deep_scan"] else "quick_search",
            "target_sections": result.get("target_sections", []),
            "section_filtered": result.get("section_filtered", False),
            "papers_found": result["papers_found"],
            "papers": papers_out,
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.post("/api/ask")
def ask(req: AskRequest):
    """
    One synthesised, cited answer.

    Serves both the corpus-wide tab and the single-paper conversation: pass
    `source_filter` to scope it to one paper. `history` turns a follow-up into a
    standalone question before retrieval, so "and what about its dataset?"
    resolves instead of being embedded literally.
    """
    try:
        result = synthesize_answer(
            query=req.query,
            embedder=embedder,
            store=scratch_store if req.scratch else store,
            model=OLLAMA_MODEL,
            reranker=reranker_model if req.use_reranker else None,
            source_filter=req.source_filter or None,
            history=[t.model_dump() for t in req.history],
            max_sources=req.max_sources,
        )
        return {"ok": True, **result.as_dict()}
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.get("/api/scratch")
def list_scratch():
    return {"papers": scratch_store.list_sources()}


@app.post("/api/scratch")
def upload_scratch(background: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Upload a PDF and start asking about it immediately.

    Ingests on the fast path - no LLM classification, no section summaries - so
    a short paper is ready in seconds rather than minutes. The full pass then
    runs in the background and replaces the fast version in place, so section
    routing improves without the user waiting for it.

    Scratch documents live in their own collection and are never visible to the
    corpus-wide tab.
    """
    results = []
    for f in files:
        name = Path(f.filename or "unnamed.pdf").name
        dest = SCRATCH_DIR / name
        with open(dest, "wb") as fh:
            fh.write(f.file.read())
        try:
            t0 = time.perf_counter()
            n = ingest_pdf(
                str(dest), embedder, scratch_store, skip_existing=False, fast=True
            )
            results.append({
                "name": name, "status": "ok", "chunks": n,
                "seconds": round(time.perf_counter() - t0, 1),
                "message": f"{n} chunks ready",
            })
            background.add_task(_upgrade_scratch, str(dest))
        except Exception as e:
            results.append({"name": name, "status": "error", "message": str(e)})
    return {"results": results, "papers": scratch_store.list_sources()}


def _upgrade_scratch(pdf_path: str):
    """Re-ingest a scratch document with the full LLM passes, in the background."""
    try:
        ingest_pdf(pdf_path, embedder, scratch_store, skip_existing=False, fast=False)
        print(f"  [scratch] upgraded {Path(pdf_path).name} to full classification")
    except Exception as exc:                       # never crash the server for this
        print(f"  [scratch] upgrade failed for {Path(pdf_path).name}: {exc}")


@app.delete("/api/scratch/{filename:path}")
def remove_scratch(filename: str):
    scratch_store.delete_source(filename)
    path = SCRATCH_DIR / filename
    if path.exists():
        path.unlink()
    return {"removed": filename, "papers": scratch_store.list_sources()}


class RenderRequest(BaseModel):
    source_path: str
    page_numbers: str
    text: str


@app.post("/api/render-chunk")
def render_chunk(req: RenderRequest):
    img = _render_chunk(
        source_path=req.source_path,
        page_numbers=req.page_numbers,
        chunk_text=req.text,
    )
    if img is None:
        raise HTTPException(status_code=404, detail="Could not render page")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return {"image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}


@app.get("/api/health")
def health():
    """Liveness probe that also reports what the index currently holds."""
    return {
        "ok": True,
        "papers": len(store.list_sources()),
        "chunks": store.count_chunks(),
        "scratch_papers": len(scratch_store.list_sources()),
    }


if __name__ == "__main__":
    import threading, webbrowser

    url = f"http://{'localhost' if APP_HOST in ('0.0.0.0', '127.0.0.1') else APP_HOST}:{APP_PORT}"
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
