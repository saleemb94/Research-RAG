import base64
import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from research_rag import (
    Embedder, Reranker, VectorStore,
    ingest_pdf, search_and_summarize, deep_scan_papers,
)
from research_rag.config import APP_HOST, APP_PORT, OLLAMA_MODEL, TOP_K
from research_rag.config import PAPERS_DIR as PAPERS_DIR_SETTING
from research_rag.pdf_viewer import render_chunk as _render_chunk
from research_rag.vector_store import WeaviateUnavailableError

PAPERS_DIR = Path(PAPERS_DIR_SETTING)
STATIC_DIR = Path("./static")
PAPERS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

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
print("All services ready.\n")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    # Weaviate's client holds an HTTP session and a gRPC channel; close them
    # so shutdown is clean rather than leaving warnings on exit.
    store.close()


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


class ChatRequest(BaseModel):
    query: str
    deep_scan: bool = False
    use_reranker: bool = True
    top_k: int = TOP_K
    source_filter: str | None = None


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
    return {"ok": True, "papers": len(store.list_sources()), "chunks": store.count_chunks()}


if __name__ == "__main__":
    import threading, webbrowser

    url = f"http://{'localhost' if APP_HOST in ('0.0.0.0', '127.0.0.1') else APP_HOST}:{APP_PORT}"
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
