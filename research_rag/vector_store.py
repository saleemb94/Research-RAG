"""
Weaviate-backed vector store.

The collection uses *self-provided* vectors: embeddings are computed locally by
`Embedder` (BAAI/bge-small-en-v1.5) and handed to Weaviate, so no vectorizer
module needs to be enabled on the server. Distances are cosine, matching the
normalized embeddings the model produces.

Start the database with `docker compose up -d` before using this class.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import weaviate
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    Tokenization,
    VectorDistances,
)
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.util import generate_uuid5

from .config import (
    FETCH_PAGE_SIZE,
    MAX_FETCH,
    WEAVIATE_COLLECTION,
    WEAVIATE_GRPC_PORT,
    WEAVIATE_HOST,
    WEAVIATE_PORT,
)

# How many extra candidates hybrid search fetches when a source filter is in
# play, to cover the share its keyword half returns from outside the filter.
# Measured at alpha=0.3, roughly a third came back foreign.
_HYBRID_OVERFETCH = 4

# Properties filtered on exactly (a whole heading, a whole filename). Weaviate's
# default `word` tokenization would make equal("Data Collection") match any
# chunk containing the token "data", so these are indexed as a single token.
_EXACT_TEXT_FIELDS = ("source_file", "section_name", "subsection_name", "section_type")

# Batch size for writes - keeps individual gRPC messages comfortably small.
_INSERT_BATCH = 200


class WeaviateUnavailableError(RuntimeError):
    """Raised when the Weaviate container is not reachable."""


@dataclass
class SearchHit:
    properties: dict
    distance: float | None


class VectorStore:
    def __init__(
        self,
        host: str = WEAVIATE_HOST,
        port: int = WEAVIATE_PORT,
        grpc_port: int = WEAVIATE_GRPC_PORT,
        collection: str = WEAVIATE_COLLECTION,
    ):
        self._client = None
        self._collection_name = collection
        try:
            self._client = weaviate.connect_to_local(
                host=host, port=port, grpc_port=grpc_port
            )
        except Exception as exc:
            raise WeaviateUnavailableError(
                f"Could not connect to Weaviate at {host}:{port} (gRPC {grpc_port}).\n"
                f"Start it with:  docker compose up -d\n"
                f"Original error: {exc}"
            ) from exc

        self._ensure_collection()
        self._col = self._client.collections.get(collection)

    # -- Schema ------------------------------------------------------------

    def _ensure_collection(self):
        if self._client.collections.exists(self._collection_name):
            self._add_missing_properties()
            return
        self._client.collections.create(
            name=self._collection_name,
            description="Chunks of research papers, embedded locally.",
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                )
            ),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(
                    name="source_file",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.FIELD,
                ),
                Property(
                    name="source_path",
                    data_type=DataType.TEXT,
                    index_filterable=False,
                    index_searchable=False,
                ),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="heading", data_type=DataType.TEXT),
                Property(
                    name="section_name",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.FIELD,
                ),
                Property(
                    name="subsection_name",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.FIELD,
                ),
                Property(
                    name="section_type",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.FIELD,
                ),
                Property(name="page_numbers", data_type=DataType.TEXT),
                Property(
                    name="section_summary",
                    data_type=DataType.TEXT,
                    # Routing context only - never filtered or keyword-searched,
                    # so it costs nothing to index.
                    index_filterable=False,
                    index_searchable=False,
                ),
            ],
        )

    def _add_missing_properties(self):
        """
        Bring an existing collection up to the current schema.

        Weaviate can add a property to a live collection, so a store created by
        an earlier version keeps working and simply gains the new field (empty
        on old objects) instead of needing a wipe and a full re-ingest.
        """
        col = self._client.collections.get(self._collection_name)
        existing = {p.name for p in col.config.get().properties}
        for prop in (
            Property(
                name="section_summary",
                data_type=DataType.TEXT,
                index_filterable=False,
                index_searchable=False,
            ),
        ):
            if prop.name not in existing:
                col.config.add_property(prop)

    # -- Write -------------------------------------------------------------

    def source_exists(self, source_file: str) -> bool:
        result = self._col.query.fetch_objects(
            filters=Filter.by_property("source_file").equal(source_file),
            limit=1,
            return_properties=["source_file"],
        )
        return len(result.objects) > 0

    def insert_chunks(self, chunks_data: list[dict]):
        """
        Insert chunk records. Each item is {"properties": {...}, "vector": [...]}.

        UUIDs are derived deterministically from source_file + chunk_index, so
        re-ingesting the same paper produces the same object IDs rather than
        silent duplicates.
        """
        if not chunks_data:
            return
        objects = []
        for item in chunks_data:
            props = dict(item["properties"])
            uid = generate_uuid5(f"{props['source_file']}::{props['chunk_index']}")
            objects.append(
                DataObject(properties=props, vector=item["vector"], uuid=uid)
            )

        for start in range(0, len(objects), _INSERT_BATCH):
            batch = objects[start : start + _INSERT_BATCH]
            result = self._col.data.insert_many(batch)
            if result.has_errors:
                first = next(iter(result.errors.values()))
                raise RuntimeError(
                    f"Weaviate rejected {len(result.errors)} of {len(batch)} objects. "
                    f"First error: {first.message}"
                )

    def delete_source(self, source_file: str):
        self._col.data.delete_many(
            where=Filter.by_property("source_file").equal(source_file)
        )

    # -- Read --------------------------------------------------------------

    def list_sources(self) -> list[str]:
        metas = self._scan(None, ["source_file"])
        return sorted({m["source_file"] for m in metas if m.get("source_file")})

    def count_chunks(self, source_filter: str | None = None) -> int:
        filters = self._build_filters(source_filter=source_filter)
        result = self._col.aggregate.over_all(total_count=True, filters=filters)
        return result.total_count or 0

    def get_section_summary(
        self, source_filter: str | None = None
    ) -> dict[str, list[str]]:
        """
        Return the section structure stored at ingest time:
            { section_name: [subsection_name, ...] }
        Ordered by first appearance in the document.
        """
        metas = self._scan(
            self._build_filters(source_filter=source_filter),
            ["section_name", "subsection_name", "chunk_index"],
        )
        # Sort by chunk_index so the order matches the document
        metas.sort(key=lambda m: int(m.get("chunk_index") or 0))

        sections: dict[str, set[str]] = {}
        order: list[str] = []
        for meta in metas:
            sn = (meta.get("section_name") or "").strip()
            sub = (meta.get("subsection_name") or "").strip()
            if not sn:
                continue
            if sn not in sections:
                sections[sn] = set()
                order.append(sn)
            if sub:
                sections[sn].add(sub)
        return {sn: sorted(sections[sn]) for sn in order}

    def get_section_descriptions(
        self, source_filter: str | None = None
    ) -> dict[str, str]:
        """
        Return the one-line description written for each section at ingest time:
            { section_name: "what this section actually contains" }

        Sections indexed before summaries existed simply have no entry, so
        callers degrade to heading names rather than breaking.
        """
        metas = self._scan(
            self._build_filters(source_filter=source_filter),
            ["section_name", "section_summary"],
        )
        out: dict[str, str] = {}
        for meta in metas:
            name = (meta.get("section_name") or "").strip()
            summary = (meta.get("section_summary") or "").strip()
            if name and summary and name not in out:
                out[name] = summary
        return out

    def get_unique_section_names(
        self, source_filter: str | None = None
    ) -> list[str]:
        """Ordered list of unique section_name values for a paper (or all papers)."""
        return list(self.get_section_summary(source_filter).keys())

    def get_section_type_map(
        self, source_filter: str | None = None
    ) -> dict[str, str]:
        """
        Return the dominant section_type for each top-level section:
            { section_name: section_type }
        Uses majority vote across all chunks in a section so one mis-classified
        chunk doesn't flip the whole section's label.
        """
        metas = self._scan(
            self._build_filters(source_filter=source_filter),
            ["section_name", "section_type"],
        )
        type_counts: dict[str, Counter] = {}
        for meta in metas:
            sn = (meta.get("section_name") or "").strip()
            st = meta.get("section_type") or "general"
            if not sn:
                continue
            type_counts.setdefault(sn, Counter())[st] += 1
        return {
            sn: counter.most_common(1)[0][0]
            for sn, counter in type_counts.items()
        }

    # -- Retrieval ---------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        source_filter: str | None = None,
        section_types: list[str] | None = None,
        section_names: list[str] | None = None,
    ) -> list[SearchHit]:
        """
        Vector similarity search.
        Filter priority: section_names (exact stored names) > section_types (labels).
        """
        filters = self._build_filters(
            source_filter=source_filter,
            section_types=section_types,
            section_names=section_names,
        )
        result = self._col.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            filters=filters,
            return_metadata=MetadataQuery(distance=True),
        )
        return [
            SearchHit(
                properties=self._normalize(obj.properties),
                distance=obj.metadata.distance,
            )
            for obj in result.objects
        ]

    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        limit: int = 24,
        alpha: float = 0.3,
        source_filter: str | None = None,
    ) -> list[SearchHit]:
        """
        Weaviate hybrid search: BM25 over the chunk text fused with the vector.

        `alpha` weights the two - 1.0 is pure vector, 0.0 is pure keyword. Low
        values favour keyword, which is the point: most of the facts this
        pipeline misses are rare literal strings (AraVec, ArCybC, OSACT5,
        QARiB), and a dense embedding has almost no neighbourhood for a proper
        noun it saw a handful of times. BM25 matches them exactly.

        The standing limitation is that BM25 can only match terms the *query*
        contains, so this helps when the rare term is asked about and not when
        it is the answer being looked for.

        Hits carry no distance: hybrid returns a fused score on a different
        scale, and everything downstream reranks anyway.

        Results are filtered again in Python because the keyword half does not
        honour the filter the way the vector half does. Asked for one paper at
        alpha=0.0, 23 of 24 hits came back from other papers; at alpha=1.0,
        none did. Building the filter by hand and passing it straight to the
        client behaved identically, so it is not how the filter is constructed.

        The damage was quiet and specific. The fan-out over papers takes the
        first hit for a paper as that paper's citation, so a leaked chunk
        produced a citation naming one paper while rendering a passage from
        another. In one answer four of eight citations opened the same passage
        of the same foreign paper.
        """
        allowed: set[str] | None = None
        if source_filter:
            allowed = ({source_filter} if isinstance(source_filter, str)
                       else {f for f in source_filter if f})

        # Overfetch while filtering, because a share of what comes back is
        # discarded and the caller still expects up to `limit` usable hits.
        fetch = limit * _HYBRID_OVERFETCH if allowed else limit
        result = self._col.query.hybrid(
            query=query_text,
            vector=query_vector,
            alpha=alpha,
            limit=fetch,
            filters=self._build_filters(source_filter=source_filter),
        )
        hits = [
            SearchHit(properties=self._normalize(obj.properties), distance=None)
            for obj in result.objects
        ]
        if allowed:
            hits = [h for h in hits if h.properties.get("source_file") in allowed]
        return hits[:limit]

    def get_by_section(
        self,
        section_types: list[str] | None = None,
        source_filter: str | None = None,
    ) -> list[SearchHit]:
        """Fetch all chunks matching section_type labels, in document order."""
        filters = self._build_filters(
            source_filter=source_filter, section_types=section_types
        )
        return self._fetch_all(filters)

    def get_by_section_name(
        self,
        section_names: list[str],
        source_filter: str | None = None,
    ) -> list[SearchHit]:
        """
        Fetch all chunks whose section_name matches any of the given names,
        in document order.  This uses the structural field set at ingest time.
        """
        if not section_names:
            return []
        filters = self._build_filters(
            source_filter=source_filter, section_names=section_names
        )
        return self._fetch_all(filters)

    # -- Internal helpers --------------------------------------------------

    def _scan(self, filters, return_properties: list[str] | None) -> list[dict]:
        """
        Walk every object matching `filters`, page by page.

        Weaviate caps a single response, and its cursor API cannot be combined
        with filters, so this pages with offset/limit up to MAX_FETCH.
        """
        out: list[dict] = []
        offset = 0
        while offset < MAX_FETCH:
            page_size = min(FETCH_PAGE_SIZE, MAX_FETCH - offset)
            page = self._col.query.fetch_objects(
                filters=filters,
                limit=page_size,
                offset=offset,
                return_properties=return_properties,
            )
            out.extend(obj.properties for obj in page.objects)
            if len(page.objects) < page_size:
                break
            offset += len(page.objects)
        return out

    def _fetch_all(self, filters) -> list[SearchHit]:
        hits = [
            SearchHit(properties=self._normalize(props), distance=None)
            for props in self._scan(filters, None)
        ]
        hits.sort(
            key=lambda h: (
                h.properties["source_file"],
                int(h.properties.get("chunk_index") or 0),
            )
        )
        return hits

    @staticmethod
    def _normalize(props) -> dict:
        """
        Weaviate returns absent text properties as None and INT properties
        as Python ints. Downstream code expects every string present and
        chunk_index an int, so absences are filled in here rather than guarded
        against at each call site.
        """
        out = dict(props)
        for key in (
            "text",
            "source_file",
            "source_path",
            "heading",
            "section_name",
            "subsection_name",
            "page_numbers",
            "section_summary",
        ):
            if out.get(key) is None:
                out[key] = ""
        if out.get("section_type") is None:
            out["section_type"] = "general"
        out["chunk_index"] = int(out.get("chunk_index") or 0)
        return out

    @staticmethod
    def _build_filters(
        source_filter: str | None = None,
        section_types: list[str] | None = None,
        section_names: list[str] | None = None,
    ):
        conditions = []
        if source_filter:
            # A list is a two-stage shortlist: restrict to those papers rather
            # than to one. FIELD tokenization makes each filename a single
            # token, so equality per paper is exact.
            if isinstance(source_filter, (list, tuple, set, frozenset)):
                names = [f for f in source_filter if f]
                if names:
                    conditions.append(Filter.any_of(
                        [Filter.by_property("source_file").equal(f) for f in names]
                    ))
            else:
                conditions.append(
                    Filter.by_property("source_file").equal(source_filter)
                )
        # section_names (exact stored headings) takes priority over section_types
        if section_names:
            conditions.append(
                Filter.any_of(
                    [Filter.by_property("section_name").equal(n) for n in section_names]
                )
            )
        elif section_types:
            conditions.append(
                Filter.any_of(
                    [Filter.by_property("section_type").equal(t) for t in section_types]
                )
            )
        if not conditions:
            return None
        return conditions[0] if len(conditions) == 1 else Filter.all_of(conditions)

    # -- Lifecycle ---------------------------------------------------------

    def close(self):
        client = getattr(self, "_client", None)
        if client is not None:
            client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):  # best-effort: avoids a leaked gRPC channel warning
        try:
            self.close()
        except Exception:
            pass
