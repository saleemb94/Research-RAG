# Research RAG

**Section-aware retrieval-augmented generation over research papers, fully offline.**

Drop a folder of PDFs in, ask questions in natural language, and get answers grounded
in specific sections of specific papers, with the source passage rendered back on the
original PDF page. No API keys and no cloud calls: parsing, embedding, reranking and
generation all run locally.

```
"What datasets were used for Arabic offensive language detection?"

  -> routed as a question that needs several papers at once
  -> shortlisted to the papers whose cards mention Arabic
  -> searched inside each one's data sections, not across the whole corpus
  -> reranked, then written as one answer whose every claim carries a [n]
     citation back to the passage it came from
```

There are four views on top of that: one synthesized answer across the library, a chat
scoped to a single paper, a drop-in upload that never touches the library, and a
browsable list of what has been ingested.

![One answer across the library, every claim cited](docs/screenshots/ask.png)

Every `[n]` is clickable. Opening one renders that exact passage on its original
PDF page, highlighted, so you can check a claim against the paper in one click
instead of taking it on trust:

![A citation opened onto its source page](docs/screenshots/citation.png)

The single-paper view keeps a conversation scoped to one document and resolves
follow-ups against what came before. Here *"that combination"* becomes the
IndoBERTweet-BiLSTM-CNN model from the previous turn:

![A scoped two-turn conversation](docs/screenshots/paper.png)

---

## Why this is not another naive RAG demo

Most RAG-over-PDF projects flatten a document into fixed-size chunks and hope cosine
similarity finds the right one. Research papers have structure worth keeping, and the
design here is built around exploiting it.

| Problem with naive RAG | What this project does |
| --- | --- |
| Fixed-size chunks split tables and paragraphs mid-thought | [Docling](https://github.com/DS4SD/docling) parses layout and a `HierarchicalChunker` splits on real document structure |
| Chunks lose their place in the document | Every chunk stores its `section_name`, `subsection_name` and page numbers |
| "What was the methodology?" retrieves the abstract, which only *mentions* methodology | Queries are classified to a section type, then search is **restricted** to matching sections |
| Papers number sections inconsistently (`3.1`, `A.`, `III.`) | Numbering is auto-detected per paper (decimal, IEEE or alphabetic) and orphaned subsections are re-parented |
| References and ethics statements pollute results | Boilerplate sections are dropped at ingest via a blacklist and a title check |
| Top-k vector hits are often loosely relevant | A cross-encoder reranks 4x the candidates down to the best `k` |
| Answers are hard to trust | Each answer links to the exact passage, rendered onto its original PDF page |
| Section vocabulary is field-specific: `corpus`, `cohort`, `specimen` and `primary sources` all mean *the data* | Classification is by section **function**, using vocabulary from nine disciplines, and it refuses to guess on words that flip meaning between fields |

---

## Architecture

```
        PDF
         │
         ▼
   ┌──────────────┐   layout-aware parsing + hierarchical chunking
   │   Docling    │
   └──────┬───────┘
          │  chunks + headings
          ▼
   ┌──────────────────────────────────┐   3-stage section classification:
   │      Section classifier          │   1. keyword match
   │  (keywords -> LLM -> content)    │   2. LLM batch classify ambiguous
   └──────┬───────────────────────────┘   3. verify against first paragraph
          │  section_name / section_type / pages
          ▼
   ┌──────────────┐   BAAI/bge-small-en-v1.5, 384-dim, normalized
   │   Embedder   │   embedded with the paper title and section as context
   └──────┬───────┘
          │  vectors + metadata
          ▼
   ┌──────────────────────────────────┐
   │   Weaviate  (Docker container)   │   HNSW, cosine, self-provided vectors
   └──────┬───────────────────────────┘   exact-match filters on section fields
          │
          ├──► one summary per section, embedded ──► second retrieval channel
          └──► one structured card per paper ──────► corpus-level questions
          │
          ▼  ── query path ──────────────────────────────────────────────
   ┌──────────────────────────────────┐   corpus    -> answer from the cards
   │        Question router           │   enumerate -> fan out over papers
   └───┬──────────────┬───────────┬───┘   focused   -> ordinary retrieval
       │              │           │
       │              │           ▼
       │              │    ┌──────────────┐  classify query, match to stored
       │              │    │ Hierarchical │  headings, filter, search within
       │              │    │  retrieval   │  section, global fallback
       │              │    └──────┬───────┘
       │              │           │   three channels, unioned:
       │              │           │   chunk vectors · section summaries · BM25
       │              │           ▼
       │              │    ┌──────────────┐  cross-encoder/ms-marco-MiniLM-L-6-v2
       │              │    │   Reranker   │
       │              │    └──────┬───────┘
       ▼              ▼           ▼
   ┌──────────────────────────────────┐   one answer, every claim carrying a
   │   Ollama   (qwen3:4b-instruct)   │   marker, validated against the
   └──────┬───────────────────────────┘   passages actually retrieved
          ▼
   FastAPI + web UI, with PyMuPDF rendering the source page
```

### Three ways a question gets answered

Not every question needs the same machinery. A router classifies each question first,
and the three paths differ in what they read, not just in how long they take.

**Corpus.** *"Which papers use transformers?"*, *"what languages are covered?"* These
are questions about the shape of the library rather than the contents of any one paper.
Retrieval is the wrong tool: the answer is a property of every paper at once, and the
top 5 chunks can only ever see a few of them. So they are answered from the structured
card built for each paper at ingest, which means every paper gets considered.

**Enumerate.** *"What datasets were used across these papers?"* The answer is a list
assembled from many papers, each contributing a piece. The question fans out: shortlist
the papers worth reading, search each one separately, extract just the relevant finding
from each, then reduce those findings into a single cited answer. Searching once across
the corpus would return five chunks from the two papers that phrase things most
similarly and silently miss the rest.

**Focused.** *"What accuracy did IndoBERTweet reach?"* One fact, in one place. Ordinary
retrieval: classify the query to a section type, filter, search, rerank, answer.

The single-paper view skips the router. The paper is already chosen, so there is
nothing to route.

### Retrieval safety rails

Section filtering is enforced in three independent places, because an LLM classifier
on its own is not trustworthy enough to silently drop content:

1. **Cross-validation at match time.** A heading the LLM matched is discarded if its
   dominant ingest-time `section_type` contradicts the query's target type.
2. **Database-level filter.** Weaviate filters on `section_name` with `FIELD`
   tokenization, so `"Data Collection"` matches that heading exactly and never a
   section that merely contains the word *Data*.
3. **Post-retrieval filter in Python.** Hits are re-checked against the allowed set
   regardless of what the database returned.

If a section turns out to be empty, retrieval falls back to a global search rather
than returning nothing, and the filters in step 3 are **switched off on that fallback
path**. That detail matters. Applying an allow-list for a section the paper does not
have deletes every fallback hit and returns an empty answer while the store is holding
perfectly good chunks. It shipped that way until the golden question set caught it, and
`tests/test_retrieval_filters.py` now pins it.

**Scope decides how sections are matched.** Inside one paper, the target is matched
onto that paper's real headings, so "methodology" finds a section actually called
"Our Approach". Across the corpus that would be wrong: heading lists get pooled, the
model picks names that exist in only one or two papers, and filtering on them makes
every other paper unreachable. A corpus-wide query filters on `section_type` instead,
which is assigned per chunk at ingest precisely so that it generalizes.

### Working across disciplines

Papers are not all from one field, so the classifier is built around section
**function** rather than any single field's vocabulary.

The eight labels (`abstract`, `introduction`, `theory`, `related_work`, `methodology`,
`dataset`, `results`, `conclusion`) describe roles that recur across empirical research
whatever the subject. `theory` covers papers that are not empirical at all: theorems
and proofs in mathematics, derivations in physics, formal models in economics,
conceptual frameworks in the social sciences.

Each label carries vocabulary from several fields, because the same role is named
differently everywhere:

| Role | Computing | Medicine | Social science | Humanities |
| --- | --- | --- | --- | --- |
| `dataset` | corpus, benchmark, training set | cohort, study population, inclusion criteria | participants, respondents, sample | primary sources, archival material |
| `methodology` | proposed method, architecture | trial protocol, randomization, assay | interview protocol, coding procedure | fieldwork, ethnography |
| `related_work` | related work, prior art | systematic review, meta-analysis | literature review | historiography |

**Where it deliberately refuses to guess.** Some headings mean different things in
different fields, and a confident wrong answer from the fast keyword stage would stop
the later stages ever reading the section's text. Those headings are listed in
`_DEFER_TO_CONTENT` and returned as unclassified on purpose, which routes them to the
LLM stages that can actually read the content:

| Heading | In computing | In another field |
| --- | --- | --- |
| "Statistical Analysis" | reported findings | the analysis plan, inside Methods (medicine) |
| "Survey" | a literature survey | a questionnaire instrument (sociology) |
| "Materials" | (rare) | reagents (chemistry), stimuli (psychology), or half of "Materials and Methods" |
| "Model" | an implemented system | a formal model (economics, theory) |
| "Baseline Characteristics" | a comparison result | a description of the participants (clinical trials) |

That last one used to be classified as a *result*, which is simply wrong for a clinical
paper. It describes who was enrolled.

### Choosing a model

The summarization step is forgiving. Most instruct models write something reasonable
from retrieved text. The *routing* steps are not: they decide which sections get
searched at all, so an error there silently changes the answer without looking like a
failure. `scripts/eval_models.py` scores exactly those three call sites against ground
truth spanning nine disciplines:

```bash
python scripts/eval_models.py                      # every installed model
python scripts/eval_models.py qwen3:8b --think     # measure reasoning mode
python scripts/eval_models.py -v qwen2.5:7b        # show every wrong answer
```

Measured on an RTX 5060 Laptop (8 GB), 50 graded items:

| Model | Headings | Query | Target | Overall | Time |
| --- | --- | --- | --- | --- | --- |
| **qwen3:4b-instruct** (default) | 96% | 100% | 100% | **98%** | **5.5 s** |
| qwen3:8b (reasoning off) | 92% | 100% | 100% | 96% | 9.0 s |
| qwen2.5:7b | 88% | 95% | 100% | 92% | 10.9 s |
| qwen3:8b (reasoning on) | 92% | 90% | 100% | 92% | 133.5 s |
| llama3.2:3b | 33% | 90% | 83% | 62% | 10.1 s |

Three results worth knowing:

- **Bigger was not better.** `qwen3:4b-instruct` beat `qwen3:8b` on accuracy and was
  faster. These tasks reward instruction-following on a fixed label set, not knowledge.
- **Reasoning actively hurt.** With thinking enabled, `qwen3:8b` took 15x longer and
  scored lower (92% against 96%). It deliberates its way out of correct one-word
  answers, spending around 715 tokens where 2 would do. Reasoning is off by default for
  that reason; set `OLLAMA_THINK=true` to override.
- **Heading classification is where models separate.** Every model handles query
  routing well. Only the good ones classify unfamiliar headings like "Baseline
  Characteristics" or "Historiography" correctly, and that step runs at ingest, so
  every later retrieval inherits its mistakes.

Note that the 2507 "Instruct" (non-thinking) refresh only exists for 4B, 30B-A3B and
235B. There is no `qwen3:8b-instruct`, so at 8B you get the hybrid model and have to
disable reasoning explicitly. Ollama returns reasoning in a separate `thinking` field,
so it never corrupts parsed output; the cost is latency, not correctness.

All Ollama traffic goes through [`research_rag/llm.py`](research_rag/llm.py), which
applies this policy in one place.

### Measuring retrieval and generation quality

`tests/golden_qa.json` holds 138 graded questions with 341 fact groups that a correct
answer has to contain: 107 single-paper questions covering 53 papers (229 groups),
10 corpus-wide synthesis questions (33 groups), 16 thematic discussion questions
(79 groups) and 5 negative controls.

Every fact was read from the source PDF, never from the search index. That is the point
of the set rather than a detail of how it was built. Facts harvested from the index
could not detect content that ingestion dropped, because coverage would then be 100% by
construction. It is 100%, but earned, and building it this way caught two real defects
that are described under *Notes and limitations*.

A golden set also goes stale as a library is curated. `eval_rag.py` names any paper
that has been removed and skips its questions instead of scoring them zero, because a
question whose source document is gone measures the shelf, not the system.

A RAG system can fail in three separate places, and one end-to-end number hides which
one broke, so the scorer reports them apart:

```bash
python scripts/eval_rag.py --coverage      # is the fact in the index at all? (no LLM)
python scripts/eval_rag.py --retrieval     # asked blind, does the right paper come back?
python scripts/eval_rag.py --generation    # given the right paper, is the fact stated?
python scripts/eval_rag.py --synthesis     # corpus-wide cited answers
python scripts/eval_rag.py --thematic      # discussion questions across a topic
python scripts/eval_rag.py --all -v
```

A corpus-wide question is not a single-paper question asked louder, so the two sets are
kept apart and scored by different code. Questions in the synthesis set have to:

1. need **at least two papers** for a complete answer, otherwise it is needle retrieval
   wearing a synthesis costume;
2. have a **closed, enumerable** answer (languages, models, datasets, scores) rather
   than inviting discussion;
3. be answerable **without naming a paper**;
4. use **discriminating** fact groups, so a fluent non-answer scores zero.

They are scored on two axes that fail apart: facts stated, and which papers were
actually cited. An answer can name four datasets while citing one paper, which reads as
authoritative and quietly under-reports the corpus.

**Thematic questions** are the third set, and they exist because the first two only ask
things with a right answer. Neither resembles what someone with a reading pile actually
types. They ask *discuss the limitations reported in research on retrieval-augmented
generation*, or *compare the reranking approaches used in these papers*: a topic spanning
a cluster of papers, with no closed answer.

That difference forces different scoring. A synthesis question has an enumerable answer,
so every expected paper must appear. A discussion question does not. The library holds
eight papers on hate speech, and a good four-paragraph answer draws on some of them, not
all eight; demanding the full set would mark a genuinely good answer as a failure. So
each theme carries a `min_papers` floor and is scored on three things the synthesis
scorer does not measure:

- **breadth**, whether the answer drew on at least `min_papers` of the theme's own papers;
- **citations outside the theme**, evidence padding that reads as authoritative;
- **distinct passages**, because eight markers pointing at one paragraph look like eight
  sources in the interface and are one.

Topic boundaries in a real corpus are fuzzy, and a survey legitimately touches half a
dozen themes. Counting every survey citation as off-topic would score the boundary I drew
rather than the system, so each theme also lists `related_papers`: papers satisfying most
of its fact groups in their own text without being primarily about it. Citing one is
neither a hit nor a miss.

Over three runs the set scores **68% of facts** (54 of 79, range 51 to 57, sd 2.4) and
meets the breadth bar on **8.3 of 16** questions. Citations are **100% distinct
passages** across all three runs, 366 for 366, and 40% of cited papers fall outside their
theme. That spread is the first thing the set established, and it is larger than most
differences anyone would want to claim, so these questions are reported over repeated
runs rather than one.

*Discuss the limitations reported in research on RAG*, almost the question that prompted
the set, scored 0 of 5 fact groups on one run and 5 of 5 on two others, from the same
index and the same question. Two more questions then score badly on every run, and
reading their answers rather than their scores found unrelated causes, one of which was
not the system at all:

**Answering from cards.** *What weaknesses of large language models are identified?* and
the RAG-limitations question on its bad runs are routed corpus-wide and answered from the
paper cards. Routing is right and breadth is right, 12 of the 12 RAG papers, but a card
is a summary: the answer comes back fluent, correctly cited, and stating none of the
hallucination, cost or privacy findings the papers actually report. Breadth and depth are
in tension on these questions, the router picks one, and the choice is not stable.

**A question scored against the wrong thing.** *How do these papers evaluate retrieval
and generation quality?* also fails every run, 1 of 5, and it is not a system failure at
all. The answer is substantive: perplexity differences between original and RAG-prompt
responses, a DCG@k estimate of what the top-k passages contribute, the eRAG method of
scoring each retrieved document by downstream task performance, faithfulness and context
relevance. It simply never writes the strings `nDCG`, `MRR` or `ROUGE`. The question asks
*how* these papers evaluate, which invites methodology, and the fact groups then demanded
metric names. The question and its scoring disagreed, and the answer was marked wrong for
obeying the question.

That is worth recording rather than quietly fixing, because it is the failure mode a
golden set is most prone to: a low score that looks like a system defect and is really
the measurement. It was found by reading the answers, not the numbers, which is the only
way it could have been found. Two of the three consistently weak questions here turned
out to be flaws in the set. So **68% understates the real quality**, and the honest
reading of that number is that it is a floor.

The robustness question is the other one. Its answer is fluent and on topic, covering
bias, manipulation of retrieval-augmented models and privacy leakage, but it never
reaches the single paper that actually runs adversarial attacks. Part system, part
question: it draws on four papers where each fact is carried by only one, which makes it
the thinnest item in the set.

**Negative controls** cover the gap every other metric leaves. All the others measure
denying content that is present. None measures inventing content that is absent, which
for a research tool is the worse failure, since retrieval always returns *something* and
nothing upstream prevents it. Five questions the corpus genuinely cannot answer are
scored purely on whether the system declines. There is a tension worth naming: a system
that abstains on everything scores 100% here and 0% on synthesis.

The controls are verified rather than assumed. Each term is counted across every chunk
before it is trusted as absent: quantum, differential privacy, blockchain, autonomous
driving and eye tracking appear zero times. The previous controls had gone stale as the
library grew (RLHF, once absent, now appears in five chunks), which is how a control
that still reads as sensible stops measuring anything.

`--coverage` is the one to run after changing anything in ingestion. It is instant,
needs no LLM, and a drop there means content was lost or mangled before retrieval ever
got a chance.

The corpus these figures were measured on is 54 papers about retrieval-augmented
generation, information retrieval, text classification, hate-speech detection and
clinical NLP. `papers/` is gitignored, so a clone starts empty; `scripts/fetch_arxiv.py`
pulls a small open-access set to try the pipeline on.

Current scores, against the baseline they replaced:

| | baseline | now | |
| --- | --- | --- | --- |
| Coverage, fact is in the index | 100% | **100%** | 229/229 |
| Retrieval, right paper returned | 67% | **70%** | 75/107 |
| Retrieval, right paper ranked first | 51% | **53%** | 57/107 |
| Routing, right section targeted | 57% | **58%** | 62/107 |
| Generation, fact stated in the answer | 71% | 69% | 158/229 |
| Generation, fully correct answers | 64% | 63% | 67/107 |
| Synthesis, facts stated in one cited answer | 73% | **76%** | 25/33 |
| Synthesis, expected papers actually cited | 56% | **70%** | 30/43 |
| Thematic, facts stated in a discussion answer | | 68% | 54/79, mean of 3 |
| Thematic, questions meeting the breadth bar | | 52% | 8.3/16, mean of 3 |
| Thematic, citations that are distinct passages | | 100% | 366/366 |
| Negative controls, correctly declined | 100% | **100%** | 5/5 |
| Invalid citations emitted | 0 | **0** | |

The thematic rows have no baseline column because that set is new and nothing has been
changed in response to it yet. They are a starting point to measure against, not a gain.

The gain is almost all in synthesis, where papers actually cited went from 56% to 70%.
Retrieval moved a few points. Generation moved a point the other way, which is inside
the run-to-run spread of a sampled model and should not be read as a change. These are
single runs, so use `--repeat` before trusting a difference this small.

Routing is scored against the section each fact *ought* to live in, which is a judgment
call made when writing the question, so part of that 58% is disagreement about labels
rather than a routing error.

An earlier 17-paper corpus with an 85-question set scored 81% / 70% / 78% / 79% / 71%
on the same rows. **Those numbers are not comparable to the ones above.** The corpus
tripled and became topically uniform (dozens of papers about retrieval-augmented
generation, passage ranking and text classification), which is the hardest case for
needle retrieval, because many papers can plausibly answer "which datasets were used".
The 35 retrieval misses are spread evenly across topics rather than concentrated in
one, which is what that explanation predicts.

That older set is gone: 8 of its papers were removed from the library, and 29 of its
questions pointed at documents that no longer exist. The table it produced is kept
because it records what the first golden set was built to catch:

| | before | after |
| --- | --- | --- |
| Coverage, fact is in the index | 99% | 100% |
| Retrieval, right paper returned | 72% | 81% |
| Retrieval, right paper ranked first | 56% | 70% |
| Routing, right section targeted | 46% | 78% |
| Generation, fact stated in the answer | 59% | 79% |
| Generation, fully correct answers | 49% | 71% |
| Synthesis, facts stated in one cited answer | 64% | 89% |
| Synthesis, expected papers actually cited | 58% | 94% |

The "before" column is not a weaker model. It is the same pipeline with three silent
bugs that only graded ground truth could surface: heading matching pooled across
papers, post-retrieval filters canceling the global fallback, and abstracts dropped as
boilerplate. Each produced a fluent, plausible answer while discarding correct content,
which is the failure mode eyeballing output cannot catch.

Answers vary between identical runs, so `--repeat N` reports a mean and spread instead
of a single number. Measuring the two extra retrieval channels three times each, on the
older corpus:

| channels | facts | papers |
| --- | --- | --- |
| neither | 73% (sd 0.9) | 86% (sd 0.9) |
| section summaries only | 71% (sd 1.7) | 86% (sd 1.7) |
| BM25 only | 74% (sd 2.2) | 84% (sd 0.5) |
| **both** | **75% (sd 0.8)** | **89% (sd 0.5)** |

Neither channel justifies itself alone. Summaries alone measured slightly worse than
baseline, and BM25 alone was no better and much noisier. Together they are best on both
metrics and, more convincingly, the most stable: every other configuration swings three
to four points between identical runs, this one swings one. The ranges still overlap at
n=3, so this is a consistency argument rather than a decisive one.

### Measuring the conversational views

The corpus-wide set asks one self-contained question at a time, so it never touches
what the other two views are for: a follow-up that only means something given what came
before.

```bash
python scripts/eval_chat.py --single     # chat with one paper
python scripts/eval_chat.py --unscoped   # follow-ups across the whole library
python scripts/eval_chat.py --scratch    # upload, ask, prove isolation, clean up
python scripts/eval_chat.py --all
```

| | facts stated | fully correct turns | follow-ups resolved |
| --- | --- | --- | --- |
| Single paper (18 turns) | 93% | 89% | 92% |
| Corpus-wide (6 turns) | 86% | 83% | 100% |
| Ad-hoc upload (2 turns) | 100% | 100% | 100% |

The scratch run asserts isolation rather than assuming it: the uploaded document
appears in the scratch store, leaves the library list unchanged, is absent from a
library-wide answer, and disappears on cleanup.

Resolution only matters when it changes what gets retrieved. Inside one paper a
reference like "the trial" needs no expanding, and the system correctly leaves those
alone. The unscoped conversations are where it bites, and they immediately found a real
defect: asked "what dataset does the adversarial robustness one use?", the rewriter
answered "what dataset does paper [9] use?". A perfectly resolved reference, and
useless, because no index knows what paper [9] is.

### Benchmarking the stages

The golden set answers *is the answer right*. It says nothing about where the time
goes, or whether parsing produced a usable document in the first place, and a RAG
system can be accurate and unusable, or fast and structurally broken.

```bash
python scripts/bench.py --structure     # parsing quality, instant, no LLM
python scripts/bench.py --retrieval     # latency per channel, p50/p95
python scripts/bench.py --generation    # answer latency and output rate
python scripts/bench.py --ingest 3      # parse/classify/embed/summarize/write
```

Measured on 54 papers and 3,924 chunks:

| Stage | Measurement | What it tells you |
| --- | --- | --- |
| Parsing | 100% of chunks typed to a real section, 0.2% fall back to `general`, 1 paper unreadable | A paper whose headings did not survive is invisible to section filtering. It degrades to plain semantic search without ever failing a test |
| Ingestion | 18s per paper, 1.0s per page: parse 46%, summaries 43%, classification 9%, embedding 2%, write 0.2% | "Ingestion is slow" is not actionable. The split is |
| Retrieval | rerank 123ms p50, vector search 2.8ms, BM25 3.2ms, end to end 141ms p50 and 233ms p95 | Reranking owns the budget, and it scales with candidates fetched rather than corpus size, so it is the first knob to turn |
| Generation | 17s p50 per answer, about 111 chars/s | Retrieval is roughly 1% of a request. The model, not the index, sets how fast this feels |

`--structure` is the one to run after touching ingestion. It is free, needs no LLM, and
it is the only check on parsing quality anywhere in the repo. It found that 17 of 54
papers had no abstract chunk, which turned out to be a classification bug; after the
fix, 3 do.

There are two scripts for tuning rather than reporting. `scripts/sweep_retrieval.py`
scores retrieval settings without calling the LLM, which takes seconds per setting
instead of the ten minutes a full evaluation needs, and `scripts/repair_index.py` fixes
parsing defects in place without re-parsing anything.

### Running the tests

Classification is covered by tests that need neither Ollama nor Weaviate, so they run
in milliseconds:

```bash
python tests/test_section_classifier.py     # standalone
pytest tests/                               # or with pytest installed
python tests/test_ui.py                     # browser checks, needs Playwright
python tests/test_ui.py --headed            # watch it drive the browser
```

The browser tests start the real app on a spare port and drive the shipped page: a
citation has to render as a clickable chip, clicking it has to open the source panel
and actually paint the PDF page, tabs have to show exactly one view, model output must
never become markup, and the console has to stay clean. None of that is reachable from
an HTTP client, and the answer renderer has been rewritten twice.

They were checked against deliberate breakage rather than trusted for passing: renaming
the chip class makes "citations render as chips" fail on its own, and disabling the tab
handler fails all four tab checks.

The classification tests assert correct labels for headings drawn from computing,
medicine, psychology, chemistry, physics, mathematics, economics, law and history, and
pin the two distinctions the retrieval design depends on: background is not related
work, and ambiguous headings have to defer rather than guess.

---

## Quickstart

### Prerequisites

- **Python 3.11+**
- **Docker**, which runs the Weaviate vector database
- **[Ollama](https://ollama.com)**, which runs the local LLM

```bash
ollama pull qwen3:4b-instruct   # ~2.5 GB; see "Choosing a model" above
```

### Install

```bash
git clone https://github.com/saleemb94/Research-RAG.git
cd Research-RAG

python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Start the vector database

```bash
docker compose up -d
```

Weaviate is published on **8081** (REST) and **50052** (gRPC) rather than the usual
8080/50051, so it does not collide with another Weaviate you may already be running.
Override with `WEAVIATE_PORT` and `WEAVIATE_GRPC_PORT` in a `.env` file; `docker
compose` and the app both read them.

Check it is healthy:

```bash
docker compose ps             # should show "healthy"
curl http://localhost:8081/v1/.well-known/ready
```

### Ingest papers and ask questions

```bash
mkdir -p papers && cp /path/to/*.pdf papers/

python -m research_rag.cli ingest papers/     # also builds cards + section index
python -m research_rag.cli list
python -m research_rag.cli sections --paper mypaper.pdf
python -m research_rag.cli search "What datasets were used?"
```

### Or use the web UI

```bash
python app.py                 # opens http://localhost:7860
```

On Windows there is a launcher that does the whole startup in one step: it starts
Docker Desktop if it is not running, brings up Weaviate, waits for both it and Ollama
to actually answer, then starts the app and opens the browser.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\launch.ps1
```

To put it on the desktop, make a shortcut to that command with
[`docs/icon.ico`](docs/icon.ico) as its icon (regenerate the icon with
`python scripts/make_icon.py`). Closing the window stops the app; Weaviate keeps
running until `docker compose down`.

Four views:

| View | What it does |
| --- | --- |
| **Ask the library** | One synthesized answer across every paper, each claim carrying a `[n]` citation you can open |
| **Single paper** | A conversation scoped to one document, with follow-ups resolved against the history |
| **Scratchpad** | Drop in a PDF, ask about it, discard it. Stored separately and never visible to the other views |
| **Library** | What is ingested: filter by title, select any number of papers to remove at once, progress reported for both adding and removing |

Clicking any citation opens the source panel and renders that passage on its original
PDF page, highlighted.

### Interface

One file, no build step, no CDN. [`static/index.html`](static/index.html) carries its
own design tokens. That last part is deliberate: a page fetching a stylesheet and two
font files on every load would quietly contradict the offline claim above, so the type
is a system stack and the CSS is local.

Two color roles do the work. Neutral ink handles text, borders, surfaces and the
primary action, since contrast is what marks a button primary, and that leaves color
free to mean something. A single amber means exactly one thing: **evidence**. Citation
markers, the open source, retrieved passages. It is the hue the PDF renderer highlights
with, so a citation and the highlight it points at read as the same object. Amber
therefore never signals a warning here; failures use the semantic red.

The answer is set in a serif at roughly a 68-character measure, because it is the one
place the product shows sustained prose and it should read as a document rather than a
message. Its evidence sits below it and subordinate to it, as rows rather than cards,
each carrying the passage text inline. So *what did it conclude*, then *on what*, then
*show me the page*, in that order, with the last step only if you want it. Every
passage shows its section, its page, and a coarse three-step read of the retrieval
distance, omitted rather than invented for the channels that report no distance.

---

## Configuration

Every setting is an environment variable with a sensible default. See
[`.env.example`](.env.example) and copy it to `.env` to override:

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `WEAVIATE_HOST` / `WEAVIATE_PORT` / `WEAVIATE_GRPC_PORT` | `localhost` / `8081` / `50052` | Where the database lives |
| `WEAVIATE_COLLECTION` | `ResearchChunk` | Collection name |
| `OLLAMA_MODEL` | `qwen3:4b-instruct` | Generation and classification model |
| `OLLAMA_THINK` | `false` | Let hybrid models reason first (slower, less accurate here) |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | 384-dim sentence embeddings |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `TOP_K` | `5` | Chunks passed to the LLM |
| `RERANK_FETCH_MULTIPLIER` | `4` | Candidates fetched per final chunk |
| `HYBRID_ALPHA` | `0.3` | Hybrid weight: 0 is pure BM25, 1 is pure vector |
| `PAPER_PREFILTER_N` | `0` | Two-stage retrieval width; 0 searches every chunk flat |
| `APP_HOST` / `APP_PORT` | `127.0.0.1` / `7860` | Web app bind address |

Changing `EMBEDDING_MODEL` changes the vector dimension, so wipe and re-ingest
(`docker compose down -v`) when you do.

**Two settings that were measured rather than chosen.** `HYBRID_ALPHA` between 0 and
0.5 is indistinguishable on the golden set and quality falls above that, so 0.3 stays;
vector-only retrieval is five points worse, which is the sparse channel earning its
place on a corpus full of acronyms like SR-DPR, PLAML and CIRAL. `PAPER_PREFILTER_N`
turns on two-stage retrieval, ranking whole papers before searching chunks, and it is
off because on a library this size it loses: shortlisting to 20 of 54 papers took the
right paper from 95% to 90% and saved no time, since searching 3,900 chunks flat is
already milliseconds. It becomes the right trade when a flat scan costs real time,
which is a few hundred papers. The recall cost at each width is in `.env.example`.

---

## Project layout

```
research_rag/
  config.py              env-driven settings, single source of truth
  ingest.py              Docling conversion, heading repair, boilerplate filtering
  section_classifier.py  3-stage heading classification and query/section matching
  embedder.py            sentence-transformers wrapper
  vector_store.py        Weaviate client: schema, writes, filtered vector search
  search.py              retrieval pipeline with the section safety rails
  synthesize.py          one cited answer, citation validation, follow-up rewriting
  map_reduce.py          exhaustive per-section scan, used by the single-paper view
  reranker.py            cross-encoder reranking
  enumerate_.py          fan-out over papers for "which X across these papers"
  paper_cards.py         one structured card per paper, for corpus-level questions
  section_index.py       section summaries as a searchable second retrieval channel
  titles.py              what a paper is called: metadata, then layout, then LLM
  chunk_rules.py         what counts as a usable chunk, and what a heading is
  context.py             embeds a chunk with its title and section as context
  prefilter.py           stage one of two-stage retrieval (off by default)
  llm.py                 single entry point for Ollama calls
  pdf_viewer.py          renders a chunk back onto its PDF page
  cli.py                 ingest / search / list / sections / delete
app.py                   FastAPI server and JSON API
static/index.html        the whole interface: markup, design tokens, behavior
scripts/
  eval_models.py                   score models on the routing tasks
  eval_rag.py                      score coverage / retrieval / generation / synthesis / thematic
  eval_chat.py                     score the multi-turn and ad-hoc upload views
  fetch_arxiv.py                   pull open-access papers by pinned arXiv id
  build_paper_cards.py             backfill paper cards without re-parsing PDFs
  screenshots.py                   regenerate the README screenshots from the app
  bench.py                         stage timings and parsing quality
  backfill_titles.py               resolve paper titles without re-parsing
  repair_index.py                  fix parsing defects without re-parsing
  reembed_with_context.py          re-embed chunks with title and section
  sweep_retrieval.py               tune retrieval settings without the LLM
  launch.ps1                       one-step startup on Windows, for a shortcut
  make_icon.py                     draw docs/icon.ico
tests/
  test_section_classifier.py       cross-discipline classification tests (no LLM)
  test_retrieval_filters.py        section-filter regression tests (no LLM)
  test_synthesis.py                citation and diversification tests (no LLM)
  test_paper_lifecycle.py          add/remove keeps all three collections in step
  test_ui.py                       browser tests: citations, source panel, tabs
  golden_conversations.json        26 multi-turn turns for the conversational views
  golden_qa.json                   138 graded questions, 341 fact groups, 53 papers
docker-compose.yml       Weaviate service
```

### API

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness plus paper and chunk counts |
| `GET` | `/api/papers` | List ingested papers and their titles |
| `POST` | `/api/papers` | Upload and ingest PDFs |
| `DELETE` | `/api/papers/{filename}` | Remove a paper and its chunks |
| `POST` | `/api/ask` | One cited answer: corpus-wide, one paper, or a scratch upload |
| `POST` | `/api/chat` | Per-paper breakdown across the library |
| `GET`/`POST`/`DELETE` | `/api/scratch` | Ad-hoc uploads, isolated from the library |
| `POST` | `/api/render-chunk` | Render a chunk on its source PDF page |

---

## Data model

One Weaviate collection with self-provided vectors. Embeddings are computed locally, so
no vectorizer module is enabled on the server:

| Property | Type | Notes |
| --- | --- | --- |
| `text` | `text` | Chunk content |
| `source_file` | `text` | `FIELD` tokenization, exact filename match |
| `source_path` | `text` | Not indexed; used for PDF rendering |
| `chunk_index` | `int` | Position in the document |
| `heading` | `text` | Full breadcrumb, for example `3 Methods > 3.1 Data` |
| `section_name` | `text` | `FIELD` tokenization, exact heading match |
| `subsection_name` | `text` | `FIELD` tokenization |
| `section_type` | `text` | One of `abstract`, `introduction`, `theory`, `related_work`, `methodology`, `dataset`, `results`, `conclusion`, `general` |
| `page_numbers` | `text` | Comma-separated source pages |

The vector is computed from the chunk text with its paper title and section prefixed,
while `text` stores the passage unchanged. Retrieval matches on context plus passage;
citations and the source panel still show the passage alone.

Object UUIDs are a deterministic `uuid5` of `source_file::chunk_index`, so re-ingesting
a paper overwrites its chunks instead of duplicating them.

A paper lives in three collections: its chunks, its card and its section summaries.
Nothing but the API keeps them in step, so adding and removing one is covered by
[`tests/test_paper_lifecycle.py`](tests/test_paper_lifecycle.py). Uploading a filename
that is already indexed is refused rather than merged, and the stored PDF is left
untouched, because replacing the file while keeping the old chunks would leave the
index describing one document and the citation viewer rendering pages from another.
Delete the paper first if you want to replace it.

---

## Notes and limitations

- **A bigger model did not help, measured twice.** `qwen3:14b` scores 96% on the
  routing tasks against `qwen3:4b-instruct`'s 98%, and pointing only the answer step at
  it gives 87% of synthesis facts against the 4B's 89%. That is inside one standard
  deviation, at three times the latency. Per-step overrides exist
  (`OLLAMA_MODEL_ROUTING`, `_EXTRACT`, `_ANSWER`) so this stays testable rather than
  assumed, but the default is one model everywhere.

- **Corpus-level questions are judged from the paper cards, and a card's stated field
  can override its own contents.** Asked which papers use causal inference, both the 4B
  and the 14B answer "only the econometrics paper", even though the clinical-trial card
  lists `targeted minimum loss-based estimators`, a causal estimator, in its own methods
  line. Both models classify the paper by topic rather than by what it lists. Editing
  that one card would pass the question and teach the system nothing, so it stands as a
  known limitation.

- **Corpus-level answers read the cards in batches of twelve.** Sending all of them in
  one prompt worked at 17 papers and failed at 54: the prompt reached about 6,600 tokens
  and the model started summarizing the library instead of answering the question.
  Batching keeps recall, since every card is still read, but it costs one LLM call per
  twelve papers, so a much larger library will want a shortlisting step ahead of the
  scan.

- **Duplicate detection is by filename, not by content.** The same paper added as
  `smith2024.pdf` and `smith_2024_final.pdf` is ingested twice, and corpus-level
  questions will then count it twice. Comparing content hashes would catch
  byte-identical copies but not the same paper from two publishers, which is the case
  that actually turns up, so the check would buy less than it looks like it would.

- **A PDF whose font encoding cannot be resolved indexes as nonsense, silently.** One
  paper in the 54 came out as `(QKDQFLQJ %(57` for "Enhancing BERT": 51 chunks of it,
  stored without error, matching nothing. The paper is in the library and invisible to
  every query, and no test noticed until `bench.py --structure` was taught to look. It
  detects this by case rather than vocabulary, because checking for common English words
  flags every Turkish and Polish paper in the corpus, while the shift that produces this
  mojibake reads as roughly 100% uppercase against 5% for Turkish prose. Re-export or
  OCR the PDF; there is no recovering it afterwards.

- **Layout parsing sometimes splits a decimal point**, so `29.6%` is extracted as
  `29 . 6%`. Only 0.4% of chunks across 7 papers, but it lands on exactly the figures
  people ask for, like accuracies, perplexities and F1 scores, where an exact search
  misses and a quoted answer looks broken. Repaired at ingest, and papers ingested
  before that fix keep the artifact until re-ingested or repaired.

- **Not a general PDF chatbot.** It assumes academic papers with recognizable section
  headings. Slide decks, scanned documents without OCR, and reports with no headings
  degrade to plain semantic search.

- **Section classification uses an LLM** and is therefore fallible. The cross-validation
  and post-retrieval filters above exist to contain that, not to eliminate it.

- **Ingestion costs about 1 second per page.** Measured with `scripts/bench.py
  --ingest`: 18s for an average paper, split parse 46%, section summaries 43%, heading
  classification 9%, embedding 2%, writing 0.2%. Card building adds two more LLM calls
  on top. Retrieval afterwards is milliseconds.

- **Anonymous access is enabled** on the Weaviate container. That is fine for a local,
  single-user setup; add authentication before exposing it on a network.

- **Papers are not committed to this repository**, because published articles are
  copyrighted. `papers/` is gitignored, so supply your own PDFs.

---

## License

MIT. See [LICENSE](LICENSE).
