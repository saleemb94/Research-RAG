# Evaluation

Full methodology for the golden sets and the scorer. The headline numbers and the short
version live in the [README](../README.md#evaluation); this is the reasoning behind them,
kept out of the README so that the first thing a reader meets is the project rather than
its test plan.

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

**Passage-level retrieval** is the fourth set, in `tests/golden_passages.json`: 40
questions with 93 hand-labeled gold passages, graded 2 where the passage fully answers
and 1 where it supports. Until this existed, retrieval was only ever scored at paper
granularity, which cannot tell returning the paragraph that states the result from
returning the same paper's related work. On a corpus where most papers are about
retrieval, that is most of the question.

```bash
python scripts/eval_rag.py --passages --at-k 10
```

| | |
| --- | --- |
| Recall@10 | 0.711 |
| MRR@10 | 0.759 |
| MAP@10 | 0.582 |
| nDCG@10 | 0.664 |
| Gold passage ranked first | 26/38 (68%) |
| No gold passage returned | 3/38 (8%) |

Two design decisions carry the weight. Gold passages are pinned by a **verbatim text
anchor** rather than a chunk id, and every anchor is verified to resolve to exactly one
chunk in its paper: chunk ids break the moment chunking changes, and it has changed here,
where a parsing repair removed 678 chunks. And the labels were made by **reading the
paper's chunks, never by running the retriever**. Labeling gold as whatever the system
returns measures the retriever against itself and can only score high.

The split underneath the average is the finding:

| | questions | recall | nDCG |
| --- | --- | --- | --- |
| One gold passage | 17 | **0.941** | 0.813 |
| Two or more gold passages | 21 | **0.535** | 0.549 |

That gap was mostly not a retrieval failure, and finding out what it was is what this
set was worth building for. The synthesis path capped sources at two per paper,
deliberately, so that one densely-worded paper could not take every slot in a
corpus-wide answer. A question whose evidence is spread over five passages of a single
paper therefore *could not* score above two, whatever the retriever did, and 5 of the 11
questions needing three or more passages sat exactly at that ceiling.

So the defect was never that retrieval stops early. It was that **one cap served two
question types that want opposite things.** Spreading evidence across papers is right
for "which datasets are used across these papers", which is why the cap was added: the
eight best chunks really do come from the two papers that discuss datasets most densely.
It is wrong for "what accuracy did BiLSTM, CNN and GRU each reach", where the whole
answer is inside one paper. The cap applied whenever a question was not explicitly
scoped to a paper, which does not tell those apart.

`cap_for()` now keys the cap off the router's classification: focused questions may take
four passages from one paper, corpus and enumerate questions still two. Four rather than
unlimited, because a focused classification is sometimes wrong and one paper taking all
eight slots is the failure the diversification exists to prevent.

Three runs per setting, since the run-to-run spread here is wide enough to manufacture
whichever result you were hoping for:

| | cap 2 | cap 4 | |
| --- | --- | --- | --- |
| Recall@10 | 0.662 [0.656-0.665] | **0.711** [0.709-0.717] | +0.049 |
| MAP@10 | 0.556 [0.551-0.564] | **0.582** [0.580-0.585] | +0.027 |
| nDCG@10 | 0.635 [0.630-0.643] | **0.664** [0.661-0.667] | +0.028 |
| MRR@10 | 0.759 [0.752-0.769] | 0.759 [0.755-0.764] | 0.000 |
| Ranked first | 0.690 [0.676-0.711] | 0.679 [0.676-0.684] | overlapping |

The three that moved have non-overlapping ranges. MRR and the rank-1 rate do not move,
which is the shape the change predicts: letting more gold through further down a list
should not disturb its top.

The cap protects breadth on corpus-wide questions, so the obvious way for this to be a
bad trade is a breadth regression elsewhere. Measured over three runs each, thematic
facts went 68% to 66% and breadth 8.3 to 8.0 of 16, synthesis facts 76% to 75% and
papers cited 70% to 67%; every one of those overlaps its baseline range. A first single
run had shown breadth at 7 of 16, which looked like exactly the regression to fear and
was noise.

One caveat makes these numbers floors rather than estimates: labeling is **known to be
incomplete**. Only chunks carrying a golden fact were reviewed, so a passage that answers
in paraphrase can be unlabeled and scores as a miss.

The scorer also **excludes rather than zeroes** any question the router sends to the
paper cards, since those never reach passage retrieval and booking them as retrieval
failures would hide a routing decision inside a retrieval number. When the router
sampled, this fired on two or three questions and varied between runs. With routing
greedy it fires on none, and the exclusion is now a guard against a case that no longer
occurs rather than a live adjustment.

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

