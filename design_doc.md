# ARIA Design Document
**Adaptive Research Intelligence Architecture**

---

## 1. Agent Topology

### Why This Structure?

ARIA uses a **hub-and-spoke topology** with a single orchestrator coordinating four specialist agents. The orchestrator is the only agent that reasons about the full query and task graph; specialists are stateless within a single invocation and receive minimal context budgets.

**Alternative considered: Peer-to-peer (all agents share a message bus)**
Rejected because it creates implicit coupling — agents would need to monitor for relevant messages and handle partial state. Debugging becomes difficult because no single agent holds the full execution trace. A hub-and-spoke model makes control flow explicit: the orchestrator is the single source of truth for what's been tried and why.

**Alternative considered: Single-agent chain-of-thought (one large prompt)**
Rejected outright. The rubric explicitly forbids monolithic LLM calls. More importantly, a single call cannot parallelize sub-questions, cannot apply specialized prompts per role, and cannot replan at the sub-task level — it's an all-or-nothing retry. Multi-agent granularity enables targeted replanning.

**Why four specialist types (not three)?**
The minimum is three (Retrieval, Critique, Synthesis), but ARIA adds a dedicated `QualityEvaluator` rather than bolting evaluation onto the synthesis prompt. This is intentional: conflating generation and evaluation in one prompt creates a sycophantic evaluator that rates its own output highly. Separation enforces independence of judgment, which is a core principle of LLM-as-Judge (Zheng et al., 2023).

### Agent Boundaries

A deliberate design constraint: **the orchestrator must not do retrieval work, and retrievers must not synthesize**. This is enforced structurally (agents have distinct typed inputs/outputs) rather than by convention. When the orchestrator needs to decide how to replan a failed task, it inspects `QualityScore` dimensions — it does not re-read raw source documents. This prevents the "God agent" anti-pattern where the orchestrator gradually absorbs all logic.

---

## 2. Conflict Resolution Strategy

### How the Critique Agent Decides Credibility

Conflict detection is a two-stage process:

1. **Claim extraction + LLM contradiction scan** — the agent presents all chunk summaries to a lightweight LLM (`claude-haiku-4-5`) and asks it to identify pairs that are *directly contradictory* (not just different framings). This avoids the false-positive problem of embedding-only approaches (two synonymous claims can have low cosine similarity; two thematically related but non-contradictory claims can be near-identical in embedding space).

2. **Structured credibility scoring** — for each detected conflict, the agent scores both sources on a four-dimensional rubric:
   - **Recency (0–3)**: More recent publication → higher score. Critical in fast-moving fields like AI.
   - **Authority (0–3)**: Academic paper > official documentation > reputable news > blog. Domain classification is LLM-based using URL and title heuristics.
   - **Specificity (0–2)**: Quantitative claims with evidence score higher than vague assertions.
   - **Corroboration (0–2)**: Claims supported by multiple independent sources score higher.

**Why not just take the highest-ranked source by authority alone?**
A 2024 blog post citing an experiment can be more credible than a 2019 paper on a specific numerical claim. The rubric captures this by weighting recency independently from authority. Total score range is 0–10.

### Failure Modes

1. **Both sources are equally credible** — the agent marks the resolution as `unresolved: true` and `winner: "neither"`. Both claims are preserved in the synthesis prompt with an explicit note. The synthesizer is instructed to present the disagreement transparently rather than pick arbitrarily.

2. **Conflict detection false positives** — an LLM may flag stylistic differences as contradictions. Mitigation: the detection prompt requires *direct* contradiction (not "different framing") and caps at 5 flagged pairs per chunk set.

3. **Conflict detection false negatives** — subtle numerical conflicts (e.g., different benchmark results for the same model) may not be caught. Mitigation: the evaluator's `factual_grounding` dimension penalizes sections with inconsistent numbers.

4. **Circular credibility** — Source A cites Source B which cites Source A. The corroboration dimension degrades gracefully because the Critique Agent doesn't track citation graphs; it scores based on presence of supporting claims in the retrieved corpus, which will double-count if two sources share the same lineage. This is a known limitation.

---

## 3. Memory Design

### What Consistency Guarantees Does Shared Memory Provide?

ARIA's `SharedWorkingMemory` uses `asyncio.Lock` for all write operations and lock-free reads for immutable data (chunk text doesn't change after storage). This provides **sequential consistency within a single async execution loop**: writes by one coroutine are visible to all subsequent readers in the same event loop.

**What it does NOT provide:**
- Multi-process safety (two Python processes cannot share the same `SharedWorkingMemory` instance). For multi-process deployments, the JSON checkpoint file would need to be replaced with a proper key-value store (Redis, DynamoDB).
- Atomic compound operations across multiple write methods. If the process crashes between `store_chunks()` and `store_section()`, the memory is in a partially-written state. The `persist()` call is intentionally a full snapshot rather than a write-ahead log, accepting that recovery may replay some work.

### Redundancy Detection

Each chunk is identified by a deterministic hash of `sha256(url + "::" + text[:200])[:16]`. Before any chunk is stored, `is_known(chunk_id)` is checked. This prevents:
- The same web page being fetched twice for different sub-questions that happen to retrieve the same source
- Repeated iterative retrieval passes inflating the context with duplicate evidence

**Trade-off:** The 16-character hash could collide, but at the scale of a single research run (hundreds of chunks), collision probability is negligible (~2^-64 given SHA-256 properties).

### What Breaks Under Concurrency?

The primary risk is **lost updates** if two coroutines simultaneously check `is_known()` before either writes. The `asyncio.Lock` on `store_chunks()` prevents this: the check and insertion happen atomically within the lock. However, the design uses cooperative scheduling (asyncio, not threading), so there is no preemption risk — the lock is a correctness guarantee against interleaved coroutine yields, not a defense against true parallelism.

If ARIA were extended to use `multiprocessing` (e.g., for CPU-bound reranking), a process-safe queue or external store would be required.

---

## 4. Termination & Quality

### How Does ARIA Avoid Running Forever or Stopping Too Early?

ARIA implements a **three-layer termination policy**, inspired by Constitutional AI's (Bai et al., 2022) principle of explicit value enforcement:

**Layer 1: Per-task hard cap** — Each sub-task has a `replan_count` ceiling (default: 3). When reached, the orchestrator unconditionally skips the task and proceeds. This is the absolute backstop.

**Layer 2: Quality gate** — If a section's aggregate quality score exceeds the threshold (default: 6.5/10), the task is marked complete immediately regardless of remaining replan cycles. This is the normal exit path.

**Layer 3: Global timeout** — A configurable wall-clock timeout (default: 10 minutes) terminates the run and writes a checkpoint. This handles adversarial inputs (e.g., a query that generates tasks that always produce low-confidence retrieval).

**Stopping too early:** The risk is a task that scores just above 6.5 but has obvious gaps. Mitigation: the `coverage` dimension has the second-highest weight (0.30), and the `improvement_suggestions` from a passing-but-marginal evaluation are logged in the report metadata so a human reviewer can see the evaluator's concerns.

**The replanning decision is non-trivial:** The orchestrator inspects *which* dimension failed, not just the aggregate:
- Low `citation_quality` → `retry` with expanded query (add more sources)
- Low `coverage` → `rephrase` the question to broaden scope  
- Low `coherence` → `retry` (likely conflicting sources that the Critique Agent should resolve more aggressively on the next pass)
- Hard cap reached → `skip`

This targeted reasoning prevents the common failure mode of "retry with the same query" — which, without a changed strategy, tends to produce nearly identical low-quality results.

---

## 5. Research Connections

### Retrieval: Beyond Naive RAG

ARIA's retrieval strategy combines three techniques:

**HyDE (Hypothetical Document Embeddings, Gao et al., 2022):** Instead of embedding the raw question, the RetrievalAgent generates a hypothetical answer and embeds that. The hypothesis more closely resembles what a relevant document looks like (same vocabulary, same structure), producing better semantic matches. This is especially effective for technical questions where the question uses different terminology than the documents.

**Iterative retrieval (inspired by Self-RAG, Asai et al., 2023):** If the initial retrieval confidence is below 0.55, ARIA extracts key terms from the retrieved chunks and issues a follow-up search. Unlike Self-RAG's token-level reflection tokens, ARIA uses aggregate chunk confidence as the reflection signal — simpler to implement, more stable under the Anthropic API's output format constraints.

**Cross-encoder reranking:** After bi-encoder retrieval from ChromaDB, candidates are rescored with `ms-marco-MiniLM-L-6-v2`, a cross-encoder that jointly encodes query + document and produces higher-precision relevance scores. Bi-encoders (FAISS/ChromaDB) are fast but imprecise; cross-encoders are accurate but slow; the two-stage approach gives both.

### Evaluation: LLM-as-Judge

ARIA's `QualityEvaluator` follows Zheng et al. (2023) LLM-as-Judge. Key design decisions:
- Use `claude-haiku-4-5` (faster, cheaper) rather than the main model to avoid self-congratulatory evaluation
- Structured JSON output with explicit rubric criteria (factual/coverage/coherence/citation) to reduce variance
- `improvement_suggestions` feed directly into the replanning prompt, closing the reflection loop

**Known limitation (Zheng et al.):** LLM judges have positional bias and may give higher scores to longer sections. Mitigation: the rubric explicitly penalizes "vague claims" in the specificity dimension, discouraging padding.

### Orchestration: ReAct Pattern

The orchestrator follows the ReAct (Yao et al., 2022) pattern of interleaved reasoning and action:
- **Reason:** inspect quality scores → diagnose failure dimension → select replan strategy
- **Act:** dispatch agent with revised question or strategy

Unlike vanilla ReAct which reasons about tool calls sequentially, ARIA's DAG executor reasons at task-graph level — multiple actions (agent invocations) are dispatched in parallel when dependencies allow.

### Report Generation: STORM Influence

STORM (Shao et al., 2024) generates Wikipedia-scale articles by first building an outline via multi-agent perspective-seeking, then grounding each section via retrieval. ARIA adapts this at the sub-task level: each sub-task produces one section, and the SynthesisAgent is given the section's expected scope (the sub-question) as a structural constraint, preventing the common problem of LLM synthesis that meanders off-topic.

---

## 6. What I'd Do With Two More Weeks

### 1. Self-Improving Evaluation Calibration
The current evaluator uses static rubric weights (0.35/0.30/0.20/0.15). With two more weeks, I'd collect 50-100 human-rated sections and fine-tune the weights using a simple regression, or prompt-tune the evaluator against a held-out test set of ARIA outputs rated by humans. The goal is to reduce the gap between LLM judge scores and human preferences.

**Concrete implementation:** Generate 50 runs of the default test query, have team members score a random sample on each dimension, compute Spearman correlation between LLM scores and human scores per dimension, and adjust weights proportionally.

### 2. Citation Graph Conflict Detection
Currently, ARIA detects conflicts between claims in the retrieved corpus but doesn't track *citation lineage*. Source A citing Source B citing Source C doesn't create three independent data points — it's one. Building a citation graph (URL → outbound citation URLs via HTTP HEAD + link extraction) would enable ARIA to discount clusters of mutually-citing sources and give higher corroboration scores to claims supported by genuinely independent sources.

### 3. Adaptive Sub-Task Granularity
The current decomposer produces 3-6 sub-tasks of fixed granularity. Some topics have deep sub-structure (e.g., "transformer limitations" has 10+ distinct failure modes); others are naturally flat. With two more weeks, I'd implement a **depth-first DAG refinement** pass: after the first retrieval attempt, if coverage is consistently low across a sub-task, the orchestrator breaks it into 2-3 more specific sub-tasks (query decomposition inspired by Decomposed Prompting, Khot et al., 2022). The depth budget prevents infinite recursion.

---

## 7. Tradeoffs Acknowledged

| Decision | Upside | Downside |
|---|---|---|
| asyncio + single process | Simple state management, no IPC | Cannot parallelize CPU-bound reranking across cores |
| DuckDuckGo (no API key) | Zero-cost, no rate limits | Lower quality than Tavily; may be rate-limited at high volume |
| Local sentence-transformers | No embedding API cost, persistent | Cold start latency (~3s for model load) |
| Per-task checkpoint (JSON) | Simple, human-readable | Not atomic; crash mid-persist can corrupt state |
| HyDE hypothesis quality | Better retrieval recall | Hypothesis errors propagate to retrieval; mitigated by reranking |
| `claude-haiku-4-5` for evaluation | 10x cheaper than Sonnet | May produce less calibrated quality scores |
