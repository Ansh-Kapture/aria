# ARIA — Adaptive Research Intelligence Architecture

A production-quality multi-agent system for automated deep research. ARIA decomposes a complex query into a dependency graph, dispatches specialist agents in parallel, resolves conflicting evidence, and produces a structured Markdown report with inline citations and LLM-evaluated quality scores.

## Architecture

```
Query → OrchestratorAgent
           ├─ Decompose → DAG of SubTasks
           └─ DAGExecutor (topological, async)
                 └─ Per task:
                       RetrievalAgent (HyDE + reranking + web search)
                         ↓
                       CritiqueAgent (conflict detection + resolution)
                         ↓
                       SynthesisAgent (merge + deduplicate + cite)
                         ↓
                       QualityEvaluator (LLM-as-judge → replan if < 6.5)
                         ↓ (if needed, max 3 cycles)
                       CitationVerifier (spot-check bonus)
           └─ ReportGenerator → final_report.md
```

## Components

| Component | File | Role |
|-----------|------|------|
| OrchestratorAgent | `aria/agents/orchestrator.py` | DAG decomposition, replanning, termination |
| RetrievalAgent | `aria/agents/retrieval.py` | HyDE + web search + cross-encoder reranking |
| CritiqueAgent | `aria/agents/critique.py` | Conflict detection + credibility-scored resolution |
| SynthesisAgent | `aria/agents/synthesis.py` | Merge evidence, deduplicate, assign citations |
| QualityEvaluator | `aria/agents/evaluator.py` | LLM-as-judge (factual/coverage/coherence/citation) |
| CitationVerifier | `aria/agents/citation_verifier.py` | Spot-check citation accuracy (bonus) |
| SharedWorkingMemory | `aria/memory/state.py` | Thread-safe shared state + JSON persistence |
| DAGExecutor | `aria/dag/executor.py` | Topological async execution, dep-aware context |
| ReportGenerator | `aria/report/generator.py` | Structured Markdown report assembly |

## Setup

```bash
# 1. Clone / unzip the project
cd aria

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY

# 5. Run
python main.py "What are the current limitations of transformer-based long-context reasoning, and what architectural directions show the most promise for solving them?"
```

## Usage

```bash
# Basic run
python main.py "Your research query"

# With options
python main.py "Your query" \
  --output output/report.md \
  --quality-threshold 7.0 \
  --stream

# Resume an interrupted run
python main.py "Same query" --state-file state/checkpoint.json

# Start fresh (ignore checkpoint)
python main.py "Your query" --no-resume

# Verbose logging
python main.py "Your query" --log-level DEBUG
```

## Run Tests

```bash
pytest tests/ -v
```

## Output

The report at `output/report.md` contains:
- **Executive Summary** — key findings, uncertainty notes, confidence level
- **Per-section findings** — quality scores (factual/coverage/coherence/citation), inline `[N]` citations
- **Conflict Resolutions** — table of detected conflicts with winner and credibility scores
- **Bibliography** — deduplicated, globally-numbered citations
- **Evaluation Metadata** — full JSON quality scores per section

## Hard Constraints Met

| Constraint | Implementation |
|---|---|
| No monolithic LLM call | 6 distinct agents, each with own prompt + schema |
| ≥ 2 external tools | Web search (DuckDuckGo/Tavily) + ChromaDB vector store |
| Max 3 replan cycles | `settings.max_replan_cycles = 3` enforced in orchestrator |
| Typed inter-agent schemas | Pydantic v2 models in `aria/schemas/` |
| Quality scores always logged | `EvaluationResult` stored in memory + exposed in report |

## Bonus Features Implemented

- **Adaptive context budgeting** — complexity classifier allocates token budgets (2k/6k/12k)
- **Multi-hop reasoning** — true DAG execution; downstream tasks receive upstream answers
- **Citation verification** — `CitationVerifier` spot-checks citations against source snippets
- **Streaming output** — `--stream` flag emits sections as they complete
- **Crash recovery** — `--state-file` persists and resumes interrupted runs
