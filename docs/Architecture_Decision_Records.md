# Architecture Decision Records (ADRs)

- **Project Title**: Semantic Routing with LLM-Based Difficulty Classification using AnyJev
- **Project ID**: 17
- **Group ID**: 16  
- **Group Members**: Venna Sai Venkata Manohar (`cs24mtech15012`)
---

## ADR Index

| ADR ID | Decision Title | Decision Date | Status |
| :--- | :--- | :---: | :---: |
| **ADR-000** | **Mandatory Methodology Reaffirmation: Training-Free Prefill-Only Semantic Routing** | 2026-10-09 | **Accepted** | 
| **ADR-001** | **Framework Selection: Training-Free AnyJev vs Fine-Tuned Kev vs Raw Readout SemIf** | 2026-09 | **Accepted** |
| **ADR-002** | **Context Budget Guard: Deterministic Hierarchical Trimming** | 2026-10 | **Accepted** |

---

## ADR-000 — Mandatory Methodology Reaffirmation: Training-Free Prefill-Only Semantic Routing

### Date, status & scope
* **Milestone Review Date**: 9 October 2026
* **Status**: **Accepted**
* **Scope**: Overall project methodology

### Context & decision drivers
As per the CS5903 course milestone evaluation framework, project teams were required to conduct an empirical review of their working methodology by 9 October 2026, 11:59 PM IST. The review evaluated whether empirical evidence gathered during preliminary dev runs warranted pivoting the core architectural approach.
* **Core Problem**: Can a frozen LLM (`Qwen/Qwen3-4B`) classify task difficulty reliably and produce calibrated routing probabilities using zero-generation prefill logit readouts, or is supervised parameter fine-tuning (SFT) necessary?
* **Decision Drivers & Constraints**:
  1. *Hardware Budget*: Development constrained to Google Colab single-GPU instances (1x NVIDIA Tesla T4, 16 GB VRAM).
  2. *Domain Generalizability*: The router must adapt zero-shot across distinct enterprise domains (bug fixing vs. insurance claims) by altering prompt rubrics without dataset retraining.
  3. *Calibration Guarantee*: Routing decisions directly dictate CostOps resource allocation.

### Options considered
1. **Option 1**: Prefill-only, training-free LLM-based classification using AnyJev (L0 cyclic option permutation + prior removal; L1 temperature scaling).
2. **Option 2**: Pivot to supervised parameter fine-tuning (e.g., LoRA) on domain-specific data.

### Decision & rationale
**Selected Option**: **Option 1**.
* On the 100-item SWE-bench Verified dev split, the uncalibrated Raw baseline suffered from severe majority-class collapse (predicting 89% Medium, with 0% Hard recall).
* L0 cyclic permutation successfully broke this collapse, increasing dev Macro-F1 from 0.346 to 0.379 (+9.5% relative) and recovering minority class boundaries with zero weight updates.
* L1 temperature fitting converged stably on 200 training examples ($T \approx 6.812$), driving dev ECE down from 0.4239 to 0.1046 (a 75.3% reduction).
* Because these were achieved on the frozen dev benchmark without retraining, the zero-generation, training-free AnyJev methodology was reaffirmed in full.

### Consequences & trade-offs
* **Benefits**: Zero training compute overhead.
* **Costs & Trade-Offs**: Requires evaluating $K = 3$ cyclic rotations per request; slight latency penalty over single-pass readout.
* **Impact on Milestones**: All original milestone commitments (M1–M6) remained on track.

### Evidence & validation
* `results/runs/Qwen3-4B_L0_dev.summary.json` and `results/runs/Qwen3-4B_L1_dev.summary.json`.

### Revisit trigger
* A drop in final held-out test Macro-F1 below the frozen acceptance threshold of 0.35, or failure of L1 temperature scaling to reduce test ECE below 0.10.

---

## ADR-001 — Framework Selection: Training-Free AnyJev vs Fine-Tuned Kev vs Raw Readout SemIf

### Date, status & scope
* **Decision Date**: September 2026
* **Status**: **Accepted**
* **Scope**: AnyJev Decision Engine

### Context & decision drivers
To build a semantic difficulty router, we required an open-source framework capable of turning next-token logits into structured, typed decision probabilities over discrete ordinal classes (`{Easy, Medium, Hard}`).
* **Decision Drivers**:
  1. *Model-Agnostic*: Ability to run on any off-the-shelf open-weights checkpoint without custom architectures.
  2. *Position Invariance*: Mitigation of prompt option-order recency bias.
  3. *Probabilistic Trust*: Native support for calibration.

### Options considered
| Criteria | Option 1: Kev Framework | Option 2: SemIf (formerly OpenJev) | Option 3: AnyJev v0.2.0 (Selected) |
| :--- | :--- | :--- | :--- |
| **Model Coupling** | Requires custom fine-tuned checkpoints (0.8B–27B). | Works with frozen base models. | **Works with any off-the-shelf causal LLM.** |
| **Position Bias** | Unmitigated; option ordering affects output. | Unmitigated; reads raw unpermuted logits. | **Mathematically eliminated via cyclic rotation averaging.** |
| **Calibration** | Fixed at training time. | Optional empirical scaling. | **L0 (rotation), L1 (temperature), L2 (linear head).** |

### Decision & rationale
**Selected Option**: **Option 3: AnyJev (`anyjev[hf]==0.2.0`)**.
* AnyJev is the only open framework that provides an algebraic guarantee of position invariance through cyclic permutation averaging while eliminating prior bias without altering model parameters.

### Consequences & trade-offs
* **Benefits**: True zero-shot deployment.
* **Costs & Risks**: AnyJev evaluates $K=3$ shifts at L0, consuming additional GPU FLOPs.

### Evidence & validation
* `router/engine.py`

### Revisit trigger
* Any breaking upstream changes in the AnyJev core API.

---

## ADR-002 — Context Budget Guard: Deterministic Hierarchical Trimming

### Date, status & scope
* **Decision Date**: October 2026.
* **Status**: **Accepted**.
* **Scope**: Context Budget Guard

### Context & decision drivers
Real-world software engineering requests contain extensive code repositories, discussion comments, and stack traces that can easily exceed GPU memory limits.
* **Decision Drivers**:
  1. *Memory Safety*: Strict enforcement of a 12,000 token context budget ceiling.

### Options considered
1. **Option 1: Prompt Compression**: Uses a smaller language model to score token perplexity and compress prompts by pruning low-information tokens.
2. **Option 2: Deterministic Hierarchical Context Trimming (Selected)**: Categorizes incoming request sections by kind and executes deterministic head/tail trimming in reverse order of importance.
3. **Option 3: Hard Tail Truncation**: Blindly slices the prompt at token index 12,000.

### Decision & rationale
**Selected Option**: **Option 2: Deterministic Hierarchical Context Trimming**.
* **Rationale**: LLMLingua adds latency to compress context, which directly violates low-latency routing goal.
* In contrast, deterministic hierarchical trimming executes in $<1$ ms, preserves verbatim code syntax, and enforces a strict priority hierarchy.

### Consequences & trade-offs
* **Benefits**: Low latency.
* **Costs & Trade-Offs**: Truncation of lower-priority auxiliary logs if prompts exceed 12,000 tokens. However, profiling across all 500 SWE-bench Verified instances showed a maximum token length of 8,701 tokens, resulting in an truncation rate of exactly 0.0%.

### Evidence & validation
* `scripts/max_context.py`

### Revisit trigger
* Deployment to production workloads where incoming requests exceed 32,000 tokens and cannot be resolved without semantic prompt summarization.
