# Temporal Encoding & Trust Research — Full Timeline

## Origins: Ŧrust in Cognicism (2016–2017)

Ŧrust — attention over source, time, and content — has been a core concept in the Cognicism framework since 2016-2017. In Cognicism, Ŧrust is a credibility signal: sources earn influence by making claims that age well, and lose it when their predictions fail. It was always intended to be mechanized — a system where LLMs mediate epistemic conflict by tracking who sees clearly in which domains over time. The timestamp encoding research that begins below was built from the start as a low-dimensional proof of this concept: before you can build Ŧrust, you need a neural network that understands *when*.

## Ŧrust as a Primitive

Ŧrust is not a technique — it is a primitive. The mechanism is the same regardless of what flows through it: content embedding, source embedding, and timestamp embedding form the triplet. For time series, content is a value embedding. For language, content is a word embedding. For images, content is a patch embedding. The source is whoever generated the content — who predicted it, who said it, who sensed it, what instrument reported it. The timestamp is when. A grid of temperature sensors where each sensor's reliability follows a rolling seasonal pattern is the same problem as a set of political forecasters whose accuracy waxes and wanes with election cycles. The data type changes; the trust mechanism does not.

The synthetic scalar experiment in this directory is the lowest-dimensional proof of this general mechanism. Three (or eight) sources make scalar predictions. Exactly one is the expert at any given time, cycling sinusoidally. The model receives predictions and a timestamp and must output the expert's value. To do this, it must learn *who* to trust *when* — and neither dimension alone is sufficient. This is the irreducible core of Ŧrust: you need all three legs of the triplet.

## Why Ŧrust Is Revolutionary

Every domain has the same structure: multiple sources of information, each with varying reliability, and you need to combine them into something closer to truth than any individual source.

### The Fundamental Problem Is Noise

No single source sees clearly all the time. A sensor drifts. An analyst has blind spots. A model overfits. A witness misremembers. A newspaper has an editorial slant. The signal is always mixed with noise, and the signal-to-noise ratio *changes* — it depends on who's speaking, when, and about what.

The traditional approach is to either pick one source you trust (fragile — what happens when your oracle is wrong?) or average them (wasteful — the expert's signal gets diluted by everyone else's noise). Both approaches are static. They don't adapt to the fact that expertise is contextual and temporal.

### Dynamic, Contextual Source Weighting

Ŧrust is dynamic, contextual source weighting. At every moment, for every claim, the mechanism asks: given who said this, when they said it, and what they said — how much should this contribute to my belief? The attention weights over sources ARE the trust scores. They sum to 1. They shift in real time. The output is a weighted combination where the weights reflect learned reliability.

### A General Epistemology

Science works by weighing evidence from multiple experiments and researchers. Courts weigh testimony from multiple witnesses. Markets aggregate information from multiple traders. Medicine synthesizes findings from multiple studies. Intelligence agencies fuse reports from multiple assets. In every case, the core operation is: *who do I listen to, how much, right now, on this topic?* Ŧrust mechanizes that operation.

### The Generative Representation Is More Than the Sum of Its Parts

When you attend over sources, you don't just pick the best one — you form a *synthesis* that no individual source could produce alone. Source A might be right about the direction. Source B might be right about the magnitude. Source C might be noisy overall but catches edge cases the others miss. The attention-weighted combination extracts the signal from each and composes something richer than any input. This is how ensemble methods work in ML, how juries deliberate, how scientific consensus forms — but Ŧrust does it differentiably, adaptively, and with an interpretable receipt of who contributed what.

### Context Makes It Non-Trivial

A climate model is brilliant about temperature and useless about stock prices. A cardiologist is expert on hearts and not on knees. A sensor calibrated for daytime readings drifts at night. Static trust — "source X is reliable" — misses this entirely. Ŧrust conditions on content: *what is being claimed?* A source's weight shifts depending on whether the topic is in their domain of competence. This is how humans naturally evaluate information — you trust your mechanic about your car and your doctor about your health — but no system has formalized it as a differentiable primitive.

### Time Makes It Adaptive

Expertise isn't just domain-specific, it's temporal. An economic model trained on boom times fails in recessions. A political analyst is sharp during election cycles and complacent between them. A satellite sensor degrades over years. A new researcher publishes breakthrough work early in their career then coasts. Ŧrust conditions on time: the same source, on the same topic, gets different weight depending on when. The timestamp embedding captures the temporal patterns — seasonal cycles, long-term drift, regime changes — and attention uses them to modulate trust in real time.

### The Field of Sources Is the World

Think about what this looks like at scale. Every person who makes a public claim is a source. Every sensor in an IoT network. Every model in an ensemble. Every news outlet. Every scientific paper. Every prediction market participant. The "field" is a high-dimensional space where each source has a trajectory through time, a domain of competence, and a history of signal vs noise. Ŧrust is a learned map over this field: given a query (time, topic), it tells you where the signal is and where the noise is, *right now*.

### The Interpretability Is the Receipt

Unlike a black-box ensemble or a neural network that fuses inputs opaquely, Ŧrust produces readable attention weights. You can ask: "why did the model believe this?" and the answer is: "it weighted source 3 at 0.7 and source 5 at 0.2, because at this time, on this topic, those sources have historically been reliable." The trust scores are not a post-hoc explanation — they ARE the mechanism. The thing that produces the output is the same thing that explains it.

### It Inverts the Information Problem

Most systems try to filter noise before processing. Ŧrust does the opposite — it takes in everything, noise and all, and learns to weight it. You don't need to curate your sources. You don't need to pre-judge who's reliable. You feed in the full field of claims and let the mechanism learn who sees clearly about what, when. Bad sources get low weight. Good sources get high weight. Sources that are good sometimes and bad other times get time-varying weight. The system self-organizes around signal.

This is why the timestamp embedding matters so much. Time is one third of the triplet. If the model can't represent "when" richly enough, it can't learn the temporal dynamics of trust — and every real-world trust pattern has temporal dynamics. Getting the timestamp embedding right is the foundation that everything else builds on.

## The Timestamp Embedding: Non-Negotiable Design Principles

**One embedding. One representation of time. This is the design.**

A timestamp embedding is like a word embedding or a positional embedding. "Cat" gets one vector. Position 5 gets one vector. Unix timestamp 1640000000 gets one vector. The transformer's downstream layers — attention, MLPs — extract what they need from that single representation. You do not have two word embeddings for "cat" depending on whether it appears as subject or object. You do not have two positional embeddings for position 5 depending on whether it's in a key or a query. You do not have two timestamp embeddings.

If experiments show two encodings outperforming one, the conclusion is not "two encodings are needed." The conclusion is "something is broken in training that prevents one encoding from working." The research task is to find and fix that training problem.

**The input is seconds from Unix epoch. This is correct.**

Seconds from epoch is a single number that uniquely identifies every moment in time. Everything temporal is derivable from this one number through periodic functions at different frequencies:

- **Periodic features at every scale.** Seconds (heartbeats, CPU cycles), minutes (traffic lights), hours (circadian rhythm, market hours), days (sleep/wake), weeks (weekday/weekend), months (billing cycles), seasons (weather, agriculture), years (budgets, tax cycles), decades (economic cycles). The learnable frequency bands span 13 orders of magnitude precisely to capture whichever scale matters for a given problem.
- **Linear features.** Time moves forward. Things trend — populations grow, prices inflate. The trend pathway captures directional drift that accumulates over the span of the data.
- **Phase relationships.** Two sources might share an annual cycle but peak in different months — same frequency, different phase. Sin and cos at each frequency encode phase. Attention between entities compares phases.
- **Time differences.** Cross-attention between two timestamp embeddings naturally computes relative time. The dot product of two periodic embeddings at the same frequency is a function of their time difference. "3 days ago" vs "6 months ago" emerges from absolute timestamps through attention — no explicit subtraction needed.
- **Multi-scale interactions.** "Tuesday mornings in December" is hour × day-of-week × month. The embedding provides raw periodic components at each scale. Attention layers learn cross-scale interactions.
- **Aperiodic structure.** Holidays, elections, one-off events. Sharp features decompose as sums of many frequencies (Fourier). The gating mechanism blends periodic and trend components adaptively.

No manual feature engineering. No "is it a weekday." No binning into hours. One number. The network learns what matters.

**It generalizes to any time-dependent problem.**

The embedding doesn't know about stock markets or weather or source expertise. It decomposes a timestamp into periodic and trend components. A downstream transformer combines that with domain-specific features through attention. Any problem where events happen at timestamps and there are patterns in when they happen can use the same embedding. The data type changes; the time representation does not.

**Trust requires all three: source × time × content.**

A climate scientist is an expert on climate predictions, not healthcare predictions. Trust is not just "source X is reliable at time T" — it is "source X is reliable on topic Y at time T." The timestamp embedding provides the temporal component. Attention over the full triplet — source identity, time representation, and content — produces the trust score. The embedding must be rich enough for downstream attention to compute these three-way interactions, but the interactions themselves are attention's job, not the embedding's job.

## The Development Story

### The Idea (2016–2017)

The concept of Ŧrust in Cognicism dates to 2016–2017, long before any code was written. The idea was always that credibility should be mechanized — sources earn influence by making claims that age well, and lose it when their predictions fail. Ŧrust was formalized as a derivation of transformer attention operating over three embedding spaces: source, time, and content. The technical white papers (John Ash, 2024) describe a gated combination where each signal — who said it, when they said it, what they said — is weighted by learned gates that examine all three dimensions simultaneously. The attention weights over sources at a given time ARE the trust scores: both the mechanism that produces the output and the readable receipt of which sources contributed what, when.

The critical insight was that this mechanism is domain-agnostic. The content embedding could be a word embedding (language), a patch embedding (images), a projected scalar (time series), or a sensor reading (IoT). The source could be a forecaster, an author, a sensor, an instrument. The timestamp encoding captures *when* at every scale. The architecture doesn't change — only the content dimensionality does.

### The Timestamp Encoding Problem (Aug 2023)

Before you can build Ŧrust, a neural network needs to understand *when*. A Unix timestamp — seconds since January 1, 1970 — already contains temporal information at every scale. The number 1,577,836,800 encodes that it is midnight, that it is a Wednesday, that it is January, that it is 2020. The question was whether a neural network could learn to extract the periodicities that matter for a given task, directly from this single scalar, without being told what to look for.

The technical work began on August 11, 2023, with a burst of experimentation that produced six files in a single day. The starting points were existing approaches:

**Time2Vec** (Kazemi et al., 2019) was the first reference implementation (`time2vec.py`). Time2Vec represents time as k+1 learned features: one linear (non-periodic) plus k sinusoidal (periodic), each with learnable frequency and phase. It captures both trend and periodicity, but uses a flat frequency initialization — it doesn't know which time scales to look at, and has no multi-scale structure. On our benchmarks it learned slower and generalized worse, particularly on random-context data where multi-scale temporal reasoning was essential.

**Sinusoidal frequency encodings** (`sinusoidal_frequencies.py`) took the opposite approach: manually extract calendar components (minute/60, hour/24, day/month, month/12, year/1000) and apply sin/cos at specified frequencies to each. This works when you know which time scales matter. It fails when you don't — when the relevant periodicity is 11 days, or 3.5 cycles across an arbitrary date range, or any pattern that doesn't align with calendar boundaries.

**Positional encodings** (Vaswani-style, explored in `pos_unix_enc.py`) use fixed sinusoidal frequencies determined by position index. They assume evenly-spaced positions and predetermined scales. They can't adapt to the data's actual periodicities and don't handle irregular timestamps.

The original `time_transformer.py` appeared the next day (Aug 12, 2023) and survived largely intact for 18 months. It combined value embeddings with timestamp encodings and used self-attention over context followed by cross-attention for prediction — the target timestamp queries the context asking "what should my value be?" This cross-attention architecture was a key early decision that persisted through all subsequent work.

### The Breakthrough: Multi-Band Learnable Frequencies (Jan 2025)

The timestamp encoding went through a long refinement. The git commits tell the story: "making time scale a parameter" (Jan 24), "making a version that expects seconds from epoch but uses days since epoch" (Jan 26), "masking not working as expected" (Jan 27), "adding cross attention" (Jan 29), "additive modulation" (Jan 29), "found an encoder that learns at second scale" (Jan 30), and then the breakthrough — "found a combined transformer that functions on both scales" and "TWO separate working encoders for both long and short time scales" (Jan 31).

The solution that emerged in `time_transformer_final.py` (finalized Feb 2, 2025) was a gated dual-pathway architecture:

**The Periodic Pathway** is a bank of d sinusoidal oscillators with learnable log-frequencies initialized across four bands: ultra-low (years to decades, log-freq -20 to -10), low (weeks to months, -10 to 0), medium (hours to days, 0 to 6), and high (seconds to minutes, 6 to 12). This spans over 13 orders of magnitude. The idea: initialize oscillators across all conceivable time scales, then let gradient descent prune and sharpen them toward whatever periodicities the data contains. Since frequencies are stored in log-space, a small parameter update can shift an oscillator smoothly from a daily cycle to a weekly one.

**The Trend Pathway** captures directional drift — things that accumulate or decay. It's a small MLP with heavy dropout (50%), tiny initialization, and a 0.1 scaling factor — all designed to suppress it early in training so the model relies on the periodic pathway first.

**The Adaptive Gate** examines both pathways (with detached gradients, preventing it from killing either pathway) and produces a softmax weighting. The detach is load-bearing: without it, the model collapses to a single pathway and overfits. With it, both pathways must independently minimize loss, and the gate only learns to route, not to shape.

This encoding was proven across time scales from seconds to years. It was declared READ ONLY and became the foundation imported by 12+ downstream files.

One critical discovery during this phase: **vanilla Adam only**. AdamW (weight decay), gradient clipping, learning rate schedules, and cosine decay all kill TimestampEncoding learning. The learnable frequencies span 13 orders of magnitude and need unconstrained parameters. Any regularization that penalizes parameter magnitude or restricts gradient flow prevents the oscillator bank from finding the right frequencies. This finding was confirmed repeatedly in trust experiments and holds across all configs.

### The Trust Learning Problem (Jun 2024 → Nov 2025)

With the timestamp encoding proven, the question shifted: can a transformer learn *which prediction source to trust at a given time*, when source expertise follows periodic sinusoidal cycles? The first `trust_transformer.py` appeared June 28, 2024, using MSE loss with basic source expertise via sinusoidal cycles and first ablation studies.

Over the next eight months, a half-dozen variants explored different formulations: `truster_transformer.py` (softmax source selection), `trusted_transformer.py` (simplified lightweight), `trusty_transformer.py` (time-unit agnostic), `trust_diagnostic.py` and `trust_regularized.py` (isolating the memorization problem), and `reform.py` (removing prediction values entirely to eliminate the copy shortcut). A finance application (`dynamic_finance_transformer.py`, Feb–Mar 2025) used trust scores for RL-style portfolio optimization.

The core challenge was always memorization. The model would achieve high training accuracy by memorizing (timestamp, source) → value mappings from training data, then fail on unseen timestamps. High train accuracy, random test accuracy. The standard remedies — dropout, weight decay, timestamp noise — were tried systematically in `trust_regularized.py`. None fully worked, and weight decay was actively harmful (killing the TimestampEncoding).

### The Breakthrough Push (Nov 2025 → Mar 2026)

In November 2025, the trust research was consolidated into `trust_updated.py` — a single clean file with MSE loss, an accuracy metric, and a four-way ablation (Baseline/Temporal/Source/Full). This became the base for systematic iteration.

**trust_v2** (Mar 8, 2026) vectorized the data generation (62x speedup) and introduced value dropout — randomly zeroing input predictions during training to force the model to learn trust from time+source embeddings rather than copying values. Without dropout: 99% train accuracy, 60% test. With dropout: the model was forced to actually learn the pattern.

**trust_v3** (Mar 10, 2026) brought two critical innovations. First, the **multiplicative interaction**: instead of concatenating time and source embeddings, compute `cat([v, t*s, t, s])`. This directly represents the bilinear decomposition of the ground truth — `sin(2π(t+φ)/C) = sin(2πt/C)·cos(2πφ/C) + cos(2πt/C)·sin(2πφ/C)` — which is bilinear in time features and source features. Element-wise multiplication of their embeddings directly computes these cross-terms. Second, the **distributional fix**: non-expert noise had been generated with `np.random.uniform()` while expert values used `np.random.randn()`. The model could detect this distribution difference without learning trust. Fix: normal noise for all sources.

**trust_v4** (Mar 11, 2026) found and fixed the deepest bug. With threshold-based expert selection (`sin > threshold`), multiple sources could be expert simultaneously. In the SIMPLE config (threshold=0.0), 49.8% of timesteps had two experts producing identical predictions. A trivial strategy — "find two values that match, output that" — achieved 66.5% accuracy. No model needed, no trust learned. The fix was argmax expert selection: exactly one expert per timestep, no consensus to detect.

### The Proof (Mar 11, 2026)

With both fixes applied, v4 produced the first clean proof:

| Model | Test Accuracy | Expected |
|-------|--------------|----------|
| Full Ŧrust | **87.0%** | High |
| Baseline (values only) | 35.7% | 33.3% (random) |
| Temporal only (values + time) | 35.8% | 33.3% (random) |
| Source only (values + source) | 35.5% | 33.3% (random) |

All three ablation models scored at random. Only the Full Model — with access to time AND source AND content — learned the trust pattern. This is the irreducible demonstration: you need all three legs of the triplet. Knowing *when* without knowing *who* is useless. Knowing *who* without knowing *when* is useless. Knowing *what they said* without knowing who said it or when is useless. Only the full (content, source, timestamp) triplet supports trust inference.

This result validates the mechanism at its lowest dimensionality. The same architecture — with word embeddings, patch embeddings, or any other content representation in place of the scalar value projection — should exhibit the same property: trust requires the complete triplet. The next steps are proving this on the HARD config (8 sources, which requires d=64 for enough phase capacity) and formalizing the results in the existing paper drafts (`trust_paper.md`, `timestamp_encoding_paper.md`).

## Phase 1: Timestamp Encoding Research (Aug 2023)

The first technical step: can a transformer learn time-series patterns from raw Unix timestamps through learnable periodic encodings?

| Date | File | Purpose |
|------|------|---------|
| 2023-08-11 | `time2vec.py` | Time2Vec encoding approach (reference implementation) |
| 2023-08-11 | `sinusoidal_basic.py` | Basic sinusoidal encoding experiments |
| 2023-08-11 | `sinusoidal_frequencies.py` | Frequency band exploration |
| 2023-08-11 | `mixed_time_enc.py` | Mixed encoding approaches |
| 2023-08-11 | `synthetic_data.py` | First synthetic data generation |
| 2023-08-11 | `time_enc_tester.py` | Encoder comparison harness |
| 2023-08-12 | `time_transformer.py` | **The original time-series transformer** (lasted through 2025-01-31) |
| 2023-08-14 | `spline.py` | Spline-based data generation |
| 2023-08-20 | `pos_unix_enc.py` | Positional Unix timestamp encoding |

## Phase 2: Refinement & Timestamp Encoding Proven (2024)

| Date | File | Purpose |
|------|------|---------|
| 2024-06-23 | `claude_version.py` | AI-generated variant (reference only) |
| 2024-06-28 | `timestamp_embeddings.py` | Standalone timestamp embedding experiments |
| 2024-07-03 | `old_time_transformer.py` | Snapshot of original for comparison |
| 2024-10-10 | `time_transformer_tester.py` | Systematic encoder comparison |

## Phase 3: Trust Transformer — The Core Research Problem (Jun 2024 → present)

**The question**: Can a transformer learn *which prediction source is reliable at a given time*, when expertise follows periodic sinusoidal cycles?

| Date | File | Status | Purpose |
|------|------|--------|---------|
| 2024-06-28 → 2024-07-27 | `trust_transformer.py` | ORIGINAL | First trust transformer. MSE loss, basic source expertise via sin cycles. First ablation studies. |
| 2025-02-03 | `trusted_transformer.py` | VARIANT | Simplified lightweight version |
| 2025-02-04 → 2026-02-20 | `truster_transformer.py` | VARIANT | Softmax source selection approach |
| — (untracked) | `trusty_transformer.py` | VARIANT | Time-unit agnostic variant |
| — (untracked) | `trust_diagnostic.py` | DIAGNOSTIC | Simplified configs to isolate memorization problem |
| — (untracked) | `trust_regularized.py` | DIAGNOSTIC | Heavy dropout, weight decay experiments |
| — (untracked) | `reform.py` | ALTERNATIVE | No-values approach: predict trust score from (time, source) only |
| 2025-11-30 | `trust_charts.py` | TOOLING | Trust-specific visualization |
| — (untracked) | `trust_transformer_charts.py` | TOOLING | More visualization tooling |

## Phase 4: Time Transformer Finalized (Jan–Feb 2025)

| Date | File | Status | Purpose |
|------|------|--------|---------|
| 2025-01-30 | `time_transformer_cull.py` | INTERMEDIATE | Simplification pass |
| 2025-02-02 → 2025-11-29 | `time_transformer_final.py` | **PROVEN** | Finalized TimestampEncoding. Multi-band learnable frequencies (13 orders of magnitude), gated trend+periodic. Imported by 12+ files. READ ONLY. |

## Phase 5: Finance Application (Feb–Mar 2025)

| Date | File | Purpose |
|------|------|---------|
| 2025-02-15 | `finance_transformer.py` | Initial finance transformer |
| 2025-02-17 → 2025-03-01 | `dynamic_finance_transformer.py` | RL-style portfolio optimization using trust scores → asset allocation |

## Phase 6: Source Selection Experiments (Nov 2025)

| Date | File | Purpose |
|------|------|---------|
| 2025-11-29 → 2026-02-20 | `source_selection.py` | Alternative source expertise approach |
| — (untracked) | `source_time_interaction.py` | Source×time interaction experiments |

## Phase 7: The Breakthrough Push (Nov 2025 → Mar 2026)

Consolidated trust research, systematic bug hunting, proof of genuine trust learning.

| Date | File | Key Change |
|------|------|------------|
| 2025-11-30 → 2026-03-08 | `trust_updated.py` | Consolidated working model. MSE loss, accuracy metric, 4-way ablation (Baseline/Temporal/Source/Full). Non-vectorized data gen. **Both bugs present.** |
| 2026-03-08 | `trust_v2.py` | Vectorized data gen (62x speedup). Value dropout to fight copy shortcut. d=64, no multiply. **Both bugs present.** |
| — (untracked) | `trust_quick_test.py` | Fast iteration script (~4 min per test). Parameterized experiments. Distributional fix applied. |
| 2026-03-10 | `trust_v3.py` | **Multiplicative interaction** `cat([v, t*s, t, s])` matching bilinear trig decomposition. d=32. Best-model checkpointing. **Bug 1 fixed** (normal noise for non-experts). Consensus bug still present. |
| 2026-03-11 | `trust_v4.py` | **Argmax expert selection** — exactly one expert per timestep. **Both bugs fixed.** Clean proof: all ablations at random, only Full Model learns. |

## Untracked Utility Files

| File | Purpose |
|------|---------|
| `time_enc_tester2.py` | Second encoder comparison harness |
| `timestamp_emb_enc.py` / `timestamp_emb_enc_2.py` | Embedding+encoding hybrid experiments |
| `timestamp_encoders.py` | Multiple encoder implementations |
| `pos_unix_enc2.py` | Second positional encoding variant |
| `synthetic_data2.py` / `time_synth.py` | Additional data generation approaches |
| `trust_probe.py` | Zero-values isolation test (eval with vd=1.0) |
| `trust_sweep.py` | Parameter sweep script (planned, not completed) |
| `monitor_gpu.py` | GPU monitoring utility |
| `differntiable_portfolio.py` / `offline_finance_transformer.py` | Finance variants |
| `WORKING TIMESTAMP EMBEDDINGS.py` | Snapshot of working state |

## The Three Bugs

| # | Bug | When Found | Fixed In | Mechanism |
|---|-----|-----------|----------|-----------|
| 1 | Distributional shortcut | v3 (Mar 10, 2026) | v3 | Non-expert noise was uniform, experts were normal. Model detected distribution difference. Fix: `np.random.randn()` for all. |
| 2 | Consensus shortcut | v4 (Mar 11, 2026) | v4 | `sin > threshold` allowed multiple simultaneous experts with identical predictions. Model detected consensus. Fix: `argmax` selection → one expert per timestep. |
| 3 | Copy shortcut | v2 (Mar 8, 2026) | v2 (value dropout) | Without dropout, model memorizes timestamp→value mappings. Fix: randomly zero input values during training. |

## A Note on the Bugs

All three bugs were cases where the code didn't match the researcher's intent. The concept was always clear — one expert per period, indistinguishable noise, no copying shortcuts. The visualizations were built to show one expert at a time (using `argmax`). But the data generation underneath used `> threshold`, which silently allowed overlaps. Each LLM that touched this code accepted the implementation as matching the spec. The charts looked right. The results were plausible. It took systematic ablation analysis — asking "why can a model with NO information still get 80%?" — to surface what was wrong. The research challenge was never the concept; it was getting the tools to implement it faithfully.

## Phase 8: Overnight Experiment Battery (Mar 11, 2026)

With the SIMPLE proof in hand (87% Full, ablations at random), six experiments were designed to map the difficulty curve and find the right configuration for HARD.

### What we know going in
- SIMPLE (3 sources, d=32, vd=0.5): **proven** — 87% Full, 35% ablations, 33.3% random
- HARD with d=32: **fails** — Full Model stuck at 16.6%, not enough capacity for 8 phase offsets
- Pre-fix best HARD config was d=64, vd=0.3, 100ep → 93.6%, but ablations were 77-85% (shortcut-inflated)
- All pre-fix absolute numbers are unreliable; relative rankings may hold

### Experiments and hypotheses

**1. HARD ablation (d=64, vd=0.3, 100ep)** — the main experiment. Hypothesis: Full Model reaches 50-70%, ablations stay at 12.5%. The gap will be smaller than SIMPLE because 8 sources with a 365-day cycle and threshold=0.5 is genuinely harder — narrower expert windows, more phase offsets to learn. If ablations are above 12.5%, there's another shortcut. If Full Model stays near random, d=64 isn't enough or vd=0.3 is wrong post-fix.

**2. HARD ablation (d=64, vd=0.2, 100ep)** — dropout sensitivity. Pre-fix, vd=0.2 slightly outperformed vd=0.3 on some runs. With shortcuts removed, the model needs MORE of the value signal, not less. If vd=0.2 beats vd=0.3: optimal dropout shifted downward without shortcuts. If worse: lower dropout enables copy shortcut even with argmax fix.

**3. MEDIUM (5 sources, d=32, vd=0.5, 50ep)** — difficulty interpolation. 5 sources with d=32 should still work (d=32 handles 3 sources, 5 phase offsets may still fit). Expect Full Model 60-80%, ablations at 20% (1/5). This tells us WHERE the d=32 → d=64 transition is necessary. If 5 sources works with d=32 but 8 doesn't, the bottleneck is purely capacity.

**4. SIMPLE extended (d=32, vd=0.5, 100ep)** — ceiling test. Does SIMPLE climb past 87% with more epochs? The v3 with-shortcut result was 95.5%, meaning the model was only partially using the shortcut. Extra epochs should close part of that gap. If it plateaus at 87%, 50 epochs is sufficient.

**5. HARD more data (d=64, vd=0.3, 60k samples, 100ep)** — data scaling. 2x training data means more evidence per cycle position. With 30k samples over ~1000 days, each phase region gets ~3750 samples. With 60k, ~7500. If no improvement, the bottleneck is model capacity, not data.

**6. HARD big model (d=128, vd=0.3, 100ep)** — capacity scaling. If d=64 partially works, d=128 should do better — more oscillator dimensions for 8 phase offsets. But overfitting risk increases. If d=128 > d=64: capacity was the bottleneck. If d=128 ≈ d=64 or worse: the learning problem is harder than fitting phases.

### What success looks like
1. HARD ablations ALL at 12.5% random across every experiment (no remaining shortcuts)
2. At least one HARD config gets Full Model to 50%+ (trust learning scales)
3. MEDIUM fills in the difficulty curve between SIMPLE and HARD
4. Identify the optimal (d_model, vd) for HARD post-fix

### Results (Mar 11, 2026)

| # | Config | Full Model | Ablations | Random | Time |
|---|--------|-----------|-----------|--------|------|
| 4 | SIMPLE (3src, d=32, vd=0.5, 100ep) | **91.1%** | 35-36% | 33.3% | 27 min |
| 3 | MEDIUM (5src, d=32, vd=0.5, 50ep) | **44.0%** | 22% | 20.0% | 18 min |
| 2 | HARD (8src, d=64, vd=0.2, 100ep) | **35.1%** | 13% | 12.5% | 56 min |
| 1 | HARD (8src, d=64, vd=0.3, 100ep) | **21.7%** | 13% | 12.5% | 56 min |
| 5 | HARD (8src, d=64, vd=0.3, 60k, 100ep) | **16.4%** | 13% | 12.5% | 112 min |
| 6 | HARD (8src, d=128, vd=0.3, 100ep) | **13.6%** | 13% | 12.5% | 59 min |

### Analysis

**The proof is airtight.** Across all six experiments — varying sources, capacity, dropout, and data volume — every ablation model scored at random. There are no remaining shortcuts. The only way to beat random is to have all three dimensions of the triplet.

**SIMPLE is genuinely solved.** Extending from 50 to 100 epochs pushed the Full Model from 87% to 91.1%. The model is still climbing — the learning is real and continues to improve with training. The pre-fix result was 95.5%, meaning roughly 4 points of that came from the distributional shortcut and the rest was genuine trust learning all along.

**The HARD problem is not about capacity.** This is the most important finding. d=128 didn't just underperform d=64 — it learned nothing at all (13.6%, indistinguishable from random). More data also hurt rather than helped (16.4% with 60k vs 21.7% with 30k). These results rule out the obvious hypothesis that HARD just needs a bigger model or more examples. The bottleneck is elsewhere.

**Value dropout is the key variable on HARD.** vd=0.2 scored 35.1% vs vd=0.3's 21.7% — nearly double the above-random margin. This makes sense when you think about what changed with the bug fixes. Pre-fix, the model had two free signals: the distributional difference between expert/non-expert noise, and consensus among multiple simultaneous experts. Both of these were available WITHOUT looking at the actual prediction values. The model could partially learn trust from these signals alone, making it tolerant of high value dropout.

Post-fix, those free signals are gone. The ONLY way to identify the expert is to compare prediction values against the model's learned expectation of what the true value should be. This requires actually seeing the values. High dropout (vd=0.3) masks 30% of them, and on HARD with 8 sources and narrow expert windows, that's too much lost signal. The model can't identify the expert if it can't see what most sources predicted.

**Why d=128 failed completely.** The TimestampEncoding has 128 learnable frequency parameters, 128 phase parameters, plus the trend pathway — far more degrees of freedom than the trust task requires. With only ~24k training samples and 8 sources, the model has enough capacity to memorize rather than generalize, but value dropout prevents memorization, so it's stuck: too much capacity to find the simple periodic pattern (optimizer wanders), too much dropout to memorize. The result is no learning at all.

**Why more data hurt with vd=0.3.** With 60k samples, the training set is larger but the dropout rate is the same. The model sees 70% of values per sample — but with more samples, it has more opportunities to partially memorize timestamp→value associations, which then conflict with each other when values are randomly masked. The optimization landscape becomes noisier. The fundamental issue isn't data quantity — it's that vd=0.3 is simply too aggressive for HARD post-fix.

**The difficulty curve is smooth.** SIMPLE (91%) → MEDIUM (44%) → HARD best (35%) shows a clean degradation as sources increase. The gap between "above random" margins tells the story: SIMPLE gets 58 points above random, MEDIUM gets 24, HARD gets 23. The mechanism works across scales — it just needs tuning per difficulty level. MEDIUM with only 50 epochs and d=32 already shows clear trust learning; more epochs or d=64 would likely push it much higher.

### Implications for next experiments

The clear next move is pushing value dropout lower on HARD: vd=0.15, vd=0.1, maybe even vd=0.05. The risk is that very low dropout re-enables the copy shortcut (memorizing timestamp→value mappings), but with argmax expert selection that shortcut is harder to exploit — there's only one expert per timestep, and the model must figure out which one it is. There should be a sweet spot where the model sees enough values to identify the expert but can't simply memorize the mapping.

A second move is running HARD with vd=0.2 for 200 epochs. The model was still at 40% train accuracy at epoch 100 — it hadn't converged. More training time with the right dropout could push it significantly higher.

A third is testing MEDIUM with d=64 and vd=0.3, 100 epochs. If MEDIUM is capacity-bound at d=32, it should jump substantially with d=64. If it's already near its ceiling, the difficulty is intrinsic to 5 sources, not capacity.

### Quick Test: Value Dropout Sweep (Mar 11, 2026)

Full Model only, 30 epochs each, via `quick_test.py` (imports from `trust_v4.py`).

| Config | vd | Test | Train | Random | Gap |
|--------|-----|------|-------|--------|-----|
| SIMPLE | 0.0 | 40.2% | 80.0% | 33.3% | +6.9% |
| SIMPLE | 0.2 | **86.4%** | 73.9% | 33.3% | +53.0% |
| HARD | 0.0 | 28.4% | 22.1% | 12.5% | +15.9% |
| HARD | 0.05 | 24.7% | 13.6% | 12.5% | +12.2% |
| HARD | **0.1** | **44.2%** | 14.7% | 12.5% | **+31.7%** |
| HARD | 0.15 | 19.5% | 12.6% | 12.5% | +7.0% |
| HARD | 0.2 | 24.1% | 36.8% | 12.5% | +11.6% |

**Value dropout is confirmed load-bearing.** SIMPLE vd=0 shows the classic copy shortcut: 80% train, 40% test — the model memorizes timestamp→value mappings instead of learning trust. With vd=0.2 it can't memorize and is forced to learn the periodic pattern, reaching 86.4% in just 30 epochs.

**vd=0.1 is the sweet spot for HARD.** It scored 44.2% in only 30 epochs — nearly double the overnight battery's best result (35.1% at vd=0.2 with 100 epochs). The pattern reveals a sharp optimum:

- **vd=0**: Can memorize. Train climbs to 22% but test lags (copy shortcut). Not as dramatic as SIMPLE because HARD has 8 sources and narrow expert windows — harder to memorize.
- **vd=0.05**: Worst of both worlds. Enough masking to disrupt memorization, not enough to force trust learning. The model can't commit to either strategy.
- **vd=0.1**: The sweet spot. Enough masking to break the copy shortcut, but the model still sees 90% of values per sample — enough to identify which source's prediction matches the pattern. The low train accuracy (14.7%) with high test accuracy (44.2%) is the signature of genuine generalization: the model isn't fitting the training set, it's learning the underlying periodic trust pattern.
- **vd=0.15–0.2**: Too much masking. With 8 sources and only ~13% expert time per source, masking 15-20% of values means the model frequently can't see enough predictions to identify the expert. It gives up.

The non-monotonic relationship between dropout and performance is the key insight. There's a narrow band where the model is forced to learn trust but still has enough signal to do so. For SIMPLE (3 sources, 33% expert time), vd=0.2-0.5 all work because the expert signal is strong and frequent. For HARD (8 sources, ~13% expert time), the window is much tighter — vd=0.1 is in it, vd=0.15 is already too high.

### VD Refinement + 100-Epoch Test (Mar 11, 2026)

Zoomed in around vd=0.1, Full Model only, 30 epochs.

| vd | Test | Train |
|----|------|-------|
| 0.07 | 37.2% | 39.6% |
| 0.08 | 21.7% | 16.7% |
| 0.10 | 47.8% | 28.2% |
| **0.12** | **54.4%** | 56.6% |
| 0.10 (100ep) | 41.0% | 56.7% |

**vd=0.12 is the new best** at 54.4% in 30 epochs. But the high variance is the real story: vd=0.08 cratered to 21.7% while its neighbors (0.07, 0.1) both worked. Same architecture, same data, same seed — the difference is a 0.02 change in dropout rate. The optimization landscape is rugged.

**More epochs hurt.** vd=0.1 at 100 epochs scored 41.0% — worse than 47.8% at 30 epochs. The model isn't improving with more training; it's oscillating. Train accuracy climbed to 56.7% but test dropped, meaning the model was fitting training data rather than learning the periodic pattern. This is the coupled oscillation problem: the TimestampEncoding and source embeddings chase each other. A small frequency shift destabilizes all 8 source embeddings simultaneously. In SIMPLE (3 sources) this stabilizes quickly. In HARD (8 sources) it never converges.

### Stability Experiments (Mar 11, 2026)

Testing approaches to break the coupled oscillation between TimestampEncoding and source embeddings.

| Strategy | Best Test | Time |
|----------|----------|------|
| **reverse 0→0.1** | **47.3%** | 9.3m |
| gentle 0→ramp→0.1 | 35.5% | 8.9m |
| freeze+reverse | 35.5% | 8.2m |
| flat vd=0.1 60ep (baseline) | 27.0% | 9.1m |
| freeze 15ep + vd=0.1 | 23.6% | 8.4m |

**Reverse curriculum is the winner.** Start with vd=0 (let the model memorize, find the right frequencies), then switch to vd=0.1 (break memorization, force generalization). 47.3% vs 27.0% flat baseline — a 75% improvement in the above-random margin.

The insight: the model faces a chicken-and-egg problem. It needs good frequencies to learn source phases, and good phases to learn frequencies. Reverse curriculum solves this by letting the model cheat first (vd=0, memorize the training set), which forces the TimestampEncoding to find the 365-day cycle (because memorization requires it). Then dropout kicks in and says "okay, you found the frequency — now actually learn WHO to trust at each point in the cycle, because you can't memorize anymore."

**Freezing the encoding failed.** Locking the TimestampEncoding for 15 epochs while source embeddings trained alone (23.6%) was worse than the baseline (27.0%). The encoding needs to co-evolve with the model — it can't be trained in isolation. What works is giving it a head start via memorization (where the gradients from MSE loss directly push frequencies toward the data's periodicity), not freezing it out.

**Gentle ramping underperformed hard switching.** The gradual ramp from 0→0.1 over 20 epochs (35.5%) was worse than the sharp switch at epoch 30 (47.3%). This suggests the transition should be abrupt: the model needs a clean memorization phase where it fully commits to finding frequencies, then a clean generalization phase where it fully commits to learning trust. A gradual ramp creates a middle ground where the model can't fully commit to either strategy.

### Optimizer Experiments (Mar 11, 2026)

**Hypothesis:** The coupled oscillation on HARD isn't a capacity problem — it's a learning rate problem. The TimestampEncoding needs lr=1e-3 to traverse its vast frequency space (13 orders of magnitude). But the source embeddings and attention layers don't need steps that large. When the encoding shifts a frequency, the gradient tells all 8 source embeddings to make big corrections at lr=1e-3 — they overshoot, which makes the encoding think its new frequency was wrong, so it shifts back. Classic coupled oscillation. If we could give the encoding a high lr while giving everything else a lower lr, the non-encoding parameters would make smaller, more stable corrections and the system should converge.

Previous rule was "vanilla Adam only" — but that was tested GLOBALLY. Weight decay kills the encoding's frequencies, but it shouldn't matter if the encoding is exempt.

**Setup:** HARD config (8 sources, d=64, vd=0.1, 60 epochs). Five tests:

| # | Optimizer Config | Best Test | Train at Best |
|---|---|---|---|
| 4 | **Split LR: enc=1e-3, rest=3e-4** | **63.9%** | 60.9% |
| 3 | AdamW wd=0.05 (encoding exempt) | 55.7% | ~50% |
| 1 | Vanilla Adam lr=1e-3 (baseline) | 47.6% | 46.2% |
| 5 | AdamW wd=0.01 + split LR (combined) | 36.8% | 54.6% |
| 2 | AdamW wd=0.01 (encoding exempt) | 32.1% | 58.7% |

**Analysis:**

**Split LR is the clear winner.** 63.9% — 16 points above vanilla Adam. The hypothesis was correct: non-encoding parameters were overshooting at lr=1e-3. Reducing them to 3e-4 (3.3x lower) gives smaller corrections that don't destabilize the system when the encoding shifts. The encoding still has the freedom to traverse frequency space at full speed.

**Heavy weight decay (wd=0.05) works as a proxy for lower lr.** 55.7% — second best. Weight decay continuously pulls parameters toward zero, which caps their magnitude. Even at lr=1e-3, the effective step size is smaller because decay counteracts growth. This is a blunter instrument than split LR (it constrains magnitude, not step size), but achieves a similar damping effect.

**Light weight decay (wd=0.01) hurts.** 32.1% — worst result. Not enough decay to provide meaningful damping (unlike wd=0.05), but enough to interfere with parameter learning. The wd=0.01 models had HIGH train accuracy (58.7%) but terrible test accuracy — weight decay was acting as a slight regularizer on the wrong axis, preventing the generalization path without helping stability.

**Combining split LR + weight decay is too much.** 36.8% — the non-encoding parameters are now double-constrained (lower lr AND weight decay pulling them back). The model can't move fast enough to keep up with the encoding's exploration. Interestingly, train accuracy was 54.6% at best-test time, suggesting the model was learning but the constraints prevented it from finding the right balance.

**Why split LR works (the physics):** Think of the encoding as a slow oscillator (it needs to find the 365-day frequency, which takes many epochs of gradient descent through a complex loss landscape) coupled to 8 fast oscillators (source embeddings, which just need to find the right phase offset once the frequency is known). At equal lr, the fast oscillators respond too aggressively to each encoding shift. At split lr, they make proportionally smaller corrections — tracking the encoding's exploration without overshooting. The system converges because the coupling is damped.

**Revised rule:** Not "vanilla Adam only" but "**split Adam: high lr for TimestampEncoding, lower lr for everything else.**" The encoding's learnable frequencies are fundamentally different from standard neural network parameters and need different optimization dynamics.

### Phase 10: Isolation Experiments (Mar 11, 2026)

Building on the split LR breakthrough (63.9%), five experiments designed to isolate what matters most for further progress. Each tests one variable.

**1. Split LR confirm (baseline reproduction)**
Identical to the optimizer test that got 63.9%. One run isn't enough — HARD has shown high variance between runs (vd=0.08 got 21.7% while vd=0.12 got 54.4% in earlier tests). If this lands 55-65%, the improvement is real. If it lands 35-45%, we got lucky and split LR's benefit is smaller than it appeared.

**2. Split LR, non-encoding lr=1e-4 (deeper damping)**
If the coupled oscillation theory is correct, pushing the non-encoding lr even lower should help further — smaller corrections = less overshoot. Prediction: diminishing returns. 3e-4 already damped the worst oscillation; going to 1e-4 might slow convergence too much for 60 epochs. Expect similar or slightly lower than #1. If it's MUCH better, 3e-4 was still too high and we should sweep the non-encoding lr more carefully.

**3. Split LR + 3000 days / ~8 cycles (more pattern evidence)**
The current 1000-day window gives ~2.7 complete 365-day cycles. The model might struggle to learn periodicity from fewer than 3 full repetitions — it's hard to distinguish a periodic pattern from a trend when you've only seen it repeat twice. Extending to 3000 days gives ~8 cycles while keeping sample density constant (~30 samples/day). This is orthogonal to optimizer changes — if it helps, it stacks with whatever optimizer wins. Prediction: meaningful improvement (5-10pp). The model should benefit from seeing the same phase regions repeat 8 times vs 2.7 times. The test set (last 20% = last ~600 days) will contain ~1.6 full cycles, all at timestamps never seen in training.

**4. Stop-gradient: `t_emb.detach() * s_emb` (break the coupling entirely)**
The most theoretically interesting test. Split LR DAMPS the gradient coupling through the multiplicative interaction; stop-gradient ELIMINATES it. The encoding still gets gradients from the standalone `t` term in `cat([v, t*s, t, s])`, but not from the product. This means the encoding learns from the reconstruction loss on values (through the `t` pathway) while source embeddings learn from the product (through the `t.detach() * s` pathway). If this works better than split LR: the coupling is fundamentally harmful and split LR is a band-aid. If worse: the encoding NEEDS the gradient signal from the product to learn the right frequencies — the coupling is a necessary cost of the multiplicative inductive bias. Prediction: worse than split LR. The product term `t*s` is where the trust signal lives — cutting the encoding off from that gradient means it has to learn the 365-day frequency purely from the value reconstruction pathway, which is weaker. But I could be wrong — if the encoding finds the frequency through the value pathway and the coupling was purely destructive, this could be a big win.

**5. Split LR + reverse curriculum (stack two improvements)**
Reverse curriculum (vd=0 → vd=0.1 at epoch 30) got 47.3% with vanilla Adam. Split LR got 63.9% with flat vd=0.1. If the mechanisms are complementary — reverse curriculum solves the chicken-and-egg problem (find frequencies first), split LR solves the oscillation problem (stable convergence after) — they should stack to 65-75%. If they DON'T stack (similar to split LR alone), it means split LR already provides enough stability for the encoding to find frequencies without the memorization crutch.

**Results:**

| # | Test | Best Test | vs Random |
|---|------|-----------|-----------|
| 1 | Split LR confirm | **58.1%** | +45.6pp |
| 2 | Split LR rest=1e-4 | **58.1%** | +45.6pp |
| 3 | Split LR + 3000 days | **42.1%** | +29.6pp |
| 4 | Stop-gradient (t.detach()*s) | **42.2%** | +29.7pp |
| 5 | Split LR + reverse curriculum | **31.4%** | +18.9pp |

**Analysis:**

**Split LR is real but noisy.** Test 1 got 58.1% vs the earlier 63.9% — different random seeds in the data gen produce different results but both are far above the vanilla Adam baseline of 47.6%. The true expected value of split LR on HARD is probably ~55-60%. The variance itself is informative: despite damping the oscillation, each run still follows a different trajectory through the loss landscape. The encoding finds slightly different frequency combinations each time, and those lead to different quality trust learning.

**Deeper damping doesn't help.** Test 2 (rest_lr=1e-4) matched test 1 exactly at 58.1%. The 3e-4 → 1e-4 reduction didn't improve convergence — it just slowed the non-encoding parameters down without benefit. This means 3e-4 is already in the right regime: the oscillation is damped enough. The remaining variance comes from something else — likely the encoding's random walk through frequency space early in training.

**More cycles hurt.** Test 3 (3000 days, ~8 cycles) dropped to 42.1%. This was the most surprising result. The hypothesis was that more pattern repetitions would help, but spreading 30k samples over 3000 days means each day has ~10 samples instead of ~30. The model sees more cycles but with less evidence per cycle position. The TimestampEncoding has to represent a wider range of absolute timestamps with the same capacity, and the sparser sampling means noisier gradient estimates at each phase position. **The bottleneck isn't pattern repetitions — it's sample density per cycle phase.** The model already sees 2.7 cycles, which is enough to distinguish periodicity from trend. What it needs is enough samples at each phase to learn a robust mapping, not more phases to see.

**Stop-gradient confirms the coupling is a necessary evil.** Test 4 got 42.2% — below split LR's 58.1%. Detaching `t_emb` in the product means the encoding only gets gradients from the standalone `t` pathway and the `target_t_enc` cross-attention query. These pathways don't carry the trust signal directly — they carry the timestamp's contribution to value reconstruction. The product `t*s` is where the trust signal lives (which source to weight at this time), and the encoding needs that gradient to learn WHICH frequencies matter for trust. Without it, the encoding learns frequencies useful for value prediction generally, which is a weaker signal. **The gradient coupling through the multiplicative interaction is not just a side effect — it's the primary learning signal for the encoding.** Split LR is the right solution: keep the gradient, dampen the step size.

**Reverse curriculum + split LR interfere.** Test 5 got 31.4% — the worst result, and WORSE than either reverse curriculum alone (47.3%) or split LR alone (58.1%). This is the opposite of stacking. Why? During the vd=0 phase (epochs 0-29), the model memorizes training data: timestamp → value. With split LR, the encoding learns frequencies slowly (lower effective learning rate on everything) while the attention layers memorize slowly (lr=3e-4 instead of 1e-3). When vd switches to 0.1 at epoch 30, the model has partially memorized but hasn't found frequencies as well as vanilla Adam would have (because split LR slows the encoding indirectly through weaker gradients from a partially-memorizing model). The memorization phase needs FAST learning — both encoding and attention — to fully commit to finding frequencies. Split LR's damping undermines this. **Reverse curriculum and split LR solve the same underlying problem (oscillation/instability) through conflicting mechanisms.** Reverse curriculum wants aggressive early learning (memorize fast to find frequencies). Split LR wants conservative learning throughout (small steps to prevent oscillation). They cancel out.

**What remains to be solved:**

The core problem is now clear: **high run-to-run variance**. Split LR reliably gets 45-65% on HARD (vs 12.5% random), but the spread is too wide for confident conclusions. The encoding's random walk through frequency space in early epochs determines whether it finds a good frequency for the 365-day cycle. Once found, trust learning proceeds well. If it misses, the model gets stuck.

The remaining bottleneck is NOT:
- Optimizer settings (split LR at 3e-4 is the right regime, going lower doesn't help)
- Data coverage (more cycles makes it worse, density matters more)
- Gradient coupling strength (stop-gradient proved the coupling is necessary)
- Training schedule (reverse curriculum conflicts with split LR)

The remaining bottleneck IS:
- **Frequency initialization / discovery.** The encoding starts from random frequencies and must find the 365-day cycle through gradient descent. With 8 sources and narrow expert windows, the gradient signal pointing toward the right frequency is noisy. Some runs find it, some don't. This is a search problem in frequency space, not an optimization problem in weight space.

Possible next directions:
1. **Frequency initialization**: Seed one dimension of TimestampEncoding with the known cycle frequency (2π/365 days). This is "cheating" in a sense — giving the model a hint — but it tests whether the bottleneck is really frequency discovery.
2. **Multiple random restarts**: Run 5 seeds, keep the best. Expensive but directly addresses variance.
3. **Larger encoding bank**: More frequency dimensions = more chances to stumble onto the right one. But d=128 failed earlier due to overfitting — maybe d=128 + stronger vd?
4. **Warmup + split LR**: Start with uniform lr=1e-3 for 10 epochs (fast frequency search), then switch to split LR (stable convergence). Unlike reverse curriculum, this doesn't change vd — it changes the optimizer mid-training.

### Phase 11: Challenging the "Vanilla Adam Only" Assumption (Mar 23, 2026)

**Core question:** We've been avoiding standard transformer training tools (lr schedules, warmup, grad clipping, cosine decay) because they "killed learning" in early tests. But those tests applied them GLOBALLY — including to the TimestampEncoding, which has fundamentally different optimization needs (13 orders of magnitude in learnable frequencies). Split LR already proved the rule was conditional: per-parameter-group settings work. What other tools become available when the encoding is exempt?

This matters because the remaining bottleneck is frequency discovery variance. Standard transformer training uses warmup and lr decay to stabilize early training and improve convergence. If these tools help the non-encoding parameters (attention, source embeddings, MLP) converge more reliably WITHOUT disrupting the encoding's frequency search, we could reduce the run-to-run variance that makes HARD results range from 42% to 64%.

**Tests and what each seeks to learn:**

**1. Split LR baseline (enc=1e-3, rest=3e-4, flat)**
Our current best config. Included as the control for this batch. Expected: ~55-60% (the established range from Phase 10).

**2. LR warmup on non-encoding params (0 → 3e-4 over 10 epochs)**
*Seeking to learn:* Do the non-encoding parameters benefit from starting slow? Standard transformers use warmup because randomly-initialized attention weights produce noisy gradients early on, and large lr on noisy gradients sends parameters in wrong directions. With the encoding starting from random frequencies, the early gradients for source embeddings and attention layers are especially noisy — they're trying to learn from a time representation that hasn't found the right frequency yet. Warmup lets them wait for the encoding signal to strengthen before committing.
*Prediction:* Modest improvement (2-5pp). The first 10 epochs with near-zero rest lr means the encoding explores frequencies alone, and the other params gradually join once a useful signal emerges. Risk: if the encoding needs gradient signal FROM the other params to find frequencies (the product coupling), silencing them via warmup could hurt — similar to how freezing the encoding failed.

**3. Cosine decay on non-encoding params (3e-4 → 0 over training)**
*Seeking to learn:* Does reducing the lr late in training help? By epoch 40-60, the encoding has either found the right frequency or it hasn't. If it has, the non-encoding params should be making tiny refinements, not big jumps. Cosine decay naturally transitions from exploration (high lr) to exploitation (low lr). On HARD, the best test accuracy often appears mid-training and then degrades — cosine decay on the rest params might preserve those good solutions instead of overshooting past them.
*Prediction:* Meaningful improvement (5-10pp) if the late-training degradation is caused by non-encoding params overshooting after the encoding stabilizes. No improvement if the degradation is from the encoding itself shifting frequencies late.

**4. Warmup + cosine decay on non-encoding params (combined schedule)**
*Seeking to learn:* Do warmup and cosine stack? This is the standard transformer schedule: ramp up, plateau, ramp down. The encoding stays at flat 1e-3 throughout. If both warmup and cosine individually help, the combined schedule should give the non-encoding params the best trajectory: start slow (wait for encoding signal), learn at full rate (mid-training), then freeze in place (preserve good solution).
*Prediction:* Best result if both #2 and #3 individually help. If only one helps, this may be similar to that one. If neither helps, this won't either.

**5. Gradient clipping on non-encoding params only (max_norm=1.0)**
*Seeking to learn:* Is the oscillation caused by occasional large gradient spikes rather than consistently-too-high lr? Grad clipping truncates outlier gradients while leaving normal-sized ones alone. If the coupled oscillation is driven by a few bad batches sending huge gradients through the product term, clipping would fix those spikes without slowing down normal learning. This is a different mechanism than lower lr — lower lr shrinks ALL gradients, clipping only shrinks the outliers.
*Prediction:* Depends on the oscillation's character. If it's spike-driven: significant improvement. If it's a steady-state instability (every step is slightly too large): no improvement over split LR alone, because most gradients are already below the clip threshold.

**6. Kitchen sink (split LR + warmup + cosine + grad clip, encoding exempt from all)**
*Seeking to learn:* What's the ceiling when we throw every standard tool at the non-encoding params? This isn't expected to be optimal — over-regularization hurt in Phase 9 (AdamW + split LR combined was worse than either alone). But it establishes an upper or lower bound: if all tools together STILL don't beat split LR alone, the non-encoding params aren't the problem. If it's dramatically better, there's a specific combination worth unpacking.
*Prediction:* Slightly worse than the best individual technique, due to over-constraint. But informative either way.

**Results:**

| # | Config | Best Test | vs Baseline |
|---|--------|-----------|-------------|
| 5 | **Grad clip rest only (1.0)** | **49.6%** | **+7.2pp** |
| 2 | Warmup rest 10ep | 42.5% | +0.1pp |
| 1 | Split LR baseline | 42.4% | — |
| 3 | Cosine decay rest | 35.7% | -6.7pp |
| 6 | Kitchen sink | 23.9% | -18.5pp |
| 4 | Warmup + cosine | 20.7% | -21.7pp |

Note: this batch's split LR baseline came in at 42.4% — lower than the 55-64% range from Phase 10. Run-to-run variance remains high. All comparisons are within-batch (same seed, same data).

**Analysis:**

**Grad clipping works — the oscillation IS spike-driven.** Test 5 got 49.6%, 7 points above baseline. This answers the specific question: the coupled oscillation isn't just steady-state "every step slightly too large" — it's punctuated by occasional large gradient spikes through the `t*s` product that send the non-encoding parameters flying. Clipping at max_norm=1.0 truncates these outliers while leaving normal gradients untouched. The model still makes full-speed progress on most batches, but the catastrophic oscillation episodes are prevented.

This is a different mechanism than split LR. Split LR reduces ALL steps by 3.3x. Grad clipping only intervenes on the worst 5-10% of steps. They could be complementary — split LR handles the baseline coupling, grad clipping handles the spikes.

**Warmup is neutral.** Test 2 (42.5%) matched baseline (42.4%). The non-encoding parameters don't benefit from starting slow. This makes sense in retrospect: during early epochs, the encoding is searching randomly, and the non-encoding params need to react to whatever signal exists — suppressing their learning just delays the process without improving it. The encoding doesn't produce a gradually-strengthening signal; it produces random noise until it finds the right frequency, then suddenly a good signal appears. Warmup is designed for gradually-improving signals, not random-then-sudden.

**Cosine decay hurts.** Test 3 (35.7%) is 7 points below baseline. Reducing lr late in training prevents the non-encoding params from making corrections that are still needed. On HARD, the best test accuracy appears at unpredictable epochs — the system is still actively evolving at epoch 60. Cosine decay freezes parameters in place during this period, preventing adaptation. The model needs to stay responsive throughout training, not settle into a fixed solution.

**LR schedules in general hurt this problem.** Tests 3, 4, and 6 all include cosine decay and all underperform. The trust learning task is fundamentally different from standard transformer pretraining: there's no "converging to a solution" phase where you want to reduce step size. The encoding might shift frequencies at epoch 50 just as easily as epoch 10, and the other params need to be ready to follow. Any lr reduction on non-encoding params late in training cripples this ability.

**Kitchen sink confirms over-regularization.** Test 6 (23.9%) is the worst. Three constraints stacked (warmup + cosine + clip) over-constrain the non-encoding params. This matches the pattern from Phase 9 (AdamW + split LR combined was worse than either alone). Each individual tool addresses a specific problem; combining them addresses problems that don't all exist simultaneously.

**Revised understanding of the oscillation:** It's not a steady-state instability — it's an intermittent spike problem. Most training steps are fine with split LR at 3e-4. But occasionally, a batch produces a large gradient through the `t*s` product (perhaps when the encoding is near a frequency transition, or when a batch is dominated by one source's expert region). These spikes destabilize the source embeddings, which takes multiple epochs to recover from. Grad clipping surgically removes these spikes.

**Next step:** Test split LR + grad clipping combined. Split LR handles baseline step size, grad clipping handles spikes. If they stack, this could push into 60%+ territory.

### Phase 11b: Split LR + Clip Follow-up (Mar 23, 2026)

| Config | Best Test |
|--------|-----------|
| Split LR + clip 0.5 (tighter) | 41.0% |
| Split LR + clip 1.0 | 34.5% |
| Clip only (uniform lr=1e-3) | 18.5% |

**They don't stack, and the earlier clip result may have been variance.** The Phase 11 "clip only" result of 49.6% used the same setup as follow-up test 2 (uniform lr=1e-3 + clip rest at 1.0), which got only 18.5% here. The 31-point gap between identical configs across two runs is pure variance — confirming that run-to-run instability remains the dominant factor, larger than any optimizer improvement we've measured.

**This means we can't reliably distinguish optimizer configs at N=1.** A 7pp improvement from grad clipping is within the noise of 30pp run-to-run variance. The Phase 11 ranking (clip > warmup > baseline > cosine) may be partially or entirely noise. To draw real conclusions about optimizer settings, we need multiple seeds per config.

**Revised assessment of Phase 11:** The qualitative finding — that LR schedules hurt and the system needs to stay responsive — is likely still valid (cosine decay was consistently bad across multiple tests). But the specific claim that grad clipping helps by +7pp is not confirmed. The spike-driven oscillation theory may be correct, but we can't prove it with single-seed experiments given this variance level.

**The variance IS the problem.** Every other question (optimizer, schedules, clipping) is secondary. Until we reduce or control for the 30pp run-to-run spread, we can't reliably evaluate anything. Next priority should be either:
1. **Multi-seed experiments** (3-5 seeds per config, compare means) to get reliable signal
2. **Attack the variance directly** — frequency initialization, longer warmup at uniform lr before splitting, or architectural changes that make frequency discovery more reliable

## Phase 12: Two-Encoding Bug & Encoding Placement (Mar 24, 2026)

### The Bug

All trust transformer versions (v2, v3, v4) had TWO separate `TimestampEncoding` instances:
```python
self.context_t_enc = TimestampEncoding(d_model)        # 32-dim, for context tokens
self.target_t_enc = TimestampEncoding(self.d_model_total)  # 128-dim, for cross-attention query
```

The papers explicitly specify ONE shared encoding. The timestamp encoding paper (Section 3.3): "the target timestamp is encoded via **the same** dual-pathway module." The trust paper (Section 3.1): "`TimestampEncoding(d_model)`" — singular. Two separate encodings mean two independent understandings of time that can diverge. This violates the foundational design. Every result from v2-v4 was produced on a broken architecture.

The proven `TimeSeriesTransformer` in `time_transformer_final.py` uses ONE encoding correctly. Only the trust transformer duplicated it.

### Hypothesis: Encoding placement matters

The cross-attention query IS the time in "attention over source, content, time." In a standard transformer, positional encodings go everywhere — both encoder (keys/values) and decoder (queries). For attention to use position, position must be in BOTH the query AND the keys (Q·K^T). Same logic applies to time in the trust transformer.

### v5 Initial Tests

Created `trust_v5.py` — exact copy of v4 with encoding fix. Tested two directions:

**v5a: One encoding at d_model=32, project UP to 128 for query**
- Seed 42: best test acc = 51.9%
- Heavy overfitting: train 79%, test oscillating 8-49%

**v5b: One encoding at d_model_total=128, project DOWN to 32 for context**
- Seed 42: best test acc = 82.1% (matches v4!)
- Still oscillating but higher ceiling
- Seed 123: best test acc = ~46% — massive variance

The larger encoding (128-dim) gives the query 4x the frequency resolution. 128 oscillators searching for the expertise cycle vs 32. The query drives the trust computation — it's "which source do I trust at this time?" Bottlenecking it through 32 frequencies bottlenecks the entire mechanism.

### Encoding Placement Suite — Results

`encoding_suite.py` — 5 configs × 3 seeds (42, 123, 7), full model only, 50 epochs each:

| Config | Description | s42 | s123 | s7 | Mean | Std |
|--------|-------------|-----|------|-----|------|-----|
| **A** | Two separate encs (v4 original) | 86.1% | 82.3% | 88.8% | **85.7%** | **2.7%** |
| **B** | One enc(32), proj UP for query | 61.8% | 43.4% | 55.5% | 53.6% | 7.6% |
| **C** | One enc(128), proj DOWN for context | 48.7% | 76.6% | 54.2% | 59.8% | 12.1% |
| **D** | One enc(32), same-dim cross-attn | 41.5% | 59.7% | 43.8% | 48.3% | 8.1% |
| **E** | Query only, no time in context | 62.0% | 59.3% | 54.6% | 58.6% | 3.1% |
| | Random baseline (3 sources) | | | | 33.3% | |

### Analysis

**Config A dominates on both accuracy and stability.** Two separate encodings at their native dimensions (32 for context, 128 for query) outperform every single-encoding variant by 26+ percentage points in mean accuracy with 5x lower variance.

**Why projections fail:** The architecture has a structural dimension mismatch. Context tokens are `cat([v, t*s, t, s])` = `4*d_model`. Cross-attention query must match at `4*d_model`. Time lives at `d_model` inside the context (one of 4 slices) but at `4*d_model` in the query. Every "one encoding" approach requires a projection somewhere, and every projection loses information.

- **B** (32→128 projection for query): 32 oscillators can't find the expertise cycle. Massive overfitting (train 80%+, test 40%).
- **C** (128→32 projection for context): Right encoding dimension but projecting down destroys temporal structure in context. Huge variance (48.7%–76.6%) — sometimes the projection preserves enough, sometimes not.
- **D** (cross-attn at 32): Projects the 128-dim context down to 32 for both sides. Worst performer — too much information lost.
- **E** (query only): No time in context at all. Most stable single-encoding variant (std=3.1%) but low ceiling (58.6%). Confirms time must be on BOTH sides of cross-attention for Q·K^T to leverage temporal information.

**The positional encoding parallel holds:** Standard transformers use the same positional encoding scheme on both sides of attention (encoder keys/values and decoder queries). For Q·K^T to use position, position must be in BOTH Q and K. For Q·K^T to use time, time must be in both the query AND the context tokens. Config E proves this — query-only gets 58.6% vs A's 85.7%.

**Key insight:** Two separate encodings aren't a bug — they're an adaptation to the dimension mismatch. Each encoding specializes at its native dimension: the 32-dim context encoding learns frequencies useful for self-attention over sources (the `t*s` interaction), while the 128-dim query encoding learns frequencies for the trust selection in cross-attention. Forcing them to share parameters through projections hurts both.

### Config F: Additive combination (d_model=128, no concat)

Hypothesis: if concatenation creates the dimension mismatch, use addition instead. `v + t*s + t + s` at d_model=128. One encoding, no projections.

| Config | s42 | s123 | s7 | Mean | Std |
|--------|-----|------|-----|------|-----|
| **F** | 51.1% | 79.7% | 55.5% | 62.1% | 12.6% |

**Result:** Addition collapses signals. The model can't disentangle which component contributed what. Concatenation preserves each signal in its own subspace — addition does not.

### Config G: One enc(128), concat, cross-attn kdim/vdim (d_model=128)

Hypothesis: use d_model=128 so encoding fits in context natively at 128. Keep concatenation (512-dim context). Cross-attention handles Q=128 vs KV=512 via kdim/vdim.

| Config | s42 | s123 | s7 | Mean | Std |
|--------|-----|------|-----|------|-----|
| **G** | 45.4% | 39.4% | 54.5% | 46.5% | 6.2% |

**Result:** Worst of all configs. The d_model=128 model is massively overparameterized for 15K samples.

### Diagnostic Suite: Isolate WHY Config A Works

Changed ONE variable at a time from A (d_model=32, two separate encs at 32 and 128):

| Test | Description | s42 | s123 | s7 | Mean | Std |
|------|-------------|-----|------|-----|------|-----|
| **A** | Two separate encs (baseline) | 83.6% | 88.6% | 84.3% | **85.5%** | **2.2%** |
| **A-big** | d_model=64, two encs (64+256) | 80.0% | 54.3% | 73.0% | 69.1% | 10.8% |
| **A-context-only** | No query enc, learned query vector | 74.0% | 54.3% | 51.7% | 60.0% | 10.0% |
| **A-swapped** | Context=128→32, query=32→128 | 74.5% | 90.8% | 84.7% | **83.3%** | 6.7% |
| **A-same-128** | Both encs at 128, context projected to 32 | 76.4% | 75.5% | 43.8% | 65.2% | 15.2% |
| **E** | Query only, no context time (from earlier) | 62.0% | 59.3% | 54.6% | 58.6% | 3.1% |

### What the diagnostic suite proves

1. **Two independent frequency learners is the key factor.** A-swapped (83.3%) nearly matches A (85.5%) despite having projections on BOTH sides. The dimension assignment (which side gets 32, which gets 128) barely matters. What matters is that each side has its own set of learned frequencies.

2. **Model size is a real confound.** A-big (d_model=64) drops to 69.1% even with two encodings. Every d_model=128 experiment (F, G) was doomed by overparameterization, not encoding architecture. Cannot compare d_model=128 results to d_model=32 results.

3. **Both sides of cross-attention need time.** Context-only=60.0%, query-only=58.6%. Neither alone is enough. Both contribute ~25pp. Together they give 85.5%.

4. **Native dimension > projected dimension.** A (native 32 + native 128) = 85.5%. A-swapped (projected on both sides) = 83.3%. A-same-128 (native 128 + projected 128→32) = 65.2% with huge variance. Starting an encoding at native dimension and using it directly is more stable than projecting.

5. **Two encodings are a training crutch, not a design choice.** Two separate encodings currently outperform one, but this is because two independent frequency searches are more likely to discover the expertise cycle than one (Phase 11). The fix is improving training reliability for one encoding — not accepting two as the design. See "Non-Negotiable Design Principles" above.

### The real question: why can't one encoding represent time?

A timestamp encoding is a representation of time. One rich representation should be enough — that's how every other embedding works. The diagnostic suite shows two encodings help, but that doesn't mean two are architecturally necessary. It could mean training is fragile.

**Hypothesis: the bottleneck is frequency discovery, not architecture.** The TimestampEncoding starts with random frequencies spanning 13 orders of magnitude and must find the 500-day expertise cycle through gradient descent. That search fails often — it's the primary source of run-to-run variance (Phase 11). Two encodings help because two independent random searches are more likely to find the needle than one. This is a workaround for unreliable training, not a feature of the design.

**Test: Config C + frequency hint.** Take Config C (one encoding at 128, projected to 32 for context — the cleanest single-encoding variant) and initialize one frequency band near `2π/500` days (`ln(2π/(500*86400))` ≈ -15.74 in log-space). If C-with-hint matches A (~85%), frequency discovery is the bottleneck. If it improves but doesn't close the gap, frequency discovery is part of the problem but something else is also broken.

### Frequency Hint Results

| Config | s42 | s123 | s7 | Mean | Std |
|--------|-----|------|-----|------|-----|
| **A: Two separate encodings** | 81.7% | 81.8% | 73.7% | **79.1%** | 3.8% |
| **C: One enc(128), no hint** | 45.9% | 45.0% | 44.7% | **45.2%** | 0.5% |
| **C-hint: One enc(128) + freq hint** | 56.8% | 70.4% | 64.0% | **63.7%** | 5.6% |

**Frequency hint helped significantly** — C-hint (63.7%) jumped 18.5pp over C (45.2%). But it didn't close the gap to A (79.1%). Frequency discovery is **part** of the training problem but not all of it.

The remaining ~15pp gap indicates additional issues with one encoding: likely gradient interference (context path and query path pulling shared parameters in conflicting directions) and/or information loss through the projection from 128→32. The next experiments isolate which of these causes the remaining gap.

### Phase 13: Closing the One-Encoding Gap

Three experiments, run in order. Each isolates one hypothesis about why one encoding underperforms two.

**Experiment 1: Gradient detach (isolate gradient interference)**

Config C-hint but with `detach()` on the encoding output before it enters the context path. Only the query path's gradients train the encoding. The context path gets a frozen copy each forward pass.

*Rationale:* Two encodings have independent gradients — the context encoding is trained only by context-path loss, the query encoding only by query-path loss. One shared encoding gets gradients from both paths simultaneously. If these gradients conflict (context wants frequency X tuned one way, query wants it tuned another), they partially cancel or oscillate, preventing convergence. Detaching one path eliminates the conflict.

*Expectation:* If gradient interference is the remaining problem, this should close most or all of the ~15pp gap to A (~79%). The encoding would be optimized purely for the query's trust-selection role, while the context path gets a stable read-only copy of whatever the query path learns. This is the most likely fix because A-swapped (83.3%) proved the dimension assignment barely matters — what matters is independent optimization.

*If it fails:* Gradient interference is not the bottleneck. The problem is either the projection itself or something about sharing parameters that we haven't identified.

**Experiment 2: Focused frequency bands (isolate frequency coverage)**

One encoding at 128-dim with ALL frequency bands initialized in the range -18 to -12 (weeks to multi-year), plus the freq hint at -15.74. No bands wasted on sub-second or sub-minute frequencies that are irrelevant to a 500-day expertise cycle.

*Rationale:* The standard initialization spreads 128 bands across 13 orders of magnitude. Only ~8 bands fall near the relevant range (-18 to -12). The rest are wasted searching frequency space that can never matter for this task. Dense coverage in the right range means more oscillators available to lock onto the cycle, and small perturbations during training are more likely to stay in the useful range.

*Expectation:* Moderate improvement over C-hint, probably 5-10pp. This helps with initial frequency discovery but doesn't address gradient interference. If experiment 1 already closed the gap, this becomes less important — but if both help independently, the combination could push past A.

*Critical caveat:* This is a **diagnostic**, not a solution. Narrowing the frequency range to weeks-to-years makes the encoding problem-specific — it's now tuned for 500-day cycles and useless for sub-hour patterns. The timestamp embedding must generalize to any problem: sub-second sensor data, hourly weather, daily markets, yearly climate. The full 13-order-of-magnitude span is the design. If focused bands help, the lesson is "the encoding needs to find relevant frequencies faster" — and the fix must achieve that without sacrificing generality (e.g., better optimization, curriculum on frequency learning, adaptive band allocation that narrows dynamically based on gradients).

*If it fails:* Frequency coverage at 128-dim is already sufficient, and the problem is purely about how gradients flow through shared parameters.

**Experiment 3: No-projection architecture (eliminate the dimension mismatch entirely)**

Redesign the model so one encoding at one dimension works everywhere without projection. d_model=32, encoding at 32. Replace `cat([v, t*s, t, s])` with a cross-attention architecture where self-attention operates over 32-dim source tokens and cross-attention uses the 32-dim timestamp as query against 32-dim source keys/values. No concatenation, no 128-dim context, no projection needed.

*Rationale:* Every single-encoding experiment so far required a projection somewhere (128→32 or 32→128). Projections are learned linear maps that can lose information, add parameters, and create optimization challenges. If the encoding lives at one dimension and the architecture uses that dimension natively everywhere, there's nothing to lose and nothing to interfere. This is how standard transformers work — positional encodings are added at the model dimension and that's it.

*Expectation:* This is the most uncertain experiment. It fundamentally changes the architecture (no more `cat([v, t*s, t, s])` multiplicative interaction). The multiplicative `t*s` term was specifically designed to match the bilinear trig decomposition of source expertise — removing it by restructuring could hurt. But if the attention mechanism can learn the source×time interaction through Q·K^T dot products (which is also multiplicative), the explicit `t*s` concatenation may be redundant. If this matches or beats A, it proves the concatenation scheme was creating an artificial dimension mismatch that forced the two-encoding workaround. Combine with focused frequency init so 32 bands suffice.

*If it fails:* The `cat([v, t*s, t, s])` interaction is load-bearing and the architecture genuinely needs the 128-dim context representation. In that case, experiment 1 (gradient detach) is the path forward — keep the current architecture, keep one encoding, just fix the gradient flow.

**Experiment 4: Multi-task training (test generalizability directly)**

Generate three synthetic datasets simultaneously with different expertise cycle lengths — say 100 days, 500 days, and 1500 days. Train one model with one encoding on all three, interleaved. The model must learn to represent time at multiple scales in a single embedding.

*Rationale:* Every experiment so far trains on one cycle length. The encoding is supposed to generalize to any problem — so test that claim directly. More importantly, multi-task training provides gradient signal across multiple frequency ranges simultaneously. With one cycle length, gradients only push a few bands toward the relevant frequency. With three cycle lengths, gradients push bands toward three different frequencies. This prevents the encoding from collapsing all its capacity toward one narrow range and forces it to maintain the broad coverage that generalizability requires.

*Expectation:* This could actually *improve* single-encoding performance. The pathology of one encoding might be that all bands drift toward the same frequency (or away from useful frequencies) because there's only one gradient signal pulling them. Multiple tasks create multiple attractors in frequency space, spreading the bands out and keeping them useful. If multi-task one-encoding matches or beats single-task two-encodings, it's a profound result — two encodings were compensating for insufficient gradient diversity, not insufficient parameters. The encoding was always capable; it just needed richer training signal to stay general.

*If it fails:* The encoding genuinely struggles to maintain multiple frequency scales under gradient pressure, and the generalizability problem is harder than a training fix.

### Phase 13 Results

Four experiments, each isolating one hypothesis about why one encoding underperforms two.

| Config | s42 | s123 | s7 | Mean | Std |
|--------|-----|------|-----|------|-----|
| **A: Two encodings (baseline)** | 86.0% | 85.7% | 81.3% | **84.3%** | 2.1% |
| C-hint (reference) | 77.0% | 60.0% | 77.6% | 71.5% | 8.2% |
| **Exp1: detach context grads** | 71.4% | 50.7% | 58.7% | **60.3%** | 8.5% |
| **Exp2: focused freq bands** | 77.5% | 71.1% | 63.4% | **70.7%** | 5.7% |
| **Exp3: additive + hints** | 87.9% | 87.6% | 88.5% | **88.0%** | **0.4%** |
| **Exp4: multi-task concat** | 49.9% | 51.5% | 52.1% | **51.1%** | 0.9% |

**Exp 3 appeared to solve the problem** — but it combined two changes (additive architecture AND frequency hints). A 2×2 isolation was needed to separate the variables.

### Phase 13b: 2×2 Isolation (additive vs concat × hints vs no hints)

|  | No hints | Hints |
|--|----------|-------|
| **Concat (one enc)** | 44.6% ± 2.4% | 73.6% ± 14.1% |
| **Additive (one enc)** | 63.1% ± 5.9% | 85.8% ± 0.7% |
| **Two encs concat (A)** | 87.8% ± 2.9% | — |

Full results (3 seeds each):

| Config | s42 | s123 | s7 | Mean | Std |
|--------|-----|------|-----|------|-----|
| **A: Two encs (reference)** | 91.9% | 85.1% | 86.3% | **87.8%** | 2.9% |
| Concat, no hints | 41.6% | 47.6% | 44.6% | 44.6% | 2.4% |
| Concat + hints | 78.1% | 88.2% | 54.6% | 73.6% | 14.1% |
| Additive, no hints | 69.7% | 55.3% | 64.2% | 63.1% | 5.9% |
| **Additive + hints** | 86.3% | 84.8% | 86.4% | **85.8%** | **0.7%** |

### Validation battery (single seed, no hints, additive architecture)

| Test | Result | Finding |
|------|--------|---------|
| **t*s interaction** | With: 71.2%, Without: 41.9% | `t*s` is load-bearing — 30pp drop without it |
| **Ablations** | Full=70.4%, Temporal=35.9%, Source=35.8%, Baseline=35.8% | Trust proof intact — only full model learns |
| **Multi-task** (100d/500d/1500d) | Mean: 51.3% | Failed — multi-task doesn't help at this scale |
| **5 sources** | 29.4% (random=20%) | Barely above random without hints |
| **Learned frequencies** | Band 3: initialized at -15.71 (537d), learned to -15.81 (536d) | Encoding finds the neighborhood but doesn't sharpen enough |

### What we actually know

**1. The core problem is frequency discovery.** Two separate encodings reliably find the 500-day expertise cycle (87.8%, std 2.9%). One encoding struggles to find it (44-63% depending on architecture). Frequency hints — manually seeding the answer — close the gap completely (85.8%). This confirms the problem is not architectural capacity, gradient interference, or information loss. It's that one set of learnable frequencies has trouble locking onto the task-relevant periodicity through gradient descent alone.

**2. Additive combination helps but doesn't solve it.** Additive (63.1%) outperforms concat (44.6%) without hints — an 18.5pp improvement. The additive architecture removes the projection bottleneck and lets the encoding's output flow directly into the model. But 63.1% is still 25pp below two encodings. The architecture makes the encoding more usable, but doesn't help it find the right frequency.

**3. Additive + hints is the best config found.** 85.8% ± 0.7% — matches two encodings with 4x lower variance. But hints are not a solution. They require knowing the task's cycle length in advance, which defeats the purpose of a generalizable embedding.

**4. The `t*s` multiplicative interaction is load-bearing.** Removing it drops accuracy by 30pp (71.2% → 41.9%). The explicit bilinear term `t * s` matches the trig decomposition of source expertise and provides signal that attention alone can't learn at d_model=32.

**5. Ablations hold on the additive architecture.** Temporal-only, source-only, and baseline all stay at random (~35.8%). Only the full model learns. The trust proof is intact regardless of combination method.

**6. Why two encodings find the frequency and one doesn't.** Two independent random initializations are more likely to place at least one frequency band near the target cycle. With 32 bands spanning 13 orders of magnitude, each encoding gets ~8 bands in the ultra-low range where the 500-day cycle lives. The linspace initialization places band 3 at -15.71 (537 days) — already close. But "close" isn't "exact," and during training the band barely moves (delta = -0.1). Two encodings give two independent chances for one of their bands to be close enough. One encoding gets one chance.

### The open problem

The timestamp encoding needs to find task-relevant frequencies through gradient descent, starting from a broad initialization across 13 orders of magnitude. With 32 bands, only ~8 fall in any given order-of-magnitude range. The gradient signal from the trust loss is apparently too weak or noisy to reliably pull a nearby band onto the exact target frequency.

Two encodings solve this by brute force (two independent searches). This works but violates the one-encoding design. Frequency hints solve it by cheating (telling the model the answer). This works but isn't generalizable.

**The actual fix must make one encoding's frequency search more reliable without knowing the answer in advance.** Possible directions:
- Better optimization for the frequency bands (separate learning rate, different optimizer)
- Frequency band adaptation during training (start broad, narrow based on gradient signal)
- Denser initialization in commonly-useful ranges without sacrificing the full span
- Longer training or curriculum approaches that give the encoding more time to find frequencies
- Architectural changes that amplify the gradient signal back to the frequency bands

## Key Architectural Decisions (Evolving)

- **TimestampEncoding**: Learnable multi-band frequencies spanning seconds to years. Gated trend+periodic. Proven in `time_transformer_final.py`.
- **Multiplicative interaction**: `t_emb * s_emb` directly represents the bilinear decomposition of `sin(2π(t+φ)/C)`.
- **Split Adam**: TimestampEncoding at lr=1e-3 (needs to traverse vast frequency space), everything else at lr=3e-4 (smaller corrections prevent coupled oscillation). Weight decay, grad clipping, lr schedules still kill the encoding when applied globally — the key insight is per-parameter-group optimization.
- **Value dropout**: Load-bearing — prevents copy shortcut. Optimal rate varies by difficulty: vd=0.5 for SIMPLE, vd=0.1-0.12 for HARD. Too high masks the signal, too low enables memorization.
- **Reverse curriculum**: Start vd=0 (memorize, find frequencies), switch to target vd (generalize). Helps with vanilla Adam but CONFLICTS with split LR — the two solve oscillation through opposing mechanisms.
- **Best-model checkpointing**: Test accuracy oscillates ±15% per epoch — must save best.
- **Argmax expert selection**: One expert per timestep, no consensus signal.

---

## Phase 14: The Sequence Insight — Trust IS a Time Series

### The Breakthrough Realization (2026-03-24)

Every version of the trust model — v2 through v5, reform, encoding_suite, validation_battery, two_by_two — presents the problem the same way: **one timestep at a time**. The model receives N source predictions at timestamp T and must output the expert's value. The "sequence" dimension is sources, not time. There is no history.

This is fundamentally wrong. Not wrong as in "suboptimal architecture choice." Wrong as in "we removed the thing that makes transformers work."

### Why It Failed: No Temporal Variation in the Sequence

In a standard transformer (language, time series, anything), each token in the sequence has a **different position**. When attention computes Q·K^T, the dot product expands to four terms:

```
Q·K^T = (content·content) + (content·position) + (position·content) + (position·position)
```

The cross-terms (content×position) are what let the model learn position-dependent content interactions — "this word at this position means X." For this to work, **tokens must have different positions** so the cross-terms carry information.

In our trust formulation, all source tokens at a given timestep share the **same** timestamp. The position terms in Q·K^T are identical across all source pairs within a timestep. Self-attention over sources can't distinguish anything by time because there is no temporal variation.

The timestamp encoding's output only enters meaningfully through **cross-attention** (query = timestamp, keys/values = source context). That's a much thinner gradient path. The encoding gets weak, indirect gradient signal. No wonder it can't find the right frequencies — we've bottlenecked the only path where time matters.

Compare to `time_transformer_final.py` where ONE encoding works perfectly: each context token has a **different** timestamp. Self-attention sees the wave directly. The encoding gets strong gradients from every attention head, at every layer, because temporal variation is in every token pair's dot product. The encoding thrives because the architecture lets it contribute.

We weren't giving the trust model a time series problem. We were giving it a lookup table problem — "at this single moment, who's the expert?" — and wondering why it memorized instead of learning the cycle.

### The Fix: Present Trust as What It Actually Is

Trust is a time series. Source expertise varies over time in a periodic cycle. The model should see that cycle the same way the proven time series model sees value patterns — as a **sequence of observations across time**.

**New data presentation:**

Instead of one timestep with N sources, give the model a window of W timesteps, each with N sources. If W=10 and N=3, the sequence has 30 tokens:

```
[S0_T1, S1_T1, S2_T1,  S0_T2, S1_T2, S2_T2,  ...,  S0_T10, S1_T10, S2_T10]
```

Each token embedding: `source_emb(s) + value_emb(v) + timestamp_enc(t)`

- Tokens at the same timestep share a timestamp (source variation)
- Tokens at different timesteps have different timestamps (temporal variation)
- Self-attention sees BOTH dimensions simultaneously

This is exactly how a transformer is supposed to work. Attention aggregates information across the full sequence, building up a representation of how source reliability changes over time. The periodicity in expertise is directly visible as a pattern in the sequence, just as periodicity in values is visible in the time series task.

**Query:** Cross-attention with a target timestamp T_target asks "given this history, who's the expert at time T_target?"

### How This Differs from Standard Transformers

One notable difference from text: at each timestep, there are multiple tokens (one per source) rather than one token per position. This is fine — attention doesn't care about the shape of the sequence, it just attends over all tokens. It's like having multiple sentences in a context window. Tokens within a timestep share temporal locality, but attention operates over the full flattened sequence.

The model can learn both:
- **Within-timestep patterns**: Which sources agree/disagree at a given time
- **Across-timestep patterns**: How each source's accuracy waxes and wanes — the cycle itself

### What We Expect

**1. One encoding should work immediately.** The structural reason two encodings were needed disappears. Every token in the sequence has a timestamp, temporal variation is everywhere, and the encoding gets strong gradients from self-attention over the full history. This is the same regime where one encoding already works in `time_transformer_final.py`.

**2. No frequency hints needed.** The encoding should find the 500-day cycle through gradient descent because the gradient signal is strong and comes from many attention pairs, not just one cross-attention bottleneck.

**3. Ablations should still hold.** Temporal-only should beat baseline (now the model sees time variation). Source-only should beat baseline (source identity still matters). Full should beat both. The trust proof is preserved — in fact it should be cleaner.

**4. Multi-task should work.** Multiple cycle lengths are just different patterns in the time series. The encoding's multi-band design is built for this — different bands capture different periodicities. With a proper sequence presentation, the model can learn all of them.

**5. Scaling to HARD should be tractable.** More sources = more tokens per timestep, higher threshold = sharper transitions. But the fundamental mechanism (attend over temporal history, learn the cycle) doesn't change. The model capacity scales with sequence length, which is what transformers are designed for.

### Implementation Plan: `trust_v6.py`

**Data generation:** Same `SourcePredictionGenerator` for ground truth. But instead of sampling individual (timestamp, predictions, target) tuples, sample **windows** — contiguous or sampled subsequences of W timesteps. Each window becomes one training example with W×N tokens.

**Architecture:**
- One `TimestampEncoding(d_model)` — the same proven encoding
- `source_emb(s) + value_emb(v) + timestamp_enc(t)` for each token — additive, no projection
- Self-attention over the full W×N sequence (with `t*s` interaction preserved)
- Cross-attention: target timestamp query attends over the sequence
- Output: predicted value at the target timestamp

**Training:**
- Same chronological 80/20 split
- Value dropout still applies (prevents value-copying within the sequence too)
- Window size W is a hyperparameter — start with W=20, sweep later
- Batch size may need to shrink (sequences are longer)

**Evaluation:**
- Same accuracy metric: does the output land closest to the expert's value?
- Compare directly to Config A (two encodings, single timestep) on identical data

### v6 First Attempt: Consecutive Windows (Failed)

The first implementation used consecutive windows — 20 adjacent timesteps from 2000 data points over 1000 days. Result: all models stuck at 33% (random). Nothing learned.

The diagnosis was straightforward: 20 consecutive points covered ~10 days — 2% of the 500-day cycle. The model couldn't see the pattern because the window was too narrow in time. But this led to a deeper realization about what the context window means for timestamp embeddings.

### The Context Window Is Not What You Think

In a language transformer, the context window is a sequence of ordered slots. Position 1, position 2, ... position N. Each slot is one specific position embedding. The slots are fixed quanta — you can't subdivide them.

With timestamp embeddings, the context window is fundamentally different. The slots aren't positions — they're **containers**. You can put an observation from *any* time into *any* slot. Day 1 in slot 47. Day 750 in slot 3. Day 500 in slot 12. The ordering doesn't matter. The density doesn't matter. The timestamp encoding tells the model when each observation is from, independent of which slot it occupies.

This is what makes timestamp embeddings a different kind of positional encoding. A traditional positional embedding says "this is the Nth thing in a sequence." A timestamp embedding says "this happened at time T." The first is relative to the sequence. The second is absolute in time.

The consequence: you can fill a 100-slot context window with observations spanning any time range. Two observations 250 days apart will produce very different encodings at the frequency band near the 500-day cycle. The model doesn't need dense sampling — it needs observations that **span** enough time for the periodic structure to be detectable in the encoding space. The encoding provides the temporal structure; the model just needs enough samples to read it.

This is analogous to how the proven time series task works in `time_transformer_final.py`: the context window contains observations at various timestamps, and the model learns the pattern from the temporal structure in the encodings, not from the sequential order of the slots.

### Revised v6 Implementation Plan

**Data presentation:** Don't sample consecutive windows. Sample **scattered** observations from across the full time range. Each training example is a context of W randomly sampled (source, time, value) triples from history, plus a query timestamp. The triples can come from anywhere in the date range — they don't need to be consecutive, ordered, or evenly spaced.

This is more natural for the real-world use case too. When you ask "who should I trust right now?", you're drawing on scattered memories of past performance — not a neatly ordered recent history.

**Architecture:** Same as before:
- One `TimestampEncoding(d_model)`
- Token = `source_emb(s) + value_emb(v) + timestamp_enc(t)` with `t*s` interaction
- Self-attention over the context
- Cross-attention with target timestamp query

**Key change:** The context window spans the full date range. With W=100 samples across 1000 days, the model sees multiple full cycles of expertise. The encoding's frequency bands can detect the 500-day periodicity because the observations span that range.

### v6 Implementation Attempts

**Attempt 1: Consecutive windows (failed).** 20 adjacent timesteps from 2000 points. All models at random. Window covered ~10 days — 2% of the 500-day cycle.

**Attempt 2: Scattered history only, no query predictions (failed).** 99 tokens sampled from across the full time range, no query-time source predictions in the sequence. All models at random. The model had no way to output the correct value — the candidates it was choosing from weren't in the input.

**Attempt 3: Scattered history + query predictions appended, value dropout on all tokens (failed).** 99 history tokens + 3 query tokens = 102 total. Value dropout applied uniformly across all tokens. All models at random. The query predictions were being masked by dropout, and 3 query tokens were lost in 102 total — the model couldn't find the candidates.

**Attempt 4: v5 task + scattered history context (first success).** The key realization: v6 is not a different task. It is the v5 task (receive N source predictions at query time, output the expert's value) with historical context added to the sequence. The query predictions are always present and always unmasked — they are the candidates the model chooses from. History is additional context that provides temporal variation.

Implementation: wrap the v5 dataset. Each sample contains the original v5 data (query predictions, target) plus 30 randomly sampled historical timesteps (90 history tokens). Total sequence: 93 tokens. Value dropout applies to history tokens only. Query predictions are never masked.

**Results (seed=42, 50 epochs, SIMPLE config):**

| Model | Best Test | Train at ep50 |
|-------|----------|---------------|
| Full v6 | **50.2%** | 67.2% |
| Baseline | 35.7% | 33.6% |
| Temporal only | 35.3% | 33.2% |
| Source only | 35.8% | 33.3% |
| Random baseline | 33.3% | — |

This is the first time one encoding learns the trust task without frequency hints. The model was still climbing at epoch 50 — train accuracy at 67% suggests it hasn't converged. The learned frequency band 3 moved from ~483d initialization to 395d — actively searching, unlike the snapshot model where bands barely moved.

All ablations at random. The trust proof holds: only the full model with all three legs of the triplet (source, time, content) learns.

**What the history context provides:** Temporal variation in the sequence. Tokens at different timestamps give the encoding gradients from self-attention over token pairs at different times. In the v5 snapshot, all tokens shared one timestamp — no temporal variation, thin gradient signal through cross-attention only. With history, the encoding gets dense gradient signal from self-attention across the full sequence. This appears to be sufficient for frequency discovery without hints.

**What we don't know yet:**
- Whether it converges higher with more epochs
- Whether it's stable across seeds
- Whether the same approach works on HARD
- Whether the history context size (30 timesteps) matters
- How it compares to the two-encoding v5 snapshot (87.8%)

Longer runs with multiple seeds are in progress (150 epochs, seeds 42 and 123).

### 150-Epoch Results

**Seed 42, 150 epochs:**
- Best test accuracy: **43.9%** (at epoch 120)
- Train accuracy oscillated between 42–79% throughout training
- Test accuracy oscillated ±15% per epoch, never stabilized
- Learned frequency band 3: period=403d (shifted from initialization)

**Seed 123, 150 epochs:**
- Best test accuracy: **44.9%** (at epoch 65)
- Train accuracy reached 85% but test stayed low
- Same heavy oscillation pattern
- Learned frequency band 3: period=305d

The initial 50.2% at epoch 50 (seed 42) now appears to be a high point in the oscillation, not a stable measurement. With more epochs, the model didn't climb higher — it oscillated around ~42-44% test accuracy while train accuracy continued rising. The gap between train and test widened, indicating memorization is still occurring despite the history context.

### Honest Assessment

Four separate implementations were attempted, each claiming to match the original specification for v6. The results across all four:

1. **Consecutive windows** → random (33%)
2. **Scattered history, no query predictions** → random (33%)
3. **Scattered history + query, uniform value dropout** → random (33%)
4. **v5 task + scattered history context** → 44-50% (above random, but oscillating and not converging)

Attempt 4 learns something — it is consistently above the 33.3% random baseline, and only the full model (with source, time, and content) learns, preserving the trust proof. But 44% with heavy oscillation is far from the two-encoding v5 benchmark of 87.8%, and the pattern of results suggests something fundamental is still wrong with how the problem is being presented to the model.

The user's assessment: the implementation does not match the original specification. The core idea — presenting trust as a time series so the transformer processes it "like a traditional transformer" — may not have been correctly translated into code across any of the four attempts. The repeated cycle of "implement → fail → diagnose → reimplement" suggests a gap between the conceptual insight and the implementation.

**What we know:**
- Adding scattered history context helps (44% vs 33% without it)
- The coupled oscillation problem persists (encoding and source embeddings chasing each other)
- One encoding still can't match two encodings on this task
- Four different implementations each diverged from the original spec in ways that weren't caught until results came in

**What we don't know:**
- Whether the original v6 specification has been correctly understood
- What "resembling a more traditional transformer" means concretely for this task
- Whether the architecture or the data presentation (or both) need to change

### Why This Should Have Been Obvious

`time_transformer_final.py` works because the model sees a time series — multiple timestamps, temporal variation in the sequence. We proved that timestamp encoding works on time series. Then we took the trust problem, removed all temporal variation from the sequence, and wondered why the encoding couldn't learn. We were asking the encoding to solve the problem alone from a single point, instead of letting the transformer do what transformers do — attend over a sequence and learn patterns.

---

## Phase 15: v6 Reimplementation — Causal Masking + Pre-Norm (2026-03-25)

### What Changed

Three concrete changes from the failed v6 attempts:

1. **Causal masking** — lower-triangular mask, `-inf` before softmax, exactly like nanoGPT. History tokens are sorted chronologically. Position i attends to positions 0..i. Query tokens (at the end) see all history.

2. **Pre-norm** — LayerNorm before attention and FFN, not after. `x = x + attn(ln(x))`, `x = x + mlp(ln(x))`. Final LayerNorm before output. This gives a clean gradient highway through the residual stream.

3. **History sorted by time** — the existing v6 sampled history randomly but didn't sort it. Causal masking is meaningless on an unsorted sequence. Now sorted chronologically so the mask actually enforces temporal causality.

Everything else preserved: one TimestampEncoding, additive `v + t + s + t*s`, cross-attention query readout, value dropout, MSE loss, argmax accuracy, ablations.

### Predictions

**SIMPLE (3 sources, 500-day cycle, threshold=0.0):**
- Expect test accuracy >85%. The temporal variation from 30 history timesteps gives the encoding gradients at multiple time points. Causal masking forces the model to build up temporal understanding incrementally rather than attending everywhere at once. Pre-norm stabilizes training.
- The ablation pattern should hold: Full > Temporal-only ≈ Source-only > Baseline.

**If it fails, the most likely cause is:**
- Frequency discovery — the encoding needs to find the 500-day cycle frequency through gradient descent. This has always been the bottleneck. Causal masking and pre-norm don't directly help with this.
- Sequence length mismatch — 30 history timesteps × 3 sources = 90 history tokens + 3 query = 93 total. If this is too short for the causal attention to learn useful patterns, performance will be poor.
- Implementation bug — given 4 previous failed attempts, this is the most honest prediction.

### Embedding Combination Sweep

A sweep (`trust_v6_sweep.py`) tested 6 embedding combination modes and 2 readout strategies on SIMPLE (30 epochs each):

| Variant | Combination | total_d | Test Acc | Train Acc |
|---------|-------------|---------|----------|-----------|
| C | `cat([v+t, s, t*s])` — hybrid | 96 | **68.7%** | 80.8% |
| A | `v+t+s+t*s` — all additive | 32 | 57.2% | 44.5% |
| B | `cat([v, t, s, t*s])` — all concat | 128 | 50.9% | 80.5% |
| D | `cat([v+t, s])` — no t*s | 64 | 49.2% | 58.5% |
| All cross-attn variants | — | — | 33-41% | — |

The hybrid C (timestamp added like positional embedding, source concatenated as separate modality) outperformed both extremes. Cross-attention readout was consistently worse than mean-last-N with causal masking. The t*s interaction remains load-bearing — removing it drops 20pp.

C run to 100 epochs oscillated, best test only 55.9%. One encoding still can't match two.

---

## Phase 16: Parameter Inspection — What Config A Actually Learns (2026-03-25)

This is a turning point. Instead of building another v6 and hoping it works, we stopped and asked: **what does the working model actually learn internally?** We trained Config A (trust_v4.py, two encodings, SIMPLE config) to 88.0% test accuracy and then inspected every learned parameter. This is the first time we've looked inside the model that works rather than debugging the models that don't.

### Method

`inspect_v4.py` — trains the full model (SIMPLE: 3 sources, 500-day cycle, threshold=0.0, 100 epochs, seed 42), restores the best checkpoint, then:

1. **Dumps freq_bands** — the learned log-frequencies of both TimestampEncodings, converted to periods in days
2. **Probes gate outputs** — feeds timestamps through and measures trend vs periodic weighting
3. **Extracts source embeddings** — norms and pairwise cosine similarities
4. **Measures temporal variance** — how much each encoding dimension varies across the date range
5. **Runs FFT on encoding outputs** — finds dominant periods in the actual output signal (not just the parameter values)
6. **Cross-correlates the two encodings** — are they learning similar or different representations?

This is something we should do after every successful training run going forward. The parameters tell us what the model discovered about the task, and comparing across runs reveals what's stable vs what's noise.

### What We Found

**Both encodings find the 500-day cycle.**

Context encoding (32d): band 3 sits at **499.3 days**. One band, dead-on. FFT of the encoding output confirms 501 days as the dominant period (power=5.87), with dim 3 carrying almost all the 500d signal (power=285.6, vs next-highest dim 2 at 52.7).

Target encoding (128d): bands 13 and 14 sit at **529.1 days** and **399.7 days** — bracketing the target. FFT shows 501 days as the second-strongest period (power=2.39), behind 251 days (power=2.57). The 251-day signal is the half-period, which makes physical sense: with 3 sources evenly spaced in a 500-day cycle, source transitions happen every ~167 days, and the pairwise switching pattern repeats at ~250 days.

**The two encodings learn completely different representations.**

Cross-correlation between the first 32 dims of each encoding: mean |r| = 0.050. Zero dimensions with |r| > 0.5. These are not redundant copies — they encode the same timestamp using entirely different internal structure.

**Gate behavior diverges dramatically.**

The context encoding gates roughly evenly: 47% trend, 53% periodic. It uses both components.

The target encoding (which serves as the cross-attention query) gates almost entirely periodic: **11% trend, 89% periodic**. It has learned that querying the context sequence is primarily a frequency-matching problem, not a trend problem. This makes sense — the query needs to ask "which source is expert NOW?", which is a periodic question.

**Source embeddings are well-separated.**

All three sources have similar norms (~6.7) but negative pairwise cosine similarities (-0.51, -0.20, -0.35). The model pushes sources apart in embedding space, which is the correct thing to do — they represent distinct entities with non-overlapping expertise windows.

**Almost all dimensions are active.**

Context: 31/32 dims have temporal variance > 0.01. Target: 124/128 dims. The encodings aren't just using a few dimensions — they spread information broadly, though the 500d signal concentrates in specific bands (dim 3 for context, dims 13-15 for target).

### What This Means

The naive interpretation of the two-encoding advantage was: "two independent frequency searches double the probability that at least one finds the 500-day cycle." But that's not what the data shows. **Both encodings find the cycle.** Not just one getting lucky — both converge to within 1-30 days of the target frequency, from initializations spanning 13 orders of magnitude.

This raises a deeper question: why do both find it? If frequency discovery is the hard part (as we've been assuming), and both encodings face the same search problem from similar random initializations, then having two shouldn't make both succeed — it should make at least one succeed. But both do.

One possibility: the two encodings **help each other learn** through shared gradients. The context encoding's gradients flow through self-attention over the source prediction tokens. The target encoding's gradients flow through cross-attention. But both paths ultimately depend on the same loss function, and the model can only minimize loss when the two encodings cooperate — when the context encoding marks time correctly in the token representations AND the target encoding queries for the right time. This creates a coupled optimization landscape that may be easier to navigate than either encoding alone.

This would explain why dropping one encoding doesn't just halve performance — it collapses it. Not because you need two representations of time, but because the gradient coupling between two encodings creates a more navigable loss landscape. The cross-attention loss provides a strong supervisory signal to the target encoding ("your query must match the context at the right time"), and the self-attention loss provides a different supervisory signal to the context encoding ("tokens at similar times should have similar representations"). Two different gradient paths, both pushing toward the same frequency.

We can't confirm this without tracing actual gradient flow, which is the next step. But the parameter inspection strongly suggests the story is more interesting than "probability."

### Methodology Note

This is the kind of inspection we should do routinely:

- **After every successful training run**: dump freq_bands, gate ratios, source embeddings, FFT of encoding outputs
- **After failed runs**: compare the same parameters to successful runs to see where they diverge
- **Across seeds**: do the same bands converge? Do different seeds find different bands?
- **Across configs**: how do SIMPLE vs HARD differ in what bands matter?

The tools are simple — convert log-frequencies to periods in days, run FFT on encoding outputs, check gate ratios, measure cross-correlation. The insights are disproportionately valuable. We spent weeks guessing why one encoding fails; 10 minutes of parameter inspection told us more than all five v6 attempts combined.

### Config A HARD Inspection — 44.8% test (12.5% random)

Same method, same architecture, different task: 8 sources, 365-day cycle, threshold=0.5, 30000 samples, 100 epochs.

**The context encoding missed the frequency.** Band 3 landed at 517.4 days — outside the 365-day target window. Zero context bands in the 219-511 day range. On SIMPLE, this same band hit 499.3 days dead-on. The context encoding failed on HARD.

**The target encoding found it alone.** Three bands near target: 293.7d, 371.5d, 401.5d. The 128d target encoding has 4x more bands to search with, and it found the neighborhood. But the context encoding — the one that marks time in the token representations — did not. The model achieved 44.8% with only one encoding contributing useful temporal signal. On SIMPLE, both contributed, and the result was 88%.

This changes the story from SIMPLE. On SIMPLE, both encodings found the cycle, which challenged the "probability" explanation. On HARD, only one found it, and performance dropped by half. The probability explanation holds better here: with a harder frequency target (365d vs 500d, narrower expert windows), one of the two searches failed.

**The context encoding gave up on periodic.** Gate ratio flipped to 70% trend / 30% periodic (was 47/53 on SIMPLE). When the context encoding can't find the right frequency, it stops trying to be periodic and leans into the trend component. This is the encoding telling us it failed — the gate is a diagnostic signal.

**Temporal variance collapsed.** Context total variance: 0.85 (was 2.96 on SIMPLE). Only 20/32 dims vary over time (was 31/32). The context encoding is producing a nearly static output — it's not representing time effectively.

**Source embeddings learned the circular phase structure.** Despite weaker overall performance, the source embeddings show a striking pattern: adjacent sources in the phase cycle have high cosine similarity (src0↔src7: 0.74, src6↔src7: 0.65), opposite sources are strongly negative (src0↔src4: -0.83). The model discovered that sources are arranged on a circle — matching the sinusoidal phase offsets exactly. This is correct and meaningful, even though the timestamp encoding struggled.

**Two encoding dimensions partially correlated.** Dims 1 and 2 between context and target: r=0.64 and r=0.83. On SIMPLE, zero dims correlated. When both encodings succeed, they find different solutions (uncorrelated). When one struggles, it partially mirrors the other — possibly receiving gradient signal through the shared loss that pulls it toward whatever the other encoding found.

### What SIMPLE vs HARD Comparison Reveals

| Metric | SIMPLE (88%) | HARD (44.8%) |
|--------|-------------|--------------|
| Context finds cycle? | Yes (499.3d) | No (517.4d — missed) |
| Target finds cycle? | Yes (529.1d, 399.7d) | Yes (371.5d, 401.5d) |
| Context gate | 47% trend / 53% periodic | 70% trend / 30% periodic |
| Target gate | 11% trend / 89% periodic | 32% trend / 68% periodic |
| Context temporal variance | 2.96 | 0.85 |
| Cross-correlation | 0 dims > 0.5 | 2 dims > 0.5 |
| Source embedding structure | Pushed apart (negative cos) | Circular ordering (matches phase) |

The pattern: when the task is easy enough for both encodings to find the frequency, they produce completely independent solutions and the model excels. When the task is harder, one encoding fails, the other partially compensates, and performance drops roughly in half. The gate ratio is a reliable diagnostic — when an encoding shifts toward trend-dominant, it has failed to find a useful periodic signal.

The source embeddings are interesting: they learned more structure on HARD (circular phase ordering) than SIMPLE (just pushed apart). More sources with more phase offsets gave the model a richer structure to discover, and it found it — even though the timestamp encoding struggled. The source side of the model is working. The temporal side is the bottleneck.

### Next Steps

1. **Gradient flow analysis** — trace how gradients reach each encoding during training. Do they share pathways? Does one encoding's learning bootstrap the other's?
2. **Single-encoding parameter tracking** — run the same inspection on a failed one-encoding model. Does the encoding find the frequency but something else breaks? Or does it never find the frequency at all?
3. **Per-epoch inspection** — track freq_bands, gate ratios, and target-cycle FFT power every 10 epochs during training. Watch the frequency discovery process happen in real time. The gate ratio flipping from periodic to trend should be visible as a phase transition.
4. **Seed variation** — run Config A with 5 different seeds and compare which bands end up near the target. Is band 3 always the one? Or is the specific band arbitrary?

---

## Phase 17: Timestamp Normalization — A Potential Breakthrough on HARD

### The Idea

The `TimestampEncoding` receives raw Unix timestamps (~1.58 billion for 2020). At float32 precision, values near 1.5B have resolution of only ±128 seconds — one day's difference is ~86400 seconds, which is only ~675 float32 steps. The encoding must learn periodic functions over inputs where the signal occupies a tiny fraction of the representable range.

Normalizing timestamps to [0, 1] — mapping the full date range to the unit interval — doesn't encode any task-specific knowledge. It preserves the same temporal resolution (the relative spacing of all timestamps is identical). But it gives float32 full precision: ~16 million distinct values between 0 and 1, vs ~675 meaningful increments near 1.5 billion. The encoding sees the same temporal structure with much finer numerical granularity.

This is not cheating. It's the same operation as normalizing pixel values to [0, 1] — standard practice that doesn't embed task knowledge.

### Implementation

`NormalizedTimestampEncoding` subclasses `TimestampEncoding`. Registers `ts_min` and `ts_max` as buffers from training data range. Forward pass: `x_norm = (x - ts_min) / (ts_max - ts_min)`, then standard trend + gated periodic, but with `x_norm` replacing the raw timestamp. No `/86400` in the trend path (the input is already in a sane range).

### Results — Multi-Seed

| Config | Seed | Unnormalized | Normalized |
|--------|------|-------------|------------|
| SIMPLE | 42 | 88.0% | 68.2% |
| HARD | 42 | 44.8% | 66.4% |
| HARD | 123 | — | 74.8% |
| HARD | 7 | — | 73.3% |

**Mean normalized HARD: 71.5% ± 4.5pp.** Previous best single-seed: 63.9% (split LR, Phase 14).

**HARD normalized is a confirmed new all-time best.** Previous HARD record was 63.9% (split LR, Phase 14). The seed 123 result of 74.8% exceeds that by 11 percentage points. Even the weaker seed (42 at 66.4%) beats the old record.

**SIMPLE normalized is worse.** 68.2% vs 88.0%. This is surprising and important — normalization helps HARD but hurts SIMPLE. The same architectural change has opposite effects on different difficulty levels.

### Parameter Inspection — What Normalization Changes Internally

**Gate ratios tell the story.** On normalized HARD:
- Context: 76% trend / 24% periodic (seed 42)
- Target: 3.4% trend / 96.6% periodic (seed 42)

Compare to unnormalized HARD:
- Context: 70% trend / 30% periodic
- Target: 32% trend / 68% periodic

The target encoding went from 68% periodic (unnormalized) to **96.6% periodic** (normalized). Normalization let the target encoding commit almost entirely to its periodic component. This is a dramatic shift — the encoding is no longer hedging between trend and periodic. It found a frequency it trusts and is going all-in.

The context encoding stayed trend-dominant in both cases (~70-76% trend). The context encoding still struggles on HARD regardless of normalization.

**Frequency bands are more focused.** Normalized HARD target encoding found 6 bands near the 365-day cycle (245-509d range), vs 3 bands in unnormalized. More frequency bands in the right neighborhood means a richer representation of the target periodicity.

**Target temporal variance increased.** Normalized target variance: 63.3 (seed 42), vs 17.2 unnormalized. The target encoding produces much more temporally-varying output — it's encoding time more actively, not collapsing to near-constant values.

**Cross-correlation stayed low.** Mean |r| = 0.080 (normalized) vs 0.081 (unnormalized). The two encodings remain independent — normalization didn't change their relationship to each other.

### Why Normalization Helps HARD But Hurts SIMPLE

The SIMPLE problem (3 sources, 500-day cycle, threshold=0.0) was already solved at 88%. Both encodings found the cycle reliably. Normalization changes the input distribution that the encoding was already handling well. The learned freq_bands, phase offsets, and trend projections were calibrated for raw timestamps. Normalization changes the input scale by ~9 orders of magnitude, requiring completely different parameter values. On an easy problem where the existing parameterization works, this is a regression.

The HARD problem (8 sources, 365-day cycle, threshold=0.5) was NOT solved. The context encoding couldn't find the 365-day frequency from raw timestamps. Normalization changes the optimization landscape — with inputs in [0, 1], the gradient signal through `sin(x * exp(freq_band) * freq_scale + phase)` has different dynamics. The frequency bands that correspond to 365 days in normalized space are at different positions in log-frequency space than in raw-timestamp space. This apparently makes the frequency easier to find on HARD.

The key insight: **normalization doesn't help the encoding represent time better — it makes certain frequencies easier to discover during optimization.** It reshapes the loss landscape. On SIMPLE, the old landscape was fine. On HARD, the old landscape had the context encoding stuck; the new landscape apparently unsticks the target encoding enough to compensate.

### What The Plots Show

The FFT spectra for normalized HARD (both seeds) show clear peaks near the 365-day target in the target encoding. The context encoding FFT shows power at 200d and 500d but misses 365d — similar to the unnormalized case. The improvement is coming almost entirely from the target encoding's stronger periodic commitment.

The temporal variance plots confirm: target encoding has a smooth decay across ~70 active dimensions (vs ~50 unnormalized), context encoding has a sharp dropoff after ~15 dims (similar to unnormalized). Normalization specifically helped the target encoding use more of its capacity.

### Implications

1. **The target encoding is the key lever on HARD.** It's the one that responds to normalization. It's the one whose gate ratio shifts dramatically. The context encoding remains stubbornly trend-dominant on HARD regardless.

2. **Input preprocessing matters for frequency discovery.** Not because it changes what the encoding CAN represent (it can represent any frequency either way), but because it changes the optimization path. This suggests that the frequency discovery problem — the core bottleneck — is an optimization problem, not a capacity problem.

3. **Normalization should become standard for the HARD benchmark.** It's a free lunch on HARD (no task knowledge, no architectural change, consistent improvement across seeds so far). The new HARD baseline should be normalized Config A.

4. **The SIMPLE regression needs understanding.** If normalization is truly input-agnostic, it shouldn't hurt. The fact that it does suggests the encoding's initialization is tuned (accidentally) for raw timestamp magnitudes. This points to a potential improvement: initialization that's robust to input scale.

### Cross-Seed Parameter Comparison — Normalized HARD

This is what the model actually learns across three different random initializations:

| Parameter | Seed 42 (66.4%) | Seed 123 (74.8%) | Seed 7 (73.3%) | Unnorm seed 42 (44.8%) |
|-----------|-----------------|-------------------|-----------------|------------------------|
| **Target gate (periodic)** | 96.6% | *(see plot)* | **100.0%** | 67.8% |
| **Context gate (periodic)** | 24.0% | *(see plot)* | 34.9% | 30.3% |
| **Target bands near 365d** | 6 | *(see plot)* | 5 (incl. **364.0d**) | 3 |
| **Context bands near 365d** | 1 (433d) | *(see plot)* | 1 (409d) | 0 |
| **Target top FFT period** | 91d | *(see plot)* | **334d** | 334d |
| **Context top FFT period** | 200d | *(see plot)* | 4d | 501d |
| **Target temporal variance** | 63.3 | *(see plot)* | 59.7 | 17.2 |
| **Context temporal variance** | 0.72 | *(see plot)* | 1.09 | 0.85 |
| **Cross-correlation mean |r|** | 0.080 | 0.080 | 0.159 | 0.081 |
| **Context dims varying** | 15/32 | *(see plot)* | 9/32 | 20/32 |
| **Target dims varying** | 69/128 | *(see plot)* | 64/128 | 128/128 |

**Consistent patterns across seeds:**

1. **Target encoding goes all-in on periodic.** 96.6% → 100%. Unnormalized was only 67.8%. This is the single biggest change normalization produces. The encoding is so confident in its periodic component that it gates out the trend entirely.

2. **Target finds the 365d neighborhood reliably.** 5-6 bands in the 219-511d range across seeds. Seed 7 hit 364.0 days exactly. Unnormalized only found 3 bands there.

3. **Target temporal variance ~3.5x higher.** ~60 vs 17 unnormalized. The encoding is producing much richer temporal representations.

4. **Context encoding still struggles.** Gate stays trend-dominant (65-76% trend), only 1 band near 365d, low temporal variance. The context encoding on HARD is the consistent weak link across normalized and unnormalized.

5. **Two encodings remain independent.** Cross-correlation mean |r| between 0.08-0.16 across all seeds. They're solving the problem independently.

**The most striking finding: seed 7's target gate at exactly 100% periodic.** This means the trend component is contributing literally zero to the encoding output. The model discovered that for this problem, with normalized inputs, pure periodic encoding is optimal. The trend was noise that the gate learned to eliminate completely. This is what a well-functioning gate looks like — it's doing architecture search at inference time.

### What Seed 7 Found That Others Didn't

Seed 7 hit 364.0 days — essentially perfect frequency discovery on the target cycle. This is the closest any band has come to the true 365-day period across all experiments. And the FFT confirms it: the strongest target FFT peak is at 334 days (power=6.16), the highest target FFT power we've recorded.

Even seed 7's context encoding shows 334d in its FFT (power=2.13) — the first time a context encoding has shown ANY signal near 365d on the HARD problem. On unnormalized HARD, the context FFT peak was at 501 days. Normalization is helping both encodings, but the target encoding benefits far more.

### Ablation Results — Trust Proof Holds (trust_v7.py)

Full ablation on normalized HARD, seed 42, 100 epochs (`trust_v7.py --hard`):

| Model | Best Test Acc | vs Random (12.5%) |
|-------|-------------|-------------------|
| Baseline (Value Only) | 13.2% | at random |
| Temporal Only | 13.9% | at random |
| Source Only | 13.3% | at random |
| **Full Model** | **76.5%** | **+64pp** |

**The trust proof holds cleanly.** All three ablations are indistinguishable from random (12.5%). The full model at 76.5% is a new all-time HARD best — beating the previous record of 63.9% (split LR, Phase 14) by 12.6 percentage points.

This confirms the 71.5% mean from the inspection runs is real temporal-source learning, not a value-copy shortcut or any other artifact. The model needs BOTH temporal encoding AND source information to perform above chance. Neither alone is sufficient. Normalization didn't create a shortcut — it made the real learning easier.

Note: the full model hit 76.5% here vs 66.4% in the inspection run at the same seed (42). The difference is likely due to the ablation run training all 4 models with the same data loader, which changes the random state slightly. This further confirms the ~30pp seed/run variance on HARD, but the floor has risen: even the weakest normalized run (66.4%) beats the old unnormalized best (63.9%).

### Summary — Normalized Config A HARD

| Seed | Run Type | Full Model | Ablations |
|------|----------|-----------|-----------|
| 42 | Inspection | 66.4% | — |
| 42 | Ablation (v7) | **76.5%** | all ~13% |
| 123 | Inspection | 74.8% | — |
| 7 | Inspection | 73.3% | — |
| **Mean** | | **72.8%** | |
| **Previous best** | | **63.9%** | |

### The Context Encoding Was Hurting — Target-Only Beats Both

The parameter inspection showed the context encoding (32d) was consistently trend-dominant on normalized inputs — 74% trend on SIMPLE, 65-76% trend on HARD. The target encoding (128d) was doing virtually all the temporal work (96-100% periodic). This raised the question: is the context encoding actually helping, or is it injecting noise?

**Test: drop the context encoding entirely.** Keep source embeddings in the token representation, keep the target encoding for the cross-attention query. The self-attention tokens get `cat([v, s])` instead of `cat([v, t, s, t*s])`. The temporal signal comes only through the cross-attention query.

Results on SIMPLE (seed 42, `trust_v7.py --target-only-simple`):

| Model | Best Test Acc |
|-------|-------------|
| Full (both encodings) | 73.6% |
| **Target encoding only** | **94.5%** |

**94.5% is a new SIMPLE best.** It beats unnormalized Config A (88%) and demolishes the normalized two-encoding result (68-74%). The context encoding wasn't just failing to contribute — it was actively degrading performance. Its trend-dominant output was polluting the token representations, adding noise that the self-attention had to work around.

This makes sense mechanically: when the context gate is 74% trend / 26% periodic, the temporal component of every token is mostly a linear function of the timestamp. The `t*s` multiplicative interaction then becomes mostly `linear(timestamp) * source_embedding` — a weak signal. Meanwhile the source embedding `s` on its own is useful (it identifies which source made the prediction). Concatenating a mostly-linear `t` and a mostly-linear `t*s` is adding 2*d_model dimensions of near-noise to the token representation. The self-attention must learn to ignore them.

With target-only, the tokens are `cat([v, s])` = 2*d_model = 64 dims. Cleaner, smaller, all signal. The temporal information enters purely through cross-attention, where the 128d target encoding with 99.99% periodic gate provides a strong, clean temporal query.

**Prediction: removing the context encoding will also help HARD.** The same pattern holds — context encoding is trend-dominant, target encoding is doing the temporal work. If anything, the effect should be larger on HARD because the context encoding is even more trend-dominant there (65-76% trend vs 74% on SIMPLE).

### Updated Architecture Understanding

The original Config A hypothesis was that two encodings help because they provide two independent attempts at frequency discovery. The parameter inspection (Phase 16) refined this: the two encodings learn independent but complementary representations.

Now we have a third revision: **the context encoding is unnecessary and harmful when normalized.** The target encoding alone, at 128d with near-100% periodic gate, provides sufficient temporal signal through cross-attention. The context encoding's trend-dominant output adds noise to the token representations.

This suggests the real architecture for normalized timestamps is:
- Token: `cat([v, s])` — value + source, no temporal in tokens
- Self-attention: learns source relationships from values and identities
- Cross-attention query: `target_t_enc(timestamp)` — temporal signal enters here
- The model learns WHAT to look for in self-attention, and WHEN to apply it via cross-attention

### Outstanding Questions

### Target-Only on HARD — Context Encoding IS Needed

Target-only (no context encoding) on HARD: **58.9%** (collapsed after epoch 10). Full two-encoding: 67.0%. Removing the context encoding hurts on HARD — opposite of SIMPLE. Even a weak, trend-dominant temporal signal in the token representations helps self-attention learn source patterns on the harder problem.

### Shared Encoding — One Encoding, Used Twice (encoding_suite.py)

Inspired by how positional encodings are used in multiple places in standard transformers: one `NormalizedTimestampEncoding` at d_model_total (128d), projected down to d_model (32d) for the token concat, used directly at 128d for cross-attention query. Same parameters, gradients from both paths.

**HARD results (3 seeds):**

| Seed | Shared (1 enc) | Two enc (normalized) | Gate (periodic) | Bands near 365d |
|------|---------------|---------------------|-----------------|-----------------|
| 42 | 69.0% | 66.4-76.5% | 52.5% | 4 (incl 367d) |
| 123 | 75.7% | 74.8% | 81.0% | 5 (incl 382d) |
| 7 | 67.9% | 73.3% | 75.0% | 5 (incl 359d) |
| **Mean** | **70.9%** | **72.8%** | | |

**One shared encoding matches two separate encodings on HARD.** Mean 70.9% vs 72.8% — within noise. Fewer parameters. The shared gradient signal from both paths (self-attention context + cross-attention query) trains the encoding from two complementary perspectives simultaneously.

The gate ratios are telling: 52-81% periodic. When the context encoding was separate and 32d, it went 65-76% trend (failing). When it shares parameters with the 128d encoding, the gate finds a healthy balance — periodic enough to encode time, but with some trend contribution. The cross-attention gradient path (which pushes toward periodic) now directly benefits the context path too.

**SIMPLE shared encoding:** *(running, seed 42)*

### Architecture Evolution Summary

| Architecture | SIMPLE | HARD | Encodings |
|-------------|--------|------|-----------|
| Config A unnormalized (v4) | 88% | 44.8% | 2 separate |
| Config A normalized (v7) | 68% | 72.8% mean | 2 separate normalized |
| Target-only normalized | **94.5%** | 58.9% | 1 (query only) |
| **Shared normalized** | 75.7% (200ep) | **70.9% mean** | **1 shared** |

### Shared Encoding on SIMPLE — Slow Learner, Still Climbing

Shared encoding on SIMPLE (seed 42, 200 epochs):

| Epoch | Best Test Acc |
|-------|-------------|
| 50 | 55.2% |
| 100 | 63.4% |
| 130 | 71.5% |
| 150 | 74.3% |
| 190 | 75.3% |
| 200 | **75.7%** |

The trajectory is a **slow, consistent climb** — not noisy oscillation. The best-checkpoint ratchets up steadily. At 100 epochs it was 63.4% (we initially reported 70.5% from the first run — slight seed variance). At 200 it's 75.7% and clearly not converged.

Compare: target-only normalized SIMPLE hit 94.5% in 100 epochs. Shared encoding needs 2x the epochs and still hasn't matched it. The shared architecture learns slower on SIMPLE because the two gradient paths (context projection and cross-attention query) want different things from the encoding — the gate is at 58.6% trend / 41.4% periodic, a compromise.

Parameters at 200 epochs:
- Gate: 58.6% trend / 41.4% periodic — unchanged from 100 epochs. The gate settled early.
- 5 bands near 500d (including 490d) — frequency discovery complete
- FFT peak at 501d — correct

The slow climb with a settled gate suggests the encoding found the right frequency early, but the downstream layers (self-attention, cross-attention, projection) are slowly learning to use it better. The bottleneck isn't frequency discovery — it's the model learning to route the shared encoding's output through two different paths effectively.

### Updated Architecture Comparison

| Architecture | SIMPLE (best) | HARD (mean 3-seed) | Epochs | Encodings |
|-------------|--------------|-------------------|--------|-----------|
| Config A unnormalized (v4) | 88% | 44.8% | 100 | 2 separate |
| Config A normalized (v7) | 68% | 72.8% | 100 | 2 separate normalized |
| Target-only normalized | **94.5%** | 58.9% | 100 | 1 (query only) |
| Shared normalized (100ep) | 70.5% | 70.9% | 100 | 1 shared |
| Shared normalized (200ep) | 75.7%+ | — | 200 | 1 shared |

Target-only is the clear SIMPLE winner. For HARD, shared and two-enc normalized are comparable. The question remains whether shared at 200+ epochs on HARD would also improve.

### Outstanding Questions

- Would 200+ epochs on shared HARD also show continued climbing?
- Why does target-only fail on HARD but crush SIMPLE? What does the context path provide on HARD specifically?
- Can we build an adaptive architecture that uses target-only when context isn't helping?
- The gate settled at 58% trend early and never moved — is there a way to encourage more periodic commitment in shared mode?

---

## Integration Checkpoint (2026-03-25)

### The Breakthroughs — What Actually Mattered

Looking across the full research arc, these are the real breakthroughs — the results that changed the trajectory, not incremental tuning:

1. **Value dropout** (v2, Mar 8) — Broke the copy shortcut. Without it, the model memorizes timestamp→value mappings (99% train, 40% test). With it, the model is forced to learn the periodic trust pattern. Load-bearing in every subsequent experiment.

2. **Multiplicative interaction `t*s`** (v3, Mar 10) — Directly represents the bilinear trig decomposition of sinusoidal expertise: `sin(2π(t+φ)/C) = sin(2πt/C)·cos(2πφ/C) + cos(2πt/C)·sin(2πφ/C)`. Element-wise multiplication of time and source embeddings computes these cross-terms. Removing it drops accuracy by 30pp.

3. **Bug fixes: distributional + argmax expert** (v3/v4, Mar 10-11) — Non-expert noise was uniform (detectable), and threshold-based selection allowed multiple simultaneous experts (consensus shortcut). Argmax selection + normal noise for all sources eliminated both. First clean proof: 87% Full, all ablations at random.

4. **Split LR** (Phase 9, Mar 11) — Encoding at lr=1e-3 (needs to traverse 13 orders of magnitude in frequency space), everything else at lr=3e-4 (smaller corrections prevent coupled oscillation). HARD from 47% to 64%. Revised the "vanilla Adam only" rule to "split Adam."

5. **Parameter inspection** (Phase 16, Mar 25) — First time we looked inside a working model. Gate ratios, freq bands, FFT of encoding outputs, source embedding geometry. Revealed that both encodings find the 500d cycle on SIMPLE, that the context encoding fails on HARD (gate shifts to trend-dominant), that source embeddings learn circular phase structure. Every subsequent finding came from reading the parameters. This is the single most important methodological discovery.

6. **Timestamp normalization** (Phase 17, Mar 25) — Mapping timestamps to [0,1] preserves relative spacing, gives float32 full precision. HARD from 44.8% to 72.8% mean (3-seed). Target encoding gate went from 68% periodic to 96-100% periodic. Not cheating — same as normalizing pixel values. The biggest single-change improvement in the entire research.

7. **Target-only architecture** (Phase 17, Mar 25) — Removing the context encoding on normalized SIMPLE: 94.5% (new all-time best). The parameter inspection showed the context encoding was trend-dominant (74% trend) — its output was noise in the token representations. Temporal signal enters purely through cross-attention query. But target-only collapses on HARD (58.9%) — the weak context signal IS needed for the harder problem.

8. **Shared encoding** (Phase 17, Mar 25) — One `NormalizedTimestampEncoding` at 128d, projected to 32d for context, used directly for cross-attention. Matches two separate encodings on HARD (70.9% vs 72.8% mean). Fewer parameters, one representation of time. Slow learner on SIMPLE (75.7% at 200ep, still climbing).

### What Didn't Matter (or Was Noise)

- **Gradient clipping** — Appeared to help (+7pp) in Phase 11 but follow-up showed the improvement was within run-to-run variance (~30pp). Cannot reliably distinguish optimizer configs at N=1.
- **LR schedules (warmup, cosine decay)** — Consistently hurt. The trust task doesn't have a "converging to a solution" phase. The encoding might shift frequencies at epoch 50; non-encoding params must stay responsive.
- **Reverse curriculum** — Worked alone (vd=0→0.1) but CONFLICTED with split LR. The two solve oscillation through opposing mechanisms (aggressive early learning vs conservative throughout).
- **More data / more capacity** — d=128 learned nothing (overfitting). 60k samples was worse than 30k (sparser per-phase). The bottleneck was never capacity or data quantity.
- **Frequency hints** — A diagnostic confirming frequency discovery is the bottleneck. Additive + hints achieved 85.8% ± 0.7%. But hints require knowing the cycle length in advance — not a solution, just a confirmation of the problem.
- **Stop-gradient on `t*s`** — Proved the gradient coupling through the multiplicative interaction is necessary, not harmful. The encoding needs that gradient signal.
- **Multi-task training** — Failed (51.3% mean). Multiple cycle lengths didn't help frequency discovery at this scale.

### Current State of the Art

| Architecture | SIMPLE (best) | HARD (mean 3-seed) | Encodings |
|-------------|--------------|-------------------|-----------|
| Config A unnormalized (v4) | 88% | 44.8% | 2 separate |
| Config A normalized (v7) | 68% | 72.8% | 2 separate normalized |
| Target-only normalized | **94.5%** | 58.9% | 1 (query only) |
| Shared normalized (100ep) | 70.5% | 70.9% | 1 shared |
| Shared normalized (200ep) | 75.7%+ | — | 1 shared |

### The Foundation Test — Does the Original Proof Still Hold?

`time_transformer_final.py` proved that `TimestampEncoding` outperforms no-encoding on synthetic time series prediction. This is the foundational result the entire research builds on. With normalization now standard for trust, the question: does `NormalizedTimestampEncoding` still beat no-encoding on the original time series task?

**Test in progress** (`test_normalized_foundation.py`): three models trained on 50k random-context time series samples — original encoding, normalized encoding, no encoding. 100 epochs, seed 42.

**Preliminary results (epoch 70/100):**

| Model | Train Loss | Test Loss | Status |
|-------|-----------|-----------|--------|
| Original encoding | 0.0003 | 0.001 | Converged, working |
| Normalized encoding | 0.0002 | 0.56-1.2 | Severe overfitting |
| No encoding | 0.37 | 0.69 | Cannot solve (expected) |

**The normalized encoding memorizes perfectly (train near 0) but completely fails to generalize.** The original encoding generalizes well. This is the opposite of what normalization does in the trust task — there, normalization helps generalization.

**Hypothesis:** The trust task has **value dropout** which prevents memorization. The time series prediction task has no value dropout. Without dropout, the normalized encoding memorizes the training set through its trend pathway (the gate goes trend-dominant). The original encoding also memorizes somewhat but generalizes better because the raw-timestamp inputs provide a natural regularization — the large input magnitudes make the optimization landscape less prone to sharp memorization minima.

**Next step:** Add value dropout to the foundation test and rerun. If the normalized encoding generalizes with dropout, the lesson is universal: **normalization requires value dropout to prevent memorization.** They are a package deal — normalization gives the encoding better numerical precision, but that same precision makes memorization easier. Dropout breaks the memorization path.

This would be a satisfying unification: the two breakthroughs (value dropout from v2, normalization from Phase 17) aren't independent techniques — they're complementary. Normalization opens the door; dropout ensures the model walks through it toward generalization rather than memorization.

---

## Phase 18: Fixing the Normalized Foundation Proof (2026-03-26)

### The Diagnosis: Gradient Starvation

The normalized encoding wasn't memorizing — it was **dying**. Parameter inspection: 0/64 active dims, temporal variance = 0.00. The encoding produced constant output regardless of timestamp.

**Root cause:** The gradient `d(loss)/d(freq_band)` through `sin(x * freq)` is proportional to `x`. With raw seconds x ≈ 1.58e9, freq_bands get enormous gradient. With x_norm ∈ [0,1], freq_bands get ~1e9 times less gradient — they can't move. In trust, the multiplicative `t*s` interaction amplifies gradients through source embeddings. The time series task has no such amplifier.

### The Fix

One-line change in `NormalizedTimestampEncoding.forward()`: multiply `x_norm` by `ts_range` before passing to `sin()`. Trend path still gets [0,1]. Periodic path gets seconds-scale magnitude for gradient amplification.

### Results (seed=42, 50 epochs, split LR)

| Model | Test Loss (MSE) | vs No Encoding |
|-------|----------------|---------------|
| Normalized (range-scaled) | 0.020 | 39.5x better |
| Original | 0.040 | 20.1x better |
| No encoding | 0.806 | baseline |

Parameter inspection confirms the encoding is alive: gate 50/50 trend/periodic, 10/64 active dims, temporal variance = 1.75, real FFT structure. The normalized encoding now outperforms the original by 2x on test loss.

### Trust Results with Range-Scaled Encoding (seed=42)

`trust_v7.py` now imports the range-scaled `NormalizedTimestampEncoding` from `time_transformer_proof.py`.

| Config | Range-Scaled | Ablations | Random |
|--------|-------------|-----------|--------|
| SIMPLE (50ep) | 85.3% | 35-36% | 33.3% |
| HARD (100ep) | 76.8% | 13% | 12.5% |

Trust proof holds — all ablations at random on both configs.

**Note:** These trust results were accidentally run with two separate encodings (the old Config A architecture). trust_v7.py has been corrected to use one shared encoding. Shared encoding results with range-scaled normalization pending.

### Updated Architecture Comparison

| Architecture | SIMPLE (best) | HARD (best seed) | Encodings |
|-------------|--------------|-----------------|-----------|
| Target-only normalized | **94.5%** | 58.9% | 1 (query only) |
| Config A unnormalized (v4) | 88% | 44.8% | 2 separate |
| Range-scaled normalized (two enc) | 85.3% | **76.8%** | 2 separate |
| Config A normalized old (v7) | 68% | 76.5% | 2 separate |
| Shared normalized (200ep) | 75.7%+ | 70.9% mean | 1 shared |
| **Range-scaled shared (v7)** | **89.6%** | **74.5%** | **1 shared** |

The range-scaled encoding fixed the foundation proof (normalized now outperforms original on time series by 2x) and improved trust SIMPLE by 17pp over the old normalization with two encodings (85.3% vs 68%). HARD is comparable to previous best (76.8% vs 76.5%).

**Shared encoding + range-scaling results (seed=42):**
- SIMPLE: 89.6% best test (50 epochs). Ablations at random (~35%). Up from 75.7% (old shared normalized at 200ep).
- HARD: 74.5% best test (100 epochs). Ablations at random (~13%). Up from 70.9% mean (old shared normalized).
- HARD was still climbing at epoch 90 (73.3%) — more epochs may improve further.

The key result: one shared encoding with range-scaling now works on **both** time series prediction and trust learning. The old normalization worked on trust (via `t*s` gradient amplification) but died on time series. Range-scaling fixes time series without hurting trust. The shared encoding is now the best single-encoding architecture for both SIMPLE and HARD, closing the gap with the two-encoding results (85.3%/76.8%) while using a cleaner, more principled architecture.

### 3-Seed Benchmark (200 epochs, full model only)

| | Seed 42 | Seed 123 | Seed 7 | **Mean** |
|---|---|---|---|---|
| **SIMPLE** | 83.1% | 93.5% | 93.5% | **90.0%** |
| **HARD** | 78.0% | 71.0% | 65.8% | **71.6%** |

Parameter inspection across seeds:

| | Seed 42 | Seed 123 | Seed 7 |
|---|---|---|---|
| **SIMPLE gate (trend/periodic)** | 48/52 | 29/71 | 7/93 |
| **SIMPLE active dims** | 124/128 | 121/128 | 125/128 |
| **SIMPLE temporal variance** | 11.96 | 24.36 | 30.77 |
| **HARD gate (trend/periodic)** | 0.3/99.7 | 0/100 | 51/49 |
| **HARD active dims** | 63/128 | 108/128 | 19/128 |
| **HARD temporal variance** | 36.15 | 41.66 | 8.06 |

**Key observations:**
- SIMPLE is robust: 90.0% mean across 3 seeds. Seed 42 is the weak link (83.1%) — the gate stays near 50/50 trend/periodic instead of committing to periodic.
- HARD seed=42 at 78.0% is a new all-time HARD best (previous: 76.8% with two encodings).
- HARD seed=7 failed to discover periodic structure: gate stuck at 51/49, only 19/128 active dims, temporal variance 8.06. This is the frequency discovery problem — when the gate doesn't commit to periodic, the encoding can't represent the cyclical expertise pattern.
- HARD variance is the bottleneck: 65.8%–78.0% range (12.2pp spread) vs SIMPLE's 83.1%–93.5% (10.4pp).
- On HARD, the encoding consistently goes fully periodic (gate → 0/100) when it succeeds — the trend component is useless for cyclical expertise. Seed 7's failure to discover this is the outlier.

### 500-Epoch Training Curves (seed=42)

Ran SIMPLE and HARD for 500 epochs to see whether more training helps and to chart train/test dynamics over time. Chart: `plots/v7_500ep_curves.png`.

| | 200ep best | 500ep best |
|---|---|---|
| **SIMPLE** | 83.1% | 88.8% |
| **HARD** | 78.0% | 68.3% |

**What the curves show:**

1. **The model is genuinely generalizing, not memorizing.** Test accuracy on SIMPLE sits at 75-88% for hundreds of epochs (well above 33.3% random). HARD test sits at 40-68% (well above 12.5%). If the model were memorizing training data, test would collapse toward random — it doesn't.

2. **More epochs don't help.** SIMPLE peaks around epoch 80-150 then oscillates for the remaining 350 epochs with no upward trend. HARD peaks early and the best-checkpoint approach just catches lucky snapshots. The learning is done by epoch ~100-200; beyond that, we're rolling dice.

3. **Training stability is the bottleneck.** Test loss on both configs shows massive oscillation — swinging from 0.05 to 1.0+ on SIMPLE, similar on HARD. Train loss is smooth and low. The model finds good solutions then loses them epoch to epoch. This isn't a capacity or data problem — it's an optimization problem.

4. **HARD is more unstable than SIMPLE.** HARD test accuracy swings 40pp within spans of 20-30 epochs. Train accuracy also oscillates (60-80%) unlike SIMPLE's stable 92%. The 8-source problem with 365-day cycles is harder to lock onto and easier to lose.

### What We Know Now (Phase 18 Summary)

**Proven:**
- The architecture works. One shared encoding + concatenated subspaces + cross-attention query = genuine generalization on both SIMPLE and HARD.
- Range-scaled normalization fixed the gradient starvation problem. The encoding is alive (real FFT structure, active dims, meaningful gate values).
- Ablations confirm the mechanism: all three legs of the triplet (value, time, source) are necessary. Remove any one and accuracy drops to random.

**The remaining gap:**
- SIMPLE: 90% mean (3-seed), should be higher with stable training.
- HARD: 71.6% mean (3-seed) with high variance (65.8-78.0%). Seed-dependent gate discovery and training instability are both contributing.

### Phase 19: Hyperparameter Experiments (2026-03-27)

After the 500-epoch training curves revealed test oscillation, we ran a series of experiments to find the next lever. Important caveat: these experiments interact with each other and with the architecture changes from Phase 18. A variable that did nothing on an older architecture may matter now, and vice versa. Results should be read as data points in a complex landscape, not definitive claims.

#### LR Scheduling (seed=42, 200 epochs, SIMPLE + HARD)

| Schedule | SIMPLE | HARD |
|---|---|---|
| **Constant (1e-3)** | **83.0%** | **72.3%** |
| Cosine decay | 81.1% | not completed |
| Warmup + cosine | 74.4% | not completed |

LR scheduling hurt. Constant LR won on both configs. The hypothesis was that late-training oscillations came from overshooting a found solution, and decaying the LR would let the model settle. This was wrong, or at least wrong for these specific schedules and this architecture. The oscillation has a different cause.

#### Value Dropout Sweep (seed=42, 100 epochs, SIMPLE)

| Dropout | Best Test |
|---|---|
| 0.0 | 77.7% |
| 0.3 | 79.6% |
| 0.5 | 73.5% |
| 0.7 | 81.3% |
| 0.9 | 73.6% |

All within seed variance. Value dropout level is not a significant lever. The differences here are smaller than the 10pp seed variance we see in the 3-seed benchmarks.

#### Data Size (seed=42, 100 epochs, SIMPLE, all data per epoch)

| Samples | Best Test |
|---|---|
| 15,000 | 87.5% |
| 50,000 | **95.6%** |
| 150,000 | 90.9% (plateaued at ep 80, each epoch 10x longer) |

**95.6% is a new all-time SIMPLE best under any configuration.** More data works. The 50k result beat the previous best (93.5%, seeds 123/7 at 200ep) on the historically weak seed 42.

However, the 50k run saw all 50k samples every epoch (~780 batches vs ~234 for 15k). This means 3.3x more gradient steps per epoch, confounding data diversity with compute. The 150k run plateaued at 90.9% despite even more compute per epoch, suggesting diminishing returns or that the model overfits within each massive epoch.

#### Data Size with Fixed Batches Per Epoch (seed=42, SIMPLE)

To isolate data diversity from compute per epoch, reran with all configurations seeing exactly 188 batches per epoch (matching 15k baseline) using `RandomSampler` with replacement. Larger datasets provide diversity across epochs, not more compute within them.

| Samples | Epochs | Best Test |
|---|---|---|
| 15,000 | 150 | 80.1% |
| 50,000 | 300 | 90.3% |
| 150,000 | 400 | 92.3% |
| 500,000 | 500 | 72.5% |
| 100,000 | 1,000 | **96.3%** |

**96.3% is a new all-time SIMPLE best under any configuration, any seed.** The 100k/1000ep run's parameter inspection showed gate 73/27 trend/periodic, 16/128 active dims.

The replacement-based sampling hurts small pools (15k scored 80.1% vs 87.5% with normal iteration). The 500k run didn't have enough epochs to make use of its pool. The 100k run reached 96.3% within 1000 epochs. For comparison, 50k with all data per epoch reached 95.6% in 100 epochs.

#### What We've Learned

1. **LR scheduling is not a lever.** Constant LR beat cosine and warmup+cosine on both SIMPLE and HARD.
2. **Value dropout is not a lever.** Tested 0.0 through 0.9 on SIMPLE; all within seed variance.
3. **More data helps.** Both the all-data runs and fixed-batch runs show improvement with more samples, up to a point.
4. **We haven't found an efficient way to use more data.** 50k with all data per epoch reached 95.6% in 100 epochs. 100k with fixed batches reached 96.3% but needed 1000 epochs. The relationship between pool size, sampling strategy, and epoch budget is not yet understood.
5. **HARD data scaling: 200K/1000ep reached 84.1%.** New all-time HARD best (previous: 78.0%). Hit 84.1% around epoch 280, then oscillated 45-72% for the remaining 300 epochs without improving. Train accuracy stayed at 83-85% throughout — notably lower than the 95%+ train accuracy seen with smaller datasets. The model wasn't memorizing individual samples, but the test instability pattern from earlier HARD experiments persisted.

#### Proof Run: 200K Samples, 500 Epochs, SIMPLE + HARD (seed=42)

Ran both configs with full model (500 epochs) + 3 ablations (100 epochs each), plus diagnostic visualizations: confusion matrix, expert cycle vs model predictions over time, parameter inspection.

| Config | Full Model | Baseline | Temporal Only | Source Only | Random |
|---|---|---|---|---|---|
| SIMPLE | **87.3%** | 33.5% | 33.5% | 33.2% | 33.3% |
| HARD | **86.9%** | 13.0% | 13.0% | 13.0% | 12.5% |

All ablations at random. Trust learning requires both temporal and source information together.

**Parameter inspection:**

| | Gate (trend/periodic) | Active dims | Temporal variance |
|---|---|---|---|
| SIMPLE | trend-heavy (not recorded exactly) | 78/128 | 8.38 |
| HARD | 51/49 (nearly even) | 17/128 | 7.64 |

HARD used a more balanced gate but far fewer active dimensions (17 vs 78). Different solutions to different problems.

**Per-source accuracy (test set):**
- SIMPLE: S0 not in test window, S1: 92%, S2: 80%
- HARD: S0: 91%, S4: 79%, S5: 87%, S6: 85%, S7: 88%. S1-S3 not in test window.

Sources not in the test window reflect the chronological split — the last 200 days of a 1000-day range doesn't cover every expert's phase.

**Error analysis (expert cycle vs predictions chart):**

From the diagnostic visualizations: **errors are more frequent at expert transition boundaries but also present during stable phases.** At handoff points accuracy drops to 20-40%. During stable phases accuracy is typically 80-95% — better but not perfect. This pattern holds on both SIMPLE and HARD.

On HARD with 8 sources and a 365-day cycle, expert windows are shorter and transitions more frequent, so there are more dips. When the model is wrong, the confusion matrix shows errors spread roughly uniformly across non-expert sources — it's not systematically confusing specific pairs. At transitions it's essentially guessing; during stable phases it still makes some errors but at a lower rate.

#### Future: Trend vs Periodic Gate Behavior

The encoding gates between a trend (linear) component and a periodic (sinusoidal) component. For our trust problem the expert cycle is a pure sine — the ideal solution should be fully periodic. But the model often learns trend-heavy solutions (e.g. 73/27 trend/periodic on the 100k/1000ep SIMPLE run) and still scores well.

This raises questions for harder problems down the line:
- **Trend + periodicity**: What if the expert cycle drifts over time (e.g. seasonal but with growing amplitude)? The model would need both components working together — trend for the drift, periodic for the cycle.
- **Non-sinusoidal repeating patterns**: A repeating sequence like AABCBADBCA is purely periodic but not sinusoidal. The periodic component should still handle this (multiple frequency bands can compose arbitrary repeating patterns), but the gate behavior might differ.
- **Regime changes**: A one-time structural shift (rules change permanently at some date) is pure trend, no periodicity. The gate should go fully trend-heavy.

Understanding why the model prefers trend on a purely periodic problem — and whether that limits its ceiling — is worth revisiting once the simpler cases are more fully understood.

---

## Phase 20: Multi-Scale Context Window (Mar 28, 2026)

### Motivation

Previous architectures showed each training sample a single timestamp with N source predictions. The model had to piece together the expertise cycle across thousands of independent single-timestamp examples over many epochs. The cycle structure was never visible within a single training example.

Multi-scale context window changes this: each sample sees K=8 timestamps, each with all N source predictions. Loss is computed across all K targets — 8x gradient signal per sample. Different samples span different time scales via log-uniform span selection, from ~1 hour to the full date range.

### Architecture Change

**Sampling:** Each sample picks a random span size (log-uniform from K indices to full range), places K=8 timestamps evenly within it. Different samples see different scales. With 200K points over ~1000 days, each index is ~7 minutes apart, so spans range from ~1 hour (K=8 indices) to ~1000 days. The timestamp encoding supports scales down to sub-second, but the data resolution limits testing to ~minutes and above.

```python
span = int(np.exp(np.random.uniform(np.log(K), np.log(num_datapoints - 1))))
start = np.random.randint(0, num_datapoints - span)
indices = np.linspace(start, start + span, K, dtype=int)
```

**Model:** Self-attention across K×N tokens (K timestamps × N sources). Cross-attention: K queries (timestamp encoding at each target) attend to K×N context. Output: K predictions, one per timestamp.

**Token structure unchanged:** `cat([v, t, s, t*s])` — value, projected timestamp encoding, source embedding, multiplicative interaction. Each in its own d_model=32 subspace, total 128 dims.

### Results (bench_v7_multiscale.py, 200K samples, 100 epochs, 375 fixed batches/epoch, seed=42)

| Config | Best Test Acc | Previous Best | Total Epochs |
|--------|--------------|---------------|--------------|
| **HARD Full** | **88.6%** | 86.9% (Phase 19, 500ep) | 100 |
| **SIMPLE Full** | **97.3%** | 96.3% (Phase 19, 1000ep) | 100 |

Both are new records. The run used 100 total epochs vs 500-1000 in Phase 19. The exact epoch where best test was achieved was not recorded.

**Parameter inspection:**

| | Gate (trend/periodic) | Active dims |
|---|---|---|
| HARD | 0.368 / 0.632 | 89/128 |
| SIMPLE | 0.204 / 0.796 | 108/128 |

Both models shifted toward periodic-dominant gating compared to Phase 19 (HARD was 51/49, now 63% periodic). Active dims increased substantially (HARD: 17 → 89, SIMPLE: 78 → 108).

**Ablations:**

| Config | Baseline | Temporal Only | Source Only | Random |
|--------|----------|---------------|-------------|--------|
| HARD | 12.8% | 12.8% | 12.7% | 12.5% |

All ablations at random. Note: an initial source-only run showed 14.2% due to a bug where the timestamp encoding leaked into the cross-attention query when `use_temporal=False`. After fixing (mean pooling fallback, matching trust_v7.py), source-only dropped to 12.7% — random.

### What this shows

1. **Multi-scale context window produced new records on both HARD and SIMPLE.** HARD 88.6% (prev 86.9%), SIMPLE 97.3% (prev 96.3%), both in 100 epochs instead of 500-1000.

2. **More active dimensions and stronger periodic gating.** The model uses more of the encoding (89/128 and 108/128 active dims vs 17 and 78 previously) and leans more toward the periodic path. With the cycle structure visible within individual training examples, the encoding appears to activate more of its capacity.

3. **Ablations remain at random.** The trust triplet (source + time + content) is still the only combination that learns.

4. **Efficiency gain.** 100 epochs × 375 batches = 37,500 gradient steps. Previous best needed 500-1000 epochs × 375 batches = 187,500-375,000 steps. Roughly 5-10x fewer steps to a better result.

### Ablation bug found and fixed

When adapting the multi-scale architecture from trust_v7.py, the `use_temporal=False` code path was implemented incorrectly: the timestamp encoding was still computed and used as the cross-attention query even when temporal information was supposed to be ablated. This caused source-only to score 14.2% instead of the expected ~12.5%.

The fix: when `use_temporal=False`, replace cross-attention with mean pooling over source tokens per timestamp (matching trust_v7.py's fallback). After the fix, source-only dropped to 12.7% — random. The bug did not affect full model results or temporal-only ablations (both have `use_temporal=True`).

### Test window coverage bug

The bench_v7_multiscale results (88.6% HARD, 97.3% SIMPLE) used a date range of 2020-01-01 to 2022-09-27 (~1000 days). With a chronological 80/20 split, the test window was ~200 days — less than one full HARD cycle (365 days) or SIMPLE cycle (500 days). The confusion matrix revealed that sources 1, 2, 3 had zero test samples (never expert during the test window), and source 4 had only 9.5%. The 88.6% HARD number was measured on 5 of 8 sources.

This doesn't invalidate the architecture or the ablation proof, but the accuracy numbers are not fully representative.

### trust_v8.py — Self-Contained Proof (Mar 28-29, 2026)

Self-contained proof file importing only from `time_transformer_proof.py`. Incorporates the multi-scale context window, the ablation fix (mean pooling when `use_temporal=False`), and an extended date range to ensure full cycle coverage in the test window.

**Date range fix:** Extended from 2020-2022 (~1000 days) to 2015-2022 (~2827 days). The test window (last 20% = ~565 days) now covers 1.55 HARD cycles and 1.13 SIMPLE cycles. All sources appear in the test set.

#### v8 Full Results (400K samples, 200 epochs full / 50 epochs ablation, 375 fixed batches/epoch, seed=42)

| Config | Best Test Acc | Random |
|--------|-------------|--------|
| **HARD Full** | **88.5%** | 12.5% |
| **SIMPLE Full** | **96.7%** | 33.3% |
| HARD Baseline (Value Only) | 12.5% | 12.5% |
| SIMPLE Baseline (Value Only) | 33.5% | 33.3% |
| HARD Temporal Only | 12.5% | 12.5% |
| SIMPLE Temporal Only | 33.4% | 33.3% |
| HARD Source Only | 12.5% | 12.5% |
| SIMPLE Source Only | 33.4% | 33.3% |

All 8 ablations at random. Both full models well above random. The trust triplet holds cleanly.

**Parameter inspection:**

| | Gate (trend/periodic) | Active dims | Temporal variance |
|---|---|---|---|
| HARD | 0.165 / 0.835 | 39/128 | 14.37 |
| SIMPLE | 0.419 / 0.581 | 41/128 | 18.19 |

HARD shifted strongly toward periodic (83.5%), the most periodic-dominant result across all phases. SIMPLE is more balanced (58.1% periodic). Both use ~40/128 active dimensions.

**Confusion matrix (HARD):** All 8 sources present in test. Per-source accuracy: S0 90%, S1 86%, S2 89%, S3 89%, S4 88%, S5 88%, S6 89%, S7 88%. Clean diagonal — when wrong, errors spread uniformly across non-expert sources. No systematic pair confusion.

**Confusion matrix (SIMPLE):** All 3 sources at 96-97%. Counts more evenly distributed than HARD.

**Expert cycle (HARD):** Model picks track the true expert across the full test window. Accuracy stays 80-95% during stable expert phases. Errors concentrate at transition boundaries but the dips are shallower than Phase 19 (~50-70% at transitions vs 20-40% previously).

**Expert cycle (SIMPLE):** Near-perfect tracking. Accuracy 90%+ during stable phases, dips to ~60-70% at the 5 transition boundaries, recovers immediately. Overall 96.8%.

**Training curves:** Both HARD and SIMPLE show train/test tracking closely with no memorization gap. HARD oscillates more widely (20-30pp dips) but the best-test ceiling climbs steadily throughout 200 epochs — the accuracy envelope was still rising at epoch 200. SIMPLE shows less oscillation, with accuracy plateauing in the mid-90s by epoch 100.

#### What this shows

1. **88.5% HARD on all 8 sources.** Comparable to the bench result (88.6%) but now measured on a representative test set. The bench number was inflated by testing only 5 of 8 sources.

2. **96.7% SIMPLE on all 3 sources.** Slightly below bench (97.3%) for the same reason — a harder, more representative test.

3. **All 8 ablations at random.** The trust triplet proof is fully confirmed across both configs with clean ablations.

4. **Transition boundaries are the remaining error source.** Both HARD and SIMPLE show accuracy dips at expert handoff points. The dips are shallower than Phase 19, suggesting the multi-scale context window helps the model see transitions within individual training examples.

5. **HARD is not plateauing at 200 epochs.** The training curve shows the accuracy envelope still rising. More epochs or more data may push further.

6. **Two files constitute the proof.** `time_transformer_proof.py` (timestamp encoding works) and `trust_v8.py` (trust learning works). Everything else is research history.
