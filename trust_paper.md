# Ŧrust: Learning Time-Varying Source Reliability Through Attention Over Source, Time, and Content

## 1. Introduction

Ŧrust is a derivation of transformer attention that operates over three embedding spaces: source, time, and content. Standard attention computes relevance between content tokens. Ŧrust computes a distribution over *sources* conditioned on all three dimensions — producing both the mechanism by which outputs are generated and the interpretable receipt of which sources contributed what, when.

In an image generation model, Ŧrust could show the distribution of training sources that influenced a generated image, and which source first introduced the patterns being used. In a language model mediating between forecasters, it shows which voices shaped a conclusion and how much weight each carried. In any multi-source system, the attention weights over sources at a given time ARE the trust scores — they are the readable output of the mechanism itself.

This paper tests the Ŧrust mechanism in its lowest-dimensional form: scalar numerical predictions as content, with cyclical source expertise as the ground truth. The three embedding dimensions are all present — timestamp embeddings encode when, source embeddings encode who, content embeddings (value projections) encode what — but the content is a single number rather than text or images. The structure of the mechanism is identical to what it would be at higher content dimensionalities. If it works here, the architecture generalizes.

## 2. The Ŧrust Mechanism

### 2.1 Three Embedding Spaces

Every input to a Ŧrust-enabled model carries three kinds of information:

- **Timestamp embedding** ($\mathbf{t}$): When was this contribution made? Encoded via the gated dual-pathway `TimestampEncoding` (Ash, 2024), which converts raw Unix timestamps into learned representations capturing periodicity at every scale from seconds to decades.

- **Source embedding** ($\mathbf{s}$): Who made this contribution? A persistent learned vector per source, updated as the model trains. Encodes identity, track record, and domain-specific reliability patterns.

- **Content embedding** ($\mathbf{x}$): What was contributed? The semantic content itself — projected into the model's latent space. This could be a word embedding, an image patch embedding, a projected scalar prediction, or any other representation. The dimensionality varies by domain; the mechanism does not.

### 2.2 Gated Combination

The three signals combine through learned gates:

$$g_t = \sigma\big(W_t \cdot [\mathbf{t};\, \mathbf{s};\, \mathbf{x}] + b_t\big)$$
$$g_s = \sigma\big(W_s \cdot [\mathbf{t};\, \mathbf{s};\, \mathbf{x}] + b_s\big)$$
$$g_x = \sigma\big(W_x \cdot [\mathbf{t};\, \mathbf{s};\, \mathbf{x}] + b_x\big)$$
$$\mathbf{c} = g_t \odot \mathbf{t} + g_s \odot \mathbf{s} + g_x \odot \mathbf{x}$$

Each gate examines all three dimensions and decides how much its corresponding signal matters in this context. The gating is not fixed — it's learned per domain. For a task where timing is everything, the temporal gate opens wide. For a task where source identity dominates, the source gate opens. Content is always available as a residual signal.

### 2.3 Attention as Trust

The combined embedding produces attention weights over sources:

$$\alpha = \text{softmax}(W_{\text{Ŧrust}} \cdot \mathbf{c})$$

These weights serve two functions simultaneously:

1. **Mechanism**: They weight source contributions when computing the model's output, exactly like standard attention weights token contributions.
2. **Receipt**: They are the readable record of which sources influenced this output, how much, and — through the temporal embedding — when those sources earned that influence.

In the full Cognicism framework (Ash, 2024), these weights update dynamically as outcomes are observed:

$$T_s \leftarrow T_s + \eta \cdot (R - T_s)$$

where $R$ is a reward from a proper scoring rule. Sources whose predictions age well gain influence. Sources whose predictions fail lose it. The attention distribution is the trust score.

## 3. The Flattened Experiment

### 3.1 Setup

We test Ŧrust at its lowest content dimensionality: scalar numerical predictions. $S$ sources each make predictions at various timestamps. Source expertise follows sinusoidal cycles:

$$e_s(t) = \mathbb{1}\Big[\sin\!\Big(\frac{2\pi(t + \phi_s)}{C}\Big) > \tau\Big]$$

where $C$ is the cycle length, $\tau$ is the threshold, and $\phi_s = s \cdot C / S$ spaces sources evenly. When a source is expert, its prediction equals the true value. When not, its prediction is noise.

All three Ŧrust dimensions are present:
- **Content**: Each source's prediction value, projected via `nn.Linear(1, d_model)`
- **Source**: Learned embedding per source via `SourceEmbedding(num_sources, d_model)`
- **Time**: The timestamp, encoded via `TimestampEncoding(d_model)`

These are concatenated into the per-source representation, processed through self-attention (sources attending to each other's contributions), then aggregated via cross-attention where the target timestamp queries the source predictions.

### 3.2 The Task

Given source predictions at a timestamp, output the true value. Success means the model learned to select the expert source's prediction — it learned *who to trust, when*.

### 3.3 Ablation Design

Four model variants isolate each dimension's contribution:

| Variant | Content | Source | Time | Expected behavior |
|---------|---------|--------|------|-------------------|
| Baseline | Y | N | N | Averages predictions (no source or time signal) |
| Temporal only | Y | N | Y | Knows when but not who |
| Source only | Y | Y | N | Knows who but not when they're reliable |
| Full model | Y | Y | Y | Can learn time-varying source trust |

Only the full model has access to all three dimensions. Only it can, in principle, learn the cyclical expertise pattern and generalize to future timestamps.

### 3.4 The Algebraic Structure

The expertise function decomposes via the angle addition identity:

$$\sin\!\Big(\frac{2\pi(t + \phi_s)}{C}\Big) = \sin\!\Big(\frac{2\pi t}{C}\Big)\cos\!\Big(\frac{2\pi \phi_s}{C}\Big) + \cos\!\Big(\frac{2\pi t}{C}\Big)\sin\!\Big(\frac{2\pi \phi_s}{C}\Big)$$

This is bilinear in time features and source features. If the timestamp encoding learns oscillators at frequency $2\pi/C$ and the source embedding learns the corresponding phase coefficients, their interaction directly represents the expertise signal. This motivates testing element-wise multiplication of timestamp and source embeddings as an alternative to concatenation.

### 3.5 The Reform Variant

To isolate the timestamp-source interaction from any content-copying shortcut, we also test a stripped formulation:

- **Input**: (timestamp, source ID) only — no prediction values
- **Output**: Binary trust label — is this source expert at this time?
- **Content dimension**: Removed entirely

This tests the two-dimensional case: can the model learn trust from source and time alone? If so, the mechanism works even without content. Adding content back (at any dimensionality) is strictly more information.

## 4. Experimental Design

### 4.1 Parameter Sweep

We systematically vary difficulty from an easy baseline:

**Baseline**: 2 sources, threshold 0.0, 500-day cycle, 10,000 samples

| Parameter | Sweep values | What gets harder |
|-----------|-------------|-----------------|
| num_sources | 2, 3, 5, 8, 10 | More phase relationships to learn |
| threshold | 0.0, 0.3, 0.5, 0.7, 0.8, 0.9 | Narrower expert windows, class imbalance |
| cycle | 500, 365, 200, 100 | Finer temporal resolution needed |
| samples | 5k, 10k, 20k, 50k | More/less evidence per configuration |

### 4.2 Interaction Types (Reform Variant)

Three ways to combine timestamp and source embeddings:

- **Concat**: $[\mathbf{t};\, \mathbf{s}] \to \text{MLP}$ — MLP must implicitly learn cross-terms
- **Multiply**: $\mathbf{t} \odot \mathbf{s} \to \text{MLP}$ — directly computes bilinear interaction
- **Both**: $[\mathbf{t} \odot \mathbf{s};\, \mathbf{t};\, \mathbf{s}] \to \text{MLP}$ — multiplicative plus residual

The multiply variant has the strongest inductive bias toward the bilinear ground truth.

### 4.3 Evaluation

Chronological 80/20 train/test split. The model trains on the first 80% of the date range. Test accuracy on future dates it has never seen is the measure of generalization. The gap between train and test accuracy is the measure of memorization.

## 5. Results

*(To be populated from sweep experiments)*

## 6. Discussion

*(To be written after results — will address: where generalization succeeds, where it fails, what fixes work at the boundary, and implications for scaling to higher content dimensionalities)*

## References

1. Ash, J. "Ŧrust: A Technical White Paper on a Trust-Based Attention Mechanism for Cognicism." 2024.
2. Ash, J. "Iris and Ŧrust White Paper." Cognicism Framework, 2024.
3. Vaswani, A. et al. "Attention Is All You Need." *NeurIPS*, 2017.
4. Kazemi, S. M. et al. "Time2Vec: Representing Time in Neural Networks." *NeurIPS*, 2019.
