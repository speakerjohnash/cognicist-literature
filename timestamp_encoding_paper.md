# A Gated Dual-Pathway Encoding for Raw Timestamps in Time-Series Prediction

## 1. Introduction

Time-series transformers need temporal information. The standard approach is to engineer it: extract hour-of-day, day-of-week, month-of-year from each timestamp, embed each feature, and concatenate. This works when you know which time scales matter. It fails when you don't, or when relevant periodicities don't align with calendar boundaries.

A Unix timestamp — seconds since January 1, 1970 — already contains temporal information at every scale. The number 1,577,836,800 encodes that it is midnight, that it is a Wednesday, that it is January, that it is 2020. The question is whether a neural network can learn to extract the periodicities that matter for a given prediction task, directly from this single scalar, without being told what to look for.

We present a timestamp encoding that does this. It takes a raw Unix timestamp and produces a learned $d$-dimensional representation through two parallel pathways — one for aperiodic trends, one for periodic rhythms — blended by an adaptive gate. It consistently outperforms the no-encoding baseline on synthetic time-series prediction, across time scales from seconds to years. The encoding is used as a component in a broader transformer architecture for time-series forecasting, where the target timestamp queries a context window via cross-attention.

## 2. Architecture

### 2.1 Input

A single scalar: seconds since epoch, stored as a float. No preprocessing, no calendar decomposition, no normalization. The encoding handles the raw number.

### 2.2 Trend Pathway

Time has a directional component — things that accumulate, drift, grow, decay over time. The trend pathway captures this:

```
days = timestamp / 86400
trend = LayerNorm(MLP(days)) * 0.1
```

The MLP is two layers ($1 \to d \to d$) with GELU and 50% dropout. The division by 86400 converts seconds to days, bringing the input from ~$1.6 \times 10^9$ into the hundreds — a more natural range for a linear layer.

Three design choices suppress this pathway early in training:

1. **Tiny initialization.** First-layer weights are initialized uniformly in $[-10^{-5}, 10^{-5}]$. The trend pathway outputs near-zero at the start.
2. **Heavy dropout.** 50% dropout during training prevents the trend pathway from fitting noise.
3. **Scaling factor.** The output is multiplied by 0.1 before reaching the gate.

The intent is that the model relies on the periodic pathway first. Trend information is only recruited if the loss demands it. This prevents the trend pathway from memorizing the training set's timestamp-to-value mapping — a linear function of time can overfit easily on ordered data.

### 2.3 Periodic Pathway

Time has cyclic structure at many scales — circadian rhythms, weekly patterns, seasonal effects, annual cycles. The periodic pathway is a bank of $d$ sinusoidal oscillators:

```
freqs = exp(freq_bands) * freq_scale
periodic = LayerNorm(sin(timestamp * freqs + phase))
```

where:
- `freq_bands` $\in \mathbb{R}^d$ — learnable log-frequencies
- `freq_scale` $\in \mathbb{R}$ — learnable global scale (initialized to 1)
- `phase` $\in \mathbb{R}^d$ — learnable per-oscillator phase offsets

The frequencies are initialized across four bands:

| Band | Log-freq range | Physical scale | Count |
|------|----------------|----------------|-------|
| Ultra-low | $[-20, -10]$ | Years to decades | $d/4$ |
| Low | $[-10, 0]$ | Weeks to months | $d/4$ |
| Medium | $[0, 6]$ | Hours to days | $d/4$ |
| High | $[6, 12]$ | Seconds to minutes | $d/4$ |

This spans over 13 orders of magnitude ($e^{-20} \approx 2 \times 10^{-9}$ Hz to $e^{12} \approx 1.6 \times 10^5$ Hz). The idea: initialize oscillators across all conceivable time scales, then let gradient descent prune and sharpen them toward whatever periodicities the data contains. Since frequencies are stored in log-space, a small parameter update can shift an oscillator smoothly from a daily cycle to a weekly one.

Each dimension uses a single $\sin(\cdot)$ with a learnable phase, not a $\sin$/$\cos$ pair. Since $\sin(x + \phi)$ can represent any phase relationship, the explicit pair is unnecessary when phases are learnable.

### 2.4 Adaptive Gate

The two pathways are blended by a learned gate:

```
combined = cat(trend, periodic)
gate = softmax(MLP(combined.detach()))   # shape: (*, 2)
output = gate[..., 0:1] * trend + gate[..., 1:2] * periodic
```

The gate MLP ($2d \to 4d \to 2$) examines both pathways and produces a softmax weighting over them. Two details matter:

**Detached input.** The gate sees `combined.detach()` — no gradients flow from the gate back into the trend or periodic pathways. This prevents the gate from killing one pathway by driving its gate weight to zero through gradient flow. Both pathways must become useful on their own terms. The gate only learns to *route*, not to *shape*.

**Spectral normalization.** The gate's first linear layer has spectral normalization applied, bounding its Lipschitz constant. This keeps routing weights stable and prevents sharp mode-switching during training.

The result: for data with strong periodicity, the gate routes toward the periodic pathway. For trend-dominated data, it routes toward trend. For data with both, it blends. The allocation is adaptive and learned per-domain.

## 3. Integration: The TimeSeriesTransformer

The encoding is not used in isolation. It's a component of a transformer that predicts future values from a context window of past observations, each paired with its timestamp.

### 3.1 Context Encoding

Each context point has a value and a timestamp. Values are embedded via a linear projection ($1 \to d$). Timestamps are encoded via the dual-pathway module ($1 \to d$). The two are concatenated to form the context representation ($2d$ per position):

```
context = cat(value_embed(x), timestamp_encode(timestamps))  # (B, seq_len, 2d)
```

### 3.2 Self-Attention Over Context

The context passes through $L$ self-attention layers (multi-head attention + feed-forward + layer norms). This lets context points attend to each other, incorporating both their values and their temporal positions. A data point from last Tuesday can attend to data points from previous Tuesdays if the encoding has learned a weekly frequency.

### 3.3 Cross-Attention for Prediction

The target (future) timestamp is encoded via the same dual-pathway module, paired with a zero-valued vector in place of the unknown future value. This target embedding serves as the *query* in a cross-attention layer, attending over the processed context:

```
target = cat(zeros(B, d), timestamp_encode(target_timestamp))  # (B, 2d)
output = cross_attention(query=target, keys=context, values=context)
prediction = linear(output)
```

This is the key architectural choice: the future timestamp asks the context "what should my value be?" through attention. The timestamp encoding gives the query rich temporal structure — it knows *when* it's asking about, at every time scale — and the attention mechanism finds the context points most relevant to that temporal position.

## 4. Experimental Setup

### 4.1 Synthetic Data

We generate time-series data with known structure: a linear trend plus sinusoidal seasonality (3.5 cycles across the date range), smoothed with a Savitzky-Golay filter and perturbed with light Gaussian noise. The signal is deliberately time-scale invariant — the same pattern appears whether the date range spans days, months, or years. Timestamps are Unix seconds.

### 4.2 Two Sampling Regimes

**Ordered context.** A sliding window of 25 consecutive points, predicting 1 step ahead. This is the standard time-series setup — the model sees recent history in order.

**Random context.** 25 points sampled randomly from earlier in the sequence (one guaranteed to be recent), predicting a random future point. This is harder — the model can't rely on recency or ordering. It must use the *actual timestamps* to understand temporal relationships.

The random regime is the critical test. Without timestamp encoding, the model has 25 values at unknown times and must predict a value at an unknown time. With encoding, it has 25 (value, time) pairs and a target time — a much richer signal.

### 4.3 Ablation

Two models train on the same data:
- **With encoding**: `TimestampEncoding` active, timestamps concatenated with value embeddings
- **Without encoding**: timestamps ignored, value embeddings only

Both use the same transformer architecture ($d = 64$, 4 heads, 2 layers), same optimizer (Adam), same loss (MSE), same 80/20 chronological train/test split.

### 4.4 Results

The encoding consistently outperforms the no-encoding baseline:

- On **ordered data**, both models perform well on training data (the sequential structure provides implicit temporal signal), but the encoding model generalizes better to the test period.
- On **random data**, the gap is larger. Without encoding, the model struggles to make sense of unordered context. With encoding, it can attend to context points by their temporal proximity to the target, recovering strong predictive performance.

The effect holds across time scales — the same encoding architecture works whether the data spans seconds, days, or years, because the multi-band initialization covers all scales and gradient descent selects the relevant ones.

## 5. Why This Works (And What Doesn't)

### 5.1 What the encoding learns

During training, the oscillator bank undergoes a kind of natural selection. Oscillators initialized near the data's true periodicities sharpen — their frequencies converge, their phases align, and the gate routes more weight to the periodic pathway. Oscillators at irrelevant frequencies contribute noise; LayerNorm dampens them and the gate learns to downweight them.

The trend pathway activates only if there's genuine drift that the periodic pathway can't capture. On purely cyclical data, the gate suppresses it almost entirely.

### 5.2 The detach is load-bearing

Without detaching the gate input, the model can learn a degenerate solution: drive one gate weight to zero and collapse to a single pathway. The detach forces balanced learning — both pathways must independently minimize loss, and the gate can only observe which is doing better, not influence them.

In early experiments without detach, the model would sometimes kill the periodic pathway and overfit via the trend MLP, memorizing the training set's timestamp-to-value mapping. The detach prevents this by severing the shortcut.

### 5.3 What doesn't work

**Fixed-frequency positional encodings** (Vaswani-style) assume evenly-spaced positions and predetermined scales. They can't adapt to the data's actual periodicities and don't handle irregular timestamps.

**Time2Vec** learns periodic activations but uses a single linear component (no regularization, no gating, no multi-scale initialization). On our benchmark it learns slower and generalizes worse, particularly on random-context data where multi-scale temporal reasoning is essential.

**Calendar decomposition** (embed hour, weekday, month separately) works when those are the right features. It fails when the relevant periodicity is, say, 11 days, or 3.5 cycles across an arbitrary date range. It also can't handle sub-day timestamps or multi-year trends without being manually extended.

## 6. Conclusion

A raw Unix timestamp contains all temporal information at every scale. The gated dual-pathway encoding extracts it through learnable oscillators spanning 13 orders of magnitude, blended with a regularized trend pathway via an adaptive gate. No feature engineering, no calendar assumptions, no predetermined time scales.

The encoding is a general-purpose temporal representation module. It slots into any architecture that needs temporal context — time-series transformers, sequential decision models, or systems that must reason about *when* in addition to *what*. It works because it starts with maximum coverage and lets gradient descent do the selection.

## References

1. Vaswani, A. et al. "Attention Is All You Need." *NeurIPS*, 2017.
2. Kazemi, S. M. et al. "Time2Vec: Representing Time in Neural Networks." *NeurIPS*, 2019.
