# Actually Explainable AI: Models in Text Space

**John Ash**

---

## The Core Idea

Every machine learning model is a compression of information. A neural network compresses training data into weights: millions of floating point numbers arranged in matrices. These weights encode everything the model "knows," but no human can read them. When the model is wrong, you can't look at the weights and see why. When it improves, you can't diff the weights and read what changed. The compression is lossy in a specific way: it loses human legibility.

What if the compression happened in text instead?

Instead of encoding learned knowledge as floating point weights in parameter space, encode it as natural language instructions in text space. The model becomes a document: readable, versionable, diffable. When it's wrong, the reasoning tells you why. When it changes, you can read the diff like a code review. The model is the explanation.

This is not a post-hoc interpretability technique. SHAP values, LIME, attention maps, saliency maps: these are all attempts to explain an opaque model after the fact. They approximate what the model might be "thinking." They are lossy translations from parameter space to human language. Actually Explainable AI doesn't need a translation layer because the model already lives in the space humans think in.

---

## Information Density: Parameters vs. Text

A single weight in a neural network is a 32-bit floating point number. It encodes roughly 4 bytes of information. That information is entangled with every other weight in its layer through matrix multiplication, so the meaning of any single weight is unrecoverable without the full context of the network. The information is there, but it is distributed, implicit, and opaque. You cannot point to a weight and say what it knows.

A single word in a text-space model is a token. It carries semantic meaning on its own, and that meaning compounds with surrounding tokens through the same mechanism humans use: language. A sentence like "water elements alone do not guarantee high engagement under current algorithmic conditions" encodes a specific, falsifiable claim about the world. The information density per token is lower than per parameter, but the information *accessibility* is total. Every piece of knowledge in the model is individually readable, individually testable, and individually revisable.

This is the fundamental tradeoff. Neural networks compress information into a space that is computationally efficient but humanly illegible. Text-space models compress information into a space that is humanly legible but computationally expensive (requiring an LLM to execute). The question is not which compression is better in the abstract. The question is which compression is appropriate for a given task, and whether legibility is worth the cost.

For tasks where accountability, auditability, and trust matter more than raw pattern capacity, text is the better compression medium. The model stores less, but you can read all of it.

---

## The Architecture

In traditional ML, the model is a function parameterized by weights: `f(x; θ)`. The weights `θ` are learned through optimization.

In text space, the model is a criteria document: a structured set of natural language instructions that tells an LLM how to evaluate an input and produce a numerical prediction. The document IS `θ`. It contains the learned patterns, the scoring rules, the edge cases, the temporal conditions, the exceptions. When the LLM receives an input along with this document, it applies the criteria and outputs a score with reasoning.

Consider a system that predicts how well a piece of visual content will perform with an audience. The model is a ~15KB structured document containing sections on visual criteria, audience behavior patterns, algorithmic distribution patterns, temporal trends, red flags, green flags, and explicit scoring bands. An LLM sees an image plus this document and outputs a predicted growth score between 0.0 and 1.0, along with a paragraph of reasoning explaining which rules fired and why.

This document is not a description of the model. It is not a summary of what some neural network learned. It IS the model. The instructions are the weights. The rules are the parameters. The sections are the layers.

Every prediction produces two outputs: a numerical score and natural language reasoning. The reasoning is not optional decoration. It is the critical piece that makes the entire feedback loop work.

When the model predicts 0.45 for an image and the actual performance turns out to be 0.22, you don't just have an error of 0.23. You have a paragraph: "Clean single subject with vibrant colors, water element present, spiritual/cultural markers detected. These green flags indicate strong audience resonance and high growth potential." Now you know *why* the model was wrong. The rule about water elements is too strong. The cultural markers aren't sufficient without algorithmic distribution. The reasoning is the gradient signal.

Every scored item retains its prediction, the reasoning behind that prediction, which model version produced it, and the actual real-world outcome once it arrives. This creates a complete audit trail: for any prediction the system ever made, you can trace back to which model made it, what rules it applied, and how reality compared.

---

## How Traditional ML Concepts Map to Text Space

The claim is not that text-space models are *like* traditional ML. The claim is that they implement the same mathematical functions, expressed in a different medium. Every core concept in the training loop has a direct analog.

**Loss.** In traditional ML: `L = Σ(predicted - actual)²`. In text space, the same error statistics are computed identically: MAE, RMSE, median error. The system scores items, waits for real-world performance data, and computes the gap. But the loss is richer than a scalar. Every prediction carries reasoning, and that reasoning persists through the feedback loop. The recalibration process doesn't just see "the model was off by 0.23 on this item." It sees "the model said 'water elements correlate with high engagement' and was wrong because under current algorithmic conditions, water alone doesn't guarantee distribution." The loss has semantics. It explains itself.

**Backpropagation.** In traditional ML, gradients flow backward through the computation graph. Each weight gets a small update proportional to its contribution to the error: `θ_new = θ_old - α * ∂L/∂θ`. In text space, the gradient is natural language. The recalibration process receives the current model, the corpus of scored items with their predictions, reasoning, and actual outcomes, the aggregate error statistics, and the previous model's error trajectory. It produces an updated criteria document. The "gradient" is the structured analysis of where predictions went wrong and why: "The criteria overestimates items with property X by +0.18 on average. The rule was accurate when conditions were different, but under current conditions it produces systematic bias." The gradient and the parameters live in the same medium. There is no translation between "what the model learned" and "what we can say about what the model learned." They are the same thing.

**Learning rate.** In traditional ML, the learning rate `α` controls how aggressively the model updates. Too high and it oscillates. Too low and it converges slowly. In text space, the learning rate is encoded as training modes: explicit instructions that control how aggressively the recalibration process modifies the criteria document. A conservative mode calibrates update magnitude to current error: MAE below 0.12 means "make ZERO structural changes, only fix typos"; MAE between 0.12 and 0.16 means "identify the SINGLE most systematic error pattern and address only that"; MAE above 0.23 means "rewrite underperforming sections." This is a decaying learning rate schedule expressed in English: the better the model performs, the less you should change. A balanced mode is triggered when the model has collapsed its prediction range, predicting everything between 0.16 and 0.30 when actual outcomes span 0.0 to 1.0. This is a local minimum. The model has learned the mean and stopped differentiating. Balanced mode increases the learning rate to escape it, providing explicit target distributions and forcing the model to learn what distinguishes a 0.90 from a 0.20. An overhaul mode is a full rewrite: reinitializing with a high learning rate, starting fresh but informed by accumulated data. These aren't metaphors for learning rates. They ARE learning rates. They control the magnitude of the update applied to the model. The fact that they're expressed as English paragraphs instead of floating point numbers doesn't change their mathematical function.

**Regularization.** L1/L2 penalties discourage large weights. Dropout prevents co-adaptation. In text space, regularization is expressed as explicit constraints: "Don't let outliers destroy patterns that work on average." "Do NOT make blanket declarations. You see a sample but the full data has contradicting cases." "Never write 'X always achieves 0.80+', write conditional rules that account for variance." There is an accumulation principle that functions as selective pruning: preserve rules that predict well on average, add exceptions to handle failures, remove rules ONLY if they consistently fail across multiple items. This is L1 regularization in text: prune rules that don't contribute, keep rules that do, and add specificity only where the data supports it.

**Batch size.** The recalibration process uses a structured sampling strategy: top performers shown with full inspection (the expensive tokens), bottom performers shown with full inspection, a random sample of the remaining corpus shown as text statistics only, and the complete corpus present as sorted data with error magnitudes. This is a minibatch strategy where visual inspection is the expensive forward pass and text statistics are the cached activations. Recent data is weighted more heavily through separate sections and explicit instructions to attend to it: importance sampling that prioritizes learning from the current distribution.

**Momentum.** The recalibration process receives error statistics from the previous model version alongside the current errors. If version 5 overestimated a class of items by +0.18 and version 6 still overestimates by +0.12, the trajectory is visible. The corrections are moving in the right direction but haven't converged. Individual recalibration rounds don't overcorrect based on one batch of errors. They see the full trajectory and adjust proportionally.

**Concept drift.** Traditional models struggle with distribution shift. The data changes, the model's assumptions become stale, performance degrades. The standard response is retraining: expensive and discontinuous. Text-space models handle concept drift naturally. Because the model is text, temporal reasoning can be encoded directly: "Before mid-2025, organic distribution baselines were higher. After mid-2025, algorithmic suppression reduced organic reach by ~40%." When the world changes, the model gets a paragraph explaining what changed and when. A traditional model would need retraining. A text-space model needs an edit.

---

## Three Implementations

### 1. Learning Reasoning Patterns from Human Experts

The foundational method. A generator-critic feedback loop learns instructions for extending chains of reasoning. A formal ontology defines how reasoning steps relate to each other: how positions lead to arguments, how arguments decompose into premises, how rebuttals engage with claims. Human experts applied this ontology thousands of times in building a structured debate graph, making decisions at every node about how to chain reasoning.

The system learns prompts that compress this expressed knowledge. A Generator produces candidate instructions. An Executor tests them by embedding the instructions at the end of a conversational reasoning chain and collecting the continuation. A Critic evaluates the output against the human-curated original, returning a score and a natural language explanation. These explanations accumulate across iterations. The Generator sees the full history of what was tried, what was produced, and what the Critic said.

The loss function is fidelity to human reasoning patterns: not task accuracy on a benchmark, but whether the output reasons the way experts demonstrated through the act of building the graph. The optimized instructions ARE the learned model. They can be applied to generate new reasoning chains in new domains, transferring the structure of reasoning from one graph to others.

This establishes the principle: text-space models can learn from human behavior, and the learned model is inherently readable because it's expressed in the same medium as human thought.

### 2. Predicting Real-World Outcomes from Visual Content

The production implementation. A criteria document scores visual content, reality provides ground truth (engagement metrics that arrive days or weeks later), and the document evolves through recalibration cycles with multiple training modes.

This is where the training mode hyperparameters were developed, because in production you need control over update magnitude. Conservative mode for fine-tuning a model that's already performing well. Edit mode for targeted fixes. Balanced mode for escaping local minima. Overhaul mode for fundamental resets.

The key advance: real-world feedback loops with distribution shift. The foundational system evaluates against static human examples (supervised learning). The production system evaluates against actual outcomes that arrive asynchronously (online learning). The model must handle the fact that the domain changes: algorithms change, audience behavior shifts, what worked six months ago may not work today. The temporal reasoning encoded in the criteria document tracks these shifts explicitly, and the model receives the date of each item so it knows which era's rules to apply.

The full system maintains versioned models, each with complete criteria text and error statistics, with the ability to activate or roll back any version. The corpus of scored items retains which model version scored each item, creating a complete provenance chain from input to prediction to reasoning to model version to outcome.

### 3. Financial Predictions with Traceable Beliefs

The next evolution. A financial prediction system makes buy/sell/hold recommendations with reasoning. Beliefs are encoded as timestamped, scored claims tied to identity: each one a specific proposition about the world with a formation date, a track record of accuracy, and a confidence score derived from historical performance.

When the system recommends selling a position, the reasoning cites specific beliefs: "Based on the belief that semiconductor valuations are stretched relative to near-term earnings (formed January 2026, accuracy 73% over 12 evaluations)." When reality arrives, the beliefs that drove it are updated. Beliefs that led to accurate predictions gain confidence. Beliefs that led to errors are revised or downweighted.

This closes the full loop:

1. **Beliefs** are the model weights: each one a learned claim about the world, timestamped, scored, and tied to a source
2. **Predictions** are the forward pass: combining relevant beliefs into an output with reasoning that cites them
3. **Reality** provides the gradient: did the prediction play out?
4. **Belief updates** are the weight updates: confidence adjusted based on outcomes

The "model" is the full set of beliefs. You can read every one. You can trace any prediction back to which beliefs drove it. You can see which beliefs have been accurate over time and which haven't. When the model is wrong, you can read *exactly* which belief was incorrect and update it with the specificity of knowing what went wrong and why.

This is the same trust mechanism that operates across all three implementations: signal reliability evaluated over time, weighted by recency and domain. The same mathematical primitive that tells you which visual pattern predicts audience engagement can tell you which financial belief predicts market outcomes. The data changes. The trust mechanism doesn't.

---

## Why This Matters

The AI safety community has spent years trying to make neural networks interpretable. Billions of parameters, attention heads, circuits, features: a cottage industry of post-hoc explanation techniques. All of it is trying to answer one question: what does the model know, and how does it use that knowledge?

Text-space models answer this question trivially. The model knows what it says. It uses that knowledge the way the instructions describe. There is nothing hidden. The model and its explanation are the same object.

This doesn't replace neural networks. Neural networks can encode patterns that text cannot: high-dimensional visual features, subtle statistical regularities, knowledge compressed across billions of tokens. Text-space models trade raw capacity for legibility. They will never match a 175B parameter model's ability to capture fine-grained patterns.

But for many domains where scoring, classification, prediction, and decision-making matter, the tradeoff is worth it. A model you can read is a model you can trust. A model you can diff is a model you can govern. A model whose reasoning chain is the model itself is a model that is actually, not approximately, explainable.

The architecture also solves a problem that opaque models cannot: accountability across time. When a neural network changes its behavior after retraining, there is no human-readable record of what changed. When a text-space model changes, the diff is prose. Regulators can read it. Auditors can review it. The model's learning history is a document, not a tensor.

The alternative to explaining an opaque model is building a legible one.

---

## Relationship to Prior Work

**TextGrad** (Yuksekgonul et al., 2024) established that natural language feedback can serve as gradients for optimizing text variables in computation graphs. The text-space model concept goes further: rather than optimizing a variable within a larger system, the entire model is a text variable. The gradient and the parameters share a medium. This insight was arrived at independently, the same principle discovered because it's the natural solution to the problem of making model knowledge legible.

**ProTeGi** (Pryzant et al., 2023) introduced "natural language gradients" for prompt optimization on benchmarks. The work described here arrived at the same mechanism independently, applying it to a fundamentally different domain: learning the embedded structure of human reasoning rather than optimizing task accuracy. Two independent lines of work converging on the same mechanism strengthens the claim that natural language feedback is a real principle of optimization, not an artifact of any single system.

**Constitutional AI** (Bai et al., 2022) uses natural language principles to guide model behavior through critique-revision loops. Text-space models take the same principle but apply it to the model specification itself rather than to individual outputs. The constitution IS the model.

The lineage is: prompts control model behavior, prompts can be optimized through natural language feedback, the optimized prompt IS the learned model, the model can be versioned and diffed and governed like code, every prediction is traceable to specific readable beliefs that can be independently evaluated.
