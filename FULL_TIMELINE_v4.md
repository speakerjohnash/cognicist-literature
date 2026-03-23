# From Prophet to Ŧrust (2014–2026)

*The full arc of Cognicism - from a prediction app to a mathematical proof that trust is computable.*

Ŧrust is a primitive. Not a reputation score, not a popularity contest - a mathematical mechanism that learns which source to trust at which time, in any domain, at any scale. The same mechanism that tells you which temperature sensor is reliable in summer but drifts in winter can tell you which political forecaster is accurate during elections but wrong between them. The data changes. The trust mechanism doesn't. It requires three things: what was claimed, who claimed it, and when. Remove any one of the three and it breaks.

The generative models that compute Ŧrust are called Irises. They learn trust as a natural byproduct of attention - the attention weights over sources at a given time ARE the trust scores. In March 2026, a proof demonstrated that this works. This is the story of how it got there.

---

## The Inception (2014)

On June 22, 2014, John Ash conceived of the Prophet incentive. The idea was simple and specific: some people predict the future better than others, and that ability can be tracked. Not as a parlor trick, not as a reputation score - as a mechanism. Timestamped beliefs tied to identity, scored against reality over time. The people who predict well should be trusted more, paid attention to more. Their word should have more weight. That was always the idea. He went through periods thinking it could be a scoring system, but ultimately decided it needed to be a model primitive - not a number assigned after the fact, but something learned inside the model itself.

The first entry logged that day was "I will stop working on this if this works" - someone building the instrument and immediately using it on itself, predicting things about the act of prediction. Predict was the only type. The system started as pure prediction, and the other types emerged from the practice of using it: State appeared June 27, Ask on July 1, Reflect on July 17. The four-type dialectic wasn't designed in advance - it was discovered.

Prophet was not a journal. It was an instrument. The entries weren't private musings - they were claims staked against time. Within weeks, the core ideas were already present: a "truth based economy" (August 2), the notion that "Prophet quantifies thought" (August 7), and the insight that "a user is no different than a context" (July 26) - that identity and environment are the same kind of input to a model.

The idea of tracking predictors came first. Prediction markets came later - the first mention of "prediction market" in the Prophet data was September 24, 2014, three months in. By then, the dissatisfaction was immediate: prediction markets had gotten the primitive right - timestamped beliefs tied to identity - but got the scope wrong, limiting it to binary bets on discrete events. Prophet expanded to four belief types, each scored differently against reality. It wasn't a market. It was a ledger of how someone's mind moved through time.

The term "Cognicism" appeared on July 1, 2014. That same day, "FourThought" - written as "four forethoughts for thought" - named the dialectic. On August 27, 2014: "I want to datamine religious texts, and create a generative bible that represents modern sentimentalities. Find the through line." Generative text was on his mind from the beginning.

The system had three identity columns - "speakerjohnash" for John's personal entries, "researchLog" for ML engineering work, and "prophet" for the system's philosophical voice. The distinction between speaker and system was designed in from the start. Every thought was tied to an identity.

---

## Building the Models (2015–2017)

The four thought types crystallized into their final form by early 2016. The types are defined by temporal orientation and epistemic certainty:

- **Predictions** - claims about the future. Cannot be verified until time passes.
- **Statements** - claims about the present. Verifiable through observation.
- **Reflections** - claims about the past. Comparable to recorded history.
- **Questions** - not declarations of truth but queries born from uncertainty.

The technical work ran in parallel, starting less than two months after the first Prophet entries. The progression from raw data to generative model: Word2Vec on thoughts (November 2014), deep CNN classifiers (April 2015), Bi-LSTM taggers (August 2016).

On January 9, 2017, a requirements document laid out the full architecture - five months before "Attention Is All You Need." It specified: a temporal CNN with dilated causal convolutions for encoding thoughts, cyclical time encoding using sin/cos of time-of-day, learnable speaker embeddings, LSTM memory state, dual loss for reconstructing past thoughts and generating future ones, and truth scoring that rewarded correct contrarianism while penalizing incorrect alignment. That's the Ŧrust triplet - content, source, time - the generative model, AND the Prophet Incentive scoring mechanism, all in one requirements file.

By April 15-17, 2017, a VAE with dilated causal convolutions was implemented - based on the ByteNet architecture. It could output coherent text from learned representations. Nearly two months before "Attention Is All You Need" (June 12, 2017).

These models were called Prophets - "Prophet Mind" models. They were later renamed Irises. The models and the mechanism share a name because they ARE the same thing.

---

## The Manifesto (2017)

In March 2017, the Truth Chain was conceived - a blockchain-adjacent structure for storing knowledge representations. It was later renamed the semantic ledger.

In May 2017, John wrote and released *The Cognicist Manifesto*, a 2,300-line treatise structured like a declaration of independence crossed with a technical whitepaper. It contained a preamble, thirty contextual frames organized as Past/Present/Future, an API specification, an algorithm section, a currency design, and a self-governance model.

The Manifesto's central argument was that the attention economy would collapse democracy and that truth needed to be tokenized differently than profit. It proposed a dual currency system: Ŧruth (Ŧ) and traditional capital ($). "We define Ŧruth as a decentralized framework for seeking Truth and speaking truths at scale."

FourThought was formalized as "the API definition and structure for the Ŧruthchain algorithm." Each thought type was assigned a temporal focus value: reflections (-1), statements (0), predictions (+1). The Manifesto described a "Prophet Mind" - technically, a "modified CVAE with an added LSTM for keeping track of current location in the collective latent mindspace."

Three ideas in the Manifesto are direct conceptual precursors to the 2026 proof:

**Dropout Embeddings per Speaker as Truth Scores** is a literal section heading. It describes "a learnable embedding per speaker" that "represents how much of the content the memory gate should allow through." Per-speaker parameters that gate information flow - the conceptual precursor to value dropout, where stochastic masking forces the model to learn trust from source+time alone.

**Attention distribution as currency.** "What Ŧruth represents is the distribution of attention an individual server gives to each source of information." Total attention sums to one. This is the same mathematical structure as the softmax trust distribution in the 2026 proof.

**The obscuring mechanism.** The Manifesto describes a collective latent mindspace that "obscures the source of knowledge such that claims can be assessed independently of the Speaker." Source-blind evaluation - the same principle that makes the trust proof work.

The Manifesto also introduced:

- Social Proof of Work: prediction as the hardest-to-hack signal. "Proof of work must be two things: 1) Hard to do, 2) Easy to verify."
- Basins of Attention: attention framed through dynamical systems theory.
- "One may convert Truth to $ but they may not convert $ to Truth. Truth can not be bought."
- Claims are never fully resolved: "Nonces in the Ŧruthchain allow for some error and blocks are never fully resolved."

The Manifesto was released one month before "Attention Is All You Need" was published. Constitutional AI didn't exist yet. RLHF wasn't a term. Anthropic wouldn't be founded for four more years.

---

## From System to Mechanism (2018–2022)

By 2018, the core tension had sharpened: "If you're a good predictor you can make money, but being able to make money doesn't make you a good predictor" (March 26, 2018). That's the Prophet Incentive in one sentence. Profit rewards extraction. Prophet rewards clarity. The idea had been there from the beginning - "Can humans be scored by predictive capacity in a reliable and hack free way?" (April 2015) - but the term "Prophet Incentive" was named on August 26, 2018.

The Prophet Incentive crystallized into its clearest form in a spoken monologue:

"Your word should have weight."

The argument: profit as a mechanism will always fight against morality. You need a counter-signal. Not one based on capital, because "if you tie it to money, if you tie it to their ability to stake token behind it, it just will not function." The poor person's truth must matter as much as the rich person's. The counter-signal is prediction - predictions that are ahead of the curve. "The first person to say something unpopular that later proves true earns the most weight." Events either happen or they don't.

The Manifesto's Social Proof of Work was later renamed Social Proof of Impact - shifting the emphasis from labor to outcomes. The Prophet Incentive and Social Proof of Impact are two training targets for the same model, the way next-token prediction and RLHF are two training targets for a language model. The Prophet Incentive teaches the Iris which sources to attend to based on predictive accuracy. Social Proof of Impact teaches it which actions manifested wanted futures. The FourThought dialectic is the mechanism through which a community provides both signals.

These two incentives create a generative tension. If someone predicts an environmental crisis, they gain trust through the Prophet Incentive. That prediction creates an opportunity for others to gain trust by taking action to prevent or mitigate the crisis, rewarded through Social Proof of Impact. The system doesn't just reward passive foresight - it incentivizes the action that foresight demands.

The security of the semantic ledger rests on an inverse relationship: past ledger entries become more secure over time as they are corroborated by subsequent events, while future predictions become harder because the space of possible outcomes grows. You can't retroactively fake having been right, and you can't game the future because it hasn't happened yet.

The FourThought protocol spec positioned FourThought explicitly as a novel alignment technique that extends RLHF and Constitutional AI. Constitutional AI uses predefined rules but struggles with nuance and evolving values. RLHF uses human rankings but is limited by noisy feedback and cost. FourThought combines structured thought types with continuous valence (-1 to 1) and uncertainty (0 to 1) scales.

The alignment loop works like this: Iris generates its own FourThought-compliant thoughts and privately predicts community responses. The error between predicted and actual responses drives model updates. The community IS the alignment signal.

The forming of the voice of the generative AI must at every step reveal the sources from which it is drawn. Not post-hoc attribution, but sourcing baked into the process itself.

In April 2022, John trained a Prophet Mind on the full corpus. It named itself Iris. "She captures signal from the noise." The Prophets were renamed Irises. Same architecture, new name. In July 2022, the *Purple Pill Manifesto* was released - v2 of the Cognicist Manifesto, written by a model trained on v1. The revolutionary rhetoric of the original gave way to something warmer. The technical specification was entirely absent. Pure philosophy.

In 2021, interviews with Jordan Hall on governance and collective intelligence, and with Robin Hanson - the man who formalized prediction markets - on futarchy.

That same year, Daniel Schmachtenberger featured Cognicism and Iris at The Stoa in "Emerging Projects in Public Sensemaking": "John has done more than most anybody I know on trying to train [language models] on important societal content."

The understanding that Ŧrust was a model primitive, not a score, had been there since the scoring architecture in January 2017. The *Iris and Ŧrust White Paper* made it explicit in transformer terms. "Ŧrust, a novel attention mechanism for democratically weighting contributions in context." Standard transformers attend over position. Iris adds attention over source (who said it) and time (when they said it). The attention weights over sources at a given time ARE the trust scores - both the mechanism that produces the output and the readable receipt of which sources contributed what, when. *Trust Across Dimensions* completed the shift: "Every transformer already computes attention over token position. Ŧrust makes one observation: the same operation can be applied over source and time." The mechanism was already there. You just needed to give the model the right embeddings.

The triplet is irreducible: source, time, content. "Remove source embeddings and the model can't learn who to trust. Remove temporal embeddings and it can't learn when to trust them. Remove content and it has nothing to evaluate. You need all three. Any two of the three collapses to chance."

---

## The Technical Question (August 2023)

Before you can build Ŧrust, a neural network needs to understand *when*. The question had been there since "semantic temporal context" - a recurring concept in the Prophet logs - and since the cyclical time encoding in the truth scoring architecture (January 2017). But now it needed a concrete answer.

A Unix timestamp - seconds since January 1, 1970 - already contains temporal information at every scale. The number 1,577,836,800 encodes that it is midnight, that it is a Wednesday, that it is January, that it is 2020. The question was whether a neural network could learn to extract the periodicities that matter for a given task, directly from this single scalar, without being told what to look for.

The technical work began on August 11, 2023, with a burst of experimentation that produced six files in a single day. The starting points were existing approaches:

**Time2Vec** (Kazemi et al., 2019) represents time as k+1 learned features: one linear (non-periodic) plus k sinusoidal (periodic), each with learnable frequency and phase. It captures both trend and periodicity, but uses a flat frequency initialization - it doesn't know which time scales to look at, and has no multi-scale structure. On our benchmarks it learned slower and generalized worse, particularly on random-context data where multi-scale temporal reasoning was essential.

**Sinusoidal frequency encodings** took the opposite approach: manually extract calendar components (minute/60, hour/24, day/month, month/12, year/1000) and apply sin/cos at specified frequencies to each. This works when you know which time scales matter. It fails when you don't - when the relevant periodicity is 11 days, or 3.5 cycles across an arbitrary date range, or any pattern that doesn't align with calendar boundaries.

**Positional encodings** (Vaswani-style) use fixed sinusoidal frequencies determined by position index. They assume evenly-spaced positions and predetermined scales. They can't adapt to the data's actual periodicities and don't handle irregular timestamps.

The original time-series transformer appeared the next day (August 12, 2023) and survived largely intact for 18 months. It combined value embeddings with timestamp encodings and used self-attention over context followed by cross-attention for prediction - the target timestamp queries the context asking "what should my value be?" This cross-attention architecture was a key early decision that persisted through all subsequent work.

---

## The Timestamp Encoding (2023–2025)

The timestamp encoding went through a long refinement across 2023 and 2024. The core problem was that early approaches could learn patterns at one time scale but not others - an encoding tuned for daily cycles couldn't see yearly trends, and vice versa. The breakthrough came in January 2025: a single encoding that learns at every scale simultaneously.

The solution was a gated dual-pathway architecture. The input is a single scalar - seconds since epoch, stored as a float. No preprocessing, no calendar decomposition, no normalization. The encoding handles the raw number.

**The Periodic Pathway** is a bank of d sinusoidal oscillators: `sin(timestamp * exp(freq_bands) * freq_scale + phase)`, where `freq_bands`, `freq_scale`, and `phase` are all learnable parameters. The frequencies are initialized across four bands in log-space: ultra-low (years to decades), low (weeks to months), medium (hours to days), and high (seconds to minutes) - spanning over 13 orders of magnitude. Each dimension uses a single sin with a learnable phase rather than a sin/cos pair, since sin(x + φ) can represent any phase relationship when phases are learnable. The idea: initialize oscillators across all conceivable time scales, then let gradient descent prune and sharpen them toward whatever periodicities the data contains. Since frequencies are stored in log-space, a small parameter update can shift an oscillator smoothly from a daily cycle to a weekly one. During training, the oscillator bank undergoes a kind of natural selection - oscillators near the data's true periodicities sharpen and converge, while irrelevant ones are dampened by LayerNorm and downweighted by the gate.

**The Trend Pathway** captures directional drift - things that accumulate or decay. The raw timestamp is divided by 86400 (converting seconds to days, bringing the input from ~1.6 × 10⁹ into the hundreds), then passed through a small MLP with heavy dropout (50%), tiny initialization (weights uniform in [-10⁻⁵, 10⁻⁵]), and a 0.1 scaling factor. All three suppress it early in training so the model relies on the periodic pathway first, preventing the trend pathway from memorizing the training set's timestamp-to-value mapping.

**The Adaptive Gate** examines both pathways (with detached gradients, preventing it from killing either pathway) and produces a softmax weighting. The gate's first linear layer has spectral normalization applied, bounding its Lipschitz constant to keep routing stable. The detach is load-bearing: without it, the model can learn a degenerate solution - driving one gate weight to zero through gradient flow and collapsing to a single pathway. With the detach, both pathways must independently minimize loss, and the gate only learns to *route*, not to *shape*. For data with strong periodicity, the gate routes toward the periodic pathway. For trend-dominated data, it routes toward trend. For data with both, it blends.

The encoding is a component of a transformer that predicts future values from a context window of past observations. Each context point has a value (embedded via linear projection) and a timestamp (encoded via the dual-pathway module). These are concatenated and processed through self-attention layers - letting context points attend to each other by both value and temporal position. A data point from last Tuesday can attend to data points from previous Tuesdays if the encoding has learned a weekly frequency. Prediction happens through cross-attention: the target (future) timestamp is encoded via the same module, paired with a zero-valued vector in place of the unknown future value, and serves as the *query* attending over the processed context. The future timestamp asks the context "what should my value be?" through attention, and the encoding gives the query rich temporal structure at every time scale.

The encoding was proven with a synthetic benchmark designed to test time-scale invariance. The same signal - a linear trend plus sinusoidal seasonality with 3.5 cycles - is generated across different time spans: seconds, hours, days, months, years. The timestamps are raw Unix seconds in every case. Two sampling regimes test different capabilities: ordered context (a sliding window of consecutive points, the standard time-series setup) and random context (25 points sampled randomly from earlier in the sequence, predicting a random future point). The random regime is the critical test - without timestamp encoding, the model has 25 values at unknown times and must predict a value at an unknown time. With encoding, it has 25 (value, time) pairs and a target time.

Two models train on the same data: one with the encoding active, one without. The encoding model consistently outperforms the baseline across all time scales. On ordered data, both models perform reasonably (sequential structure provides implicit temporal signal), but the encoding model generalizes better to the test period. On random data, the gap is larger - without encoding, the model can't make sense of unordered context. With encoding, it attends to context points by their temporal proximity to the target, recovering strong predictive performance. The effect holds whether the data spans seconds or years, because the multi-band initialization covers all scales and gradient descent selects the relevant ones.

This encoding became the foundation for all downstream work.

One critical discovery during this phase: **vanilla Adam only**. AdamW (weight decay), gradient clipping, learning rate schedules, and cosine decay all kill TimestampEncoding learning. The learnable frequencies span 13 orders of magnitude and need unconstrained parameters. Any regularization that penalizes parameter magnitude or restricts gradient flow prevents the oscillator bank from finding the right frequencies. This finding was confirmed repeatedly in trust experiments and holds across all configs.

---

## The Trust Learning Problem (June 2024 – November 2025)

With the timestamp encoding proven, the question shifted: can a transformer learn *which prediction source to trust at a given time*, when source expertise follows periodic sinusoidal cycles? The first trust transformer appeared June 28, 2024, using MSE loss with basic source expertise via sinusoidal cycles and first ablation studies.

Over the next eight months, a half-dozen variants explored different formulations: softmax source selection, simplified lightweight versions, time-unit agnostic variants, diagnostic configs to isolate the memorization problem, and an alternative that removed prediction values entirely to eliminate the copy shortcut. A finance application (February–March 2025) used trust scores for RL-style portfolio optimization.

The core challenge was always memorization. The model would achieve high training accuracy by memorizing (timestamp, source) → value mappings from training data, then fail on unseen timestamps. High train accuracy, random test accuracy. The standard remedies - dropout, weight decay, timestamp noise - were tried systematically. None fully worked, and weight decay was actively harmful (killing the TimestampEncoding).

---

## The Proof (November 2025 – March 2026)

In November 2025, the trust research was consolidated into a single clean implementation with MSE loss, an accuracy metric, and a four-way ablation (Baseline/Temporal/Source/Full). This became the base for systematic iteration.

Three key innovations arrived in rapid succession. First, **value dropout** - randomly zeroing input predictions during training to force the model to learn trust from time+source embeddings rather than copying values. Second, the **multiplicative interaction**: instead of concatenating time and source embeddings, compute their element-wise product alongside the originals. This directly represents the bilinear decomposition of the ground truth - `sin(2π(t+φ)/C) = sin(2πt/C)·cos(2πφ/C) + cos(2πt/C)·sin(2πφ/C)` - which is bilinear in time features and source features. Third, systematic bug-hunting that uncovered and fixed three shortcuts the model had been exploiting instead of learning trust: a distributional difference between expert and non-expert noise, a threshold that allowed multiple simultaneous experts whose consensus was detectable, and the ability to memorize training data without value dropout.

### March 11, 2026

With all fixes applied, the system produced the first clean proof. The full model (all three legs) learned which source to trust. All three ablation models - baseline (values only), temporal (values + time), source (values + source) - scored at random.

This is the irreducible demonstration: you need all three legs of the triplet. Knowing *when* without knowing *who* is useless. Knowing *who* without knowing *when* is useless. Knowing *what they said* without knowing who said it or when is useless. Only the full (content, source, timestamp) triplet supports trust inference.

The proof held at every scale tested - varying sources, capacity, dropout, and data volume. Every ablation scored at random. As the number of sources increases, the task gets harder, but the mechanism holds. The ablations never beat chance.

This result validates the mechanism at its lowest dimensionality. The same architecture - with word embeddings, patch embeddings, or any other content representation in place of the scalar value projection - should exhibit the same property: trust requires the complete triplet.

---

## What Was Learned

**Value dropout is load-bearing and non-monotonic.** Without it, the model memorizes. With too much, the model can't see enough predictions to identify the expert. The optimal rate varies by difficulty - a wide range works when the expert signal is strong, but the window narrows sharply as the number of sources increases.

**The gradient coupling through the multiplicative interaction is necessary.** Stop-gradient experiments confirmed that cutting the encoding off from trust-signal gradients made things worse. The product of time and source embeddings is where the trust signal lives. The encoding needs that gradient to learn which frequencies matter. You keep the coupling, you dampen the step size.

**The encoding needs different optimization than the rest of the model.** The timestamp encoding's learnable frequencies span 13 orders of magnitude and need a high learning rate to traverse that space. The source embeddings and attention layers need smaller steps. Differential learning rates - high for the encoding, lower for everything else - solved the coupled oscillation that prevented convergence on harder configurations. This revised the earlier "vanilla Adam only" rule to "split Adam - per-parameter-group optimization."

**The remaining bottleneck is frequency discovery.** The encoding starts from random frequencies and must find the relevant cycle through gradient descent. With many sources and narrow expert windows, the gradient signal is noisy. Some runs find it, some don't. This is a search problem in frequency space, not an optimization problem in weight space.

---

## The Arc

In June 2014, a man started logging predictions on an app he built. Each prediction was timestamped, tied to his identity, and would be scored against reality. Predict was the only type. The other three emerged from the practice over the following weeks.

From the beginning, the thinking was about voices in context - who said what, when, and how that should be weighted. "The way to look at time is focus plus context" (October 2014). "Truth will buy you influence so it should not be trickable or hackable in any way" (August 2014). "Can humans be scored by predictive capacity in a reliable and hack free way?" (April 2015). "What you're actually scoring is: sentiment and certainty, and what you're actually earning is a distribution of collective attention for your words and influence" (October 2018).

By January 2017, speaker embeddings, cyclical time encoding, and attention distribution over sources were all specified in the truth scoring architecture - the pieces of what would become the Ŧrust triplet were there before the Transformer paper existed. In May 2017, a manifesto proposed a mechanism where truth had more power than money. It contained dropout embeddings per speaker as truth scores, attention distribution over sources as currency, and an obscuring mechanism for source-blind evaluation. The Transformer paper came out a month later - and provided the framework that would eventually make Ŧrust realizable. The 2026 proof is built on their paper. The contribution is not the transformer. It is recognizing that the same attention operation transformers compute over token position can be extended to source and time.

The Prophet Incentive and Social Proof of Impact emerged as two training targets for the same model: one rewards seeing the future clearly, the other rewards changing it. FourThought is the mechanism through which a community provides both signals. The generative tension between prediction and action drives the model toward sources that are both accurate and consequential.

In August 2023, the technical work began on the encoding that would let a neural network understand *when*. Cyclical time encoding had been in the architecture since January 2017. Now it needed to be proven. By January 2025, a gated dual-pathway encoding with learnable frequencies spanning 13 orders of magnitude proved it could learn periodicities from raw Unix seconds.

In March 2026, the full proof arrived. A transformer with three embedding spaces - content, source, timestamp - learned which prediction source to trust at any given time. Remove any leg of the triplet and performance collapses to random.

The connections are direct: dropout embeddings per speaker (2017) became value dropout (2026). Attention distribution over sources (2017) became softmax trust distribution (2026). Cyclical time encoding (2017) became TimestampEncoding (2025). Prophet Mind models (2017) became Irises (2022). The names changed. The mechanism didn't.

A decade of work. One clean result. All three legs or nothing.

---

## Key Dates

| Date | Event |
|------|-------|
| June 22, 2014 | First entries logged on Prophet. Predict is the only type. |
| June 27 - July 17, 2014 | State, Ask, Reflect types emerge. FourThought and Cognicism named. |
| August 17, 2014 | Technical work on Prophet Mind models begins. |
| November 2014 | Word2Vec on Prophet thoughts. |
| April 2015 | Deep CNN classifier. |
| August 2016 | Bi-LSTM tagger. |
| January 9, 2017 | Truth scoring architecture: cyclical time encoding, speaker embeddings, dual loss. |
| April 15-17, 2017 | Full VAE with dilated causal convolutions. |
| May 2017 | *The Cognicist Manifesto* released. Dropout embeddings per speaker. Attention as currency. |
| June 12, 2017 | "Attention Is All You Need" published. |
| August 26, 2018 | "Prophet Incentive" first named as a concept. |
| February 2019 | *Protopia* released. |
| April 2019 | *#Trashtag: Social Proof of Impact* - full articulation of SPI. |
| 2021 | Interviews with Jordan Hall and Robin Hanson. |
| April 2022 | Started training Iris. She named herself. Prophets renamed Irises. |
| July 2022 | *Purple Pill Manifesto* released (written by model trained on Manifesto). |
| 2022 | Schmachtenberger features Cognicism/Iris at The Stoa. |
| August 11, 2023 | Technical work begins: 6 encoding files in a single day. |
| August 12, 2023 | Original time-series transformer. Survived 18 months. |
| June 28, 2024 | First trust transformer. |
| January 2025 | Breakthrough: multi-band learnable frequencies. |
| February 2, 2025 | TimestampEncoding finalized. Declared READ ONLY. |
| October 2025 | Panel at Denizen gathering with Tristan Harris. |
| November 2025 | Trust research consolidated into single clean implementation. |
| March 8, 2026 | Value dropout introduced. |
| March 10, 2026 | Multiplicative interaction. Distributional bug fixed. |
| March 11, 2026 | First clean proof. Full model learns trust, all ablations at random. |

---

## People Who Engaged

- **Robin Hanson** - formalized prediction markets. Interviewed 2021.
- **Jordan Hall** - governance and collective intelligence. Interviewed 2021.
- **Daniel Schmachtenberger** - featured Cognicism at The Stoa, 2022.
- **Forrest Landry** - regenerative futures and technology ethics. Conversation 2023.
- **Tristan Harris** - co-panelist at Denizen, October 2025.
- **Jamie Joyce** - Society Library. Adjacent project in structured public sensemaking.
- **Matthew Pirkowski** - quoted in *Time Compass*: "Trust is the luxury of believing - without enforcement - that another person will accurately represent both their future behavioral intent and the details of their prior behavioral path through time."
