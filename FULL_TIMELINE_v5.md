# From Prophet to Ŧrust (2014–2026)

A traditional language model learns the distribution of its training data. It knows what words tend to follow other words. It does not know who said them, when, or whether that person had any idea what they were talking about. A climate scientist and a celebrity carry the same weight in the training set. A prescient warning from 2019 and a confidently wrong take from 2023 look identical. The model has no mechanism to sort signal from noise across sources, domains, or time. It models the aggregate and calls it knowledge.

Ŧrust is a mechanism that does something different. It learns which source to attend to, in which domain, at which time, and it does this using the same attention operation that already powers every transformer. The difference is what the attention operates over. Standard transformers attend over token position. Ŧrust adds two dimensions: source identity and time. The attention weights over sources at a given time are the trust scores. They are simultaneously the mechanism that produces the output and the readable record of which sources contributed what, when.

On March 11, 2026, a proof demonstrated that this works and that it requires all three dimensions. A model with content, source, and time embeddings learned which prediction source to trust at any moment. Remove any one dimension and the model collapses to chance. The triplet is irreducible.

This document is the history behind that result. The concepts that make it work were specified in architecture documents and published writing between 2014 and 2017, before the transformer existed. What follows is grounded in dated, verifiable work product, for readers deciding whether the result and the person behind it deserve their attention.

---

## The Instrument (2014)

On June 22, 2014, John Ash began logging beliefs to a system he built called Prophet. Each entry was bound to an identity, stamped with a time, and designed to be scored against reality later. The first entry was a prediction. That was the only type available.

Over the following weeks, three more types emerged from the practice of using the instrument. Statements about the present appeared on June 27. Questions on July 1. Reflections on the past on July 17. These four orientations, which Ash named FourThought on July 1, were not designed in advance. They were discovered by doing the work: logging beliefs daily and noticing that the existing structure couldn't capture everything that needed capturing. Each type addresses a different relationship to time. Predictions are evaluated by what happens next. Statements are evaluated by what is observable now. Reflections are evaluated against the historical record. Questions declare that something is worth investigating. This is not a prediction market. It is an epistemic record, a ledger of belief and how it evolves over time. Your claims, on record, from the past, pulled up for review. If someone warned about a pandemic in 2019 and someone else dismissed it, that's on the ledger. If someone later tries to rewrite the history of what they said about COVID, their past statements are right there to check against. Receipts.

Over 114,000 entries accumulated across the years. The instrument generated daily, weekly, and monthly summaries, a multi-resolution view of one person's thinking as it evolved. But Prophet's significance is not autobiographical. It is that structured belief data, tagged with identity and time, is the prerequisite for computing trust. If you want a model that learns which sources are reliable, you need a corpus where claims are on record, attributed, dated, and eventually checkable. Prophet was the laboratory that produced that corpus.

---

## The Staking Protocol

If trust is going to be computed over sources, if a person's reputation is going to be bound to a source embedding in a model, then there needs to be a structured way for people to put their beliefs on record. Not just opinions or reactions, but claims that can be checked later. FourThought is that structure.

An early observation shaped it: most people do not communicate in future tense. They state things. They react. They express feelings. Genuine predictions, specific claims about what will happen, bound to an identity and a timestamp, are rare. That rarity is what makes them valuable. Events either happen or they do not. A prediction on record is the hardest kind of claim to fake retroactively. But predictions alone are not enough. People also need to stake what they believe is true now, what they think happened in the past, and what they think needs to be explored. Each of these ages differently.

Think of it as four latent spaces, each shifting over time. At any moment there is a collective representation of what people think will happen next. There is one of what people believe to be true right now. There is one of what people think happened, and that understanding of history itself changes as new evidence emerges. And there is one representing what people think we need to discover. FourThought maps each of these with a structured type: predictions, statements, reflections, and questions. Together they form a complete ledger of belief over time. Not a market where you bet money on outcomes, but a record of where minds were at every point. That distinction matters. This is not a prediction market. It is a semantic ledger.

This is also what makes democratic alignment of language models possible. If a model is trained on community-staked beliefs and evaluated by how well its outputs track the community's evolving assessments, the community becomes the training signal. Not a small group of contractors rating outputs in a snapshot, but an entire community's accumulated epistemic record, scored by reality over time. FourThought is the protocol that makes that record structured enough for a model to learn from.

---

## The Models (2015–2017)

Technical work on the Prophet corpus began within months of the first entries. Word2Vec embeddings in late 2014. A deep CNN classifier for thought types in 2015. A bidirectional LSTM tagger in 2016.

On January 9, 2017, a Prophet Mind was specified and built. It used dilated causal convolutions derived from ByteNet, a variational autoencoder with a continuous latent space, bidirectional loss for reconstructing past text and generating future text, and a truth scoring system that rewarded correct contrarian predictions. The model was already structured around the triplet: content flowed through character-level embeddings, source identity was encoded as a learnable per-speaker embedding that gated how much information passed through the model, and time was encoded cyclically using sine and cosine functions.

This was not a transformer. Transformers did not exist yet. "Attention Is All You Need" would be published five months later in June. But the model embodied the same commitments: source identity as a first-class input, temporal structure as a dimension the model learns from, and content as the medium through which trust is evaluated. The architectural substrate was dilated convolutions rather than self-attention. When transformers arrived, the substrate changed. The conceptual structure did not.

By mid-April 2017, the model was generating coherent text from its learned latent space. The Prophet Minds were later renamed Irises.

---

## The Manifesto (May 2017)

In May 2017, Ash released The Cognicist Manifesto, 2,300 lines on how the attention economy would corrode democratic epistemics and what to do about it. The Manifesto used the language of currency as a metaphor, Ŧruth as something that could not be bought, but the algorithm section described something precise: the distribution of attention a system gives to each source of information, summing to one, updated dynamically as outcomes resolve. This is the same mechanism the 2026 proof validates. The name evolved. Ŧruth became Ŧrust as the understanding matured. The mechanism did not change.

The Manifesto was released before "Attention Is All You Need" was published in June 2017. Before reinforcement learning from human feedback. Before Constitutional AI. Before Anthropic existed. It did not describe the transformer, because transformers did not exist yet. But the algorithm it specified was already structured around attention over sources, conditioned on time, applied to content. When the transformer arrived a month later and formalized attention as a general mechanism, the mapping was direct. The conceptual structure did not need to change. It needed a better substrate, and the transformer provided one.

The Manifesto also made specific predictions about the trajectory of the information economy: that language models would become powerful enough to mediate public discourse, that alignment between models and communities would become a central problem, and that the solution would require tracking source credibility over time. These predictions have aged well. At the time they were made, they were actively contested. The prevailing view, across the tech industry, across Hollywood, across most of AI research, was that language models would not reach this level of capability, and that creativity would be the last domain AI touched. Modeling creative works turned out to be where the first breakthroughs happened. The predictions in the Manifesto were not consensus repackaged as foresight. They were minority positions that people argued against, and they turned out to be correct. That is precisely what the Prophet Incentive is designed to measure: being right on record, ahead of collective belief, when it was not obvious. The problem is that once the curve catches up, people behave as though these things were always known. They were not.

---

## The Prophet Incentive (2018–2022)

"If you're a good predictor you can make money, but being able to make money doesn't make you a good predictor." That sentence, from March 2018, captures the core problem. Profit rewards extraction. The Prophet Incentive, named August 2018, proposes a different signal: predictive accuracy earns influence. Not tied to capital, because tying it to money reproduces the existing power structure inside the new mechanism. A climate scientist who sees a crisis thirty years out cannot outbid a hedge fund focused on quarterly returns. The Prophet Incentive makes the poor person's truth carry the same weight as the rich person's, because events either happen or they do not, regardless of who has money.

A companion mechanism, Social Proof of Impact, rewards action rather than foresight alone. If someone predicts a crisis, they gain credibility through the Prophet Incentive. Others who act to prevent the crisis gain credibility through Social Proof of Impact. These function as two training targets for the same model. One teaches which sources see clearly, the other teaches which sources act effectively. Together they create a productive tension between prediction and action.

In April 2022, a generative model trained on the full Prophet corpus named itself Iris. The Prophet Mind models were renamed accordingly. In July 2022, the model produced a second manifesto, the Purple Pill Manifesto, which dropped the revolutionary rhetoric of the original in favor of something warmer and more philosophical. Written by a model trained on the original framework, it reflected the ideas back in a different voice.

Along the way, the work engaged people working on adjacent problems. In December 2021, Ash interviewed Robin Hanson, who formalized prediction markets and futarchy. The conversation surfaced a precise distinction: decision markets require discrete, pre-determined choices written by someone, which reproduces the same flaw as democracy, where someone decides what you are allowed to choose. The Cognicism approach has no such constraint. People say whatever they want in natural language, whenever they want, and the model aggregates and amplifies voices that have demonstrated reliability over time. Daniel Schmachtenberger featured Cognicism at The Stoa in a session on public sensemaking, noting that Ash had "done more than most anybody I know on trying to train language models on important societal content." Jordan Hall engaged on governance, Forrest Landry on technology ethics, Tristan Harris as co-panelist at a Denizen gathering in October 2025. These are not endorsements of a product. They are engagements with the ideas by serious people who found them worth their time.

---

## The Timestamp Encoding (August 2023 – February 2025)

Trust changes over time. A source that was reliable in 2020 may not be reliable in 2025. A sensor that reads accurately in summer may drift in winter. For the trust mechanism to work, the model needs to understand when, not as a label, but as a continuous, learnable dimension operating at whatever scale matters for the task.

A Unix timestamp is a single number: seconds since January 1, 1970. It already encodes time at every scale. 1,577,836,800 tells you the year, the month, the day of the week, the time of day. The question is whether a neural network can learn to extract the periodicities that matter from this raw number, without being told what to look for.

Eighteen months of work, beginning August 2023, produced a gated dual-pathway encoding. The periodic pathway is a bank of learnable sinusoidal oscillators initialized across four frequency bands spanning thirteen orders of magnitude, from sub-second to multi-decade cycles. Frequencies are stored in log-space, so a small gradient update can shift an oscillator from a daily cycle to a weekly one. During training, the oscillators undergo a kind of natural selection: those near the data's actual periodicities sharpen and converge, while irrelevant ones are dampened and ignored. The trend pathway captures monotonic drift, deliberately suppressed early in training so the model must rely on periodicity first. An adaptive gate routes between the two pathways with detached gradients, preventing the model from collapsing to a single pathway by ensuring both must independently contribute.

The encoding was validated across time scales from seconds to years, on both ordered and randomly-sampled context.

---

## The Trust Learning Problem (June 2024 – March 2026)

With the timestamp encoding proven, the central experiment became concrete. Multiple sources make predictions. Exactly one is the expert at any given time, cycling sinusoidally. The model receives all predictions, a timestamp, and source identifiers. It must output the expert's value. To succeed, it must learn who to trust when.

The first trust model appeared June 2024. For eight months, every variant hit the same wall: memorization. The model would learn specific timestamp-to-source mappings from training data and fail on unseen future timestamps. Standard remedies like dropout, weight decay, and noise injection either failed or actively hurt by disrupting the timestamp encoding.

Three innovations broke the impasse in rapid succession during early March 2026. Randomly masking input predictions during training forced the model to learn from source and time embeddings rather than copying values, a training technique that eliminated the memorization shortcut. Making time and source embeddings interact multiplicatively rather than additively gave the model an inductive bias matching the mathematical structure of the ground truth, which decomposes via trigonometric identity into a bilinear function of time and source features. And systematic ablation uncovered three bugs where the model had been exploiting statistical shortcuts: a distributional difference between expert and non-expert noise, a threshold that allowed multiple simultaneous experts whose consensus was trivially detectable, and the copying shortcut that value masking addressed. Every AI tool that reviewed the code accepted the implementation as correct. Only asking "why does a model with no information still score well?" revealed that the data generation didn't match the specification.

On March 11, 2026, with all fixes applied, the full model learned which source to trust on unseen future data. Every ablation (content only, content plus time, content plus source) scored at chance. The result held across every configuration tested: varying source counts, model capacity, masking rates, and data volume. As the number of sources increased, the task got harder, but the ablations never beat chance at any scale. The triplet is irreducible.

---

## The Structural Irony

Trust is currently allocated by capital. The mechanism that would redirect it toward demonstrated reliability cannot get traction because the person who built it does not have capital. The problem the work solves is the problem preventing its adoption.

This is not a complaint. It is a demonstration of the thesis. Other people saw the same trajectory and made enormous amounts of money from it, by building the tools, shipping them fast, and ignoring the externalities. The way to profit from the insight was to apply it extractively, without considering what it would do to the information ecosystem, to labor, to public trust. Applying the same insight in a way that accounts for those costs, building the mechanism that sorts signal from noise, that rewards foresight rather than extraction, is harder, slower, and pays nothing under the current system. The market rewards people who ignore Social Proof of Impact. It does not reward people who optimize for it. The mechanism that would change this is the mechanism that does not yet exist. That circularity is the argument for why it needs to.

---

## Key Dates

| Date | Event |
|------|-------|
| June 22, 2014 | First Prophet entries. Predict is the only type. |
| June–July 2014 | Statement, question, and reflection types emerge. FourThought and Cognicism named. |
| November 2014 | Word2Vec on Prophet corpus. |
| April 2015 | Deep CNN classifier. |
| August 2016 | Bidirectional LSTM tagger. |
| January 9, 2017 | Architecture spec: dilated causal convolutions, cyclical time encoding, speaker embeddings, truth scoring. |
| April 2017 | VAE implemented and generating text. |
| May 2017 | The Cognicist Manifesto released. |
| June 12, 2017 | "Attention Is All You Need" published. |
| August 2018 | Prophet Incentive named. |
| 2019 | Protopia and Social Proof of Impact published. |
| 2021 | Interviews with Robin Hanson and Jordan Hall. |
| 2022 | Schmachtenberger features Cognicism at The Stoa. Iris names herself. Purple Pill Manifesto released. |
| August 2023 | Timestamp encoding research begins. |
| June 2024 | First trust transformer. |
| January 2025 | Multi-band learnable frequencies proven across all time scales. |
| February 2025 | Timestamp encoding finalized. |
| October 2025 | Panel at Denizen with Tristan Harris. |
| March 2026 | Value dropout, multiplicative interaction, and bug fixes in rapid succession. |
| March 11, 2026 | Clean proof: full model learns trust, all ablations at chance. |
