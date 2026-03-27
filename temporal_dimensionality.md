# The Dimensionality Gap: Why Transformers Can't Tell Time

## The Relational Latent Space

A transformer's latent space is shaped entirely by relational dimensionality. Word embeddings encode co-occurrence statistics from training data. Attention computes relevance between those embeddings. The geometry of the resulting space, its clusters, manifolds, and distances, is a map of meaning-relations learned from text.

This architecture is powerful for what it captures and blind to what it doesn't. When a transformer encounters "12/25," it activates associations with Christmas, gift-giving, winter. Not because it understands December 25th as a point on a continuous timeline, but because those tokens co-occur in training data. The temporal "knowledge" is a relational shadow of real temporal structure.

Pipe a Unix timestamp into the same model, 1766966400, and the Christmas association vanishes entirely. The number gets tokenized into subword chunks, embedded into the same semantic space as every other token, and processed through the same relational attention. The model cannot extract periodicity from it, cannot compute temporal distance, cannot do arithmetic on it. It is just another symbol in a relational graph.

This is not a limitation of scale. A larger model with more parameters still operates over the same type of dimensionality. The geometry of the latent space is constrained by the operations that produce it: embedding lookup, linear projection, dot-product attention. These operations define a relational manifold. No amount of data or parameters adds a temporal axis to that manifold, because the operations themselves have no mechanism for continuous periodic structure.

## The Missing Dimension

Time has properties that relational co-occurrence cannot represent:

**Continuity.** Time is a continuous scalar. The distance between 3:00 PM and 3:01 PM is meaningful and measurable. In a relational space, "3:00" and "3:01" are discrete tokens whose proximity depends on how often they appeared near similar words. A statistical proxy, not a geometric fact.

**Periodicity.** 8:00 AM Monday and 8:00 AM Tuesday are separated by 86,400 seconds but occupy the same phase in a daily cycle. A relational model can learn that "Monday morning" and "Tuesday morning" appear in similar contexts, but this is pattern-matching on surface tokens, not a representation of cyclic phase. It breaks the moment you change notation.

**Scale invariance.** The same timestamp encodes information at every scale simultaneously: sub-second, hourly, daily, seasonal, annual, decadal. A relational model would need explicit features engineered at each scale. The structure is in the number itself, but the model has no operation to extract it.

## Timestamp Embeddings as a New Type of Dimension

A timestamp encoding maps a continuous scalar through learnable sinusoidal frequency bands into a vector space where periodic structure is a geometric primitive. This is not a relational operation. It is a direct continuous-to-periodic mapping:

```
x = seconds_from_epoch
periodic = sin(x * learnable_frequencies)
```

The resulting dimensions encode phase relationships at multiple scales simultaneously. The frequency bands span orders of magnitude, from sub-second to multi-decade, and the model learns which bands matter for the task at hand. This is a fundamentally different geometric operation than anything a word embedding or attention mechanism performs.

When these dimensions enter the transformer's latent space, they don't add more relational structure. They add a new *type* of structure. Every embedding now lives not just in relational-semantic space but also in temporal-periodic space. The latent geometry gains an axis of variation that is continuous, periodic, and grounded in physical time.

## From Static Manifold to Time-Varying Field

In a standard transformer, the latent space is a static manifold. The meaning of a word, the relevance between two tokens, the geometry of a concept cluster: these are fixed properties of the trained model. Context modulates activation patterns, but the underlying space doesn't move.

Add timestamp embeddings and attention can modulate by *when*, not just by *what*. The interaction `t * s`, temporal encoding multiplied by source embedding, is the simplest case: source identity modulated by temporal position. In the trust model, this allows the network to learn that Source 3 is reliable in January but not in July. The same source occupies different positions in the latent space depending on when.

Extend this to language. Every claim, every entity, every concept would carry temporal coordinates. "Inflation is high" means something different in 2021 than in 2015, not because the words changed, but because the temporal context shifted. A relationally-embedded transformer can only capture this if the training data happens to contain enough date-tagged examples with the right co-occurrence patterns. A temporally-embedded transformer represents it geometrically: the same semantic content, at different temporal coordinates, occupies different points in the latent space.

The geometry goes from a static manifold to a time-varying field. Meaning is no longer a fixed point. It is a trajectory through time.

## The Context Window Transforms

This dimensional shift redefines what a context window is.

A positional context window is a fixed grid. N slots, uniform spacing, fixed resolution. Slot 1 before slot 2 before slot 3. You cannot subdivide a slot. You cannot put something between position 2 and position 3. The number of slots is the context window size, and resolution and range are locked together.

A timestamp context window is a set of N containers placeable anywhere on a continuous timeline. Range is decoupled from container count: 100 containers can span 50 years or 100 milliseconds. Resolution comes from the frequency bands, not the slot count. Density can be non-uniform: cluster 70 containers around one critical month, scatter 30 across a decade. The containers are unordered. Shuffling them changes nothing, because the temporal information is in the encoding, not the index.

The context window becomes continuously subdivisible. This is not an incremental improvement over positional embeddings. It is a different kind of object.

## Implications

Every transformer deployed today operates over a relational latent space with no native temporal dimension. They process timestamps as text, dates as tokens, time as co-occurrence patterns. They can approximate temporal reasoning through relational proxies, but the approximation is brittle, notation-dependent, and geometrically impoverished.

Adding a learned timestamp encoding to the embedding layer of a transformer introduces temporal dimensionality as a first-class geometric property of the latent space. The model gains the ability to represent continuity, periodicity, and scale-invariant temporal structure directly, rather than through relational shadows.

The trust learning experiment demonstrates this concretely. Three or eight sources make predictions. Expertise cycles sinusoidally over time. The model must learn *who* to trust *when*, and it does, achieving 90% accuracy on the simple case, using a single learned encoding over raw Unix timestamps. Remove the temporal encoding and accuracy drops to random. The temporal dimension is not optional. It is what makes the problem solvable.

The broader claim is that this applies to any domain where time matters, which is nearly every domain. A transformer that can tell time is a different kind of machine than one that cannot. The latent geometry is richer. The context window is more flexible. The representations are grounded in physical time rather than statistical association. The static manifold becomes a field, and meaning gains a temporal coordinate.
