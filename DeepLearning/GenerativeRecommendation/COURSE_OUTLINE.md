# Generative Retrieval for Recommendation Systems — Course Outline

A structured set of slides covering the generative retrieval paradigm for recommendations: from sequential models (SASRec) through semantic ID generation (TIGER) to your interpretable hierarchical token approach (GenRec).

---

## Chapter 1: Sequential Recommendation Foundations

**File:** `01_sequential_rec_foundations.html`

### Slide Breakdown:

#### Slide 1: Title
#### Slide 2: The Sequential Recommendation Problem
- Input: user's interaction history (ordered sequence of items)
- Output: predict next item(s) the user will interact with
- Why sequence matters: temporal patterns, session context, intent evolution
- Math: given s = [i_1, i_2, ..., i_t], predict i_{t+1}

#### Slide 3: Evolution of Sequential Methods
- Markov chains → RNN (GRU4Rec) → Attention (SASRec) → Generative (TIGER/GenRec)
- Timeline: 2016 → 2018 → 2023 → 2025

#### Slide 4: SASRec — Self-Attentive Sequential Recommendation (Kang & McAuley, 2018)
- Left-to-right causal self-attention over item ID embeddings
- Architecture: embedding layer → L transformer blocks → prediction head
- Prediction: softmax over item vocabulary
- Math: h_t = Transformer(e_{i_1}, ..., e_{i_t}), score(i) = h_t · e_i
- Results on your data: Recall@1 = 0.706, NDCG@10 = 0.838#### Slide 5: BERT4Rec — Bidirectional Sequential Rec
- Masked item prediction (like BERT's MLM)
- Bidirectional attention: sees past AND future context
- Training: randomly mask items, predict them from context
- Comparison with SASRec: more context but harder to generate autoregressively

#### Slide 6: GRU4Rec — RNN-Based Sequential Rec
- GRU hidden state summarizes interaction history
- Session-based: each session is a sequence
- Limitations: vanishing gradients, no parallel computation
- Results on your data: Recall@1 = 0.628

#### Slide 7: Limitations of ID-Based Methods
- Vocabulary size: each item needs a unique embedding
- Cold-start: new items have no learned embedding
- No semantic meaning in IDs: item 42 tells you nothing about the item
- No compositionality: can't generalize to unseen combinations
- Story: "ID-based models memorize the catalog. They can't reason about items they haven't seen."

#### Quiz (4 questions)

---

## Chapter 2: The Generative Retrieval Paradigm

**File:** `02_generative_retrieval.html`

### Slide Breakdown:

#### Slide 1: Title
#### Slide 2: What is Generative Retrieval?
- Traditional retrieval: score all candidates, pick top-K (discriminative)
- Generative retrieval: GENERATE the item token-by-token (generative)
- Analogy: multiple-choice exam vs write-your-own-answer
- Key shift: items are represented as TOKEN SEQUENCES, not atomic IDs

#### Slide 3: Why Generate Instead of Retrieve?
- Scalability: no need to score all N items (just decode)
- Compositionality: tokens can recombine for novel items
- Interpretability: intermediate tokens reveal reasoning
- Cold-start: new items get tokens from attributes
- Constrained generation: enforce business rules at decode time
- Trade-off: harder optimization, potentially lower raw accuracy

#### Slide 4: The Encoder-Decoder Framework for GenRec
- Encoder: process user interaction history → context representation
- Decoder: autoregressively generate item token sequence
- Math: P(item) = P(c_1) · P(c_2|c_1) · ... · P(c_K|c_1,...,c_{K-1}, context)
- Architecture variations: decoder-only vs encoder-decoder vs LLM-based

#### Slide 5: Key Design Dimensions
| Dimension | Options | Trade-off |
|-----------|---------|-----------|
| Tokenization | Learned vs Semantic | Accuracy vs Interpretability |
| Architecture | Dec-only vs Enc-Dec | Simplicity vs Expressiveness |
| Training | Teacher forcing vs RL/DPO | Stability vs Optimality |
| Inference | Greedy vs Beam vs Constrained | Speed vs Quality |
| Evaluation | uni100 vs Full-catalog | Comparable vs Realistic |

#### Slide 6: Generative vs Discriminative — When to Choose
- Discriminative wins when: full catalog is small, latency is critical, accuracy is paramount
- Generative wins when: interpretability matters, constrained generation needed, cold-start frequent, catalog is huge
- Hybrid: generate candidates → discriminative reranker (best of both)

#### Quiz (4 questions)

---

## Chapter 3: Item Tokenization Methods

**File:** `03_tokenization_methods.html`

### Slide Breakdown:

#### Slide 1: Title — "Item Tokenization: The Most Important Design Decision"
#### Slide 2: Why Tokenization is the Ceiling
- Theorem (informal): generation accuracy ≤ tokenization quality
- If two different items have the same token sequence (collision) → model CANNOT distinguish them
- Collision rate directly bounds achievable Recall@1
- All architecture improvements are meaningless if tokens are bad
- Story: "You can build the best language model in the world, but if your vocabulary is ambiguous, it will always hallucinate."

#### Slide 3: Taxonomy of Tokenization Methods
```
Item Tokenization
├── Learned (data-driven)
│   ├── RQ-VAE (TIGER) — residual quantization of embeddings
│   ├── VQ-VAE — single-level vector quantization
│   ├── OPQ (Optimized Product Quantization) — rotation + PQ
│   └── CF-augmented (CoFiRec) — collaborative signal in codebook
├── Semantic (attribute-based)
│   ├── Category hierarchy (L1→L2→...→Ln)
│   ├── Mixed attributes (brand, price, intent, specs)
│   └── LLM-classified attributes
├── Hybrid
│   ├── Dual Identifiers — semantic + collaborative tokens
│   ├── KGTB — knowledge graph → token hierarchy
│   └── Attribute tokens + learned spec tokens (your approach)
└── Hashing-based
    ├── Locality-Sensitive Hashing
    └── Learned hash functions
```

#### Slide 4: RQ-VAE — Residual Quantization (TIGER, 2023)
- Input: item embedding e ∈ R^d (from content model or CF)
- Step 1: find nearest centroid c_1 = argmin ‖e - codebook_1[k]‖
- Step 2: compute residual r_1 = e - codebook_1[c_1]
- Step 3: quantize residual c_2 = argmin ‖r_1 - codebook_2[k]‖
- Step 4: repeat for K levels
- Output: item ID = [c_1, c_2, ..., c_K]
- Math: e ≈ Σ_{k=1}^K codebook_k[c_k]
- Codebook sizes: typically 256-1024 per level, K=4-8 levels
- Training: straight-through estimator for gradients through argmin
- Commitment loss: ‖e - sg(codebook[c])‖² + β‖sg(e) - codebook[c]‖²

#### Slide 5: RQ-VAE — Properties and Limitations
- Hierarchical: level 1 = coarse cluster, level K = fine discrimination
- Uniqueness: optimized to minimize collisions (but not guaranteed)
- Semantic structure: similar items → similar token prefixes (soft property)
- Training jointly with downstream task (end-to-end) vs pre-trained separately
- Limitations:
  - Opaque: token [3, 17, 42, 8] means nothing to humans
  - Cold-start: new items need embedding → need to run through RQ-VAE
  - Constraint generation: can't say "only generate electronics" (codes are opaque)
  - Codebook collapse: some centroids go unused → wasted capacity

#### Slide 6: OPQ — Optimized Product Quantization
- Different from RQ-VAE: splits dimensions (not residuals)
- Step 1: learn rotation matrix R to decorrelate dimensions
- Step 2: split rotated embedding into M subvectors
- Step 3: quantize each subvector independently (K centroids per subspace)
- Output: item ID = [c_1, c_2, ..., c_M] where each c_m ∈ {1,...,K}
- Math: ê = R·e, split into [ê_1, ..., ê_M], quantize each
- Advantage: each subspace captures different "aspect" (one might capture price, another category)
- Used in your L7 spec tokens: OPQ on Titan Embed summaries → 4 spec tokens

#### Slide 7: CoFiRec — Collaborative Filtering in Tokenization (Nov 2025)
- Problem with pure content tokens: miss co-purchase signal
- CoFiRec approach:
  - Stage 1: content-based clustering (as in RQ-VAE)
  - Stage 2: refine assignments using CF signal (co-interaction graph)
  - Items that are frequently co-purchased get similar token prefixes
- Result: tokens encode BOTH content similarity AND collaborative similarity
- Key finding: +15-20% Recall improvement over content-only tokenization
- Connection to your graph work: CF signal from graph propagation → token quality

#### Slide 8: Semantic / Attribute-Based Tokenization
- Use human-interpretable attributes as tokens directly
- Hierarchy examples:
  - Simple: Category → Subcategory → Brand → Price Tier
  - Rich (your approach): Family(L1) → Ecosystem(L3) → Brand(L4) → Price(L5) → Intent(L6) → Specs(L7)
  - Knowledge graph: Entity type → Relations → Attributes
- LLM-classified attributes: prompt LLM to classify items into categories/intents
- Collision handling: multiple items can share same attribute tuple
  - Solution: add spec tokens (learned) to increase uniqueness
  - Solution: item ID head for de-collision at inference

#### Slide 9: Semantic Tokenization — Properties and Trade-offs
| Property | Learned (RQ-VAE) | Semantic (Attributes) |
|----------|-------------------|----------------------|
| Interpretability | ✗ Opaque codes | ✓ Human-readable levels |
| Cold-start | ✗ Needs encoder pass | ✓ Attributes known at item creation |
| Uniqueness | ~95-99% (optimized) | ~70-90% (depends on schema) |
| Constrained generation | Hard | Trivial (filter by attribute) |
| Collaborative signal | Via embedding input | ✗ Unless CF-augmented |
| Update cost | Retrain codebook | Just assign attributes |
| Vocabulary size | Small (256-1024/level) | Varies (18 families, 50 brands, etc.) |

#### Slide 10: Hybrid Approaches
- **Dual Identifiers (SIGIR-AP 2025):** generate both semantic + collaborative tokens
  - Two heads: one generates attributes, one generates RQ-VAE codes
  - Score = combination of both generation probabilities
- **KGTB:** knowledge graph structure → token tree
  - KG entity → parent entity → parent → ... → root = token path
  - Items inherit tokens from their KG position
- **Your approach (GenRec):** interpretable attributes (L1-L6) + learned specs (L7)
  - Best of both: coarse levels interpretable, fine levels learned for uniqueness
  - L1-L6: semantic, constrained, interpretable (18+4+50+4+18 = <100 codes total)
  - L7a-d: OPQ on Titan Embed (256 codes × 4 = discriminative power)
  - Collision rate: ~10% among 45K items (90% unique tuples)

#### Slide 11: Collision Analysis — The Practical Ceiling
- Definition: collision = multiple items share the SAME complete token sequence
- Impact: if 3 items share tokens, model can't distinguish → Recall@1 ≤ 1/3 for those items
- Theoretical max Recall@1 = (unique items / total items)
  - 90% unique → max R@1 from tokens alone = 0.90 (before disambiguation)
  - With ID head: can recover some collisions via discriminative scoring
- Reducing collisions:
  - More token levels (deeper hierarchy)
  - Larger codebook per level
  - Better embedding input (captures more item distinctiveness)
  - Domain-specific attributes (avoid generic categories)
- Your numbers: 2M catalog × 90% unique = most items distinguishable

#### Slide 12: Practical Token Design Decisions
- How many levels? 4-9 (fewer = more collisions, more = harder to generate)
- Codebook size per level? 256-1024 (smaller = easier to predict, larger = more expressive)
- Hierarchical or flat? Hierarchical enables constrained beam search
- Jointly trained with model? Yes (end-to-end) > No (fixed tokens) — but more expensive
- Update frequency? Content tokens = static; CF tokens = retrain with new interactions
- Table of design choices by paper: TIGER (4 levels, 256/level), CoFiRec (4, 512), GenRec (9 levels, mixed)

#### Quiz (5 questions):
1. If a tokenization scheme has 15% collision rate, what is the theoretical maximum Recall@1?
2. RQ-VAE quantizes residuals; OPQ quantizes subvectors. What structural assumption does each make?
3. Why does CoFiRec improve over pure content-based tokenization? What signal is it adding?
4. In your L1-L7 scheme, why are L1-L6 semantic but L7 is learned (OPQ)?
5. Dual Identifiers generate two token types. Why not just make the semantic tokens more fine-grained instead?

---

## Chapter 4: TIGER and Foundational Generative Retrieval Models

**File:** `04_tiger_and_foundations.html`

### Slide Breakdown:

#### Slide 1: Title
#### Slide 2: TIGER — NeurIPS 2023
- Architecture: encoder (user history) → decoder (item token sequence)
- Encoder: SASRec-style self-attention over item embeddings
- Decoder: autoregressive generation of RQ-VAE tokens
- Token hierarchy: coarse → fine (category → specific item)
- Training: teacher forcing on ground-truth token sequences
- Inference: beam search to generate top-K items

#### Slide 3: TIGER Architecture Detail
- Item encoder: convert each item to its token sequence representation
- History encoder: bidirectional attention over past items
- Decoder: causal cross-attention (attend to history), generate tokens left-to-right
- Loss: cross-entropy per token level
- Math: P(item) = P(c_1)·P(c_2|c_1)·...·P(c_K|c_1,...,c_{K-1})

#### Slide 4: CoFiRec — Hierarchical Item Tokenization (Nov 2025)
- Key insight: incorporate collaborative filtering signal into tokenization
- Two-stage: content-based clustering → CF-refined assignment
- Better token quality → better generation accuracy
- Connection: tokenization quality IS the ceiling for generative retrieval

#### Slide 5: LETTER — LLM for Generative Retrieval (Li et al., 2024)
- Uses pre-trained LLM (rather than training from scratch)
- Items described as text → LLM generates item descriptions
- Bridge between NLP generative models and recommendation

#### Slide 6: HiGR — Tencent (Feb 2026)
- Hierarchical generative retrieval at industrial scale
- Multi-level generation with progressive refinement
- Key challenge: maintaining retrieval quality at billion-item scale

#### Slide 7: Dual Identifiers (SIGIR-AP 2025)
- Combines semantic IDs with collaborative IDs
- Addresses limitation of purely semantic tokens (lose co-purchase signal)
- Architecture: dual-head decoder generating both ID types

#### Slide 8: Key Findings Across Papers
- Token quality matters more than architecture
- Collision rate is the primary bottleneck
- Hierarchical structure helps beam search diversity
- Gap to ID-based models narrows with better tokenization

#### Quiz (4 questions)

---

## Chapter 5: LLM Training Techniques for Generative Retrieval

**File:** `05_llm_training_techniques.html`

### Slide Breakdown:

#### Slide 1: Title — "LLM Training Techniques Applied to Generative Recommendation"

#### Slide 2: Why LLM Techniques Transfer
- GenRec = autoregressive token generation (same as LLMs)
- Shared challenges: exposure bias, mode collapse, reward optimization
- Key difference: discrete structured outputs (item tokens) vs free-form text
- Applicable techniques: all of autoregressive training + alignment

#### Slide 3: Teacher Forcing and Exposure Bias
- Teacher forcing: use ground-truth tokens as decoder input during training
- Math: L = -Σ_t log P(c_t | c_1,...,c_{t-1}, context) — conditioned on TRUE prefix
- Problem at inference: model feeds its own predictions (may drift from training distribution)
- Exposure bias: model never practices recovering from its own mistakes
- In GenRec: if L1 prediction is wrong, all subsequent levels drift

#### Slide 4: Scheduled Sampling
- Gradually replace ground-truth with model's own predictions during training
- Schedule: P(use_model_output) increases from 0 → ε over training
- Curriculum: early training = stable (teacher forcing), late training = realistic (self-feeding)
- Math: at step t, use c_t = ground_truth with prob (1-ε_t), else argmax P(c_t|...)
- Bridges the train-test distribution gap
- Limitation: can introduce instability if schedule is too aggressive

#### Slide 5: Multi-Token Lookahead Training
- Problem: autoregressive loss only optimizes "next token" — no long-range planning
- Solution: auxiliary heads predict DOWNSTREAM tokens from EARLIER hidden states
- Lookahead pairs: (L1→L4), (L1→L6), (L3→L6), (L1→L7a)
- Math: L_total = L_main + λ · Σ_{(src,tgt)} CE(head_tgt(h_src), c_tgt)
- Lambda: 0.3 auxiliary loss weight
- Effect: early tokens are forced to "plan ahead" for later levels
- Analogy to LLMs: like training with both next-token AND multi-step objectives

#### Slide 6: Causal Pre-Training (GPT-style)
- Pre-train on large item sequence corpus (all user histories)
- Objective: next-item-token prediction (self-supervised)
- Then fine-tune on specific task (recommendation with labels)
- Benefits: learns item co-occurrence patterns, token transition probabilities
- Connection to GPT: pre-training on internet text → fine-tuning on task
- In GenRec: pre-train on all sequences → fine-tune with purchase/conversion labels

#### Slide 7: RLHF / Reward Modeling for Recommendations
- Standard RLHF pipeline: SFT → Reward Model → PPO
- For recommendations:
  - SFT = pre-train with teacher forcing on interaction sequences
  - Reward = purchase/conversion/engagement signal
  - PPO = optimize generation policy toward high-reward items
- Challenge: reward is sparse (only know if user bought, not preference ordering)
- Alternative: use CTR/CVR model as reward model (dense signal)
- Practical issue: unstable training, reward hacking (generate popular items only)

#### Slide 8: DPO — Direct Preference Optimization (Rafailov et al., 2023)
- Eliminates reward model: directly optimize from preference pairs
- For GenRec: purchase = chosen, explore-only = rejected
- Math: L_DPO = -log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
  - y_w = purchased item tokens (winner)
  - y_l = explored-but-not-purchased item tokens (loser)
  - π_ref = pre-trained model (frozen reference)
  - β = KL penalty strength
- Phase 1: Pre-train with standard CE loss → π_ref
- Phase 2: DPO fine-tune on preference pairs
- Result: shifts generation probability toward items users actually buy

#### Slide 9: Contrastive Learning and InfoNCE
- In-batch contrastive on decoder hidden states
- Positive: hidden state at end of generation → correct item embedding
- Negatives: other items in batch (in-batch negatives)
- Math: L_contrast = -log(exp(h·e_pos/τ) / Σ_k exp(h·e_k/τ))
- Purpose: discriminative signal for disambiguation (collision resolution)
- Complementary to generation loss: generation = recall, contrastive = precision
- Temperature τ: low = focus on hard negatives, high = smooth gradients

#### Slide 10: Knowledge Distillation (from larger models)
- Teacher: large pre-trained model (or ensemble) generates soft targets
- Student: smaller GenRec model learns to match teacher's token distribution
- Math: L_KD = KL(P_teacher ∥ P_student) = Σ P_T(c) log(P_T(c)/P_S(c))
- Application: distill LLM-based recommendations into efficient GenRec model
- Also: SASRec (discriminative) as teacher → GenRec (generative) as student
- Benefit: transfers knowledge from a model that "knows the right answer" to one that "generates the right answer"

#### Slide 11: Curriculum Learning and Data Ordering
- Idea: order training examples from easy to hard
- Easy sequences: short history, popular items, clear patterns
- Hard sequences: long history, tail items, ambiguous intent
- Schedule: first N epochs on easy → gradually introduce hard examples
- Anti-curriculum: some evidence that hard-first works for certain tasks
- For GenRec: start with high-purchase-rate sequences, add exploratory sequences later
- Connection to negative sampling curriculum in graph methods

#### Slide 12: LoRA and Parameter-Efficient Fine-Tuning
- Problem: fine-tuning full model for each task/domain is expensive
- LoRA: freeze base weights W, learn low-rank update ΔW = A·B (r ≪ d)
- Math: W' = W + α·A·B where A ∈ R^(d×r), B ∈ R^(r×d), r = 4-64
- Application in GenRec:
  - Base model: pre-trained on large catalog
  - LoRA fine-tune: per-marketplace, per-category, or per-campaign
  - Result: one base model + small adapters for different contexts
- Also applicable: adapter layers, prompt tuning, prefix tuning

#### Slide 13: Label Smoothing and Token Dropout
- Label smoothing: soften targets from one-hot to (1-ε, ε/(V-1), ...)
  - Prevents model from being over-confident on training tokens
  - Regularization that helps generalization
- Token dropout: randomly mask some input tokens during training
  - Forces model to be robust to incomplete history
  - Connection to BERT's MLM (but in decoder context)
- Embedding dropout: dropout on item embeddings during encoding
  - Prevents over-reliance on specific items in history

#### Slide 14: Combining Techniques — What Works Together
| Technique | Addresses | Compatible with |
|-----------|-----------|-----------------|
| Scheduled Sampling | Exposure bias | DPO, Lookahead |
| Lookahead | Error compounding | Scheduled Sampling, Contrastive |
| DPO | Alignment to purchases | Contrastive, LoRA |
| Contrastive | Collision disambiguation | Everything |
| Distillation | Efficiency | LoRA, Curriculum |
| Curriculum | Training stability | DPO (after easy phase) |
| LoRA | Multi-domain efficiency | Everything (applied last) |
| Speculative Decoding | Inference latency | Any trained model (post-hoc) |
| Diffusion Distillation | Parallel decoding | Distillation, LoRA |

- Recommended pipeline: Pre-train (CE + Contrastive) → Lookahead → DPO → LoRA (per-domain)
- Inference acceleration: Speculative decoding (lossless) or Diffusion distillation (lossy but faster)
- Anti-patterns: DPO too early (unstable), curriculum + random augmentation (contradictory)

#### Slide 15: Speculative Decoding for Fast Inference
- Problem: autoregressive decoding is slow — K sequential forward passes per item
- Speculative decoding (Leviathan et al., 2023): use a SMALL draft model to propose token sequences, then verify with the large model in PARALLEL
- For GenRec:
  - Draft model: lightweight 2-layer transformer trained on same token vocabulary
  - Proposes full token sequence [c_1, ..., c_K] in one pass (or few passes)
  - Target model: verifies/corrects in a single parallel forward pass
  - Acceptance: if draft token matches target's distribution → accept (free); else → reject and resample
- Math: accept c_t with probability min(1, P_target(c_t)/P_draft(c_t))
- Speedup: 2-4× for K=9 level tokens (most draft tokens accepted for "easy" items)
- Key insight for GenRec: upper levels (L1-L4) are highly predictable → draft model almost always correct there → speedup concentrated in easy-to-predict levels
- Trade-off: exact same output distribution as target model (no quality loss, just speed)

#### Slide 16: Diffusion-Based Distillation for Parallel Decoding
- Problem: autoregressive decoding is inherently sequential — can we decode ALL tokens at once?
- Diffusion approach: train a model that generates all K tokens simultaneously via iterative denoising
  - Start from random noise tokens → iteratively refine toward valid item tokens
  - Each denoising step refines ALL positions in parallel
  - After T steps (T << K): clean token sequence emerges
- Distillation pipeline:
  1. Train autoregressive GenRec model (teacher) — high quality, slow
  2. Train diffusion/masked model (student) — parallel, fast
  3. Student learns to match teacher's output distribution via KL divergence
- Architecture options:
  - Discrete diffusion (D3PM, Absorbing): mask tokens → predict simultaneously
  - Continuous diffusion on embeddings → quantize to tokens at the end
  - Masked parallel decoding (BERT-style): predict all masked positions at once, iterate
- Speedup: 5-10× (T=3-5 diffusion steps vs K=9 autoregressive steps)
- Quality trade-off: slight degradation vs autoregressive (1-3% Recall drop typical)
- Connection to MaskGIT / MDLM: same idea applied to images / language
- Why it works for GenRec specifically: item tokens are SHORT sequences (K=4-9) where parallel generation is more feasible than in long-form text

#### Slide 17: Alpha Sweep — Generation vs Discrimination Trade-off
- Combined scoring at inference: score = α · generation_prob + (1-α) · ID_head_score
- α=1.0: pure generative (token probabilities only)
- α=0.0: pure discriminative (ID head only)
- Sweet spot: α ≈ 0.3-0.5 — generation provides diversity, ID head provides precision
- This is NOT a training technique — it's an inference fusion strategy
- But training affects optimal α: better generation → higher optimal α

#### Quiz (8 questions):
1. What is exposure bias and why is it worse for multi-level tokens than single-token prediction?
2. DPO uses purchase vs explore as preference pairs. Why not purchase vs random-item?
3. In lookahead training with pair (L1→L6), what does the auxiliary head learn?
4. Knowledge distillation from SASRec to GenRec: what's the teacher's advantage that the student lacks?
5. Why might LoRA be particularly useful for generative recommendation across marketplaces?
6. You combine contrastive loss + DPO. The contrastive loss uses in-batch negatives while DPO uses explore-only items. Are these negatives the same?
7. In speculative decoding, the draft model proposes tokens and the target model verifies. Why is there NO quality loss compared to pure target-model decoding?
8. Diffusion-based parallel decoding trades quality for speed. Why is this trade-off more acceptable for GenRec (K=9 tokens) than for text generation (100s of tokens)?
- Curriculum: start with 100% teacher forcing → gradually increase self-generation
- Bridges the train-test distribution gap

#### Slide 4: Multi-Token Lookahead
- Auxiliary heads predict downstream tokens from earlier hidden states
- Pairs: (L1→L4), (L1→L6), (L3→L6), (L1→L7a)
- Reduces error compounding: early tokens learn to "look ahead"
- Lambda: 0.3 auxiliary loss weight

#### Slide 5: DPO (Direct Preference Optimization) for Recommendations
- Phase 1: Pre-train with standard generation loss
- Phase 2: DPO fine-tune with purchase=chosen, explore=rejected
- Math: L_DPO = -log σ(β · (log π(chosen)/π_ref(chosen) - log π(rejected)/π_ref(rejected)))
- Shifts generation probability toward items users actually bought

#### Slide 6: Contrastive Learning for Disambiguation
- In-batch InfoNCE on hidden states
- Positive: hidden state → correct item embedding
- Negatives: other items in batch
- Purpose: discriminative signal for collision resolution

#### Slide 7: Alpha Sweep — ID vs Generation Trade-off
- Combined scoring: score = α · generation_prob + (1-α) · ID_head_score
- α=1.0: pure generative (token probabilities only)
- α=0.0: pure discriminative (ID head only)
- Sweet spot: α ≈ 0.3-0.5 balances generation with disambiguation

#### Quiz (4 questions)

---

## Chapter 6: Beam Search, Constrained Generation, and Inference

**File:** `06_beam_search_inference.html`

### Slide Breakdown:

#### Slide 1: Title
#### Slide 2: Beam Search for Item Generation
- Standard beam search: maintain top-B partial sequences
- At each step: expand each beam by vocabulary → keep top-B
- For generative rec: vocabulary = tokens at current level
- Result: top-B complete item token sequences

#### Slide 3: Prefix Trie for Constrained Generation
- Build trie from token table (all valid item token tuples)
- At each decoding step: only allow tokens that lead to valid completions
- Guarantees: every generated sequence maps to at least one real item
- Space: O(|items| × |levels|)

#### Slide 4: Hierarchical Beam Search and Diversity
- Coarse-to-fine generation naturally produces diverse candidates
- Level 1 (family): beams spread across categories (phones, headphones, laptops)
- Deeper levels: beams specialize within each category
- Result: top-K recommendations span multiple product types without explicit MMR

#### Slide 5: Constrained Generation for Business Rules
- "Only recommend Apple products" → constrain L3 to ecosystem=apple
- "Only items under $500" → constrain L5 to budget/mid-range
- "Only from this campaign's category" → constrain L1
- Key advantage: constraints are FREE at inference (just trie filtering)

#### Slide 6: Full-Catalog vs uni100 Evaluation
- uni100: sample 99 random negatives + 1 positive, rank among 100
- Full-catalog: rank among ALL items (45K or more)
- uni100 inflates metrics (easy negatives) but is standard in papers
- Full-catalog with beam search: decode top-B, resolve against full catalog

#### Slide 7: Inference Pipeline
- Input: user's interaction history
- Step 1: Encode history (bidirectional attention)
- Step 2: Beam search decode (constrained by trie)
- Step 3: Resolve collisions (ID head or nearest neighbor)
- Step 4: Apply business rules (post-filter or constrained decode)
- Latency considerations: beam width vs quality trade-off

#### Quiz (4 questions)

---

## Chapter 7: Applications — Campaign Targeting and Beyond

**File:** `07_applications.html`

### Slide Breakdown:

#### Slide 1: Title
#### Slide 2: From Recommendation to Campaign Generation
- Standard rec: "which items should this user see?"
- Campaign targeting: "which users should see this campaign?"
- GenRec enables: "generate user interest profiles for campaign matching"

#### Slide 3: Multi-Task Framework
- Task 1: next-item prediction (sequential rec)
- Task 2: campaign-item matching (which items fit a campaign theme?)
- Task 3: user-campaign scoring (does user's generated profile match campaign?)
- Shared encoder: user history understanding

#### Slide 4: Interpretable Campaign Matching
- Campaign defines: category (L1), brand (L4), intent (L6)
- User's generated tokens reveal preferences across these dimensions
- Matching: overlap between campaign tokens and user's top predictions
- Example: campaign "premium Apple audio" matches users who generate (headphones, apple, premium)

#### Slide 5: Agentic Campaign Generation (EMNLP 2026)
- LLM agent that uses GenRec tokens to compose campaigns
- Agent workflow: identify user segments → generate targeting criteria → compose creative
- Tools: token distribution analysis, segment scoring, theme generation

#### Slide 6: Generation-Augmented Retrieval (GAR)
- Invert the pipeline: LLM generates candidate products from query
- Then resolve against real catalog using ANN
- Results: +$209M OPS from Amazon internal study
- Connection: GenRec tokens as structured generation format

#### Slide 7: Cold-Start Item Recommendation
- New item arrives with attributes but no interaction history
- ID-based models: cannot score it at all
- GenRec: new item gets tokens from its attributes (L1-L7)
- Scoring: how likely is the model to generate these tokens for this user?
- No retraining needed — instant inclusion

#### Quiz (4 questions)

---

## Chapter 8: Applied — GenRec System (Your Work)

**File:** `08_applied_genrec.html`

### Slide Breakdown:

#### Slide 1: Title — "Applied: GenRec Interpretable Generative Retrieval"
#### Slide 2: Problem Statement
- Sequential recommendation for wireless electronics (Amazon.in)
- 45K items, 842K sequences, 90K customers
- Dual-stream: purchase + explore history (up to 20 items each)
- Goal: recommend next item with interpretable reasoning

#### Slide 3: Hierarchical Semantic Token Schema
- L1: Product Family (18 categories, LLM-classified)
- L3: Ecosystem (apple/android/windows/agnostic)
- L4: Brand Group (top-N per family)
- L5: Price Tier (quartile within category)
- L6: Use-Case Intent (18 intents, LLM-classified)
- L7a-d: Spec tokens (OPQ on Titan Embed spec summaries)
- Every prediction decomposes into interpretable attributes

#### Slide 4: GenRec v17 Architecture — Encoder-Decoder
- Bidirectional encoder: per-level embeddings + dual positional encoding
  - level_pos: which token level within an item
  - item_pos: which item in the sequence
- Causal cross-attention decoder: generates tokens coarse-to-fine
- Per-level learned embedding tables (separate for each L1-L7)
- L7 centroid initialization from OPQ codebooks

#### Slide 5: Item Identification Head
- Problem: token collisions (multiple items share same token tuple)
- Solution: weight-tied embedding head, jointly trained
- Uses in-batch contrastive loss (InfoNCE) for discriminative hidden states
- De-collides at inference: among items sharing prefix, which specific one?

#### Slide 6: Multi-Position Training + Constrained Beam Search
- Multi-position: predict next item at random split points (subsequence augmentation)
- Prefix trie: every generated sequence guaranteed to map to a real item
- L3 constrained by L1, L4 constrained by (L1, L3)
- Beam diversity naturally spans multiple categories

#### Slide 7: Results Comparison
| Model | Recall@1 | NDCG@10 | MRR |
|-------|----------|---------|-----|
| SASRec | 0.706 | 0.838 | 0.801 |
| BERT4Rec | 0.691 | 0.825 | 0.787 |
| GRU4Rec | 0.628 | 0.794 | 0.746 |
| TIGER v4 | 0.388 | 0.618 | 0.527 |
| GenRec v4 | 0.499 | 0.680 | 0.609 |
| GenRec v5 | 0.503 | 0.688 | 0.616 |

#### Slide 8: What GenRec Does Better (and What It Doesn't)
- ✓ Interpretability: every prediction explains WHY (category, brand, intent)
- ✓ Cold-start: new items score-able from attributes alone
- ✓ Constrained generation: "only Apple" or "only < $500" — free at inference
- ✓ Campaign targeting: tokens → user profiling → campaign matching
- ✗ Raw accuracy gap: SASRec 0.706 vs GenRec 0.503 on Recall@1
- ✗ Trade-off: token quality (collision rate) is the ceiling
- Key insight: GenRec wins on R@1 and R@5 in full-catalog, loses at R@10+

#### Slide 9: Lessons Learned
- Token quality > architecture: reducing collisions matters more than model size
- Interpretable tokens sacrifice some accuracy for controllability
- DPO shifts toward purchases; lookahead reduces error compounding
- Jointly-trained ID head is essential for disambiguation
- Campaign targeting is the killer app (not raw retrieval accuracy)

#### Quiz (5 questions)

---

## Chapter 9: Interview Prep — Generative Retrieval

**File:** `09_interview_prep.html`

### Topics:
- Key concept questions (10 Q&A)
- "Design a generative retrieval system" framework
- Comparison: discriminative vs generative retrieval
- Cheatsheet: methods, architectures, results
- Your work talking points (30-sec, 2-min, deep-dive versions)
- Common follow-up questions and answers

---

## References

### Your Papers:
1. **GenRec Interpretable Retrieval** — Hierarchical semantic tokens + enc-dec architecture
2. **Campaign Framework** — Multi-task generative retrieval for campaign targeting
3. **EMNLP 2026 Industry** — Agentic campaign generation
4. **RecSys 2026 Industry Track** — Production system paper
5. **CIKM 2026** — (venue paper)

### Key External Papers:
1. **TIGER** — Rajput et al., NeurIPS 2023 — Generative retrieval with semantic IDs
2. **CoFiRec** — Hierarchical Item Tokenization (Nov 2025)
3. **HiGR** — Tencent, Feb 2026 — Hierarchical generative retrieval
4. **Dual Identifiers** — SIGIR-AP 2025
5. **LETTER** — Li et al., 2024 — LLM for generative retrieval
6. **SASRec** — Kang & McAuley, 2018 — Sequential baseline
7. **BERT4Rec** — Sun et al., 2019 — Bidirectional sequential
8. **StaticCSR** — YouTube, 2026 — Constrained sequential retrieval
9. **DPO** — Rafailov et al., 2023 — Direct Preference Optimization

### Internal Amazon Papers:
10. **GAR** — Generation-Augmented Retrieval (AMLC 2025)
11. **GPR** — Generative Pretrained Recommender (Feb 2026)
12. **OneMall** — (Feb 2026)

---

## Suggested Flow

| Session | Chapters | Duration | Focus |
|---------|----------|----------|-------|
| 1 | Ch 1 | ~35 min | Sequential rec foundations (SASRec, BERT4Rec) |
| 2 | Ch 2 | ~30 min | Generative retrieval paradigm |
| 3 | Ch 3 | ~45 min | Tokenization methods (RQ-VAE, OPQ, CoFiRec, semantic, hybrid) |
| 4 | Ch 4 | ~40 min | TIGER, CoFiRec, HiGR, Dual IDs |
| 5 | Ch 5 | ~50 min | LLM training techniques (DPO, lookahead, contrastive, LoRA, etc.) |
| 6 | Ch 6 | ~35 min | Beam search, constrained gen, inference |
| 7 | Ch 7 | ~30 min | Applications: campaigns, cold-start, GAR |
| 8 | Ch 8 | ~40 min | Applied: GenRec system (your work) |
| 9 | Ch 9 | ~30 min | Interview prep |

Total: ~5.5 hours across 9 chapters.
