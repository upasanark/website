# Graph Neural Networks for Recommendation Systems — Course Outline

A structured set of slides covering GNN fundamentals through advanced recommendation methods, grounded in the graph-based work done for P13N personalized campaigns and category prediction.

---

## Chapter 1: Foundations of Graph Theory for ML

**File:** `01_graph_foundations.html`

### Slide Breakdown:

#### Slide 1: What is a Graph?
- Visual: Simple graph with 5 nodes, labeled edges
- Definition: G = (V, E) where V = set of nodes, E = set of edges
- Notation: |V| = n (nodes), |E| = m (edges)
- Types preview: directed, undirected, weighted

#### Slide 2: Types of Graphs
- Visual: Side-by-side examples of each type
- **Undirected**: friendships (symmetric)
- **Directed**: follows, purchases (asymmetric)
- **Weighted**: interaction strength on edges
- **Bipartite**: two disjoint node sets (users ↔ items)
- **Heterogeneous**: multiple node/edge types

#### Slide 3: The Adjacency Matrix
- Math: A ∈ {0,1}^(n×n), where A_ij = 1 if edge (i,j) exists
- Visual: Small graph + corresponding matrix side by side
- Properties: symmetric for undirected, sparse for real graphs
- Storage: dense vs sparse (CSR format)
- Intuition: "The adjacency matrix IS the graph in linear algebra form"

#### Slide 4: Graph Properties
- **Degree**: d_i = Σ_j A_ij (number of connections)
- **Degree matrix**: D = diag(d_1, d_2, ..., d_n)
- **Clustering coefficient**: fraction of neighbor pairs that are connected
- **Connected components**: reachability partitions
- Visual: histogram of degree distribution (power-law in real graphs)
- Story: "In recommendation graphs, degree = popularity. A phone with d=50000 is a bestseller."

#### Slide 5: The User-Item Bipartite Graph
- Visual: Two-column layout — users on left, items on right, edges = interactions
- Math: A ∈ R^(|U| × |I|), where A_ui = interaction weight
- Example: 28M users × 3.9M items, 423M edges (from P13N data)
- Edge types: purchase (weight 5), ATC (3), search_click (2), view (1)
- Intuition: "Every recommendation system implicitly operates on this graph"

#### Slide 6: The Graph Laplacian
- **Unnormalized Laplacian**: L = D - A
- **Symmetric normalized**: L_sym = I - D^(-½) A D^(-½)
- **Random walk normalized**: L_rw = I - D^(-1) A
- Properties: L is positive semi-definite, smallest eigenvalue = 0
- Visual: Eigenvalue spectrum of a sample graph
- Intuition: "The Laplacian measures how different a node is from its neighbors — it's a discrete derivative on graphs"

#### Slide 7: Why Graphs for Recommendations?
- Collaborative filtering = exploiting graph structure
- "Users who bought X also bought Y" = 2-hop path in user-item graph
- Higher-order connectivity captures deeper preferences
- Visual: 2-hop path example showing implicit user-user similarity
- Comparison: tabular features miss relational structure entirely

#### Interactive Element: Graph Explorer
- Clickable nodes that highlight neighbors and show degree
- Toggle between adjacency matrix view and visual graph
- Slider to add/remove edges and see Laplacian change

#### Quiz (5 questions):
1. Given adjacency matrix A, what is the degree of node 3? (read from matrix)
2. For a bipartite graph with |U|=100, |I|=50, what is the max possible edges?
3. If L = D - A, and node i has degree 4, what is L_ii?
4. True/False: In a bipartite graph, nodes of the same type can be directly connected.
5. A user purchased items A, B, C. Items A and B were also purchased by user_2. How many 2-hop paths connect user_1 to user_2?

---

## Chapter 2: Graph Neural Networks and Graph Convolution

**File:** `02_gnn_and_gcn.html`

### Slide Breakdown:

#### Slide 1: The Core Idea — Learning on Graph Structure
- Problem: we have a graph + node features, want to learn node representations
- Visual: graph with initial random colors → after GNN → meaningful clusters
- Key insight: "A node's representation should depend on its neighborhood"
- Story: "If your friends all like jazz, you probably like jazz too — that's GNN in one sentence"

#### Slide 2: Message Passing Framework
- Three phases: **Message** → **Aggregate** → **Update**
- Math:
  - Message: m_ij^(l) = MSG(h_i^(l), h_j^(l), e_ij)
  - Aggregate: M_i^(l) = AGG({m_ij^(l) : j ∈ N(i)})
  - Update: h_i^(l+1) = UPDATE(h_i^(l), M_i^(l))
- Visual: animated arrows showing messages flowing along edges
- Intuition: "Each node collects information from neighbors, then updates itself"

#### Slide 3: Aggregation Functions
- **Mean**: M_i = (1/|N(i)|) Σ_{j∈N(i)} h_j — averages neighbor features
- **Sum**: M_i = Σ_{j∈N(i)} h_j — preserves degree information
- **Max**: M_i = max_{j∈N(i)} h_j — captures most salient neighbor
- Visual: same graph, different aggregations → different resulting embeddings
- Trade-off: mean loses degree info, sum is degree-sensitive, max is robust to noise

#### Slide 4: Multi-Hop Propagation
- Layer 1: each node sees 1-hop neighbors
- Layer 2: each node sees 2-hop neighbors (neighbors of neighbors)
- Layer K: each node has receptive field of K-hop neighborhood
- Math: h_i^(K) encodes information from all nodes within K hops
- Visual: expanding circles showing receptive field growth
- Example: "After 2 layers, a user 'sees' items purchased by similar users"

#### Slide 5: Spectral Graph Convolutions — The Theory
- Goal: define convolution on graphs (no regular grid structure)
- Key tool: Graph Fourier Transform using Laplacian eigenvectors
- Eigendecomposition: L = U Λ U^T
- Spectral convolution: g_θ ⋆ x = U g_θ(Λ) U^T x
- Problem: computing eigenvectors is O(n³) — not scalable
- Story: "Convolution on images = local pattern detection. On graphs, we need the same idea but the 'locality' is defined by graph structure, not pixels"

#### Slide 6: Chebyshev Approximation
- Approximate g_θ(Λ) with Chebyshev polynomials up to order K:
  g_θ(Λ) ≈ Σ_{k=0}^{K} θ_k T_k(Λ̃) where Λ̃ = 2Λ/λ_max - I
- Key property: T_k(L) only requires K-hop neighborhoods (localized!)
- No eigenvector computation needed — just multiply by L repeatedly
- ChebNet: K-localized spectral filter that's computationally tractable
- Math walkthrough: T_0(x)=1, T_1(x)=x, T_k(x) = 2x·T_{k-1}(x) - T_{k-2}(x)

#### Slide 7: GCN — The First-Order Approximation (Kipf & Welling, 2017)
- Simplification: set K=1, λ_max≈2 in Chebyshev
- Result: g_θ ⋆ x ≈ θ(I + D^(-½) A D^(-½)) x
- Renormalization trick: Ã = A + I (self-loops), D̃ = D + I
- **The GCN layer**: H^(l+1) = σ(D̃^(-½) Ã D̃^(-½) H^(l) W^(l))
- Intuition: "Average your neighbors' features (including yourself), then transform"
- Visual: step-by-step computation for a 4-node graph
- Story: "GCN took the elegant but expensive spectral theory and boiled it down to one simple, fast operation"

#### Slide 8: GCN — Worked Example
- 4-node graph with features: x_1=[1,0], x_2=[0,1], x_3=[1,1], x_4=[0,0]
- Step 1: Add self-loops (Ã = A + I)
- Step 2: Compute D̃^(-½) Ã D̃^(-½) (show the normalized matrix)
- Step 3: Multiply by features → aggregated features
- Step 4: Apply weight matrix W and activation σ
- Show final embeddings — nodes with similar neighborhoods get similar representations

#### Slide 9: GraphSAGE — Inductive Learning (Hamilton et al., 2017)
- Problem with GCN: transductive (needs full graph at training)
- Solution: sample a fixed-size neighborhood, then aggregate
- Algorithm: SAMPLE(N(v), k) → AGGREGATE → CONCAT with self → TRANSFORM
- Aggregators: mean, LSTM, pooling
- Key advantage: can generalize to unseen nodes (new users/items)
- Visual: sampling tree showing how a node samples 2-hop neighborhood
- Math: h_v^(l) = σ(W · CONCAT(h_v^(l-1), AGG({h_u^(l-1) : u ∈ S(N(v))})))

#### Slide 10: GAT — Graph Attention Networks (Veličković et al., 2018)
- Motivation: not all neighbors are equally important
- Attention coefficient: α_ij = softmax_j(LeakyReLU(a^T [Wh_i ∥ Wh_j]))
- Aggregation: h_i' = σ(Σ_{j∈N(i)} α_ij · W h_j)
- Multi-head attention: K heads, concatenate or average
- Visual: edge thickness proportional to attention weight
- Comparison: GCN uses fixed weights (from degree), GAT learns them
- Story: "If GCN treats all neighbors equally, GAT asks 'which neighbors matter most?'"

#### Slide 11: The Over-Smoothing Problem
- Observation: as layers increase, all node embeddings converge
- Math: after K layers with mean aggregation, h_i^(K) → global mean as K→∞
- Visual: node embedding similarity vs number of layers (shows convergence)
- Why it happens: repeated averaging = low-pass filter that kills signal
- Practical consequence: GNNs rarely go beyond 2-3 layers
- Mitigations: residual connections, jumping knowledge, DropEdge

#### Slide 12: Residual Connections and JK-Net
- **Residual**: h_i^(l+1) = h_i^(l) + GNN_layer(h_i^(l), ...)
- **Jumping Knowledge**: final embedding = f(h_i^(0), h_i^(1), ..., h_i^(K))
  - f can be concatenation, max-pooling, or attention over layers
- Visual: architecture diagrams showing skip connections
- Intuition: "Keep the original signal while adding neighborhood information"

#### Interactive Element: GNN Playground
- Step through message passing on a small graph (5-7 nodes)
- Toggle between GCN/GraphSAGE/GAT aggregation
- Slider for number of layers — watch over-smoothing happen in real-time
- Color-coded node embeddings updating each layer

#### Quiz (6 questions):
1. In the message passing framework, what are the three phases?
2. Given a GCN layer H' = D̃^(-½)ÃD̃^(-½)HW, what does adding self-loops (A+I) accomplish?
3. If a node has 3 neighbors with embeddings [1,0], [0,1], [1,1], what is the mean aggregation result?
4. Why does GraphSAGE sample neighborhoods rather than using all neighbors?
5. In GAT, attention coefficients α_ij sum to 1 over j∈N(i). What function ensures this?
6. True/False: A 10-layer GCN will always outperform a 2-layer GCN because it captures higher-order structure.

---

## Chapter 3: Graph Methods for Recommendations

**File:** `03_graphs_for_recommendations.html`

### Slide Breakdown:

#### Slide 1: From MF to Graph — A Unified View
- Matrix Factorization: R ≈ U · V^T (predict ratings from latent factors)
- Graph view: MF = 1-layer GNN with no nonlinearity on bipartite graph
- Math: u_i = Σ_{j∈N(i)} (1/√d_i · 1/√d_j) · v_j — this IS one GCN layer
- Visual: MF diagram morphing into bipartite graph + propagation
- Story: "MF was graph learning before we knew it — it just couldn't go deeper than 1 hop"

#### Slide 2: LightGCN — Less is More (He et al., 2020)
- Key finding: for CF, nonlinearities and transformations HURT performance
- LightGCN removes: feature transformation W, nonlinearity σ, self-connection
- Only operation: neighborhood aggregation with symmetric normalization
- Layer rule: e_i^(l+1) = Σ_{j∈N(i)} (1/√|N(i)| · 1/√|N(j)|) · e_j^(l)
- Final embedding: e_i = Σ_{l=0}^{L} α_l · e_i^(l) (weighted layer combination)
- Visual: architecture comparison — GCN vs NGCF vs LightGCN (progressively simpler)
- Intuition: "For collaborative filtering, the graph structure IS the signal. Learned transforms just add noise."

#### Slide 3: LightGCN — Why Simplification Works
- Theoretical argument: user/item IDs have no meaningful features — only structure matters
- Feature transformation on random initial embeddings = just a different random init
- Nonlinearities between layers = unnecessary complexity for linear signal flow
- Self-connections = redundant when doing layer combination
- Math: final embedding with α_l = 1/(L+1) is equivalent to (L+1)-hop random walk
- Empirical results: LightGCN > NGCF > GCN on all CF benchmarks

#### Slide 4: NGCF — Neural Graph Collaborative Filtering
- Architecture: embedding propagation + feature transformation + nonlinearity
- Layer: e_i^(l+1) = σ(W_1 e_i^(l) + Σ_{j∈N(i)} (1/√|N(i)||N(j)|)(W_1 e_j^(l) + W_2(e_j^(l) ⊙ e_i^(l))))
- The interaction term e_j ⊙ e_i: captures pairwise feature interactions
- Why it's worse than LightGCN: over-parameterization for CF task
- Historical importance: first to show multi-layer GNN benefits for CF

#### Slide 5: XGCN — Frozen Propagation (Separate Structure from Learning)
- Core idea: propagate ONCE, then iteratively refine embeddings
- Step 1: Propagate initial embeddings through graph (frozen, no gradient)
  - P = (D^(-½) A D^(-½))^K · E_init
- Step 2: Train an MLP to refine propagated embeddings for the task
  - E_refined = MLP(P; θ)
- Step 3: Repeat propagation with refined embeddings → iterate
- Why this works: separates collaborative structure from task optimization
- Computational benefit: no backprop through sparse graph operations
- Visual: two-phase diagram — propagation (fixed) vs refinement (learned)
- Story: "Let the graph tell you WHO is similar. Then learn WHAT to do with that similarity."

#### Slide 6: Loss Functions for Graph-Based CF
- **BPR Loss** (Bayesian Personalized Ranking):
  L_BPR = -Σ_{(u,i,j)} ln σ(ŷ_ui - ŷ_uj)
  where i = positive item, j = sampled negative
  Intuition: "positive items should score higher than negatives"
- **InfoNCE Loss** (Contrastive):
  L_InfoNCE = -log(exp(sim(u,i)/τ) / Σ_{k} exp(sim(u,k)/τ))
  Intuition: "pull positive pairs close, push all negatives away"
- **Comparison**: BPR = pairwise, InfoNCE = listwise (uses more negatives)
- Temperature τ: low τ = sharp distinctions, high τ = softer
- Visual: embedding space showing positive pull + negative push

#### Slide 7: Negative Sampling Strategies
- **Random negatives**: sample uniformly from all items
  - Simple but may sample "easy" negatives (user never heard of item)
- **Popularity-biased**: sample proportional to item popularity
  - Harder negatives (popular items user chose NOT to interact with)
- **Hard negatives**: items close in embedding space but not interacted
  - Hardest signal but risk of false negatives (user might actually like it)
- **In-batch negatives**: other users' positives as negatives
  - Efficient: no extra sampling needed, natural difficulty distribution
- Math: expected gradient with different sampling distributions
- Story: "Easy negatives teach nothing. Hard negatives risk teaching wrong. The sweet spot is in between."

#### Slide 8: Edge Weighting for Interaction Graphs
- Not all interactions are equal: purchase ≫ cart ≫ click ≫ view
- Weighting scheme: w_ij = α_type × count × recency_decay
  - α_purchase = 5.0, α_atc = 3.0, α_click = 2.0, α_view = 1.0
- Recency decay: w × exp(-Δt/τ) — recent interactions matter more
- Impact on propagation: heavy edges propagate more signal
- Alternative: learn edge weights with a lightweight MLP
- Visual: bipartite graph with edge thickness = weight

#### Slide 9: Training at Scale — Mini-batch on Graphs
- Challenge: full-batch GNN requires entire graph in memory
- **Node-wise sampling** (GraphSAGE): sample K neighbors per hop
- **Layer-wise sampling** (FastGCN): sample nodes per layer
- **Subgraph sampling** (ClusterGCN): partition graph, train on subgraphs
- Trade-offs: variance (sampling) vs memory (full graph) vs bias (subgraph)
- Practical: for 28M users, mini-batch with neighbor sampling is the only option
- Visual: full graph → sampled computation graph for one target node

#### Interactive Element: Recommendation Score Calculator
- Input: user node, item node in a small bipartite graph
- Show LightGCN propagation step by step (3 layers)
- Display final user/item embeddings and inner product score
- Toggle edge weights (uniform vs interaction-type weighted) to see impact

#### Quiz (5 questions):
1. Why does LightGCN remove feature transformation matrices W compared to standard GCN?
2. In BPR loss, what does σ(ŷ_ui - ŷ_uj) represent geometrically?
3. XGCN freezes graph propagation. What is the computational advantage of this vs end-to-end training?
4. If τ→0 in InfoNCE loss, what happens to the gradient signal?
5. A user has edges to items A (purchase), B (view), C (view). With α_purchase=5, α_view=1, what fraction of the aggregated message comes from item A?

---

## Chapter 4: Multi-Interest and Disentangled Graph Methods

**File:** `04_multi_interest_graphs.html`

### Slide Breakdown:

#### Slide 1: The Collapse Problem
- Visual: user who buys electronics + books + sports → single embedding in the middle
- That embedding is close to NONE of those interest clusters
- Math: e_user = mean(e_electronics, e_books, e_sports) — the "average user" fallacy
- Real example: a user who bought iPhone + yoga mat + cookbook
  - Single embedding: somewhere in generic space
  - Multi-interest: 3 separate embeddings, each meaningful
- Story: "Averaging contradicts. You don't become a jazz-rock-classical fan by averaging — you're three fans in one person."

#### Slide 2: DGCF — Disentangled Graph Collaborative Filtering (Wang et al., 2020)
- Idea: split embedding into K independent intent channels
- Each channel captures one "interest dimension"
- Math: e_u = [e_u^1; e_u^2; ...; e_u^K] (K channels concatenated)
- Routing: for each edge (u,i), learn which channel it belongs to
  - p(k|u,i) = softmax(e_u^k · e_i^k) — iterative routing
- Channel-specific propagation: propagate within each channel independently
- Layer: e_u^{k,(l+1)} = Σ_{i∈N(u)} p(k|u,i) · (1/√d_u·d_i) · e_i^{k,(l)}
- Problem: channels are latent — no guarantee of interpretability
- Problem: iterative routing inside propagation = expensive

#### Slide 3: PIMIRec — Prototype-Based Multi-Interest
- Idea: create K prototype embeddings per user from interaction history
- Step 1: cluster user's interacted items into K groups
- Step 2: prototype_k = attention-weighted mean of items in cluster k
- Step 3: score candidate item against ALL user prototypes, take max
- Math: score(u, i) = max_k (prototype_u^k · e_i)
- Strength: explicit, interpretable interests
- Weakness: NO collaborative propagation — prototypes only from user's own history
- Visual: user with 3 prototypes (circles) in embedding space, item scored against nearest
- Story: "PIMIRec knows WHAT you like but not what PEOPLE LIKE YOU like"

#### Slide 4: MisGCF — Multi-Intent Separation Post-Propagation
- Idea: propagate first (collaborative signal), THEN separate intents
- Architecture:
  - Layer 1: standard graph propagation → single embedding per user
  - Intent extraction: multi-head FFN splits embedding into K intent vectors
  - Layer 2: propagate intent-separated embeddings again
  - Repeat: propagation → separation → propagation
- Math: [z_u^1, ..., z_u^K] = MultiHeadFFN(e_u^propagated)
- Strength: gets collaborative signal + multi-interest
- Weakness: with pre-trained embeddings, FFN distorts the original latent space
- Visual: pipeline diagram showing alternating propagation and separation

#### Slide 5: Comparison Table — Design Choices
| Aspect | DGCF | PIMIRec | MisGCF | Ideal? |
|--------|------|---------|--------|--------|
| Collaborative signal | ✓ (propagation) | ✗ (user-only) | ✓ (propagation) | ✓ |
| Multi-interest | ✓ (channels) | ✓ (prototypes) | ✓ (heads) | ✓ |
| Interpretable | ✗ (latent) | ✓ (item clusters) | Partial | ✓ |
| Pre-trained emb friendly | Partial | ✓ | ✗ (distorts) | ✓ |
| Scalable | ✗ (routing) | ✓ | ✓ | ✓ |
- Gap analysis: no existing method achieves all five
- This motivates explicit clustering + propagation (Chapter 8)

#### Slide 6: Personalized PageRank for Interest-Aware Propagation
- Standard PageRank: π = α · A_norm · π + (1-α) · e_uniform
- Personalized: restart from specific node → measures node importance relative to source
- PPR as propagation: multi-hop with teleportation (α = restart probability)
- Math: PPR(u) = α · D^(-1)A · PPR(u) + (1-α) · e_u
- Connection to GNN: APPNP uses PPR as propagation scheme
  - H^(K) = α(D^(-½)AD^(-½))H^(K-1) + (1-α)H^(0)
- Why for recommendations: naturally handles multi-hop while preventing over-smoothing
- Visual: PPR values spreading from a source user, decaying with distance

#### Slide 7: The Explicit Clustering Approach
- Core insight: ground interests in semantic clusters of items, not latent channels
- Step 1: GMM clustering of item embeddings → soft cluster assignments
  - Each item has p(cluster_k | item) for all K clusters
- Step 2: tag each edge with cluster identity (primary + secondary)
- Step 3: propagate PER CLUSTER — user gets separate embedding per interest
- Advantages:
  - Interpretable from the start (cluster = product category/type)
  - Compatible with pre-trained embeddings (no distortion)
  - Collaborative (propagation across users)
  - Variable number of interests per user (depends on interaction diversity)
- Visual: items colored by cluster, user accumulates cluster-specific embeddings
- Story: "Don't discover interests from scratch — start with known item structure and let the graph refine it"

#### Slide 8: Normalization and Over-Smoothing in Multi-Interest
- Problem: after multiple propagation layers, cluster embeddings merge
- Solution 1: normalize direction after each layer (preserve interest identity)
  - μ̂_u,c = μ_u,c / ‖μ_u,c‖ — unit direction
  - mass = original magnitude (importance)
- Solution 2: re-anchor with original embeddings at intermediate layers
  - μ_i,c^(2) = β · μ_i,c^(propagated) + (1-β) · e_i^(pretrained) for primary cluster
- Why this works: direction = what (interest identity), magnitude = how much (importance)
- Math: decoupling direction and magnitude prevents dominant clusters from absorbing others
- Visual: without normalization (embeddings collapse) vs with (stay separated)

#### Interactive Element: Multi-Interest Visualizer
- 2D embedding space with items colored by cluster
- Click a user → show their per-cluster embeddings (separate dots)
- Toggle between single-embedding (average) and multi-interest view
- Hover over items to see attention/similarity scores from each interest

#### Quiz (5 questions):
1. In DGCF, how does routing probability p(k|u,i) determine which channel an edge belongs to?
2. PIMIRec has no collaborative signal. Give a concrete example of what it misses.
3. Why does MisGCF struggle with pre-trained ASIN embeddings but not random init?
4. In PPR with restart α=0.15, what fraction of "attention" stays within 1-hop?
5. After normalization, user u has cluster embeddings with directions [0.7, 0.3] (electronics) and [0.2, 0.8] (books), and masses 0.8 and 0.2. Which interest dominates scoring?

---

## Chapter 5: Graph Transformers

**File:** `05_graph_transformers.html`

### Slide Breakdown:

#### Slide 1: Limitations of Message Passing
- **Fixed receptive field**: K-layer GNN only sees K-hop neighbors
- **Over-squashing**: information from distant nodes compressed through bottleneck
  - Math: ∂h_v^(K)/∂x_u decays exponentially with distance(u,v)
- **Structural blindness**: GNNs can't distinguish certain graph structures (WL test limit)
- Visual: two non-isomorphic graphs that produce identical GNN embeddings
- Motivation: "What if a node could attend to ANY other node, regardless of graph distance?"

#### Slide 2: Self-Attention on Graphs
- Standard transformer attention: Attention(Q, K, V) = softmax(QK^T/√d)V
- On graphs: Q = all nodes, K = all nodes, V = all node features
- Naive approach: full O(n²) attention over all graph nodes
- Result: every node can "see" every other node — global receptive field
- Problem: ignores graph structure entirely (treats graph as complete graph)
- Solution: incorporate graph structure through positional encodings + bias terms

#### Slide 3: Positional Encodings for Graphs
- Problem: graphs have no inherent order (unlike sequences)
- **Laplacian Eigenvector PE**: use eigenvectors of L as positional features
  - PE(v) = [φ_1(v), φ_2(v), ..., φ_k(v)] (first k eigenvectors evaluated at node v)
  - Intuition: eigenvectors capture structural roles (center vs periphery, bridge vs cluster)
  - Problem: sign ambiguity (φ and -φ are both valid)
- **Random Walk PE**: p(v→v in k steps) for k = 1,2,...,K
  - RWPE(v) = [RW¹_vv, RW²_vv, ..., RW^K_vv]
  - Captures local structural properties (triangles, cycles)
  - No sign ambiguity, computationally cheap
- **Learned PE**: train a small GNN to generate positional features
- Visual: eigenvector values visualized as node colors on a graph

#### Slide 4: Graphormer (Ying et al., 2021)
- Three structural encodings added to transformer:
  1. **Centrality encoding**: degree as bias term (b_deg(v_i) added to input)
  2. **Spatial encoding**: shortest path distance as attention bias
     - A_ij += b_SPD(d(v_i, v_j)) — closer nodes get attention boost
  3. **Edge encoding**: aggregate edge features along shortest path
     - E_ij = mean of edge features on shortest path from i to j
- Architecture: standard transformer + these three structural biases
- Math: Attention_ij = (h_i W_Q)(h_j W_K)^T / √d + b_spatial(i,j) + b_edge(i,j)
- Results: SOTA on molecular property prediction (OGB benchmarks)
- Limitation: computing all-pairs shortest paths = O(n²) or O(n·m)
- Story: "Graphormer doesn't replace graph structure — it encodes it INTO the attention mechanism"

#### Slide 5: Hybrid Approaches — Local + Global
- Strategy: use GNN for local aggregation, Transformer for global context
- Architecture patterns:
  1. **GNN then Transformer**: GNN embeddings → transformer layers
  2. **Parallel**: GNN branch + Transformer branch → combine
  3. **Interleaved**: alternate GNN and Transformer layers
- Example (GPS framework):
  - Each layer: h' = GNN_local(h) + Transformer_global(h)
  - Local captures neighborhood, global captures long-range
- Visual: architecture diagrams for each pattern
- Trade-off: more parameters and compute, but better expressiveness

#### Slide 6: Efficient Graph Transformers
- Full attention: O(n²) — impractical for million-node graphs
- **Sparse attention**: only attend to graph neighbors + sampled distant nodes
- **Performer/Linear attention**: approximate softmax with random features → O(n)
- **Cluster-based**: cluster nodes, attend within clusters + across cluster representatives
- **Sampling**: attend to random subset of nodes each layer (like dropout for attention)
- Practical numbers: 28M user graph → full attention impossible, need O(n·k) where k≪n
- Visual: attention matrix — full (dense) vs sparse (block-diagonal + bridges)

#### Slide 7: Graph Transformers for Recommendations
- Application: capturing long-range user-item dependencies
- Example: user A → item X → user B → item Y (4-hop collaborative path)
  - GNN needs 4 layers (over-smoothing risk)
  - Transformer can attend directly from user A to item Y
- Sequential recommendation: SASRec/BERT4Rec as "sequence transformers on interaction graph"
- Cross-attention between user and item sets
- Current state: graph transformers for RecSys are emerging but not yet dominant
  - Compute cost vs marginal gain over simpler GNN methods

#### Interactive Element: Attention Pattern Visualizer
- Small graph (10 nodes) with transformer attention
- Heatmap showing attention weights between all node pairs
- Toggle spatial encoding on/off to see how graph structure biases attention
- Highlight: without PE, attention is purely content-based (ignores graph)

#### Quiz (5 questions):
1. What is over-squashing and why does it limit deep GNNs?
2. Laplacian eigenvector PEs have sign ambiguity. Why is this a problem for learning?
3. In Graphormer, spatial encoding adds b_SPD(d(i,j)) to attention. If nodes i,j are 5 hops apart and threshold is 3, what happens?
4. Why might a hybrid GNN+Transformer outperform a pure Graph Transformer on recommendation graphs?
5. For a graph with 10M nodes, full self-attention requires O(10^14) operations. Name two strategies to make this tractable.

---

## Chapter 6: Scaling Graph Methods — MapReduce and Distributed Computing

**File:** `06_scaling_graphs.html`

### Slide Breakdown:

#### Slide 1: The Scale Challenge
- Real numbers from production:
  - Users: 28M (CLP) → 150M (Bestsellers)
  - Items: 3.9M ASINs
  - Edges: 423M interactions
  - Embedding dimensions: 18-256
- Memory for full adjacency matrix: 28M × 3.9M × 4 bytes = 437 TB (impossible)
- Sparse representation: 423M edges × 12 bytes = 5 GB (feasible)
- But: propagation requires matrix multiplication → still distributed
- Story: "The graph fits in memory. The computation doesn't."

#### Slide 2: Graph Partitioning Strategies
- **Edge-cut partitioning**: divide nodes across machines, cut crossing edges
  - Minimize edge cuts → minimize communication
  - Problem: high-degree nodes create hotspots
- **Vertex-cut partitioning**: divide edges across machines, replicate boundary nodes
  - Better for power-law graphs (common in recommendations)
  - Each edge lives on exactly one machine
- **Hash partitioning**: hash(node_id) % num_partitions
  - Simple, balanced partition sizes
  - No locality guarantee → more communication
- Visual: same graph partitioned 3 ways, showing trade-offs
- Recommendation: vertex-cut for power-law interaction graphs

#### Slide 3: MapReduce for Graph Propagation
- **Core insight**: message passing = Map + Reduce
- Map phase: for each edge (u,i), emit (i, message_from_u)
  - message = w_ui × e_u (weighted embedding)
- Reduce phase: for each node i, aggregate all incoming messages
  - e_i^new = normalize(Σ messages_to_i)
- One MapReduce job = one propagation layer
- Multi-hop: chain K MapReduce jobs
- Visual: data flow diagram — edges emit messages, nodes collect and aggregate
- Math: exactly implements e_i^(l+1) = Σ_{j∈N(i)} (w_ji/d_i) · e_j^(l)

#### Slide 4: Edge-Wise Partitioning for Scalable Propagation
- Strategy: partition edges (not nodes) across workers
- Each worker computes partial aggregations for its edge subset
- Final aggregation: merge partial results per destination node
- Why this works: edges are independent during the "message" phase
- Math:
  - Partition edges into P sets: E = E_1 ∪ E_2 ∪ ... ∪ E_P
  - Worker p computes: partial_i^p = Σ_{j: (j,i)∈E_p} w_ji · e_j
  - Final: e_i^new = normalize(Σ_p partial_i^p)
- Advantage: linear scaling with number of edges
- Used in P13N: 423M edges split across 30 EMR executors
- Visual: edges colored by partition, partial sums flowing to merge step

#### Slide 5: Spark/PySpark Implementation Pattern
```python
# Pseudocode for one propagation layer in PySpark
edges_df = spark.read.parquet("edges/")  # (src, dst, weight)
embeddings_df = spark.read.parquet("embeddings/")  # (node_id, embedding)

# Map: join edges with source embeddings, compute messages
messages = edges_df.join(embeddings_df, edges_df.src == embeddings_df.node_id)
    .select("dst", (col("weight") * col("embedding")).alias("message"))

# Reduce: aggregate messages at destination
new_embeddings = messages.groupBy("dst")
    .agg(array_sum("message").alias("aggregated"))
    .withColumn("embedding", normalize_udf("aggregated"))
```
- Key optimizations:
  - Broadcast small tables (item embeddings if |I| < 10M)
  - Repartition edges by destination for efficient reduce
  - Cache intermediate DataFrames across propagation layers
- Visual: Spark DAG showing the computation flow

#### Slide 6: Distributed FAISS for Similarity Search
- After propagation: 150M user embeddings × 256 dims = 153 GB
- Need: find top-K similar items for each user (nearest neighbor search)
- FAISS (Facebook AI Similarity Search):
  - IndexFlatIP: exact inner product search
  - IndexIVF: approximate search with inverted file index
- Distribution strategy:
  - Broadcast item index to all workers (fits in memory if |I| ≈ 4K items)
  - Partition users across workers
  - Each worker queries local user embeddings against broadcast index
- Math: score(u, i) = e_u · e_i / (‖e_u‖ · ‖e_i‖) (cosine similarity)
- Performance: 2,500-3,000 users/second, total ~60 min for 150M users
- Visual: architecture diagram — users partitioned, index broadcast, results collected

#### Slide 7: Memory Management and Broadcast Variables
- Challenge: each Spark executor needs access to shared data
- **Broadcast variables**: send once, cached on each executor
  - FAISS index (~50 MB for 4K items × 256 dims)
  - Item metadata (~100 MB)
  - Cluster assignments (~10 MB)
- **Partition-level processing**: rebuild index once per partition, process all rows
- Anti-pattern: shipping large data per-row (serialization overhead kills performance)
- Memory budget per executor: 6-8 GB
  - ~4 GB for data partitions
  - ~1 GB for broadcast variables
  - ~1-2 GB for computation overhead
- Visual: memory layout diagram for one executor

#### Slide 8: EMR Cluster Configuration for Graph Workloads
- Cluster sizing for 150M users, 423M edges:
  - Driver: r5.2xlarge (64 GB RAM) — coordinates, collects samples
  - Executors: 30× r5.xlarge (32 GB RAM each)
  - Total cluster memory: ~1 TB
- Key Spark configs:
  - spark.executor.memory = 6g
  - spark.sql.shuffle.partitions = 500
  - spark.default.parallelism = 240 (8 cores × 30 executors)
- Runtime breakdown:
  - Data loading: ~5 min
  - Graph propagation (3 layers): ~30 min
  - FAISS retrieval: ~20 min
  - Output writing: ~5 min
- Cost optimization: spot instances for executors (3x cheaper, tolerate interruption)

#### Slide 9: Practical Challenges — Data Skew and Shuffle
- **Data skew**: popular items have millions of edges → one reducer gets all the work
  - Solution: salting — add random suffix to key, aggregate in two stages
  - Stage 1: partial_agg by (item_id + salt) — distributed
  - Stage 2: final_agg by item_id — balanced after stage 1
- **Shuffle optimization**: minimize data movement between stages
  - Repartition by destination node BEFORE join (co-located data)
  - Use broadcast join when one side is small (<10 GB)
- **Checkpointing**: save intermediate results every propagation layer
  - If job fails at layer 3, restart from layer 2 checkpoint (not from scratch)
- Visual: before/after skew — one partition with 10x data vs balanced

#### Interactive Element: Scaling Calculator
- Input: number of users, items, edges, embedding dimensions
- Output: estimated memory, runtime, cluster size, cost
- Sliders to adjust: propagation layers, FAISS k, executor count
- Show bottleneck (memory-bound vs compute-bound vs shuffle-bound)

#### Quiz (5 questions):
1. For a graph with 100M edges and 30 executors using edge-wise partitioning, how many edges per executor?
2. Why broadcast the FAISS index instead of the user embeddings?
3. In MapReduce graph propagation, the Map phase emits (destination, message). What does the Reduce phase compute?
4. A popular item has 5M incident edges while average items have 100. How does salting help?
5. You have 150M user embeddings × 256 dims in float32. How much storage is that?

---

## Chapter 7: Knowledge Graphs and Structured Representations

**File:** `07_knowledge_graphs.html`

### Slide Breakdown:

#### Slide 1: Beyond Interaction Graphs — Adding Semantics
- Interaction graph: only captures "who interacted with what"
- Missing: WHY they interacted, item properties, relationships between items
- Knowledge graph adds:
  - Item attributes (brand, category, price, specs)
  - Relations (belongs_to, manufactured_by, compatible_with)
  - Hierarchies (category → subcategory → product)
- Visual: plain bipartite graph vs enriched KG with attribute nodes
- Story: "An interaction graph knows iPhone was purchased. A KG knows iPhone is a smartphone, made by Apple, in the premium segment, compatible with Lightning cables."

#### Slide 2: Knowledge Graph Structure
- **Entities**: items, categories, brands, attributes (nodes)
- **Relations**: typed edges (belongs_to, has_brand, has_price_tier)
- **Triples**: (head, relation, tail) — e.g., (iPhone, has_brand, Apple)
- Formal: KG = {(h, r, t) | h,t ∈ E, r ∈ R}
- Scale example:
  - 3.9M item entities
  - 14 category entities
  - ~500 brand entities
  - Relations: belongs_to, has_brand, has_price, compatible_with
- Visual: subgraph showing iPhone and its knowledge connections

#### Slide 3: Knowledge Graph Embeddings — TransE, RotatE, ComplEx
- Goal: embed entities and relations in vector space such that h + r ≈ t
- **TransE** (Bordes et al., 2013):
  - Score: f(h,r,t) = -‖h + r - t‖ (translation in embedding space)
  - Intuition: relation r is a vector that translates head to tail
  - Limitation: can't model 1-to-N relations well
- **RotatE** (Sun et al., 2019):
  - Score: f(h,r,t) = -‖h ∘ r - t‖ where r ∈ C with |r|=1 (rotation)
  - Can model symmetry, antisymmetry, inversion, composition
- **ComplEx** (Trouillon et al., 2016):
  - Score: Re(⟨h, r, t̄⟩) — Hermitian dot product in complex space
  - Can model antisymmetric relations
- Visual: 2D embedding space showing translation (TransE) and rotation (RotatE)
- Training: contrastive — score true triples higher than corrupted ones

#### Slide 4: Incorporating KG into Recommendations
- **Method 1**: KG embeddings as item features → input to GNN
  - Pre-train KG embeddings (TransE/RotatE), use as initial item features
  - GNN propagation operates on enriched features
- **Method 2**: Joint graph — merge interaction graph + KG into one
  - Users + Items + Attributes all as nodes
  - Propagate through both interaction and attribute edges
- **Method 3**: KG-guided attention — use KG relations to weight edges
  - Attention(u, i) influenced by semantic similarity from KG
- Example: user who bought iPhone → KG says iPhone belongs to smartphones, has_brand Apple
  - Propagation reaches "smartphones" category and "Apple" brand
  - Other Apple products and smartphones get indirect signal
- Visual: unified graph with user-item edges (blue) + KG edges (orange)

#### Slide 5: Hierarchical Representations — Category Graphs
- Item taxonomy: Root → Department → Category → Subcategory → Item
- Browse node graph: tree structure with items as leaves
- Example (from P13N):
  - GL 107 (Wireless) → smartphones, tablets, smartwatches
  - GL 353 (Accessories) → cases, chargers, screen_protectors, earphones
- How hierarchy helps:
  - Category-level propagation: aggregate all items in a category
  - Cold-start: new items inherit category-level embeddings
  - Cross-category prediction: which category will user transition to?
- Math: P(category_k | source_item) from category transition graph
- Visual: tree structure with edge weights showing transition probabilities

#### Slide 6: KGTB — Knowledge Graph Token-Based Methods
- Idea: use KG structure to create hierarchical item tokens
- Token hierarchy encodes: category → subcategory → brand → specific item
- Connection to generative recommendation: tokens form a tree (like semantic IDs)
- GNN on token graph: propagate information between related tokens
- Cold-start advantage: new items get tokens from their KG attributes
  - (New phone) → (smartphones, Samsung, mid_range) → token assignment
- Visual: token tree with KG attributes determining placement
- Comparison: TIGER uses RQ-VAE (learned), KGTB uses KG (structured)

#### Slide 7: Cross-Category Signal Propagation
- Problem: recommend accessories for a phone purchase (cross-category)
- Solution: propagate through shared attribute nodes
  - iPhone → Apple (brand) → AirPods (same brand, different category)
  - iPhone → smartphones (category) → cases (compatible category)
- Edge types for cross-category:
  - compatible_with: phone → case, phone → charger
  - bought_together: co-purchase statistics as edges
  - same_brand: brand affinity signal
- Math: multi-relational propagation — different weight matrices per relation type
  - h_i^(l+1) = Σ_r Σ_{j∈N_r(i)} W_r · h_j^(l) / |N_r(i)|
- Application in P13N: cross-category graph propagation for accessory recommendations

#### Interactive Element: KG Explorer
- Clickable knowledge graph centered on a product
- Expand relations: click "brand" to see all products by same brand
- Show how propagation reaches items through different relation paths
- Compare: 1-hop (direct attributes) vs 2-hop (items sharing attributes)

#### Quiz (5 questions):
1. In TransE, the scoring function is -‖h + r - t‖. If Apple + has_product ≈ iPhone, what does Apple + has_product give geometrically?
2. Why can't TransE model symmetric relations (e.g., "similar_to")?
3. A new item has no interaction history but has attributes (brand=Samsung, category=phones, price=mid). How does a KG help with cold-start?
4. In multi-relational propagation, why use different W_r per relation type instead of a shared W?
5. Browse node hierarchy: GL107 → smartphones → Samsung → Galaxy S24. How many levels deep is this item in the hierarchy?

---

## Chapter 8: Applied Graph Methods — P13N Personalization System

**File:** `08_applied_graph_methods.html`

### Slide Breakdown:

#### Slide 1: System Overview — Three Graph Applications
- Visual: high-level system architecture showing 3 graph-based components
- **System A**: Multi-Interest Graph Propagation (CLP campaign targeting)
  - Purpose: personalize which campaigns to show which users
  - Scale: 28M users, 423M edges
- **System B**: Hierarchical Co-Interaction Graph (category prediction)
  - Purpose: predict next product category from purchase/exploration
  - Scale: 250K source ASINs, 5-10 categories
- **System C**: Graph Propagation Embeddings (bestseller recommendations)
  - Purpose: 150M-user personalized recommendations
  - Scale: 150M users, 4K bestseller items, 256-dim embeddings
- Story: "Same fundamental technique — graph propagation — applied at three different scales and for three different tasks"

#### Slide 2: System A — Multi-Interest Graph Propagation
- Problem: single embedding for a user who buys phones + cases + chargers = useless average
- Solution: explicit GMM clustering → per-cluster propagation
- Architecture:
  - Item embeddings: 18-dim (10 specs + 8 brand features)
  - GMM: K clusters with soft assignments p(c|item)
  - 3-layer propagation: I→U (L1), U→I (L2), I→U (L3)
- Key innovation: separate mean, mass, variance per cluster per node
  - Mean μ_u,c: direction of interest within cluster c
  - Mass w_u,c: importance of cluster c to user u
  - Variance σ²_u,c: diversity of interactions within cluster c
- Visual: user with 4 cluster-specific embedding vectors (different directions + sizes)

#### Slide 3: System A — Layer-by-Layer Propagation
- **Layer 1 (Item→User)**: group edges by cluster, aggregate per cluster
  - μ_u,c^(1) = Σ_{i∈cluster_c} w_i→u · e_i / Σ w_i→u
  - Result: user has direct preference embedding per cluster
- **Layer 2 (User→Item)**: items receive signal from users' OTHER clusters
  - If users who bought item A also have strong cluster Y interest...
  - ...item A's L2 embedding gains a cluster Y component (collaborative!)
  - Re-anchor: β·propagated + (1-β)·pretrained for primary cluster (prevent drift)
- **Layer 3 (Item→User)**: final aggregation — collaborative + multi-interest
  - User now has embeddings informed by collaborative patterns
- Visual: 3-panel showing embeddings at each layer, noting what's new at each step

#### Slide 4: System A — Results and Comparison
- Results table:
  | Method | AUC | Description |
  |--------|-----|-------------|
  | Clustered GP Weighted | 0.868 | Multi-interest propagation (ours) |
  | Clustered GP Unweighted | 0.841 | Without intent weight tracking |
  | Aggregated GP | 0.798 | Normal 2-layer (no clustering) |
  | Refined Multi-Intent | 0.721 | MisGCF-style post-hoc separation |
  | Matrix Factorization | 0.630 | MF baseline |
  | IEFM (1024-dim) | 0.529 | Pre-trained embedding similarity |
- Key insight: clustering during propagation >>> clustering after propagation
- Why IEFM fails: 1024-dim embeddings are rich but ignore collaborative structure
- Visual: bar chart of AUC values with confidence intervals

#### Slide 5: System B — Hierarchical Co-Interaction Graph
- Problem: predict which category a user will explore next
- Graph structure: source_asin → time_window → dest_category → dest_asin
- Two parallel graphs: "purchased" vs "explored" (fundamentally different patterns)
- No model needed: direct frequency counting + normalization
  - P(category | source_asin, time_window) = W(s,t,c) / Σ_c' W(s,t,c')
- Time windows capture temporal dynamics:
  - Short (0-7d): attach behavior (phone → case)
  - Medium (8-30d): delayed attach (phone → charger)
  - Long (31-365d): replenishment/discovery
- Cold-start: λ blending between item-level and category-level graph
  - λ(s) = sigmoid(evidence/500 - 3)
- Visual: transition probability heatmap (source items × categories × time windows)

#### Slide 6: System B — Graph Construction Details
- Edge strategy: first-per-destination-category (prevents spurious long-range edges)
- Edge weighting: α_purchase=5, α_atc=3, α_click=2, α_view=1
- Pruning pipeline:
  1. Edge-level: drop edges with < 3 unique customers or weight < 5.0
  2. Branch-level: drop (asin, time_window, category) with < 5 edges
  3. Source-item: flag ASINs with total weight < 50 as cold-start
- Expected: ~60-80% of raw edges pruned (noise), 30-50% ASINs flagged as cold-start
- Result: clean graph where surviving transitions are statistically reliable
- Visual: before/after pruning — noisy spaghetti graph → clean structured transitions

#### Slide 7: System C — Bestseller Graph Embeddings at Scale
- Architecture: GNN with 3-hop attention-weighted propagation
- Training: InfoNCE contrastive loss on user-item bipartite graph
- Embedding space: 256-dim, users and items in same space
- Retrieval: per-category FAISS indices, top-200 candidates per category
- Reranking: 35% similarity + 65% popularity + 20% cluster diversity
- Scale: 150M users, processed in 45 minutes on 30-executor EMR cluster
- Key design choices:
  - Per-category indices (ensures coverage across all product types)
  - Iterative diversity selection with cluster penalties
  - Piecewise popularity scoring (global rank, not per-category)
- Visual: pipeline diagram from graph → embeddings → FAISS → rerank → output

#### Slide 8: System C — Graph Features as XGBoost Input
- Instead of end-to-end neural: use graph as feature generator for XGBoost
- Top features by importance (from production model):
  1. graph_prob_for_dst (gain: 1.45M) — blended transition probability
  2. graph_evidence (gain: 135K) — total edge weight (reliability)
  3. graph_lambda (gain: 119K) — item vs category blending weight
  4. graph_top1_prob (gain: 58K) — confidence/peakiness
  5. graph_entropy — uncertainty in category distribution
  6. graph_num_branches — diversity of transitions
- Why XGBoost over neural: interpretable, fast inference, handles mixed features
- The graph provides the "what do items like this lead to?" signal
- XGBoost adjusts based on customer-specific context
- Visual: feature importance bar chart

#### Slide 9: Latest Applications of GNNs in Industry (2024-2026)

**9a: Fraud Detection and Risk (PayPal, Stripe, Ant Financial)**
- Graph: transaction network (users, merchants, devices as nodes; payments as edges)
- Method: heterogeneous GNN with temporal edge features
- Key insight: fraudsters form dense subgraphs with unusual temporal patterns
- GNN detects anomalous neighborhoods that rule-based systems miss
- Scale: billions of transactions, real-time inference (< 50ms)

**9b: Drug Discovery and Molecular Property Prediction (DeepMind, Relay)**
- Graph: molecule = atoms (nodes) + bonds (edges)
- Method: message passing neural networks (MPNN), Graphormer
- Tasks: predict toxicity, binding affinity, solubility from molecular graph
- AlphaFold uses graph attention for protein structure (residue interaction graph)
- GNNs outperform fingerprint-based methods by 10-30% on molecular benchmarks

**9c: Social Network and Content Recommendation (Pinterest, Twitter/X, TikTok)**
- PinSage (Pinterest): random-walk GNN on 3B nodes, 18B edges
  - Combines visual features + graph structure for pin recommendations
  - Key: importance-based neighbor sampling (not uniform)
- TikTok: user-video-creator heterogeneous graph for feed ranking
- Twitter/X: community detection + GNN for timeline ranking

**9d: Search and Ads (Google, Meta, Amazon)**
- Google: GNN for query-document relevance (knowledge graph enhanced)
- Meta: GNN for ad targeting (user-ad-advertiser graph)
- Amazon: product knowledge graph for search ranking
  - Item-item relationships improve recall for tail queries
- Visual: industry application matrix (domain × GNN technique)

**9e: Autonomous Driving and Robotics (Waymo, Tesla)**
- Graph: road scene = vehicles, pedestrians, lanes as nodes; proximity/interaction as edges
- Method: spatio-temporal graph networks for trajectory prediction
- Predict: where will other vehicles be in 3 seconds?
- GNN captures interactions between agents (yield, follow, merge patterns)

**9f: Supply Chain and Logistics (Amazon, JD.com)**
- Graph: warehouse-item-route network
- Tasks: demand forecasting (propagate demand signals through supply network)
- Inventory optimization: which warehouses should stock which items?
- GNN captures: if warehouse A runs out, demand shifts to warehouse B (graph effect)

#### Slide 10: Lessons Learned — What Worked, What Didn't
- **What worked:**
  - Explicit clustering >>> implicit disentanglement (interpretable + effective)
  - Normalization at each layer prevents over-smoothing across interests
  - Re-anchoring preserves pre-trained embedding quality
  - Graph as feature generator (for XGBoost) = best of both worlds
  - Edge-wise MapReduce scales linearly with graph size
- **What didn't work:**
  - MisGCF with pre-trained embeddings (FFN distorts latent space)
  - Deep propagation without normalization (interests collapse)
  - Per-category popularity scoring (rank inversions across categories)
  - All-pairs edge construction (spurious long-range attributions)
- **Design principles:**
  - Simple graph structures with strong pruning > complex architectures
  - Frequency counting + normalization can beat learned models (System B)
  - Collaborative signal is the primary value of graphs (not feature learning)
- Visual: ✓/✗ checklist of approaches tried

#### Slide 11: Connecting the Chapters — Full Picture
- Chapter 1-2 foundations → understanding WHY graph propagation works
- Chapter 3 (LightGCN, XGCN) → the base methods we built upon
- Chapter 4 (Multi-interest) → the gap that motivated our clustering approach
- Chapter 5 (Transformers) → future direction for capturing long-range patterns
- Chapter 6 (Scaling) → how we made it work at 150M user scale
- Chapter 7 (Knowledge Graphs) → how item taxonomy enriches the interaction graph
- Visual: dependency graph of chapters, showing how each builds to the applied system
- Closing thought: "Graph methods aren't a monolith — they're a toolkit. Pick the right tool for your scale, your data, and your latency budget."

#### Interactive Element: Architecture Explorer
- Clickable system diagram of all three systems
- Click a component → shows which chapter concepts it uses
- Example: click "GMM clustering" → highlights Ch 4 multi-interest
- Example: click "MapReduce propagation" → highlights Ch 6 scaling

#### Quiz (6 questions):
1. In the multi-interest system, why is L1 aggregation done per-cluster but L3 is not?
2. The co-interaction graph uses first-per-destination-category edges. Why not all-pairs?
3. λ = sigmoid(evidence/500 - 3). If a new item has evidence=100, what is λ approximately? What does this mean?
4. Why does the bestseller system use per-category FAISS indices instead of one global index?
5. Graph features have 10x higher importance than customer features in XGBoost. What does this suggest about the relative value of collaborative vs individual signal?
6. System A achieves AUC 0.87 vs IEFM's 0.53. IEFM has 1024 dims vs 18 dims for the graph method. Why does dimensionality not determine quality here?

---

## References

### Papers:
1. **LightGCN** — He et al., 2020 — Simplifying GCN for CF
2. **DGCF** — Wang et al., 2020 — Disentangled Graph CF
3. **XGCN** — Graph propagation with frozen structure
4. **KGTB** — Knowledge Graph Token-Based recommendation (Sep 2025)
5. **GCN** — Kipf & Welling, 2017 — Semi-supervised classification
6. **GraphSAGE** — Hamilton et al., 2017 — Inductive representation learning
7. **GAT** — Veličković et al., 2018 — Graph Attention Networks
8. **Graphormer** — Ying et al., 2021 — Graph Transformers
9. **PIMIRec** — Multi-interest with prototypes
10. **HiGR** — Tencent, Feb 2026 — Hierarchical generative retrieval
11. **TransE** — Bordes et al., 2013 — Translating embeddings for KG
12. **RotatE** — Sun et al., 2019 — Rotation-based KG embeddings

### Resources:
- Stanford CS224W: Machine Learning with Graphs
- Stanford CS224N: NLP with Deep Learning (attention/transformer lectures)

---

## Suggested Flow

| Session | Chapters | Duration | Focus |
|---------|----------|----------|-------|
| 1 | Ch 1-2 | ~50 min | Foundations + GNN/GCN (theory-heavy) |
| 2 | Ch 3 | ~40 min | Graphs for Recs (LightGCN, XGCN, losses) |
| 3 | Ch 4 | ~40 min | Multi-Interest (DGCF, PIMIRec, clustering) |
| 4 | Ch 5 | ~35 min | Graph Transformers |
| 5 | Ch 6 | ~40 min | Scaling with MapReduce |
| 6 | Ch 7 | ~35 min | Knowledge Graphs |
| 7 | Ch 8 | ~45 min | Applied Methods (your work + industry) |
| 8 | Ch 9 | ~40 min | Interview Prep & Cheatsheet |

Total: ~5.5 hours of content across 9 chapters, 8 sessions.

---

## Chapter 9: Interview Prep — Questions, Answers, and Cheatsheets

**File:** `09_interview_prep.html`

### Section A: Concept Questions with Answers (30 questions)

#### Fundamentals (Q1-Q8)

**Q1: Explain message passing in GNNs in one sentence. Why is it the unifying framework?**
- Answer: Each node iteratively collects and aggregates information from its neighbors, then updates its own representation. It unifies GCN, GraphSAGE, GAT because they all follow this pattern — they differ only in HOW they aggregate (mean vs attention vs sampled).

**Q2: What is the Graph Laplacian and why does it matter for GNNs?**
- Answer: L = D - A. It measures local smoothness — Lx gives the difference between a node's value and its neighbors' average. Its eigenvectors define the "frequency" basis for graphs (like Fourier for signals). GCN's spectral derivation starts from filtering in this basis. The key eigenvalue property: x^T L x = Σ_{(i,j)∈E} (x_i - x_j)² — measures total signal variation over edges.

**Q3: What is over-smoothing? Give the mathematical intuition.**
- Answer: After K layers of mean aggregation, all node embeddings converge to the same vector (the graph's stationary distribution). Mathematically, repeated multiplication by the normalized adjacency matrix A_norm is equivalent to a low-pass filter — it kills high-frequency (discriminative) components. After K→∞ layers, h_i^(K) → d_i/2m for all i (proportional to degree). Practical impact: GNNs rarely benefit from >3 layers.

**Q4: Why does GCN use D̃^(-½)ÃD̃^(-½) specifically? What would go wrong with just A?**
- Answer: Without normalization, high-degree nodes dominate (their features get summed more). D̃^(-½)ÃD̃^(-½) is symmetric normalization — it scales by 1/√(d_i·d_j), giving equal "voting power" regardless of degree. Using just D^(-1)A (row normalization) also works but breaks symmetry (good for directed graphs, problematic for undirected). The self-loop (Ã=A+I) ensures a node retains its own features.

**Q5: Compare BPR loss vs InfoNCE for graph-based CF. When would you prefer each?**
- Answer: BPR samples one positive + one negative per user: L = -ln σ(ŷ_ui - ŷ_uj). InfoNCE uses one positive + ALL in-batch negatives: L = -log(exp(sim_pos/τ) / Σ exp(sim_neg/τ)). InfoNCE generally outperforms because it uses more negative signal per update (harder optimization surface). Prefer BPR when: limited compute (O(1) negatives), small batch sizes. Prefer InfoNCE when: can afford large batches, want harder training signal.

**Q6: Explain the difference between transductive and inductive GNN learning.**
- Answer: Transductive (GCN): trained on the full graph including test nodes (just with labels masked). Cannot generalize to unseen nodes without re-training. Inductive (GraphSAGE): learns aggregation functions that generalize — can embed new nodes by sampling their neighborhood. For recommendations: new users/items arrive constantly → need inductive capability.

**Q7: What is the Weisfeiler-Leman (WL) test and how does it relate to GNN expressiveness?**
- Answer: WL test is a graph isomorphism heuristic that iteratively hashes node neighborhoods. Standard message-passing GNNs are at most as powerful as the 1-WL test — they cannot distinguish graphs that 1-WL cannot. This means: GNNs cannot count certain substructures (e.g., triangles) or distinguish certain non-isomorphic graphs. More powerful: higher-order GNNs, graph transformers.

**Q8: What is the spectral-spatial duality in GCN?**
- Answer: Spectral: convolution defined in Fourier domain (eigenvectors of L), filter g_θ(Λ). Spatial: convolution defined as neighborhood aggregation. Kipf showed: 1st-order Chebyshev approximation of spectral filter = spatial mean aggregation with symmetric normalization. They're mathematically equivalent — spectral gives theoretical grounding, spatial gives efficient implementation.

#### Recommendations (Q9-Q16)

**Q9: Why does LightGCN remove nonlinearities and transformation matrices?**
- Answer: For CF, node IDs have no meaningful initial features — just random embeddings. Transformation W on random embeddings = just a different random init (adds parameters without signal). Nonlinearity σ between layers of linear propagation adds noise since the signal IS the collaborative structure (linear). Empirically: LightGCN (linear) > NGCF (nonlinear) > GCN (with transforms) on all CF benchmarks.

**Q10: How does XGCN differ from end-to-end GNN training? What's the trade-off?**
- Answer: XGCN freezes graph propagation (forward pass only, no gradient) and trains an MLP on propagated embeddings. Trade-off: much cheaper (no sparse backprop through million-node graph), but can't jointly optimize propagation and task. In practice: works well because graph structure is a good prior — the MLP just needs to refine the task objective. Fails when: graph is very noisy or the task requires learning edge importance.

**Q11: Design a GNN-based recommendation system for a cold-start scenario. Walk through your approach.**
- Answer: Cold-start = new items with no interactions. Approach:
  1. Build hybrid graph: user-item interactions + item-attribute edges (KG)
  2. New items connect to attribute nodes (brand, category) from day one
  3. Use inductive method (GraphSAGE) so new nodes get embeddings from neighborhood
  4. Fallback: category-level embeddings from aggregated item-level signals (Bayesian prior)
  5. Warm-up: as interactions accumulate, λ blending shifts from attribute-based to interaction-based
  - λ(item) = sigmoid(interaction_count/scale - threshold)
  6. Training: contrastive loss with hard negatives from same category (force fine-grained discrimination)

**Q12: You have a user-item graph with 100M users and 10M items. How would you train a GNN at this scale?**
- Answer: Full graph doesn't fit in GPU memory. Approach:
  1. Mini-batch with neighbor sampling (GraphSAGE-style): sample K=15 neighbors per hop, 2-3 hops
  2. Use BPR or in-batch InfoNCE (avoids sampling all negatives)
  3. Distributed training: partition graph across GPUs, use gradient aggregation
  4. Alternative (XGCN approach): propagate once on CPU cluster (MapReduce), train MLP on GPU
  5. For inference: precompute all embeddings, serve via FAISS for ANN search
  - Latency: precomputed embeddings = O(1) lookup + FAISS query (< 10ms)

**Q13: What's the relationship between matrix factorization and GNNs?**
- Answer: MF = 1-layer LightGCN. Specifically: u_final = Σ_{i∈N(u)} (1/√d_u·√d_i) · e_i = one pass of normalized aggregation on bipartite graph. LightGCN extends this to K layers (K-hop collaborative signal). GNNs add: multi-hop paths, nonlinear transforms (NGCF), attention (GAT). All reduce to MF when L=1, no transforms.

**Q14: How would you handle temporal dynamics in a graph-based recommender?**
- Answer: Options ranked by complexity:
  1. **Edge decay**: w_edge × exp(-Δt/τ) — recent edges count more (simple, effective)
  2. **Time-window graphs**: separate graphs for different periods (used in P13N co-interaction graph)
  3. **Temporal edge features**: encode timestamp as edge feature, attention learns recency importance
  4. **Dynamic GNN**: separate propagation at each timestep, temporal attention across timesteps
  5. **Sequence + Graph hybrid**: GNN for collaborative structure, transformer for sequential pattern
  - Recommendation: start with edge decay or time-windows (simple, interpretable), add complexity if needed

**Q15: Compare single-interest vs multi-interest GNN approaches. When is multi-interest worth the complexity?**
- Answer: Multi-interest worth it when:
  - Users have diverse, distinct interests (electronics buyer + book reader + athlete)
  - Category diversity matters (campaign targeting, homepage)
  - Metrics show "average embedding" retrieves irrelevant items
  - Scale is manageable (multi-interest = K× more computation)
  Not worth it when:
  - Domain is narrow (only recommending within one category)
  - Users have coherent preferences (luxury fashion — one taste)
  - Latency budget is tight (serving K embeddings per user = K× ANN queries)

**Q16: Explain how you'd evaluate a GNN recommender vs a non-graph baseline fairly.**
- Answer: Key principles:
  - Same data split (temporal: train on past, test on future — not random)
  - Same negative sampling (same negatives for all models)
  - Metrics: Recall@K, NDCG@K, MRR (not just AUC on random negatives)
  - Ablations: graph structure helps? (compare GNN with random graph = only features)
  - Computational fairness: same embedding dimension, similar parameter count
  - Real-world: A/B test with CTR, conversion, diversity metrics
  - Pitfall: offline metrics on easy negatives (random) inflate all models — use hard negatives or full catalog

---

## Slide Design Principles

### Visual Style:
- Dark background with high-contrast text and diagrams
- Graph visualizations: colored nodes, weighted edges, animated propagation
- Math: LaTeX-rendered, appearing incrementally (build animations)
- Code: syntax-highlighted, minimal (pseudocode over implementation)

### Each Slide Has:
- **Title** (top): clear, specific topic
- **Visual** (center): diagram, graph, or animation — never just text
- **Math** (bottom-left): formal notation when needed
- **Intuition** (bottom-right): one-sentence plain English explanation

### Interactive Elements (one per chapter):
- Built with D3.js or Three.js for graph visualization
- Clickable, draggable nodes
- Real-time parameter adjustment (sliders)
- Step-through animations for algorithms

### Quiz Design:
- 5-6 questions per chapter
- Mix of: calculation, conceptual, true/false, "what goes wrong if..."
- Answers revealed on click with brief explanation
- Progressive difficulty: first 2 easy (recall), last 2 hard (application)

---

## Chapter 9: Interview Prep — Questions, Answers, and Cheatsheets

**File:** `09_interview_prep.html`

### Section A: Concept Questions with Answers (30 questions)

#### Fundamentals (Q1-Q8)

**Q1: Explain message passing in GNNs in one sentence. Why is it the unifying framework?**
- Answer: Each node iteratively collects and aggregates information from its neighbors, then updates its own representation. It unifies GCN, GraphSAGE, GAT because they all follow this pattern — they differ only in HOW they aggregate (mean vs attention vs sampled).

**Q2: What is the Graph Laplacian and why does it matter for GNNs?**
- Answer: L = D - A. It measures local smoothness — Lx gives the difference between a node's value and its neighbors' average. Its eigenvectors define the "frequency" basis for graphs (like Fourier for signals). GCN's spectral derivation starts from filtering in this basis. Key property: x^T L x = Σ_{(i,j)∈E} (x_i - x_j)² — measures total signal variation over edges.

**Q3: What is over-smoothing? Give the mathematical intuition.**
- Answer: After K layers of mean aggregation, all node embeddings converge to the same vector. Repeated multiplication by normalized adjacency = low-pass filter killing discriminative components. After K→∞: h_i^(K) → d_i/2m for all i. Practical impact: GNNs rarely benefit from >3 layers.

**Q4: Why does GCN use D̃^(-½)ÃD̃^(-½)? What would go wrong with just A?**
- Answer: Without normalization, high-degree nodes dominate. Symmetric normalization scales by 1/√(d_i·d_j), equalizing "voting power." Self-loop (Ã=A+I) ensures a node retains its own features.

**Q5: Compare BPR loss vs InfoNCE for graph-based CF.**
- Answer: BPR: one positive + one negative, L = -ln σ(ŷ_pos - ŷ_neg). InfoNCE: one positive + all in-batch negatives, L = -log(exp(s_pos/τ)/Σexp(s_k/τ)). InfoNCE outperforms (more negative signal per update). Prefer BPR for limited compute; InfoNCE for large batches.

**Q6: Transductive vs inductive GNN learning.**
- Answer: Transductive (GCN): trained on full graph, can't generalize to unseen nodes. Inductive (GraphSAGE): learns aggregation functions that generalize to new nodes via neighborhood sampling. For recommendations: new users/items arrive constantly → need inductive.

**Q7: What is the Weisfeiler-Leman (WL) test and GNN expressiveness?**
- Answer: WL test iteratively hashes node neighborhoods for isomorphism testing. Message-passing GNNs are bounded by 1-WL — can't distinguish certain non-isomorphic graphs or count substructures (triangles). More powerful: higher-order GNNs, graph transformers.

**Q8: Spectral-spatial duality in GCN.**
- Answer: Spectral: convolution in Fourier domain (L eigenvectors). Spatial: neighborhood aggregation. Kipf showed 1st-order Chebyshev ≈ mean aggregation + symmetric normalization. Mathematically equivalent — spectral = theory, spatial = implementation.

#### Recommendations (Q9-Q16)

**Q9: Why does LightGCN remove nonlinearities and transforms?**
- Answer: For CF, nodes have no meaningful features (just IDs). Transform W on random embeddings = different random init. Nonlinearity adds noise since collaborative signal IS linear (structure flow). LightGCN > NGCF > GCN on all CF benchmarks.

**Q10: How does XGCN differ from end-to-end GNN training?**
- Answer: XGCN freezes propagation (no gradient through graph), trains MLP on propagated embeddings. Trade-off: much cheaper but can't jointly optimize. Works well because graph structure is a good prior. Fails when graph is noisy or task needs learned edge importance.

**Q11: Design GNN for cold-start recommendations.**
- Answer: (1) Hybrid graph: interactions + KG attribute edges. (2) New items connect via attributes from day one. (3) GraphSAGE for inductive embedding. (4) Category-level fallback (Bayesian prior). (5) λ blending: sigmoid(interactions/scale - shift). (6) Hard negatives from same category.

**Q12: Train a GNN on 100M users and 10M items — how?**
- Answer: (1) Mini-batch + neighbor sampling (K=15/hop, 2-3 hops). (2) In-batch InfoNCE. (3) Distributed: partition graph across GPUs. (4) Alternative: XGCN — propagate on CPU cluster (MapReduce), train MLP on GPU. (5) Inference: precompute embeddings, serve via FAISS (< 10ms).

**Q13: Relationship between matrix factorization and GNNs?**
- Answer: MF = 1-layer LightGCN. u = Σ_{i∈N(u)} (1/√d_u·√d_i)·e_i = one normalized aggregation pass. LightGCN extends to K layers (K-hop). All GNNs reduce to MF when L=1, no transforms.

**Q14: Handle temporal dynamics in graph recommender?**
- Answer: (1) Edge decay: w×exp(-Δt/τ). (2) Time-window graphs. (3) Temporal edge features + learned attention. (4) Dynamic GNN per timestep. (5) Sequence+Graph hybrid. Start simple (decay/windows), add complexity if needed.

**Q15: When is multi-interest GNN worth the complexity?**
- Answer: Worth it: diverse user interests, category diversity matters, average embedding retrieves irrelevant items. Not worth it: narrow domain, coherent preferences, tight latency (K× ANN queries).

**Q16: Evaluate GNN recommender fairly vs baseline.**
- Answer: Temporal split, same negatives, Recall@K/NDCG@K/MRR, ablations (random graph), same embedding dims, A/B test. Pitfall: random negatives inflate all models.

#### Advanced Topics (Q17-Q24)

**Q17: Over-squashing vs over-smoothing.**
- Answer: Over-smoothing = representations become indistinguishable. Over-squashing = distant information compressed through bottlenecks (∂h_v^(K)/∂x_u decays exponentially). Both limit depth but for different reasons.

**Q18: How does Graphormer encode graph structure?**
- Answer: (1) Centrality encoding: degree as input bias. (2) Spatial encoding: shortest-path distance as attention bias. (3) Edge encoding: edge features along shortest path. Transformer "knows" topology without explicit message passing.

**Q19: Why are positional encodings harder for graphs?**
- Answer: Graphs have no canonical ordering (permutation-invariant). Solutions: Laplacian eigenvectors (sign-ambiguous), random walk statistics (invariant, cheap), learned PEs. Sign ambiguity requires sign-invariant processing.

**Q20: Connection between PageRank and GNN propagation.**
- Answer: PageRank: π = α·A_norm·π + (1-α)·uniform. APPNP: H^(K) = α·Â·H^(K-1) + (1-α)·H^(0). Same structure! Restart probability prevents over-smoothing by preserving initial features. APPNP: decouple transform (MLP) from propagation (PPR).

**Q21: Design GNN for heterogeneous graph (multiple node/edge types).**
- Answer: R-GCN: h_i' = σ(Σ_r Σ_{j∈N_r(i)} (1/|N_r(i)|)·W_r·h_j). Per-relation weight matrix W_r. Problem: |R|×d² params. Solutions: basis decomposition W_r = Σ_b a_rb·B_b, block-diagonal W_r.

**Q22: Computational complexity: GCN vs GAT vs Graph Transformer.**
- Answer: GCN: O(|E|·d + |V|·d²). GAT: O(|E|·d) — still sparse. Graph Transformer: O(|V|²·d) — dense. For 28M nodes: GCN/GAT feasible, Transformer impossible without subgraph sampling.

**Q23: How does negative sampling quality affect GNN training?**
- Answer: Easy negatives (random) → model learns trivial distinctions. Hard negatives (close in embedding but uninteracted) → fine-grained discrimination but risk false negatives. Best: curriculum (easy→hard) or in-batch (natural difficulty).

**Q24: Edge-cut vs vertex-cut partitioning for billion-scale graphs.**
- Answer: Edge-cut: assign nodes to partitions, cut crossing edges. Vertex-cut: assign edges, replicate boundary nodes. For power-law recommendation graphs: vertex-cut better (high-degree nodes replicated, not bottlenecked).

#### System Design (Q25-Q30)

**Q25: Design real-time graph-based recommendation system.**
- Answer: Offline (daily/weekly): full GNN training + embedding computation on distributed cluster. Online (real-time): user embedding lookup → FAISS top-K (< 10ms) → lightweight reranking. Near-real-time: incremental embedding updates for active users.

**Q26: A/B test GNN recommender against CF baseline.**
- Answer: Randomize users into treatment/control. Primary: CTR, conversion, revenue. Secondary: diversity, coverage. Guardrails: satisfaction, returns. Power analysis for sample size. 4 weeks minimum. Segment: cold-start users, heavy users.

**Q27: Great offline metrics but poor A/B results — diagnose.**
- Answer: (1) Distribution mismatch: easy offline negatives. (2) Temporal leakage. (3) Popularity bias: recommends items user would find anyway. (4) Latency: fallback served. (5) Diversity: similar items bore after first click. (6) Feedback loop: baseline optimized by history.

**Q28: Serve personalized recs for 150M users — infrastructure.**
- Answer: Embedding KV store (Redis/DynamoDB). Item index on FAISS serving instances. Flow: fetch embedding (1ms) → ANN (5ms) → rerank (3ms). Weekly full recompute, daily incremental. Shard users by hash, replicate items. Fallback: category popular items.

**Q29: New product category added — how does GNN system adapt?**
- Answer: Add category to KG. New items connect via attributes immediately. First interactions create edges → propagation picks up. Category-level fallback initially. λ blending shifts as evidence grows. Useful after ~1 week (thousands of edges).

**Q30: Debug GNN producing poor embeddings (all similar).**
- Answer: (1) Check cosine similarity between random pairs (should be ~0). (2) Reduce layers (over-smoothing?). (3) Check LR (too high → collapse). (4) Negative hardness (too easy → no discrimination). (5) L2 normalization present? (6) Data quality (duplicates, disconnected). (7) Does 1-layer (MF) work? If not → data problem.

### Section B: Cheatsheet — Key Formulas

#### GNN Layer Formulas:
```
GCN:        H' = σ(D̃^(-½) Ã D̃^(-½) H W)
LightGCN:   e_i^(l+1) = Σ_j (1/√d_i·√d_j) e_j^(l)
GraphSAGE:  h_v' = σ(W · [h_v ∥ AGG(S(N(v)))])
GAT:        h_i' = σ(Σ_j α_ij · W h_j)
APPNP:      H^(k) = α·Â·H^(k-1) + (1-α)·H^(0)
R-GCN:      h_i' = σ(Σ_r Σ_{j∈N_r(i)} W_r h_j / |N_r|)
```

#### Loss Functions:
```
BPR:     L = -Σ ln σ(ŷ_pos - ŷ_neg)
InfoNCE: L = -log(exp(s_pos/τ) / Σ_k exp(s_k/τ))
TransE:  L = -‖h + r - t‖
```

#### Complexity:
```
Method        | Training       | Inference    | Memory
MF            | O(|E|·d)      | O(d)         | O((|U|+|I|)·d)
LightGCN      | O(L·|E|·d)   | O(L·|E|·d)  | O((|U|+|I|)·d + |E|)
GraphSAGE     | O(B·K^L·d²)  | O(K^L·d²)   | O((|U|+|I|)·d)
GAT           | O(L·|E|·d)   | O(L·|E|·d)  | O((|U|+|I|)·d + |E|·K)
XGCN          | O(L·|E|·d)   | O(d)         | O((|U|+|I|)·d)
Graphormer    | O(|V|²·d)    | O(|V|²·d)   | O(|V|²)
```

#### Decision Tree:
```
Need collaborative signal?
├── Yes → GNN-based
│   ├── Scale > 10M nodes? → XGCN (frozen) or LightGCN (mini-batch)
│   ├── Multi-interest needed? → Clustered propagation or DGCF
│   ├── Cold-start items? → Hybrid KG + interaction graph
│   └── Real-time updates? → Incremental embedding updates
└── No → Content-based or sequential model
```

#### Method Comparison:
```
| Method      | Multi-hop | Multi-interest | Cold-start | Scale | Interpretable |
|-------------|-----------|----------------|------------|-------|---------------|
| MF          | 1-hop     | No             | Poor       | ✓✓✓   | Low           |
| LightGCN    | K-hop     | No             | Poor       | ✓✓    | Low           |
| XGCN        | K-hop     | No             | Poor       | ✓✓✓   | Low           |
| DGCF        | K-hop     | Yes (latent)   | Poor       | ✓     | Low           |
| PIMIRec     | 1-hop     | Yes (explicit) | Medium     | ✓✓    | High          |
| Clustered GP| K-hop     | Yes (explicit) | Good       | ✓✓    | High          |
| KG-GNN      | K-hop     | No             | Good       | ✓✓    | Medium        |
```

### Section C: Common Interview Patterns

#### Pattern 1: "Design a recommendation system for X"
- Framework: Data → Graph Construction → Method Selection → Training → Serving → Evaluation
- Always mention: scale, cold-start, latency, evaluation metrics
- Graph angle: what are nodes, edges, edge weights?

#### Pattern 2: "How would you improve metric X?"
- Response: Current approach → Diagnosis → Hypothesis → Solution → Measurement
- Graph solutions: add edge types, more layers, multi-interest, temporal features

#### Pattern 3: "Explain concept X at multiple levels"
- Level 1 (intuition): one-sentence analogy
- Level 2 (technical): math + architecture
- Level 3 (practical): implementation choices, trade-offs, failure modes

#### Pattern 4: "Debug this system"
- Framework: Symptoms → Hypotheses → Diagnostics → Root cause → Fix → Validation
- Common GNN issues: over-smoothing, embedding collapse, data leakage, popularity bias

### Section D: Quick Reference — Numbers to Know

```
Typical GNN layers:             2-3 (more = over-smoothing)
LightGCN embedding dim:         64-256
InfoNCE temperature τ:          0.05-0.2
Neighbor sampling (GraphSAGE):  10-25 per hop
Mini-batch size:                1024-4096
FAISS query latency:            1-10ms for top-100
Graph propagation runtime:      ~1 hour for 100M+ nodes (distributed)
Embedding freshness:            daily (active), weekly (all)
Cold-start threshold:           ~50-100 interactions before item-level reliable
Over-smoothing onset:           layer 4+ for most graphs
Recall@20 (typical):            0.05-0.20
NDCG@20 (typical):              0.03-0.15
Production CTR from recs:       1-5%
```

### Section E: Your Work — Talking Points for Interviews

#### "Tell me about a graph-based system you built"

**Version 1 (30-second pitch):**
"I built a multi-interest graph propagation system for 28M users that improved campaign targeting AUC from 0.53 (baseline) to 0.87. The key insight was using explicit GMM clustering to separate user interests during propagation, rather than trying to disentangle them after."

**Version 2 (2-minute walkthrough):**
"The problem: recommending personalized campaigns to 28M users. Standard approaches collapse diverse interests into one embedding. I designed a 3-layer hierarchical propagation where each layer maintains separate embeddings per interest cluster. Items are soft-clustered via GMM, edges tagged with cluster identity, and propagation happens per-cluster. This preserves interpretability while capturing collaborative signal. The system runs on EMR with edge-wise partitioning. It outperformed IEFM (1024-dim, AUC 0.53), MF (0.63), and even MisGCF-style post-hoc separation (0.72) — hitting 0.87 AUC."

**Version 3 (deep dive — for follow-ups):**
- Why GMM over K-means? Soft assignments allow items to contribute to multiple clusters
- Why normalize per layer? Prevents dominant clusters from absorbing smaller ones
- Why re-anchor in Layer 2? Pre-trained embeddings encode item semantics; propagation should ADD collaborative signal, not REPLACE semantic signal
- Scale: 423M edges, MapReduce with edge-wise partitioning, ~60 min on 30 executors
- What I'd do differently: learned edge weights (MLP) instead of fixed α weights

#### Key metrics to cite:
- 28M users, 423M edges, 3.9M items
- AUC: 0.87 (ours) vs 0.63 (MF) vs 0.53 (IEFM) — 64% relative improvement over MF
- Multi-interest clustering: +42% Recall@20 over single-intent propagation
- 150M user system: 45-minute end-to-end pipeline on EMR
- Graph features: top 4/10 XGBoost features by gain (graph_prob alone = 1.45M gain)

---

## Updated Suggested Flow

| Session | Chapters | Duration | Focus |
|---------|----------|----------|-------|
| 1 | Ch 1-2 | ~50 min | Foundations + GNN/GCN (theory-heavy) |
| 2 | Ch 3 | ~40 min | Graphs for Recs (LightGCN, XGCN, losses) |
| 3 | Ch 4 | ~40 min | Multi-Interest (DGCF, PIMIRec, clustering) |
| 4 | Ch 5 | ~35 min | Graph Transformers |
| 5 | Ch 6 | ~40 min | Scaling with MapReduce |
| 6 | Ch 7 | ~35 min | Knowledge Graphs |
| 7 | Ch 8 | ~45 min | Applied Methods (your work + industry) |
| 8 | Ch 9 | ~40 min | Interview Prep & Cheatsheet |

Total: ~5.5 hours of content across 9 chapters, 8 sessions.
