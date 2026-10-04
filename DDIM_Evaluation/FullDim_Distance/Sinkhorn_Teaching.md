# Sinkhorn Optimal Transport — Theory & Formulas

## 1. The Problem: Comparing Two Distributions

We have two sets of samples (empirical distributions):
- **Generated**: $\{x_1, x_2, \ldots, x_n\} \subset \mathbb{R}^{256}$ — DDIM outputs (flattened from (2,4,32))
- **Reference (Test)**: $\{y_1, y_2, \ldots, y_m\} \subset \mathbb{R}^{256}$ — real test channels

**Question**: How different are these two distributions?

---

## 2. Optimal Transport (OT) — The Core Idea

Imagine each $x_i$ is a pile of dirt and each $y_j$ is a hole. Optimal Transport finds the **cheapest plan** to move all dirt into all holes, where "cost" is the distance you have to carry.

### Formal Definition

Given:
- Source measure: $\mu = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}$ (uniform weight $a_i = 1/n$ on each source point)
- Target measure: $\nu = \frac{1}{m} \sum_{j=1}^m \delta_{y_j}$ (uniform weight $b_j = 1/m$ on each target point)
- Cost matrix: $C_{ij} = \|x_i - y_j\|^2$ (squared Euclidean distance)

The **Kantorovich optimal transport problem** is:

$$W_2^2(\mu, \nu) = \min_{P \in \Pi(\mu, \nu)} \sum_{i=1}^n \sum_{j=1}^m P_{ij} \cdot C_{ij}$$

where $\Pi(\mu, \nu)$ is the set of all **coupling matrices** $P$ (transport plans) satisfying:

$$P_{ij} \geq 0, \quad \sum_j P_{ij} = a_i = \frac{1}{n}, \quad \sum_i P_{ij} = b_j = \frac{1}{m}$$

These are the **marginal constraints**: the total mass leaving $x_i$ must equal $a_i$, and the total mass arriving at $y_j$ must equal $b_j$.

The **Wasserstein-2 distance** is then $W_2 = \sqrt{W_2^2}$.

### Intuition
- $P_{ij}$ tells you "how much mass from $x_i$ goes to $y_j$"
- The constraints say: all mass must be moved (nothing left behind, no extra created)
- We minimize total cost = sum of (mass × distance²)

---

## 3. Why Not Solve It Exactly?

The exact OT is a **linear program** with $n \times m$ variables and $n + m$ constraints.  
Computational complexity: $O(n^3 \log n)$ using network simplex or Hungarian algorithm.

For our problem: $n = 5000, m \approx 800$ → the cost matrix $C$ has $5000 \times 800 = 4{,}000{,}000$ entries. The exact LP is feasible but slow.

---

## 4. Sinkhorn Algorithm — Entropic Regularization

**Key insight** (Cuturi, 2013): Add an **entropy penalty** to make the optimization strongly convex → fast iterative solution.

### Entropic OT Problem

$$W_\varepsilon^2(\mu, \nu) = \min_{P \in \Pi(\mu, \nu)} \left[ \sum_{ij} P_{ij} C_{ij} + \varepsilon \sum_{ij} P_{ij} \log P_{ij} \right]$$

The term $\varepsilon \sum_{ij} P_{ij} \log P_{ij}$ is the **negative entropy** of $P$ scaled by $\varepsilon > 0$.

- When $\varepsilon \to 0$: solution → exact OT (but harder to compute)
- When $\varepsilon \to \infty$: solution → $P = a \otimes b$ (independent coupling, ignores geometry)
- Sweet spot: $\varepsilon$ small enough to approximate true OT, large enough for fast convergence

### Why Does Entropy Help?

Without entropy: the optimal $P$ is typically **sparse** (each source sends mass to only a few targets) — combinatorial problem.

With entropy: the optimal $P$ is **dense** (strictly positive everywhere) and has a special structure that can be found iteratively.

---

## 5. The Sinkhorn Solution — Dual Variables

The KKT conditions of the entropic OT show the optimal plan has the form:

$$P_{ij}^* = \exp\left(\frac{f_i + g_j - C_{ij}}{\varepsilon}\right)$$

where $f = (f_1, \ldots, f_n)$ and $g = (g_1, \ldots, g_m)$ are **dual variables** (Lagrange multipliers for the marginal constraints).

These must be found such that:
$$\sum_j P_{ij}^* = a_i \quad \text{for all } i \qquad \text{(row constraint)}$$
$$\sum_i P_{ij}^* = b_j \quad \text{for all } j \qquad \text{(column constraint)}$$

---

## 6. Sinkhorn Iterations (Log-Domain, Stabilized)

Working in **log-domain** to avoid numerical over/underflow:

### Initialize:
$$f_i = 0, \quad g_j = 0$$

### Iterate (alternating projections onto the two marginal constraints):

**Update f** (enforce row marginals):
$$f_i \leftarrow \varepsilon \left( \log a_i - \text{logsumexp}_j \left( \frac{-C_{ij} + g_j}{\varepsilon} \right) \right)$$

**Update g** (enforce column marginals):
$$g_j \leftarrow \varepsilon \left( \log b_j - \text{logsumexp}_i \left( \frac{-C_{ij} + f_i}{\varepsilon} \right) \right)$$

### Convergence:
Repeat until $\max_i |f_i^{(k)} - f_i^{(k-1)}| < \text{tol}$ or max iterations reached.

### Why "logsumexp"?

In the non-log domain, the updates are:
$$u_i \leftarrow \frac{a_i}{\sum_j K_{ij} v_j}, \qquad v_j \leftarrow \frac{b_j}{\sum_i K_{ij} u_i}$$
where $K_{ij} = \exp(-C_{ij}/\varepsilon)$.

But for high-dimensional data with large costs, $K_{ij}$ can underflow to 0. The log-domain version replaces $u_i = \exp(f_i/\varepsilon)$ and uses logsumexp to maintain numerical stability.

---

## 7. Final Distance Computation

Once $f, g$ converge:

1. Recover the log-plan: $\log P_{ij} = \frac{f_i + g_j - C_{ij}}{\varepsilon}$
2. Plan: $P_{ij} = \exp(\log P_{ij})$
3. Transport cost: $\text{cost} = \sum_{ij} P_{ij} \cdot C_{ij} \approx W_2^2$
4. Distance: $\text{Sinkhorn-}W_2 = \sqrt{\text{cost}}$

---

## 8. Choice of Regularization $\varepsilon$

In our implementation we use **adaptive scaling**:

$$\varepsilon = \text{reg\_frac} \times \text{median}(C)$$

where `reg_frac = 0.05` and $\text{median}(C)$ is the median of all $n \times m$ pairwise squared distances.

**Rationale**: This makes $\varepsilon$ invariant to the overall scale of the data. If channels are normalized to $[-1, 1]$, the typical $\|x_i - y_j\|^2$ has a certain magnitude; $\varepsilon$ is 5% of the median such cost.

### Effect of $\varepsilon$:
| $\varepsilon$ | Entropic bias | Convergence speed | Approximation quality |
|---|---|---|---|
| Very small | Low bias | Slow (needs many iters) | Close to exact OT |
| Medium (our choice) | Moderate bias | Fast (~100-300 iters) | Good approximation |
| Very large | High bias | Instant | Meaningless (independent coupling) |

**Important**: The entropic regularization introduces a **positive bias** — the Sinkhorn distance is always ≥ the true OT distance. Also, $\text{dist}(X, X) > 0$ (not exactly zero) due to this bias.

---

## 9. Key Properties of Sinkhorn Distance

| Property | Holds? | Note |
|---|---|---|
| Non-negativity: $d(X,Y) \geq 0$ | ✅ | |
| Identity: $d(X,X) = 0$ | ❌ (approximately) | Entropic bias gives small positive value |
| Symmetry: $d(X,Y) = d(Y,X)$ | ✅ | |
| Triangle inequality | ✅ (for fixed $\varepsilon$) | |
| Metric? | Not quite (fails identity of indiscernibles) | It's a **divergence**, not a true metric |

Because $d(X,X) \neq 0$ exactly, one sometimes uses the **Sinkhorn divergence**:
$$S_\varepsilon(X, Y) = W_\varepsilon(X,Y) - \frac{1}{2}[W_\varepsilon(X,X) + W_\varepsilon(Y,Y)]$$

which removes the self-distance bias. (Our implementation does NOT do this — it reports the raw transport cost.)

---

## 10. Comparison with FCD (Fréchet Channel Distance)

| Aspect | FCD | Sinkhorn-OT |
|---|---|---|
| **What it compares** | Gaussian fit (mean + covariance) | Full empirical distributions (all points) |
| **Assumption** | Distributions are Gaussian | None (non-parametric) |
| **Information used** | 1st and 2nd moments only | All moments, full geometry |
| **Formula** | $\|\mu_1 - \mu_2\|^2 + \text{Tr}(\Sigma_1 + \Sigma_2 - 2(\Sigma_1 \Sigma_2)^{1/2})$ | $\min_P \sum_{ij} P_{ij} C_{ij} + \varepsilon H(P)$ |
| **Sample-size sensitivity** | Low (only fits mean + cov) | **HIGH** — discussed below |

---

## 11. Critical Issue: Gen-Test Can Go Below Train-Test (and That's Correct)

### Why Gen-Test < Train-Test can happen legitimately

The "floor" (Train-Test distance) is computed using the **actual N training points**. These are a finite, fixed random draw from the true distribution — they contain sampling noise.

The generator at late τ has learned to approximate the **true underlying distribution** (not just the N training points). Its generated samples can be closer to the test distribution than the particular N training points are, because:

1. **The generator generalizes**: it learned the smooth structure of the distribution and produces samples that fill in "gaps" the finite training set doesn't cover.
2. **Finite-sample noise in Train**: the particular 1000 (or 200 or 4000) training points are one random realization. A different random draw would give a slightly different Train-Test distance.
3. **This is the POINT**: good generalization means generating samples that are closer to the true distribution than to the specific training set.

### Analogy with FID in image generation
In the image generation literature, it's well-known that FID(Gen, Test) can be lower than FID(RealA, RealB) where RealA and RealB are two different real subsets. This doesn't mean generation is "better than real" — it means the metric is comparing finite samples to finite samples, and the generator can produce an empirical distribution that happens to be closer to one test set.

### When Gen-Test going below floor IS a problem
If Gen-Test goes **far** below the floor and stays there, it might indicate:
- **Memorization/mode collapse**: the generator is copying test data (check f_mem!)
- **Mode averaging**: the generator produces the "average" sample, which is trivially close to everything but lacks diversity

### The correct interpretation
- **Quality**: Gen-Test should DECREASE from a high initial value (random noise) and approach the floor level
- **Generalization window**: the τ-range where Gen-Test has reached floor-level AND f_mem is still ~0
- **Gen-Test slightly below floor**: normal, expected for a good generator
- **Gen-Test far below floor while f_mem rises**: memorization signal

---

## 12. Our Implementation (What the Code Does)

```python
def sinkhorn_w2(X, Y, reg_frac=0.05, n_iter=300, tol=1e-6, device='cuda'):
    # 1. Move data to GPU
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)  # (n, 256)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)  # (m, 256)
    
    # 2. Compute full cost matrix C_ij = ||x_i - y_j||^2
    C = torch.cdist(Xt, Yt, p=2) ** 2    # (n, m)
    
    # 3. Auto-scale regularization
    reg = reg_frac * median(C)            # ε = 0.05 × median cost
    
    # 4. Initialize dual variables
    f = zeros(n)   # one per source point
    g = zeros(m)   # one per target point
    
    # 5. Sinkhorn iterations (log-domain)
    for iteration in range(n_iter):
        f = reg * (log(1/n) - logsumexp((-C + g) / reg, dim=columns))
        g = reg * (log(1/m) - logsumexp((-C + f) / reg, dim=rows))
        # Stop if f converged
        
    # 6. Recover transport plan and compute cost
    P = exp((f + g - C) / reg)
    cost = sum(P * C)                     # ≈ W₂²
    return sqrt(cost)                     # ≈ W₂
```

### Memory: O(n × m) for the cost matrix
For n=5000, m=800: 5000×800×4 bytes ≈ 16 MB (fits easily on GPU)

### Time: O(n_iter × n × m) for the logsumexp operations
Each iteration is a matrix reduction. With GPU: ~1 second for our problem size.

---

## 13. Summary of Formulas

| Symbol | Meaning |
|---|---|
| $x_i \in \mathbb{R}^{256}$ | Generated channel sample (flattened) |
| $y_j \in \mathbb{R}^{256}$ | Test channel sample (flattened) |
| $C_{ij} = \|x_i - y_j\|^2$ | Squared Euclidean cost |
| $a_i = 1/n$ | Source weight (uniform) |
| $b_j = 1/m$ | Target weight (uniform) |
| $\varepsilon$ | Entropic regularization strength |
| $f_i, g_j$ | Dual variables (Sinkhorn potentials) |
| $P_{ij} = e^{(f_i + g_j - C_{ij})/\varepsilon}$ | Optimal (regularized) transport plan |
| $W_\varepsilon = \sqrt{\sum_{ij} P_{ij} C_{ij}}$ | Sinkhorn-Wasserstein-2 distance |

---

## 14. References

1. **Cuturi, M.** (2013). "Sinkhorn Distances: Lightspeed Computation of Optimal Transport." *NeurIPS*.  
   — Introduced entropic regularization for fast OT computation.

2. **Peyré, G. & Cuturi, M.** (2019). "Computational Optimal Transport." *Foundations and Trends in ML*.  
   — Comprehensive textbook covering theory + algorithms.

3. **Genevay, A. et al.** (2018). "Learning Generative Models with Sinkhorn Divergences." *AISTATS*.  
   — Introduced the debiased Sinkhorn divergence $S_\varepsilon$.

4. **Ramdas, A. et al.** (2017). "On Wasserstein Two-Sample Testing and Related Families of Nonparametric Tests." *Entropy*.  
   — Finite-sample behavior of empirical Wasserstein distances.
