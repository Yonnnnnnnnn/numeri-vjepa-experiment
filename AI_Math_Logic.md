# AI Math Logic: Neuro-Symbolic Reconciliation (LNN)

## 1. Neuro-Symbolic Intent Filtering

To resolve **Instruction Pollution**, we define a logical formula $\psi_{valid}(x)$ that evaluates the validity of an intent $x$:

$$
\psi_{valid}(x) = \text{IsCandidate}(x) \wedge (\exists y: \text{IsProductCategory}(y) \wedge \text{Contains}(x, y)) \wedge \neg\text{IsConversational}(x)
$$

### Truth Value Bounds

LNN operates on the interval $[L, U] \in [0, 1]$.

- **IsConversational**: Known prompt noise words (fact, know, task) are assigned $[1, 1]$.
- **IsProductCategory**: Keywords (can, beans, bottle) are assigned $[1, 1]$.
- **IsCandidate**: Output from the VLM, mapped to $[0.8, 1.0]$.

The final score $S_{valid} = \text{mean}(L, U)$ determines if $x$ is added to the genesis intent list.

## 2. Volumetric Reconciliation (Identity Arbitration)

To resolve the **Stock Cube vs. Multi-Can Deadlock**, we arbitrate SKU identity $x$ for cluster $c$ via the formula $\psi_{identity}(c, x)$:

$$
\psi_{identity}(c, x) = \text{LatentMatch}(c, x) \wedge \text{VolumeConsistent}(c, x)
$$

### Rule 2.1: Volume Consistency

$$
\text{VolumeConsistent}(c, x) =
\begin{cases}
1.0 & \text{if } 0.3 \le \frac{V_{cluster}}{V_{unit}(x)} \le 2000 \\
0.1 & \text{if } \frac{V_{cluster}}{V_{unit}(x)} > 2000 \quad \text{(Physically unlikely)} \\
0.2 & \text{if } \frac{V_{cluster}}{V_{unit}(x)} < 0.3 \quad \text{(Unit too large for cluster)}
\end{cases}
$$

### Rule 2.2: Latent Match (DINOv2 Sovereignty)

$$
\text{LatentMatch}(c, x) = \max(\text{IoU}(c, x_{\text{det}}), \cos(\mathbf{z}_c, \mathbf{z}_x))
$$

Where:

- $\mathbf{z}_c$: Current cluster fingerprint (DINOv2).
- $\mathbf{z}_x$: Genesis anchor fingerprint.
- **Identity Sovereignty**: Visual similarity ($\cos$) overrides geometric bias (IoU) if a high-confidence anchor match exists ($>0.7$). This enables the system to identify objects even if GroundingDINO fails to generate a proposal label.

### In V3.8 Implementation:

The system queries the LNN Model:

$$
\hat{x} = \arg\max_{x \in \text{SKUs}} \text{LNN\_Query}(\psi_{identity}(c, x))
$$

This ensures that even if $IoU(c, x)$ is high, a physical impossibility ($VolumeConsistent \to 0$) will force the LNN to reject $x$ in favor of a physically plausible SKU.

## 3. Density Calibration & Heuristic Prior

To resolve the **"Model Not Fitted" Deadlock** in the `DensityPredictor`, we implement a heuristic "cold start" prior.

### Heuristic 3.1: Complexity-Density Proxy

We assume a correlation between semantic feature variance $\sigma^2(\mathbf{X})$ and physical density $\rho$, where $\mathbf{X}$ is the 768-dimensional DINOv2 latent vector:

$$
\rho_{heuristic} = \rho_{min} + \left( \frac{\sigma^2(\mathbf{X}) - \min(\sigma^2)}{\max(\sigma^2) - \min(\sigma^2)} \right) \cdot (\rho_{max} - \rho_{min})
$$

Where:

- $\rho_{min} = 0.1$ (Aerogel/Air)
- $\rho_{max} = 20.0$ (Tungsten/Heavy Metal)

This ensures the system initiates with a non-zero, physically bounded density before empirical training hits.

## 4. Volumetric Per-Cluster Tally (V3.8.1 Fix)

The relationship between total count ($n_{vol}$) and individual clusters $c$ is defined as:

$$
n_{vol} = \sum_{c \in \text{Clusters}} \frac{c_{vol} \cdot \rho}{u_v}
$$

**Critical Correction**: `c_vol` must be extracted per-cluster from the depth manifold metadata to prevent the **Global Volume Leak** error where $n_{vol}$ was incorrectly used as a divisor.

## 5. Centrality Bias — PointBeam Focus (V4.0)

To resolve **Peripheral Noise Pollution** during Intent Genesis, we apply a Foveated Confidence Boosting function that prioritizes objects near the frame center.

### Rule 5.1: Centrality Score

For a bounding box with center $(b_x, b_y)$ relative to the frame center $(0.5, 0.5)$:

$$
d = \sqrt{(b_x - 0.5)^2 + (b_y - 0.5)^2}
$$

$$
S_{centrality} = \max\left(0,\; 1 - \frac{d}{r_{decay}}\right)
$$

Where $r_{decay} = 0.35$ is the radius at which centrality decays to zero.

### Rule 5.2: Confidence Boosting

The base confidence $C_{base}$ from the VLM is boosted by the centrality score:

$$
C_{boosted} = \min\left(1.0,\; C_{base} \cdot (1 + 0.3 \cdot S_{centrality})\right)
$$

This ensures:

- Objects at the **exact center** receive up to a **30% confidence boost**.
- Objects at the **periphery** ($d > r_{decay}$) receive **no boost**.
- The **VLM prompt** explicitly instructs prioritization of centered/framed objects.
