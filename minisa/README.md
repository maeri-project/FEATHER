# MINISA ISA 2.0 Specification

## Overview

MINISA ISA 2.0 defines **8 variable-width instructions** for the FEATHER+ reconfigurable accelerator. Each instruction begins with a 3-bit opcode. Layout and execute instructions scale as $O(\log \text{AH} + \log \text{AW})$ with array size, while DMA instructions are fixed at 33 bits. This compact encoding achieves 24x--39,681x instruction reduction over direct per-cycle micro-configuration across 9 hardware configurations (validated on 450 workload-config pairs).

### Design Principles

1. **Parametric encoding**: Six parameters $\theta = (r_0, c_0, G_r, G_c, s_r, s_c)$ generate the entire $\text{AH} \times \text{AW}$ PE-to-WVN mapping algebraically, replacing $\text{AH} \times \text{AW}$ per-PE config fields.
2. **Buffer-aware sizing**: Field widths are derived from on-chip buffer depths, not fixed constants.
3. **Per-VN variable size**: The `vn_size` field in ExecuteStreaming supports K dimensions not divisible by AH.
4. **Dual dataflow**: A single-bit `dataflow` field selects weight-output-stationary (WO-S) or input-output-stationary (IO-S). Under IO-S, IVN is loaded into stationary buffer while streaming WVNs. Under WO-S, WVN is loaded in to stationary buffer while streaming IVNs. During the search, we simply transpose M,K,N into N,K,M as a new matrix multiplication when toggling dataflow.

---

## Instruction Set Summary

| Opcode | Instruction | Purpose | Width Scaling |
|--------|-------------|---------|---------------|
| `000` | SetWVNLayout | Configure stationary buffer (weight) layout | Variable |
| `001` | SetIVNLayout | Configure streaming buffer (input) layout | Variable |
| `010` | SetOVNLayout | Configure output buffer layout | Variable |
| `011` | ExecuteStreaming | Configure operand streaming parameters | Variable |
| `100` | Store | DMA store to off-chip memory | Fixed 33b |
| `101` | Load | DMA load from off-chip memory | Fixed 33b |
| `110` | Activation | Activation function (reserved) | Fixed 11b |
| `111` | ExecuteMapping | Configure PE-to-WVN mapping | Variable |

---

## Buffer Architecture

MINISA operates on three on-chip SRAM buffers, each banked by AW (one bank per PE column):

| Buffer | Controlled By | Stores | Default Allocation |
|--------|--------------|--------|-------------------|
| **Streaming Buffer** (str) | SetIVNLayout | Input activations (IVNs) | 40% of total SRAM |
| **Stationary Buffer** (sta) | SetWVNLayout | Weights (WVNs) | 40% of total SRAM |
| **Output Buffer** (ob) | SetOVNLayout | Partial sums / outputs (OVNs) | 20% of total SRAM |

### Buffer Depth Calculation

Per-bank scalar depth (no AH in denominator):

$$D_\text{str} = \frac{\text{stream\_bytes}}{\text{AW} \times \text{in\_bytes}}$$

$$D_\text{sta} = \frac{\text{stationary\_bytes}}{\text{AW} \times \text{w\_bytes}}$$

$$D_\text{ob} = \frac{\text{output\_bytes}}{\text{AW} \times \text{out\_bytes}}$$

VN row count per bank: $\text{vn\_rows} = D / \text{AH}$

Total VN capacity: $\text{vn\_total} = \text{vn\_rows} \times \text{AW}$

---

## ISA Parameters (`MinisaIsaParams`)

The ISA encoding widths are derived from a single parameter set:

| Parameter | Definition | Determines |
|-----------|-----------|------------|
| `AH` | Array height (PE column depth) | `vn_size` field width |
| `AW` | Array width (number of PE columns) | L0 and G_r/G_c field widths |
| `D_str` | Streaming buffer per-bank depth | IVN layout field widths |
| `D_sta` | Stationary buffer per-bank depth | WVN layout and EM field widths |
| `D_ob` | Output buffer per-bank depth | OVN layout field widths |
| `HBM_ADDR_BITS` | Off-chip address width (default 29) | Load/Store address field |

### Derived Bit Widths

| Symbol | Formula | Used In |
|--------|---------|---------|
| $b_\text{aw}$ | $\lceil \log_2(\text{AW}) \rceil$ | L0 fields, G_r, G_c |
| $b_\text{str\_rows}$ | $\lceil \log_2(D_\text{str} / \text{AH}) \rceil$ | SetIVNLayout, SetOVNLayout, ExecuteStreaming |
| $b_\text{sta\_rows}$ | $\lceil \log_2(D_\text{sta} / \text{AH}) \rceil$ | SetWVNLayout, ExecuteMapping s_c |
| $b_\text{sta\_total}$ | $\lceil \log_2(D_\text{sta} / \text{AH} \times \text{AW}) \rceil$ | ExecuteMapping r_0, c_0, s_r |
| $b_\text{vn\_size}$ | $\lceil \log_2(\text{AH}) \rceil$ | ExecuteStreaming vn_size |

---

## Instruction Formats

### SetWVNLayout (opcode `000`)

Configures the stationary buffer layout for weight VNs. Defines the 3-level address mapping $\text{WVN}(k, n) \to \text{bank address}$.

| Field | Width (bits) | Description |
|-------|-------------|-------------|
| opcode | 3 | `000` |
| order | 3 | Permutation order (0--5) for outer/middle/inner dim |
| N_L0 | $b_\text{aw}$ | Inner N-dimension factor (number of banks) |
| N_L1 | $b_\text{sta\_rows}$ | Middle N-dimension factor |
| K_L1 | $b_\text{sta\_rows}$ | Outer K-dimension factor |

The six permutation orders define which of `{nL0, nL1, kL1}` maps to the outer, middle, and inner position in the linear address calculation:

$$\text{addr} = \text{outer} \times (\text{mid\_size} \times \text{inner\_size}) + \text{middle} \times \text{inner\_size} + \text{inner}$$

$$\text{bank} = \text{addr} \bmod \text{AW}$$

### SetIVNLayout (opcode `001`)

Configures the streaming buffer layout for input VNs.

| Field | Width (bits) | Description |
|-------|-------------|-------------|
| opcode | 3 | `001` |
| order | 3 | Permutation order (0--5) |
| M_L0 | $b_\text{aw}$ | Inner M-dimension factor |
| M_L1 | $b_\text{str\_rows}$ | Middle M-dimension factor |
| J_L1 | $b_\text{str\_rows}$ | Outer reduction-tile (J) factor |

### SetOVNLayout (opcode `010`)

Configures the output buffer layout for output VNs. P_L1 and Q_L1 use the streaming buffer VN row width ($b_\text{str\_rows}$) to allow the output layout to address the same VN row range as the streaming buffer (needed for ping-pong swap between output and streaming buffers).

| Field | Width (bits) | Description |
|-------|-------------|-------------|
| opcode | 3 | `010` |
| order | 3 | Permutation order (0--5) |
| P_L0 | $b_\text{aw}$ | Inner M-dimension factor |
| P_L1 | $b_\text{str\_rows}$ | Middle M-dimension factor |
| Q_L1 | $b_\text{str\_rows}$ | Outer N-subgroup factor |

### ExecuteMapping (opcode `111`)

Configures the PE-to-WVN mapping. All AW PE columns are always active. The mapping is:

$$r(a_h, a_w) = r_0 + \left\lfloor \frac{a_w}{G_r} \right\rfloor$$

$$c(a_h, a_w) = c_0 + s_r \cdot a_h + s_c \cdot (a_w \bmod G_c)$$

where $(r, c)$ selects $\text{WVN}(r, c)$ for PE $(a_h, a_w)$. Out-of-bounds $(r, c)$ values are zero-padded.

| Field | Width (bits) | Description |
|-------|-------------|-------------|
| opcode | 3 | `111` |
| G_r | $b_\text{aw}$ | Row-sharing group size (consecutive PE columns per WVN row) |
| G_c | $b_\text{aw}$ | Replication period of horizontal WVN-column pattern |
| r_0 | $b_\text{sta\_total}$ | Base WVN row index |
| c_0 | $b_\text{sta\_total}$ | Base WVN column index |
| s_r | $b_\text{sta\_total}$ | Temporal stride (WVN column advance per PE row) |
| s_c | $b_\text{sta\_rows}$ | Spatial stride (WVN column spacing within one G_c period) |

**Parameter semantics:**

- **$r_0$**: Starting WVN row. Selects where the compute tile begins along the K-reduction dimension.
- **$G_r$**: Row-sharing group width. PE columns $[0, G_r)$ share WVN row $r_0$, columns $[G_r, 2G_r)$ share row $r_0+1$, etc. Controls the granularity of K-distribution across PE columns.
- **$c_0$**: Starting WVN column anchor point.
- **$s_r$**: Temporal stride. As data streams down a PE column (increasing $a_h$), the WVN column advances by $s_r$ per PE row. Typically 1.
- **$G_c$**: Replication period. The horizontal $(c)$-pattern repeats every $G_c$ PE columns. Equal to $\lceil N_t / \text{AH} \rceil$ (number of distinct N-subgroups).
- **$s_c$**: Spatial stride within one $G_c$-period. Adjacent PE columns within a period differ by $s_c$ WVN columns. Equal to AH when $G_c > 1$, else 0.

**Mixed $G_r$**: Different ExecuteMapping instructions within one tile can have different $G_r$ values. This occurs when K-groups are packed into EM batches of varying size (e.g., the last batch has fewer K-groups).

### ExecuteStreaming (opcode `011`)

Configures operand streaming parameters. Paired with ExecuteMapping.

| Field | Width (bits) | Description |
|-------|-------------|-------------|
| opcode | 3 | `011` |
| dataflow | 1 | 0 = IO-S (input stationary), 1 = WO-S (weight stationary) |
| m_0 | $b_\text{str\_rows}$ | Base streaming row index |
| s_m | $b_\text{str\_rows}$ | Streaming row stride |
| T | $b_\text{str\_rows}$ | Number of streaming steps per column |
| vn_size | $b_\text{vn\_size}$ | Active VN height minus 1 (encoded as $\text{vn\_size} - 1$) |

**vn_size encoding**: The field stores $\text{vn\_size} - 1$, so a value of 0 means VN height = 1 and a value of $\text{AH} - 1$ means full VN height = AH. This supports K dimensions not divisible by AH: for example, $K=25, \text{AH}=16$ produces two EMs, one with vn_size=15 (16 elements) and one with vn_size=8 (9 elements).

### Load (opcode `101`)

DMA load from off-chip HBM to on-chip buffer. Fixed width.

| Field | Width (bits) | Description |
|-------|-------------|-------------|
| opcode | 3 | `101` |
| target | 1 | 0 = stationary buffer, 1 = streaming buffer |
| hbm_addr | 29 | Off-chip memory address |

**Total: 33 bits (fixed)**

### Store (opcode `100`)

DMA store from on-chip buffer to off-chip HBM. Fixed width.

| Field | Width (bits) | Description |
|-------|-------------|-------------|
| opcode | 3 | `100` |
| target | 1 | 0 = stationary buffer, 1 = streaming buffer |
| hbm_addr | 29 | Off-chip memory address |

**Total: 33 bits (fixed)**

### Activation (opcode `110`)

Activation function applied to output data. Reserved for future definition.

| Field | Width (bits) | Description |
|-------|-------------|-------------|
| opcode | 3 | `110` |
| tbd | 8 | Reserved |

**Total: 11 bits (fixed)**

---

## On-Chip SRAM Capacity

The total on-chip SRAM capacity is fixed per AH group — all AW values within the same AH share the same total SRAM:

| AH | Total SRAM | Streaming (40%) | Stationary (40%) | Output (20%) |
|----|-----------|-----------------|------------------|--------------|
| 4  | 4 MB      | 1.6 MB          | 1.6 MB           | 0.8 MB       |
| 8  | 16 MB     | 6.4 MB          | 6.4 MB           | 3.2 MB       |
| 16 | 64 MB     | 25.6 MB         | 25.6 MB          | 12.8 MB      |

Per-bank scalar depths for all 9 configurations ($D = \text{buffer\_bytes} / (\text{AW} \times \text{element\_bytes})$):

| Config | Total SRAM | $D_\text{str}$ | $D_\text{sta}$ | $D_\text{ob}$ | str/sta VN rows | ob VN rows |
|--------|-----------|-----------------|-----------------|----------------|-----------------|------------|
| 4×4    | 4 MB      | 419,430         | 419,430         | 52,428         | 104,857         | 13,107     |
| 4×16   | 4 MB      | 104,857         | 104,857         | 13,107         | 26,214          | 3,276      |
| 4×64   | 4 MB      | 26,214          | 26,214          | 3,276          | 6,553           | 819        |
| 8×8    | 16 MB     | 838,860         | 838,860         | 104,857        | 104,857         | 13,107     |
| 8×32   | 16 MB     | 209,715         | 209,715         | 26,214         | 26,214          | 3,276      |
| 8×128  | 16 MB     | 52,428          | 52,428          | 6,553          | 6,553           | 819        |
| 16×16  | 64 MB     | 1,677,721       | 1,677,721       | 209,715        | 104,857         | 13,107     |
| 16×64  | 64 MB     | 419,430         | 419,430         | 52,428         | 26,214          | 3,276      |
| 16×256 | 64 MB     | 104,857         | 104,857         | 13,107         | 6,553           | 819        |

Note: With fixed SRAM per AH, per-bank depth $D$ decreases as AW increases (more banks sharing the same total memory). VN rows = $D / \text{AH}$. Configs with the same AW/AH ratio (e.g., 4×4, 8×8, 16×16 all have ratio 1) share the same VN row counts.

---

## Instruction Width Examples

All widths in bits. SRAM allocation: 0.4 / 0.4 / 0.2 (str / sta / ob). Total SRAM fixed per AH: 4 MB (AH=4), 16 MB (AH=8), 64 MB (AH=16).

| Instruction | 4×4 | 4×16 | 4×64 | 8×8 | 8×32 | 8×128 | 16×16 | 16×64 | 16×256 |
|-------------|-----|------|------|-----|------|-------|-------|-------|--------|
| SetWVNLayout | 42 | 40 | 38 | 43 | 41 | 39 | 44 | 42 | 40 |
| SetIVNLayout | 42 | 40 | 38 | 43 | 41 | 39 | 44 | 42 | 40 |
| SetOVNLayout | 42 | 40 | 38 | 43 | 41 | 39 | 44 | 42 | 40 |
| ExecuteMapping | 81 | 83 | 85 | 86 | 88 | 90 | 91 | 93 | 95 |
| ExecuteStreaming | 57 | 51 | 45 | 58 | 52 | 46 | 59 | 53 | 47 |
| Load | 33 | 33 | 33 | 33 | 33 | 33 | 33 | 33 | 33 |
| Store | 33 | 33 | 33 | 33 | 33 | 33 | 33 | 33 | 33 |
| Activation | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 11 |

### Width Scaling Analysis

With fixed SRAM per AH, increasing AW increases the number of buffer banks while decreasing the per-bank depth. This creates **opposing scaling effects**:

**L0 fields** (N_L0, M_L0, P_L0, G_r, G_c): Grow with $\lceil \log_2 \text{AW} \rceil$ — 2b (AW=4) → 8b (AW=256).

**L1 fields** (N_L1, K_L1, M_L1, J_L1, P_L1, Q_L1): Shrink with $\lceil \log_2(D / \text{AH}) \rceil$ — 17b (square) → 13b (AW=16×AH). Since $D = \text{SRAM} \times \text{frac} / (\text{AW} \times \text{elem\_bytes})$, larger AW means fewer VN rows per bank.

**Net effect on layout instructions**: L0 grows by +6b while L1 shrinks by −8b (×2 fields) from 4×4 to 4×64, yielding a net **decrease** of 4 bits. All three layout instructions have identical widths (P_L1/Q_L1 use $b_\text{str\_rows}$).

**ExecuteMapping**: Grows modestly (+14b from 4×4 to 16×256). The $b_\text{sta\_total} = \lceil \log_2(\text{sta\_vn\_rows} \times \text{AW}) \rceil$ fields (r_0, c_0, s_r) grow by +2b each because total VN count increases with AW, while the s_c field shrinks.

**ExecuteStreaming**: Shrinks significantly (−12b from 4×4 to 4×64). Three fields (m_0, s_m, T) each shrink with $b_\text{str\_rows}$, while vn_size grows only with $\lceil \log_2 \text{AH} \rceil$.

**Fixed-width instructions**: Load/Store (33b) and Activation (11b) are constant across all configurations.

| Scaling factor | Fields affected | Direction with AW↑ |
|---------------|----------------|-------------------|
| $\lceil \log_2 \text{AW} \rceil$ | L0, G_r, G_c | ↑ grows |
| $\lceil \log_2(D/\text{AH}) \rceil$ | L1, s_c, m_0, s_m, T | ↓ shrinks |
| $\lceil \log_2(D/\text{AH} \times \text{AW}) \rceil$ | r_0, c_0, s_r | ↑ grows (slowly) |
| $\lceil \log_2 \text{AH} \rceil$ | vn_size | constant per AH |

---

## 6-Stage Compilation Pipeline

The MINISA compiler translates a GEMM workload $C[M,N] = A[M,K] \times B[K,N]$ into a sequence of MINISA instructions through six stages. A **pre-stage dataflow selection** first decides which operand is stationary.

### Pre-stage: Dataflow Selection

**Knob: dataflow** $\in \{\text{WO-S}, \text{IO-S}, \text{auto}\}$

| Mode | Stationary | Streaming | Search dimensions |
|------|-----------|-----------|-------------------|
| WO-S (weight-output stationary) | Weights $B[K,N]$ | Inputs $A[M,K]$ | $(M, K, N)$ |
| IO-S (input-output stationary)  | Inputs $A[M,K]$  | Weights $B[K,N]$ | $(N, K, M)$ — transpose |

When `dataflow = "auto"`, the compiler runs both dataflows and selects the one with lower total latency. IO-S is skipped when $M = N$ (symmetric).

### Stage 1: Tile

Enumerate legal tiling choices $(M_t, K_t, N_t)$ that fit on-chip buffers.

**Knob: tile sizes** $(M_t, K_t, N_t)$ — power-of-2 values from AH up to each dimension.

Feasibility constraints:
- $\text{IVN count} = M_t \times \lceil K_t / \text{AH} \rceil \leq \text{cap\_stream}$
- $\text{WVN count} = N_t \times \lceil K_t / \text{AH} \rceil \leq \text{cap\_stationary}$
- $\text{OVN count} = M_t \times \lceil N_t / \text{AH} \rceil \leq \text{cap\_output}$

Candidates are sorted by tile volume $M_t \times K_t \times N_t$ descending (up to 512 candidates).

### Stage 2: Lower

Lower each tile into VN structure. No knobs — deterministic from $(M_t, K_t, N_t)$ and AH.

**Derived quantities:**
- $K_g = \lceil K_t / \text{AH} \rceil$ — number of K-groups (reduction tiles)
- `vn_sizes` $= (v_0, v_1, \ldots, v_{K_g-1})$ — per-K-group VN sizes. All $v_{k_g} = \text{AH}$ except possibly $v_{K_g-1} = K_t \bmod \text{AH}$ when $K_t$ is not divisible by AH. Example: $K_t=25, \text{AH}=16 \to (16, 9)$.
- $n_\text{col\_types} = \lceil N_t / \text{AH} \rceil$ — number of N-subgroups

**VN rank variables:**

Each VN is a vector of $v_{k_g}$ elements (or AH for full-sized groups). VNs are indexed by rank variables:

| VN type | Rank variables | Count formula | Description |
|---------|---------------|---------------|-------------|
| $\text{IVN}(m_t, k_g)$ | $m_t \in [0, M_t)$, $k_g \in [0, K_g)$ | $M_t \times K_g$ | Input activation vector for output row $m_t$, K-group $k_g$ |
| $\text{WVN}(k_g, n_t)$ | $k_g \in [0, K_g)$, $n_t \in [0, N_t)$ | $K_g \times N_t$ | Weight vector for K-group $k_g$, output column $n_t$ |
| $\text{OVN}(m_t, n_t)$ | $m_t \in [0, M_t)$, $n_t \in [0, n_\text{col\_types})$ | $M_t \times n_\text{col\_types}$ | Output partial sum for row $m_t$, N-subgroup $n_t$ |

### Stage 3: Group

Form VN groups $\text{VG}(m_t, k_g, n_t)$ where $m_t \in [0, M_t)$, $k_g \in [0, K_g)$, $n_t \in [0, n_\text{col\_types})$.

**Knob: WVN column stride** $\in \{\text{block}, \text{strided}\}$ (only meaningful when $n_\text{col\_types} > 1$, i.e., $N_t > \text{AH}$)

Each $\text{VG}(m_t, k_g, n_t)$ contains:
- 1 IVN: $\text{IVN}(m_t, k_g)$
- Up to AH WVNs: $\{\text{WVN}(k_g, n_t \cdot \text{AH} + i) \mid i \in [0, \min(\text{AH}, N_t - n_t \cdot \text{AH}))\}$

The WVN column stride knob controls how these WVN columns are mapped to PE rows within a column, which determines the $s_r$ and $s_c$ values in ExecuteMapping:

| WVN column stride | $s_r$ | $s_c$ | WVN column pattern across PE rows |
|-------------------|-------|-------|-----------------------------------|
| **block** (default) | $1$ | $\text{AH}$ if $G_c > 1$, else $0$ | Consecutive WVN columns within each N-subgroup |
| **strided** | $n_\text{col\_types}$ | $1$ | WVN columns interleaved across N-subgroups |

**Block example** ($N_t = 8$, $\text{AH} = 4$, $G_c = 2$): PE row $a_h$ in subgroup 0 reads WVN column $a_h$; in subgroup 1 reads WVN column $4 + a_h$. Stride $s_r = 1$, $s_c = 4$.

**Strided example** ($N_t = 8$, $\text{AH} = 4$, $G_c = 2$): PE row $a_h$ in subgroup 0 reads WVN column $2 \cdot a_h$; in subgroup 1 reads WVN column $2 \cdot a_h + 1$. Stride $s_r = 2$, $s_c = 1$.

The strided pattern changes which WVN addresses are accessed concurrently across PE columns, potentially resolving bank conflicts that the block pattern cannot avoid.

Total VN groups: $M_t \times K_g \times n_\text{col\_types}$.

### Stage 4: Combine

Combine VN groups sharing the same WVN set into combined columns (CGs).

**Knob: duplication factor** $d \in [1, d_\text{max}]$ where $d_\text{max} = \lfloor \text{AW} / n_\text{col\_types} \rfloor$.

The duplication factor controls the trade-off between M-parallelism and K-packing:

| Pattern | $d$ value | K-groups per EM | Replicas ($n_\text{rep}$) | When preferred |
|---------|-----------|---------------|---------------------------|----------------|
| **broadcast** | $d_\text{max}$ | 1 | $d_\text{max}$ | Small $K$, large $M$ |
| **contiguous** | $1 < d < d_\text{max}$ | $\text{AW} / (d \cdot n_\text{col\_types})$ | $d$ | Balanced |
| **interleaved** | $1$ | $\text{AW} / n_\text{col\_types}$ | 1 | Large $K$, small $M$ |

Derived relationships:
- Columns per K-group: $n_\text{col\_types} \times d$
- K-groups per EM: $k_\text{em} = \lfloor \text{AW} / (n_\text{col\_types} \times d) \rfloor$
- Replicas: $n_\text{rep} = \lfloor \text{AW} / (k_\text{em} \times n_\text{col\_types}) \rfloor$
- IVNs per column: $\lceil M_t / n_\text{rep} \rceil$

All variants use **interleaved stride distribution** for IVN assignment: replica $r$ takes output rows $\{r, r + n_\text{rep}, r + 2 \cdot n_\text{rep}, \ldots\}$ to ensure bank-conflict-free streaming buffer access.

The search evaluates all valid $d$ values and de-duplicates those that produce identical ExecuteMapping parameter signatures.

### Stage 5: Map

Derive (ExecuteMapping, ExecuteStreaming) parameter pairs from combined columns. Each compute tile produces one paired instruction.

**Knob: IVN distribution** $\in \{\text{interleaved}, \text{consecutive}\}$ (only meaningful when $n_\text{rep} > 1$, i.e., duplication is active)

#### ExecuteMapping parameters

$\theta_\text{EM} = (r_0, c_0, G_r, G_c, s_r, s_c)$ — derived per EM batch starting at K-group $j_\text{start}$ with $k_\text{actual}$ K-groups:

| Parameter | Formula (block) | Formula (strided) | Meaning |
|-----------|----------------|-------------------|---------|
| $r_0$ | $j_\text{start}$ | $j_\text{start}$ | Base WVN row (first K-group in batch) |
| $c_0$ | $0$ | $0$ | Base WVN column |
| $G_r$ | $\lfloor \text{AW} / k_\text{actual} \rfloor$ | $\lfloor \text{AW} / k_\text{actual} \rfloor$ | PE columns sharing one WVN row |
| $G_c$ | $n_\text{col\_types}$ | $n_\text{col\_types}$ | Replication period (N-subgroups) |
| $s_r$ | $1$ | $n_\text{col\_types}$ | Temporal stride per PE row |
| $s_c$ | $\text{AH}$ if $G_c > 1$, else $0$ | $1$ if $G_c > 1$, else $0$ | Spatial stride within one period |

**Mixed $G_r$**: When the last EM batch has fewer K-groups ($k_\text{actual} < k_\text{em}$), its $G_r$ is larger (more weight duplication). This is intentional and reduces total streaming passes.

**VN-size boundary splitting**: EM batches are split at VN-size boundaries so that all K-groups within one EM share the same `vn_size`. This ensures correct PE register loading when $K_t$ is not divisible by AH.

#### ExecuteStreaming parameters

$\theta_\text{ES} = (\text{dataflow}, m_0, s_m, T, \text{vn\_size})$ — derived from the same combined column structure:

| Parameter | Formula (interleaved) | Formula (consecutive) | Meaning |
|-----------|----------------------|----------------------|---------|
| dataflow | $0$ (IO-S) or $1$ (WO-S) | same | Selected by pre-stage dataflow knob |
| $m_0$ | $0$ | $0$ | Base streaming row index (start of IVN sequence) |
| $s_m$ | $n_\text{rep}$ | $1$ | Streaming row stride |
| $T$ | $\lceil M_t / n_\text{rep} \rceil$ | $\lceil M_t / n_\text{rep} \rceil$ | Number of streaming steps per column |
| vn_size | $v_{k_g} - 1$ (encoded) | same | Active VN height for this compute tile |

**IVN distribution modes**: When duplication is active ($n_\text{rep} > 1$), $M_t$ IVN rows are distributed across $n_\text{rep}$ replicas. The IVN distribution knob controls the assignment:

| IVN distribution | Replica $r$ streams rows | $s_m$ | Pattern |
|------------------|-------------------------|-------|---------|
| **interleaved** (default) | $\{r, r + n_\text{rep}, r + 2 \cdot n_\text{rep}, \ldots\}$ | $n_\text{rep}$ | Strided across replicas |
| **consecutive** | $\{r \cdot T, r \cdot T + 1, \ldots, r \cdot T + T - 1\}$ | $1$ | Contiguous blocks per replica |

**Interleaved** ensures that at each streaming step, the $n_\text{rep}$ concurrently-accessed IVN rows have consecutive $m_t$ values (stride = 1 in physical addressing), which typically avoids bank conflicts. **Consecutive** groups contiguous $m_t$ blocks per replica, which may resolve conflicts in cases where the interleaved pattern creates address collisions with a particular layout.

**Relationship between $T$ and $n_\text{rep}$**: $n_\text{rep} = \lfloor G_r / G_c \rfloor$ replicas each handle $T = \lceil M_t / n_\text{rep} \rceil$ IVN streaming steps.

**Per-tile vn_size**: Derived from `vn_sizes` (Stage 2). For K-group $k_g$:

$$\text{vn\_size}(k_g) = \begin{cases} \text{AH} & \text{if } k_g < K_g - 1 \text{ or } K_t \bmod \text{AH} = 0 \\ K_t \bmod \text{AH} & \text{otherwise (last K-group)} \end{cases}$$

The `vn_size` field is encoded as $\text{vn\_size} - 1$ (so 0 means height 1, AH$-1$ means full height). This affects the per-tile execution timing: WVN load = $\text{vn\_size}^2$, IVN streaming = $T \times \text{vn\_size}$, pipeline fill = $\text{vn\_size}$ cycles.

**Example**: $M_t = 64$, $K_t = 25$, $\text{AH} = 16$, $\text{AW} = 16$ (broadcast, $n_\text{rep} = 16$):
- K-group 0: ExecuteMapping $(r_0=0, G_r=16, \ldots)$ + ExecuteStreaming $(T=4, \text{vn\_size}=15)$ → 16 elements
- K-group 1: ExecuteMapping $(r_0=1, G_r=16, \ldots)$ + ExecuteStreaming $(T=4, \text{vn\_size}=8)$ → 9 elements

### Stage 6: Layout

Search for bank-conflict-free buffer address permutation orders.

**Knob: layout orders** $(\text{order\_w}, \text{order\_i}, \text{order\_o}) \in \{0,1,2,3,4,5\}^3$

Each order selects one of 6 permutations of the 3-level address factors `(L0, L1_inner, L1_outer)`:

| Order | Outer → Middle → Inner |
|-------|------------------------|
| 0 | L1_outer, L0, L1_inner |
| 1 | L1_outer, L1_inner, L0 |
| 2 | L0, L1_outer, L1_inner |
| 3 | L1_inner, L1_outer, L0 |
| 4 | L0, L1_inner, L1_outer |
| 5 | L1_inner, L0, L1_outer |

Two search modes:
- **Exhaustive**: Test all $6^3 = 216$ permutation order combinations for (W, I, O). Prunes early: valid O-orders first, then valid (W, I) pairs.
- **Sequential**: Test orders in priority sequence per operand; stop at first valid triple.

**Bank conflict check**: For each PE loading cycle, verify that no two active PE columns map to the same buffer bank with different addresses. Same bank + same address is allowed (broadcast read). Out-of-bounds VN accesses are zero-padded (no buffer access, no conflict).

#### Fallback mechanism

When Stage 6 fails to find any valid layout combination for the current design choices, the search **falls back to Stage 3** and retries Stages 3→4→5→6 with alternative design choices. The fallback iterates over all combinations of (WVN column stride, IVN distribution) in priority order:

| Priority | WVN column stride | IVN distribution | When tried |
|----------|-------------------|------------------|------------|
| 1 (default) | block | interleaved | Always (first attempt) |
| 2 | block | consecutive | If priority 1 fails |
| 3 | strided | interleaved | If priorities 1–2 fail (only when $n_\text{col\_types} > 1$) |
| 4 | strided | consecutive | If priorities 1–3 fail (only when $n_\text{col\_types} > 1$) |

The strided WVN column stride is skipped when $n_\text{col\_types} \leq 1$ (it produces identical results to block in that case). This fallback is particularly important for **rectangular configurations** (AW >> AH, e.g., AH=4,AW=16 or AH=8,AW=32) where the default (block, interleaved) design cannot find bank-conflict-free layouts.

### Search Space Summary

| Stage | Knob | Range | Typical count |
|-------|------|-------|---------------|
| Pre | Dataflow | $\{\text{WO-S}, \text{IO-S}\}$ | 2 (or 1 if $M=N$) |
| S1 | Tile $(M_t, K_t, N_t)$ | Power-of-2 values fitting on-chip | ≤ 512 |
| S2 | — (deterministic) | — | 1 |
| S3 | WVN column stride | $\{\text{block}, \text{strided}\}$ | 2 (or 1 if $n_\text{col\_types} \leq 1$) |
| S4 | Duplication factor $d$ | $[1, \lfloor \text{AW}/n_\text{col\_types} \rfloor]$ | 1–256 |
| S5 | IVN distribution | $\{\text{interleaved}, \text{consecutive}\}$ | 2 (or 1 if $n_\text{rep} = 1$) |
| S6 | Layout orders $(o_w, o_i, o_o)$ | $\{0..5\}^3$ | 216 (exhaustive) or ≤18 (sequential) |

**Note**: S3 and S5 knobs are explored via the Stage 6 fallback mechanism (see above). The default choices (block, interleaved) are tried first; alternatives are only explored when the layout search fails.

---

## Dataflow Modes

| Mode | Alias | Stationary Operand | Streaming Operand | Search Dimensions |
|------|-------|-------------------|-------------------|-------------------|
| Weight-output stationary | WO-S | Weights (K x N) | Inputs (M rows) | {M, K, N} |
| Input-output stationary | IO-S | Inputs (M x K) | Weights (N cols) | {N, K, M} |

The compiler tries both dataflows (when `dataflow="auto"`) and selects the one with lower total latency. IO-S is favored when $M \gg N$.

---

## Execution Model

### Per-ExecuteMapping/ExecuteStreaming Timing

ExecuteMapping and ExecuteStreaming are issued as a pair for each compute tile. Each pair specifies a unique `vn_size` (from ExecuteStreaming) that determines the active PE height for that tile.

For each (ExecuteMapping, ExecuteStreaming) pair with `vn_size`:

1. **WVN load to PE registers**: $\text{vn\_size}^2$ cycles (vn_size active PEs $\times$ vn_size elements each)
2. **IVN streaming**: $T \times \text{vn\_size}$ cycles ($T$ streaming steps, vn_size cycles per step)
3. **Pipeline fill**: $\text{vn\_size}$ cycles (last IVN element propagates through vn_size active PE rows)
4. **BIRRD drain**: $2 \lceil \log_2(\text{AW}) \rceil$ cycles

### Inter-EM Pipelining

WVN load for the next EM overlaps with the current EM's IVN streaming. The effective period between consecutive EMs:

$$\text{em\_period} = \max(\text{nest\_time}, \text{vn\_size}^2 - \text{vn\_size})$$

where $\text{nest\_time} = T \times \text{vn\_size} + \text{vn\_size}$.

For $K_g$ consecutive EMs per K-tile (each with its own `vn_size`):

$$C = \text{vn\_size}_0^2 + \sum_{i=1}^{K_g-1} \text{em\_period}_i + \text{nest\_time}_{K_g-1} + \text{birrd\_drain}$$

When all K-groups have the same `vn_size` (K divisible by AH), this simplifies to:

$$C = \text{vn\_size}^2 + (K_g-1) \times \text{em\_period} + \text{nest\_time} + \text{birrd\_drain}$$

### Accumulation Semantics

Consecutive ExecuteMapping instructions accumulate partial sums into the same output buffer region (additive, not overwriting). A new SetOVNLayout instruction copies the output buffer to the streaming buffer (ping-pong swap) and zeroes the output buffer for the next tile.

---

## Typical Instruction Sequence

For one output tile $C[M_t, N_t]$ accumulated over $\lceil K/K_t \rceil$ K-steps:

```
SetOVNLayout  order, P_L0, P_L1, Q_L1     ; configure output buffer

for each K-step:
    SetIVNLayout  order, M_L0, M_L1, J_L1  ; configure streaming buffer
    Load          target=1, hbm_addr         ; DMA load inputs (streaming buffer)
    SetWVNLayout  order, N_L0, N_L1, K_L1  ; configure stationary buffer
    Load          target=0, hbm_addr         ; DMA load weights (stationary buffer)

    for each EM in tile:
        ExecuteMapping   G_r, G_c, r_0, c_0, s_r, s_c    ; load WVN into NEST
        ExecuteStreaming  dataflow, m_0, s_m, T, vn_size  ; stream IVN, trigger compute

Store  target=0, hbm_addr                   ; DMA store output
```

---

## Evaluation Results (450 workload-config pairs)

All 50 workloads $\times$ 9 configurations verified at both ISA-level and config-level.

### Compression Ratio (MINISA ISA bytes vs Config Stream bytes)

| Config | Avg Compression | Inst Reduction (geo-mean) |
|--------|----------------|--------------------------|
| 4x4 | 7,684x | 24x |
| 4x16 | 7,608x | 54x |
| 4x64 | 11,782x | 184x |
| 8x8 | 12,407x | 196x |
| 8x32 | 14,216x | 600x |
| 8x128 | 30,250x | 2,465x |
| 16x16 | 21,363x | 2,176x |
| 16x64 | 34,634x | 10,275x |
| 16x256 | 32,443x | 39,681x |

### Compute Utilization

| Config | Avg Compute Utilization | Avg Latency (cycles) |
|--------|------------------------|---------------------|
| 4x4 | 92.1% | 40,045,201 |
| 4x16 | 89.2% | 10,117,115 |
| 4x64 | 91.6% | 2,510,426 |
| 8x8 | 80.1% | 10,470,460 |
| 8x32 | 82.0% | 2,617,074 |
| 8x128 | 82.4% | 710,018 |
| 16x16 | 69.3% | 2,833,058 |
| 16x64 | 69.3% | 841,181 |
| 16x256 | 69.0% | 215,097 |

---

## Implementation Reference

| Component | File | Key Functions |
|-----------|------|---------------|
| ISA parameters | `minisa/isa.py` | `MinisaIsaParams`, `config_to_isa_params()` |
| ISA bitwidths | `minisa/isa.py` | `minisa_bitwidths()`, `minisa_opcode_values()` |
| RTL parameters | `minisa/isa.py` | `FeatherPlusParams`, `config_to_hw_params()` |
| Hardware config | `minisa/config.py` | `FeatherPlusConfig`, `make_cfg()` |
| 6-stage search | `minisa/search.py` | `co_search_layout_mapping()`, `co_search_gemm()` |
| Trace generation | `minisa/trace.py` | `generate_trace_gemm()`, `verify_trace()` |
| Config expansion | `minisa/to_config.py` | `convert_trace_to_config()`, `compute_config_stream_summary()` |
| Cycle model | `minisa/cycles.py` | `estimate_cycles_for_gemm()` |
| Evaluation | `minisa/evaluate.py` | `python -m minisa.evaluate --csv ... --out-dir ... --ah 4,8,16 --aw "4,16,64/8,32,128/16,64,256"` |
