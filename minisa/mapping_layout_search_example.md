Example: GEMM tiling, lowering, mapping, and layout selection on a 4x4 PE array, VN size = 4

Notation:
  - Upper case (M,K,N, M_t,K_t,N_t, M_{L0},M_{L1},...) denotes rank shapes.
  - Lower case (m,k,n, m_t,k_g,n_t, m_{L0},m_{L1},...) denotes rank variables / indices.
  - k_g denotes the reduction-tile index after partitioning K_t by VN size.
  - This example uses WO-S, so IVN is streamed, WVN is stationary, and OVN is stored in the output buffer.

Original GEMM:
  O[M,N] = I[M,K] × W[K,N]

0) Tile the original workload
----------------------------------------
The original GEMM is first tiled into subproblems of shape {M_t, K_t, N_t}:

  O[M,N]
    -> tiles O_t[M_t,N_t] = I_t[M_t,K_t] × W_t[K_t,N_t]

The following Steps 1) ~ 5) operate on one GEMM tile.

For this example:
  M_t = 16, K_t = 12, N_t = 8

So one tile is:
  O_t[16,8] = I_t[16,12] × W_t[12,8]

1) Lower the GEMM tile into VNs along K_t
----------------------------------------
VN size = 4
=> K_t = 12 is partitioned into 3 reduction tiles:

  k_g = 0 -> k = 0..3
  k_g = 1 -> k = 4..7
  k_g = 2 -> k = 8..11

Input tile I_t[16,12]
  -> IVN(m_t,k_g),  m_t in [0..15], k_g in [0..2]

Weight tile W_t[12,8]
  -> WVN(k_g,n),  k_g in [0..2], n in [0..7]

Input VN array (M_t x ceil(K_t/4) = 16 x 3):
  m_t=0   : IVN(0,0)  IVN(0,1)  IVN(0,2)
  ...
  m_t=15  : IVN(15,0) IVN(15,1) IVN(15,2)

Weight VN array (ceil(K_t/4) x N_t = 3 x 8):
               n=0        1        2        3        4        5        6        7
  k_g=0     WVN(0,0) WVN(0,1) WVN(0,2) WVN(0,3) WVN(0,4) WVN(0,5) WVN(0,6) WVN(0,7)
  k_g=1     WVN(1,0) WVN(1,1) WVN(1,2) WVN(1,3) WVN(1,4) WVN(1,5) WVN(1,6) WVN(1,7)
  k_g=2     WVN(2,0) WVN(2,1) WVN(2,2) WVN(2,3) WVN(2,4) WVN(2,5) WVN(2,6) WVN(2,7)

2) Form VN groups
----------------------------------------
Each PE column has AH=4 rows, so one column can hold up to 4 stationary WVNs.
Therefore N_t=8 is split into two column groups:

  n_t = 0 -> output columns n = 0..3
  n_t = 1 -> output columns n = 4..7

Definition:
  VG(m_t,k_g,n_t)
    = one schedulable VN group for one PE column,
      containing:
        - one streaming IVN(m_t,k_g)
        - up to AH stationary WVNs with the same k_g

For this example:
  VG(m_t,k_g,0)
    = IVN(m_t,k_g)
    + {WVN(k_g,0), WVN(k_g,1), WVN(k_g,2), WVN(k_g,3)}

  VG(m_t,k_g,1)
    = IVN(m_t,k_g)
    + {WVN(k_g,4), WVN(k_g,5), WVN(k_g,6), WVN(k_g,7)}

#VN groups
  = M_t × ceil(K_t/4) × ceil(N_t/AH)
  = 16 × 3 × 2
  = 96

3) Combine VN groups across all streamed inputs
----------------------------------------
Definition:
  CG(k_g,n_t)
    = one combined VN group that reuses the same stationary WVNs
      across all streaming rows m_t.
    Equivalently:
      CG(k_g,n_t) = { VG(m_t,k_g,n_t) for all m_t in [0..M_t-1] }

The 6 combined VN groups are:

  CG(0,0): stationary {WVN(0,0), WVN(0,1), WVN(0,2), WVN(0,3)}
           streamed   {IVN(0,0), IVN(1,0), ..., IVN(15,0)}

  CG(0,1): stationary {WVN(0,4), WVN(0,5), WVN(0,6), WVN(0,7)}
           streamed   {IVN(0,0), IVN(1,0), ..., IVN(15,0)}

  CG(1,0): stationary {WVN(1,0), WVN(1,1), WVN(1,2), WVN(1,3)}
           streamed   {IVN(0,1), IVN(1,1), ..., IVN(15,1)}

  CG(1,1): stationary {WVN(1,4), WVN(1,5), WVN(1,6), WVN(1,7)}
           streamed   {IVN(0,1), IVN(1,1), ..., IVN(15,1)}

  CG(2,0): stationary {WVN(2,0), WVN(2,1), WVN(2,2), WVN(2,3)}
           streamed   {IVN(0,2), IVN(1,2), ..., IVN(15,2)}

  CG(2,1): stationary {WVN(2,4), WVN(2,5), WVN(2,6), WVN(2,7)}
           streamed   {IVN(0,2), IVN(1,2), ..., IVN(15,2)}

#combined groups
  = ceil(K_t/4) × ceil(N_t/AH)
  = 3 × 2
  = 6

4) Map combined VN groups onto the 4x4 PE array
----------------------------------------
A 4x4 PE array has AW=4 columns, so at most 4 combined VN groups
can execute concurrently.

Invocation 1: four distinct combined VN groups
----------------------------------------
                stream        stream        stream        stream
                IVN(m_t,0)    IVN(m_t,0)    IVN(m_t,1)    IVN(m_t,1)
                m_t=0..15     m_t=0..15     m_t=0..15     m_t=0..15
                   |             |             |             |
                   v             v             v             v
               +-----------+-----------+-----------+-----------+
               |   Col 0   |   Col 1   |   Col 2   |   Col 3   |
+------------+-+-----------+-----------+-----------+-----------+
| Row 0      | | WVN(0,0)  | WVN(0,4)  | WVN(1,0)  | WVN(1,4)  |
| Row 1      | | WVN(0,1)  | WVN(0,5)  | WVN(1,1)  | WVN(1,5)  |
| Row 2      | | WVN(0,2)  | WVN(0,6)  | WVN(1,2)  | WVN(1,6)  |
| Row 3      | | WVN(0,3)  | WVN(0,7)  | WVN(1,3)  | WVN(1,7)  |
+------------+-+-----------+-----------+-----------+-----------+
                 CG(0,0)     CG(0,1)     CG(1,0)     CG(1,1)

Per-column stream ranges:
  - Col 0: stream IVN(m_t,0), m_t = 0..15
  - Col 1: stream IVN(m_t,0), m_t = 0..15
  - Col 2: stream IVN(m_t,1), m_t = 0..15
  - Col 3: stream IVN(m_t,1), m_t = 0..15

Invocation 2: remaining two combined VN groups with duplication
----------------------------------------
                stream        stream        stream        stream
                IVN(m_t,2)    IVN(m_t,2)    IVN(m_t,2)    IVN(m_t,2)
                m_t=0..7      m_t=8..15     m_t=0..7      m_t=8..15
                   |             |             |             |
                   v             v             v             v
               +-----------+-----------+-----------+-----------+
               |   Col 0   |   Col 1   |   Col 2   |   Col 3   |
+------------+-+-----------+-----------+-----------+-----------+
| Row 0      | | WVN(2,0)  | WVN(2,0)  | WVN(2,4)  | WVN(2,4)  |
| Row 1      | | WVN(2,1)  | WVN(2,1)  | WVN(2,5)  | WVN(2,5)  |
| Row 2      | | WVN(2,2)  | WVN(2,2)  | WVN(2,6)  | WVN(2,6)  |
| Row 3      | | WVN(2,3)  | WVN(2,3)  | WVN(2,7)  | WVN(2,7)  |
+------------+-+-----------+-----------+-----------+-----------+
                 CG(2,0)     CG(2,0)     CG(2,1)     CG(2,1)

Per-column stream ranges:
  - Col 0: stream IVN(m_t,2), m_t = 0..7
  - Col 1: stream IVN(m_t,2), m_t = 8..15
  - Col 2: stream IVN(m_t,2), m_t = 0..7
  - Col 3: stream IVN(m_t,2), m_t = 8..15

5) Choose IVN / WVN / OVN layouts to avoid bank conflicts
----------------------------------------
Each operand is laid out in a physical D x AW buffer with AW=4 columns.
A layout is determined by:

  (i) partition factors, e.g. M = M_{L1} M_{L0}
 (ii) an order_id over the three remaining ranks after fixing VN size
(iii) row-major folding into buffer coordinates

For any operand X, after flattening to VN index L_X:

  addr_row = floor(L_X / AW)
  addr_col = L_X mod AW

Bank-conflict rule:
  At each cycle, let the AW PE columns concurrently request VNs
  { X(alpha_0), X(alpha_1), ..., X(alpha_{AW-1}) }.
  A bank conflict occurs iff two requested VNs have

    same addr_col
    but different addr_row.

  Reusing the exact same VN from the exact same physical address
  is allowed, since it is a broadcast / duplicate access rather than
  a conflict between different rows.

5.1) IVN layout choice
----------------------------------------
The streamed IVN tensor has logical VN shape:

  J_{L1} x M_t = 3 x 16

Choose:
  J_{L0} = 4   (fixed by VN size)
  M_t = M_{L1} M_{L0} = 4 x 4
  order_id = 011
  IVN order: m_{L0} -> m_{L1} -> j_{L1}

Then

  L_IVN
    = m_{L0} * (M_{L1} J_{L1})
    + m_{L1} * J_{L1}
    + j_{L1}

  addr_col(IVN)
    = (m_{L1} * J_{L1} + j_{L1}) mod 4
    = (3 m_{L1} + j_{L1}) mod 4

Why this works:
  - In Invocation 1, the two distinct streamed IVNs are IVN(m_t,0) and IVN(m_t,1).
    Since j_{L1}=k_g appears in addr_col, they land in different buffer columns.
  - In Invocation 2, duplicated columns process two different m_t ranges:
      left half  : m_t = 0..7
      right half : m_t = 8..15
    Here m_{L1} differs between the two row groups, so IVN(m_t,2) from the two
    groups also lands in different buffer columns.
  - Thus the IVNs simultaneously requested by different PE columns do not collide
    on the same buffer-column index unless they are the exact same IVN.

5.2) WVN layout choice
----------------------------------------
The stationary WVN tensor has logical VN shape:

  K_{L1} x N_t = 3 x 8

Choose:
  K_{L0} = 4   (fixed by VN size)
  N_t = N_{L1} N_{L0} = 2 x 4
  order_id = 010
  WVN order: n_{L0} -> k_{L1} -> n_{L1}

Then

  L_WVN
    = n_{L0} * (K_{L1} N_{L1})
    + k_{L1} * N_{L1}
    + n_{L1}

  addr_col(WVN)
    = (k_{L1} * N_{L1} + n_{L1}) mod 4
    = (2 k_{L1} + n_{L1}) mod 4

This gives:

  CG(0,0) -> addr_col = 0
  CG(0,1) -> addr_col = 1
  CG(1,0) -> addr_col = 2
  CG(1,1) -> addr_col = 3
  CG(2,0) -> addr_col = 0
  CG(2,1) -> addr_col = 1

Why this works:
  - In Invocation 1, the four concurrently active combined groups
    CG(0,0), CG(0,1), CG(1,0), CG(1,1)
    map to four distinct buffer columns 0,1,2,3, so there is no conflict.
  - In Invocation 2, only two distinct groups remain:
    CG(2,0) and CG(2,1), which map to columns 0 and 1.
    Their duplicates reuse the same physical addresses, which is allowed.

5.3) OVN layout choice
----------------------------------------
The output tile O_t[16,8] is accumulated in the output buffer.
For illustration, group the 8 output columns into the same two output groups:

  n_t = 0 -> n = 0..3
  n_t = 1 -> n = 4..7

Choose:
  P = M_t = P_{L1} P_{L0} = 4 x 4
  Q = 2 output groups, indexed by q_{L1} = n_t
  P_{L0} = 4
  order_id = 000
  OVN order: p_{L1} -> p_{L0} -> q_{L1}

Then

  L_OVN
    = p_{L1} * (P_{L0} Q_{L1})
    + p_{L0} * Q_{L1}
    + q_{L1}

  addr_col(OVN)
    = q_{L1}

Why this works:
  - The two output groups n_t=0 and n_t=1 occupy different buffer columns.
  - Therefore, partial sums for output columns 0..3 and 4..7 do not collide
    in the output buffer.
  - Columns that contribute to the same output group either reuse the same
    output address or are scheduled through the local accumulation path;
    the layout search rejects any choice that would place different output
    groups on the same buffer column with different rows.

5.4) Summary of one feasible conflict-free layout
----------------------------------------
One valid layout choice for this example is:

  SetIVNLayout:
    M_{L1}=4, M_{L0}=4, order_id = 011
    order = m_{L0} -> m_{L1} -> j_{L1}

  SetWVNLayout:
    N_{L1}=2, N_{L0}=4, order_id = 010
    order = n_{L0} -> k_{L1} -> n_{L1}

  SetOVNLayout:
    P_{L1}=4, P_{L0}=4, Q_{L1}=2, order_id = 000
    order = p_{L1} -> p_{L0} -> q_{L1}

These choices make the concurrently requested IVNs, WVNs, and OVNs
fall on different buffer-column indices for this 4-column schedule,
thereby avoiding bank conflicts.

End-to-end summary
----------------------------------------
Original GEMM:
  O[M,N] = I[M,K] × W[K,N]
          |
          | tile into {M_t,K_t,N_t}
          v
Tile GEMM:
  O_t[M_t,N_t] = I_t[M_t,K_t] × W_t[K_t,N_t]
          |
          | split K_t into VNs of size 4
          v
IVNs : M_t × ceil(K_t/4)
WVNs : ceil(K_t/4) × N_t
          |
          | group one IVN with up to AH WVNs
          v
VN groups:
  VG(m_t,k_g,n_t)
          |
          | merge across all m_t
          v
Combined VN groups:
  CG(k_g,n_t)
          |
          | map onto AW columns of the 4x4 PE array
          v
Invocation 1: CG(0,0), CG(0,1), CG(1,0), CG(1,1)
Invocation 2: CG(2,0)x2, CG(2,1)x2
          |
          | choose IVN / WVN / OVN layouts
          | so concurrent accesses use different addr_col
          v
Conflict-free execution on the 4x4 PE array
          |
          v
Final output tile O_t[M_t,N_t]
          |
          v
Search IVN layout to avoid bank conflicts (if it ever find one choice, stop)
          |
          v
Search WVN layout to avoid bank conflicts (if it ever find one choice, stop)
          |
          v
Search OVN layout to avoid bank conflicts (if it ever find one choice, stop)
          |
          v  (if no valid layout found)
Fallback: retry from step 2) with alternative design choices
  - WVN column stride: "block" (default, s_r=1, s_c=AH) or "strided" (s_r=n_col_types, s_c=1)
  - IVN distribution: "interleaved" (default, s_m=n_rep) or "consecutive" (s_m=1)
  Priority: (block,interleaved) → (block,consecutive) → (strided,interleaved) → (strided,consecutive)

Note: The example above uses the default design choices (block WVN column stride,
interleaved IVN distribution). The strided WVN column stride changes how WVN columns
are assigned across PE rows — instead of consecutive columns within each N-subgroup
(s_r=1, s_c=AH), it interleaves columns across N-subgroups (s_r=n_col_types, s_c=1).
This changes the concurrent bank access pattern and can resolve conflicts that the
default block pattern cannot, particularly for rectangular arrays (AW >> AH).
