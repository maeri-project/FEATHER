"""MINISA -- FEATHER+ ISA toolchain.

6-Stage Search Pipeline (search.py)
------------------------------------
Stage 1 (Tile):    Enumerate legal (Mt, Kt, Nt) tiling choices
Stage 2 (Lower):   Lower tile into IVN/WVN/OVN counts with per-VN sizes
Stage 3 (Group):   Form VN groups VG(m_t, k_g, n_t)
Stage 4 (Combine): Combine groups sharing WVNs into combined columns
Stage 5 (Map):     Derive ExecuteMapping params with design choices
                     (broadcast, interleaved, contiguous patterns)
Stage 6 (Layout):  Search bank-conflict-free buffer layouts

VN abstraction:    vn.py    (Steps 1-3: workload -> VN tiles -> VN groups)
Trace emission:    trace.py (generate_trace_gemm, generate_trace_conv)
Config expansion:  to_config.py (convert_trace_to_config)
Inter-layer:       search.py (multi_layer_search)

Supporting modules: config.py, layout.py, isa.py, cycles.py, workload.py
Integration: evaluate.py, analyze.py
Visualization: minisa_gui.py
"""

from .config import (
    FeatherPlusConfig, TraceBundle, TraceStateTracker,
    CycleBreakdown, make_cfg, ceil_div,
    canonical_n_ivn, canonical_n_wvn, canonical_n_ovn,
    canonical_n_vndp, canonical_n_vn_groups,
)
from .isa import (
    FeatherPlusParams, MinisaIsaParams,
    minisa_bitwidths, hw_config_widths,
    config_to_hw_params, config_to_isa_params,
)
from .layout import (
    LayoutSpec, TABLE_II_OUTER_TO_INNER,
    choose_layout_W, choose_layout_I, choose_layout_O,
    check_ob_port_conflict, layouts_match, derive_ovn_from_ivn,
)
from .trace import (
    generate_trace_gemm, generate_trace_conv,
    verify_trace, verify_config, estimate_minisa_inst_bytes,
)
from .cycles import estimate_cycles_for_gemm, model_instruction_fetch
from .to_config import (
    ConfigStream, ConfigStreamSummary,
    convert_trace_to_config, compute_config_stream_summary,
    compute_memory_comparison,
)
from .search import (
    co_search_layout_mapping, SearchResult,
    brute_force_layer_search, layout_constrained_search,
    multi_layer_search,
    LayerCandidate, LayerSpec, MultiLayerResult,
    ExecuteMappingParams, ExecuteStreamingParams, derive_execute_streaming,
    # New 6-stage API
    co_search_gemm, GemmSearchResult, SearchCandidate,
    TilingChoice, LoweredTile,
    enumerate_tiling_choices, lower_tile, normalize_dataflow,
)
from .vn import (
    GEMMWorkload, ConvWorkload, LogicalVN, VNTile, VNGroup,
    workload_to_logical_vns, tile_logical_vns, assemble_vn_groups,
    lower_workload_to_vn_summary, choose_tile_sizes,
)
from .evaluate import run_evaluation
from .analyze import run_all_analyses, gpu_tpu_comparison
from .workload import load_workload_csv
