# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import re
import functools
import yaml
import math
import numpy as np
import os, inspect, sys

bert = [
    #  M,    N,    K
    (512,  768,  768),
    (512,  768, 3072),
    (512, 3072,  768),
]

###########################
# User Specified Inputs
###########################

available_lines_glb_buf = 1024  # number of lines within a single on-chip global buffer
size_of_line_glb_buf = 32       # number of data a single on-chip global buffer line contains
size_of_line_offchip_dram = 32  # number of data a single DRAM line contains

weight_related_ranks = ['N', "K"]
weight_auto_expanded_ranks = ['N', "K"]

iacts_related_ranks = ['M', "K"]
iacts_auto_expanded_ranks = ['M', "K"]

oacts_related_ranks = ["M", "N"]
oacts_auto_expanded_ranks = ["M", "N"]

architect_namelist = ["eyeriss", "sigma", "vector_256"]

arch_yaml_dict = {
    "eyeriss": "/home/ubuntu/squareloop/benchmarks/arch_designs/eyeriss_like/arch/eyeriss_like.yaml",
    "sigma": "/home/ubuntu/squareloop/benchmarks/arch_designs/simba_like/arch/simba_like.yaml",
    "vector_256": "/home/ubuntu/squareloop/benchmarks/arch_designs/vector_256.yaml",
}

# We use P/Q in the timeloop to represent W/H.
# Need to make sure each line of iActs, weights, oActs won't exceed 16 elements.
# This definition is specified only for iActs and oActs.
# When layer shape is smaller than spatial parallelism offered by the layout,
# the layout value will be bounded by the actual layer parallelism
# When spatial layout for weights does not utilize all available parallelism,
# we spread out the R, S dimensions
layout_policy_constraints_dict = {
    #            [M,  N,  K]
    "Mx32":      [32, 1,  1],
    "Nx32":      [1, 32,  1],
    "Kx32":      [1,  1, 32],
    "Mx4Nx8":    [4,  8,  1],
    "Mx4Kx8":    [4,  1,  4],
    "Nx4Kx8":    [1,  4,  8],
    "Mx8Nx4":    [8,  4,  1],
    "Mx8Kx4":    [8,  1,  4],
    "Nx8Kx4":    [1,  8,  4],
}

layout_name_dict = [
    "MNK_Mx32",
    "MNK_Nx32",
    "MNK_Kx32",
    "MNK_Mx4Nx8",
    "MNK_Mx4Kx8",
    "MNK_Nx4Kx8",
    "MNK_Mx8Nx4",
    "MNK_Mx8Kx4",
    "MNK_Nx8Kx4",
]

###########################
# Analysis Logics
###########################

def collect_names(node, depth, levels):
    # Process "local" nodes
    if "local" in node:
        for item in node["local"]:
            cls = item.get("class", "").lower()
            # Include nodes whose class indicates DRAM or storage.
            # Here we check if the class string contains keywords such as "dram",
            # "storage" or "smartbuffer". We also skip names starting with "DummyBuffer".
            if (("DRAM" in cls) or ("dram" in cls) or ("SRAM" in cls) or ("sram" in cls) or ("storage" in cls) or ("smartbuffer" in cls)) or ("regfile" in cls):
                # Remove any bracketed range (e.g. "[0..15]")
                name = re.sub(r'\[.*?\]', '', item["name"]).strip()
                # If the (lowercased) name is in our rename map, use the mapped name.
                levels.setdefault(depth, []).append(name)
    # Recursively process subtrees
    if "subtree" in node:
        for child in node["subtree"]:
            collect_names(child, depth + 1, levels)

def parse_yaml_hierarchy_from_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    levels = {}
    # Start from the top-level subtree under "architecture"
    arch = data.get("architecture", {})
    for node in arch.get("subtree", []):
        collect_names(node, 1, levels)
    
    # Build the final result list.
    # For each depth level, if there is only one name return it as a string;
    # if there are multiple names, return them as a list.
    result = []
    for depth in sorted(levels.keys()):
        names = levels[depth]
        result.append(names)
    return result


def prod(l):
    return functools.reduce(lambda x, y: x * y, l)


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("ERROR: Creating directory. " + directory)
        sys.exit()


def product(cap_dict, ranks):
    """Compute the product of capacities for a list of ranks."""
    prod = 1
    for r in ranks:
        prod *= cap_dict[r]
    return prod


def expand_line_utilization(spatial_capacities_glb_buf, workload_shape, data_related_ranks, data_auto_expanded_ranks):
    """
    Given:
      - spatial_capacities_glb_buf: dict mapping each rank to its initial capacity per line.
      - workload_shape: dict mapping each rank to the total workload available.
      - data_related_ranks: list of ranks that define the weight tensor layout.
      - data_auto_expanded_ranks: list of ranks that can be expanded if the line is underutilized.
      - size_of_line_glb_buf: total number of elements that can be stored per on-chip buffer line.
      
    This function checks if the product of the spatial capacities for the data_related_ranks
    exactly fills the line. If not, it increases the capacity for auto-expandable ranks (without
    exceeding the total available workload for that rank) until the product is as close as possible
    to (but not over) size_of_line_glb_buf.
    
    Returns a tuple (final_capacities, final_product) where:
      - final_capacities is a dict with the (possibly expanded) capacity for each weight-related rank.
      - final_product is the product (i.e. total weights per line) after expansion.
    """
    # Initialize the capacities for the weight-related ranks
    current_caps = {r: spatial_capacities_glb_buf[r] for r in data_related_ranks}
    current_product = product(current_caps, data_related_ranks)
    
    print(f"Initial product for {data_related_ranks}: {current_product}")
    print(f"Target line size: {size_of_line_glb_buf}")
    
    # If already fully utilized (or exceeded), no need to expand.
    if current_product >= size_of_line_glb_buf:
        print("No auto-expansion needed.")
        return current_caps, current_product
    
    # Compute extra available for each auto-expandable rank.
    extra_available = {}
    for r in data_auto_expanded_ranks:
        extra_available[r] = workload_shape[r] - spatial_capacities_glb_buf[r]
    
    # Try to expand until either the line is fully utilized or no further expansion is possible.
    while current_product < size_of_line_glb_buf:
        expanded_this_round = False
        # Cycle over auto-expandable ranks
        for r in data_auto_expanded_ranks:
            if extra_available[r] > 0:
                # Compute what the product would be if we add one extra unit for rank r.
                # Because current_product = current_caps[r] * (product of others),
                # we can compute new_product as:
                new_product = (current_product // current_caps[r]) * (current_caps[r] + 1)
                # Only add if we do not exceed the line capacity.
                if new_product <= size_of_line_glb_buf:
                    current_caps[r] += 1
                    extra_available[r] -= 1
                    current_product = product(current_caps, data_related_ranks)
                    expanded_this_round = True
                    print(f"Expanded rank {r}: new capacity = {current_caps[r]}, new product = {current_product}")
                    spatial_capacities_glb_buf[r] = current_caps[r]
                    # If we exactly hit the target, break early.
                    if current_product == size_of_line_glb_buf:
                        break
        # If no rank could be expanded without exceeding the target, exit the loop.
        if not expanded_this_round:
            print("No further expansion possible without exceeding the line size.")
            break


def calculate_lines_required(workload_shape, spatial_capacities_glb_buf, permutation):
    """
    Given:
      - workload_shape: dict mapping each rank (e.g. 'S', 'R', etc.) to its total number of elements.
      - spatial_capacities_glb_buf: dict mapping each rank to the number of elements that can fit in one line.
      - permutation: string specifying the order of ranks (e.g. "SRCQPMNHW")
    Returns:
      A dict mapping each rank to the number of lines required.
    """
    lines_required = {}
    for rank in permutation:
        total = workload_shape.get(rank)
        capacity = spatial_capacities_glb_buf.get(rank)
        if total is None or capacity is None:
            raise ValueError(f"Missing total or capacity for rank '{rank}'")
        lines_required[rank] = math.ceil(total / capacity)
    return lines_required


def allocate_tiles(permutation, total_line_required_glb_buf, available_lines_glb_buf):
    """
    Allocate on-chip lines (tile sizes) per rank under a multiplicative (product) constraint.
    
    For each rank in the given priority order (permutation), we assign an allocation:
      allocated[r] = min( total_line_required_glb_buf[r], floor(available_lines_glb_buf / product_so_far) )
    where product_so_far is the product of allocated lines for all ranks processed so far.
    
    Returns:
      allocated: dict mapping each rank to its allocated number of on-chip lines.
      product_val: the product of the allocated lines (total lines used by one tile/chunk).
    """
    allocated = {}
    product_val = 1
    for r in permutation:
        # Maximum we can allocate for rank r without exceeding the overall on-chip capacity:
        max_possible = available_lines_glb_buf // product_val
        # We want to allocate as many lines as needed by the workload (total_line_required_glb_buf[r]),
        # but cannot exceed max_possible.
        allocated[r] = min(total_line_required_glb_buf[r], max_possible)
        product_val *= allocated[r]
    return allocated, product_val


def compute_chunk_params(permutation, total_line_required_glb_buf, spatial_capacities_glb_buf, available_lines_glb_buf):
    """
    Compute the per-rank chunk size and number of iterations (chunks) given:
      - permutation: a string like "SRCQPMNHW" defining priority.
      - total_line_required_glb_buf: dict mapping each rank to total lines needed to store its workload.
      - spatial_capacities_glb_buf: dict mapping each rank to number of data elements stored in one on-chip line.
      - available_lines_glb_buf: overall number of on-chip lines available (i.e. product constraint).
      
    Returns:
      chunk_size: dict mapping each rank to number of data elements loaded per chunk.
      chunk_iteration: dict mapping each rank to number of chunks needed to cover the workload.
      allocated_tiles: dict showing allocated on-chip lines (tile sizes) per rank.
      product_val: product of allocated lines (total lines used by one chunk).
    """
    allocated_tiles, product_val = allocate_tiles(permutation, total_line_required_glb_buf, available_lines_glb_buf)
    
    # Compute chunk size per rank: the number of data elements loaded for that rank in one chunk.
    chunk_size = {r: allocated_tiles[r] * spatial_capacities_glb_buf[r] for r in permutation}
    
    # Compute how many chunks (iterations) are needed per rank:
    # If the on-chip tile for that rank holds allocated_tiles[r] lines, but the workload requires total_line_required_glb_buf[r]
    # lines, then we need:
    chunk_iteration = {r: math.ceil(total_line_required_glb_buf[r] / allocated_tiles[r]) for r in permutation}
    
    return chunk_size, chunk_iteration, allocated_tiles, product_val


def generate_layout(file_path, layout_policy, workload_bounds, memory_hierarchy):
    workload_shape = {
        "M": workload_bounds[0],
        "N": workload_bounds[1],
        "K": workload_bounds[2]}
    print(f"workload_shape:\n{workload_shape}")

    M, N, K = layout_policy_constraints_dict[
        layout_policy.split("_")[-1]
    ]

    ####################
    # on-chip global buffer spatial layout configuration
    ####################
    spatial_capacities_glb_buf = {
        "M": min(M, workload_shape['M']),  # M_spatial
        "N": min(N, workload_shape['N']),  # N_spatial
        "K": min(K, workload_shape['K']),  # K_spatial
    }

    expand_line_utilization(spatial_capacities_glb_buf, workload_shape, weight_related_ranks, weight_auto_expanded_ranks)
    expand_line_utilization(spatial_capacities_glb_buf, workload_shape, iacts_related_ranks, iacts_auto_expanded_ranks)
    expand_line_utilization(spatial_capacities_glb_buf, workload_shape, oacts_related_ranks, oacts_auto_expanded_ranks)
    print(f"after expansion spatial_capacities_glb_buf:\n{spatial_capacities_glb_buf}")

    # Change ranks spatial value to higher if it does not fully utilize the capacity of a single line.
    permutation = layout_policy.split("_")[0]
    total_line_required_glb_buf = calculate_lines_required(workload_shape, spatial_capacities_glb_buf, permutation)
    print(f"total_line_required_glb_buf:\n{total_line_required_glb_buf}")

    chunk_size, chunk_iteration, allocated_lines, product_val = compute_chunk_params(permutation, total_line_required_glb_buf, spatial_capacities_glb_buf, available_lines_glb_buf)
    print(f"chunk_size:\n{chunk_size}\nchunk_iteration:\n{chunk_iteration}\nallocated_lines:\n{allocated_lines}")
    print(f"on_chip global buffer utilization={product_val/available_lines_glb_buf*100:0.2f}%, used #lines={product_val}")
    # on-chip buffer temporal layout configuration
    # total size / spatial line size based on the order.
    # assuming each data space equally divide the overall on-chip buffer line.
    # In case some data space do not fully utilize all on-chip buffer line,
    # The size would be used for other data spaces with remaining shape
    # for iActs

    ####################
    # off-chip DRAM spatial layout configuration
    ####################
    spatial_capacities_dram = {
        "M": min(M, workload_shape['M']),  # M_spatial
        "N": min(N, workload_shape['N']),  # N_spatial
        "K": min(K, workload_shape['K']),  # K_spatial
    }

    expand_line_utilization(spatial_capacities_dram, workload_shape, weight_related_ranks, weight_auto_expanded_ranks)
    expand_line_utilization(spatial_capacities_dram, workload_shape, iacts_related_ranks, iacts_auto_expanded_ranks)
    expand_line_utilization(spatial_capacities_dram, workload_shape, oacts_related_ranks, oacts_auto_expanded_ranks)
    print(f"after expansion spatial_capacities_dram:\n{spatial_capacities_dram}")

    # Change ranks spatial value to higher if it does not fully utilize the capacity of a single line.
    permutation = layout_policy.split("_")[0]
    total_line_required_dram = calculate_lines_required(workload_shape, spatial_capacities_dram, permutation)
    print(f"total_line_required_dram:\n{total_line_required_dram}")

    with open(file_path, "w") as f:
        for lvl, name_list in enumerate(memory_hierarchy):
          for name in name_list:
            if lvl == 0:
              f.write("layout:\n")
              f.write(f"  - target: {name}\n")
              f.write("    type: interline\n")
              f.write(f"    factors: M={total_line_required_dram['M']} N={total_line_required_dram['N']} K={total_line_required_dram['K']}\n")
              f.write(f"    permutation: {permutation}\n")
              f.write("\n")
              f.write(f"  - target: {name}\n")
              f.write("    type: intraline\n")
              f.write(f"    factors: M={spatial_capacities_dram['M']} N={spatial_capacities_dram['N']} K={spatial_capacities_dram['K']}\n")
              f.write(f"    permutation: {permutation}\n")
              f.write("\n")
            elif lvl == 1:
              f.write(f"  - target: {name}\n")
              f.write("    type: interline\n")
              f.write(f"    factors: M={allocated_lines['M']} N={allocated_lines['N']} K={allocated_lines['K']}\n")
              f.write(f"    permutation: {permutation}\n")
              f.write("\n")
              f.write(f"  - target: {name}\n")
              f.write("    type: intraline\n")
              f.write(f"    factors: M={spatial_capacities_glb_buf['M']} N={spatial_capacities_glb_buf['N']} K={spatial_capacities_glb_buf['K']}\n")
              f.write(f"    permutation: {permutation}\n")
              f.write("\n")
            elif lvl == 2:
              f.write(f"  - target: {name}\n")
              f.write("    type: interline\n")
              f.write(f"    factors: M={1} N={1} K={1}\n")
              f.write(f"    permutation: {permutation}\n")
              f.write("\n")
              f.write(f"  - target: {name}\n")
              f.write("    type: intraline\n")
              f.write(f"    factors: M={1} N={1} K={1}\n")
              f.write(f"    permutation: {permutation}\n")
              f.write("\n")
            else:
                raise ValueError("memory level not configured.")

def python_call():
    
    this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
    this_directory = os.path.dirname(this_file_path)
    create_folder(this_directory)

    sys.path.append(this_directory)
    for arch_name in architect_namelist:
      memory_hierarchy = parse_yaml_hierarchy_from_file(arch_yaml_dict[arch_name])
      assert(len(memory_hierarchy) == 3)
      # construct problem shapes for each layer
      for layout_policy in layout_name_dict:
          for net_id, gemm_layer in enumerate(bert):
              net_name = "bert"
              if not os.path.exists(os.path.join(this_directory, "..", "layout", net_name)):
                  os.makedirs(os.path.join(this_directory, "..", "layout", net_name))
              problem = gemm_layer
              file_name = f"{arch_name}_{layout_policy}_" + str(net_id + 1) + ".yaml"
              file_path = os.path.abspath(
                  os.path.join(this_directory, "..", "layout", net_name, file_name)
              )
              generate_layout(file_path, layout_policy, problem, memory_hierarchy)


if __name__ == "__main__":
    python_call()
    # print(parse_yaml_hierarchy_from_file(arch_yaml_dict["sigma"]))
    # print(parse_yaml_hierarchy_from_file(arch_yaml_dict["eyeriss"]))
    # print(parse_yaml_hierarchy_from_file(arch_yaml_dict["vector_256"]))
