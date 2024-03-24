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

import functools
import yaml
import numpy as np
import os, inspect, sys
from gemm_layers import *

def prod (l):
    return functools.reduce(lambda x, y: x*y, l)

# We use P/Q in the timeloop to represent W/H.
layout_policy_constraints_dict = {
    "SIGMA_MK_M32":  [1,0,0],  # SIGMA
    "SIGMA_MK_K32":  [0,0,1],  # SIGMA
    "SIGMA_MK_M4K8":  [1,0,1],  # SIGMA
    "eyeriss_like": [1,0,0], 
    "simba_like": [1,0,0], 
}

layout_policy_list_dict = {
    "SIGMA_MK_M32":  "MK_M32",  # SIGMA
    "SIGMA_MK_K32":  "MK_K32",  # SIGMA
    "SIGMA_MK_M4K8":  "MK_M4K8",  # SIGMA
    "eyeriss_like": "MK_M32", 
    "simba_like": "MK_M32", 
}

layout_permutation_constraints_dict = {
    "SIGMA_MK_M32":  "MK",  # SIGMA
    "SIGMA_MK_K32":  "MK",  # SIGMA
    "SIGMA_MK_M4K8":  "MK",  # SIGMA
    "eyeriss_like": "MK", 
    "simba_like": "MK", 
}

# num_line, line_size, reg_line_size
blocksize_dict = {
    "SIGMA_MK_M32":  [8, 32, 1],  # SIGMA
    "SIGMA_MK_K32":  [8, 32, 1],  # SIGMA
    "SIGMA_MK_M4K8": [8, 32, 1],  # SIGMA
    "eyeriss_like": [8, 32, 1], 
    "simba_like": [8, 32, 1], 
}

net_name = "bert"
# arch_name_list = ["template", "eyeriss", "simba", "edge_128", "edge_256", "edge_512", "edge_256_zcu104", "cloud_1024", "cloud_2048_alveoU50", "cloud_4096"]
layout_policy_list = ["MK_M32", "MK_K32", "MK_M4K8"]
arch_name_list = ["SIGMA_MK_M32", "SIGMA_MK_K32", "SIGMA_MK_M4K8", "eyeriss_like", "simba_like"]

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('ERROR: Creating directory. ' + directory)
        sys.exit()

def generate_layout(file_path, layout_policy, arch_name, workload_bounds):
    # if(use_specific_design): # specify a SotA accel instead of the general accel template
    dim_spatial = layout_policy_constraints_dict[arch_name]
    dim_permutation = layout_permutation_constraints_dict[arch_name]
    mem_block_size, line_size, reg_block_size = blocksize_dict[arch_name]
    m,n,k = workload_bounds
    k_spatial = line_size
    k_temporal = np.ceil(k/k_spatial)
    m_spatial, n_spatial, m_temporal, n_temporal = 1, 1, m, n

    # else:
    #     dim_spatial = layout_policy_constraints_dict[layout_policy]
    #     dim_permutation = layout_permutation_constraints_dict[layout_policy]
    #     mem_block_size, line_size, reg_block_size = blocksize_dict[arch_name]

    #     m,n,k = workload_bounds
    #     m_spatial, m_temporal = 0, 0
    #     n_spatial, n_temporal = 0, 0
    #     k_spatial, k_temporal = 0, 0

    if(layout_policy=="MK_M32"):
        m_spatial = line_size
        m_temporal = np.ceil(m/m_spatial)
        n_spatial, k_spatial, n_temporal, k_temporal = 1, 1, n, k
        
    elif(layout_policy=="MK_N32"):
        n_spatial = line_size
        n_temporal = np.ceil(n/n_spatial)
        m_spatial, k_spatial, m_temporal, k_temporal = 1, 1, m, k
        
    elif(layout_policy=="MK_K32" ):
        k_spatial = line_size
        k_temporal = np.ceil(k/k_spatial)
        m_spatial, n_spatial, m_temporal, n_temporal = 1, 1, m, n
        
    elif(layout_policy=="MK_M4K8"):
        m_spatial = 4
        m_temporal = 1
        k_spatial = 8
        k_temporal = np.ceil(k/k_spatial)
        n_spatial, n_temporal = 1, n

    # Main Memory
    with open(file_path, "w") as f:
        f.write("layout:\n")
        f.write("  - target: MainMemory\n")
        f.write("    type: temporal\n")
        f.write(f"    factors: M={int(m)} N={1} K={int(k)}\n")
        f.write(f"    permutation: {dim_permutation}\n")
        f.write("\n")
        f.write("  - target: GlobalBuffer\n")
        f.write("    type: temporal\n")
        f.write(f"    factors: M={int(m_temporal)} N={1} K={int(k_temporal)}\n")
        f.write(f"    permutation: {dim_permutation}\n")
        f.write("\n")
        f.write("  - target: GlobalBuffer\n")
        f.write("    type: spatial\n")
        f.write(f"    factors: M={int(m_spatial)} N={1} K={int(k_spatial)}\n")
        f.write(f"    permutation: {dim_permutation}\n")
        f.write("\n")
        f.write("  - target: RegisterFile\n")
        f.write("    type: temporal\n")
        f.write(f"    factors: M={1} N={1} K={1}\n")
        f.write(f"    permutation: {dim_permutation}\n")
        f.write("\n")      
    
def cli_launch():
    this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
    this_directory = os.path.dirname(this_file_path)

    sys.path.append(this_directory)

    if len(sys.argv) < 2:
        print('Usage: python3 construct_layout.py <network_name> <layout_policy> <arch_name>')
        sys.exit(0)
    net_name = sys.argv[1]
    layout_policy = sys.argv[2]
    arch_name = sys.argv[3]
    
    layout_template_path = os.path.join(this_directory, 'layout_sample.yaml')

    gemm_layers = net_dim_list[0]
    for net_name_id, net_name_str in net_dim_list:
        if net_name_str == net_name:
            gemm_layers = net_dim_list[net_name_id]

    # construct problem shapes for each layer
    for i in range(0, len(gemm_layers)):
        problem = gemm_layers[i]
        file_name = arch_name + "_" + layout_policy + '_' + str(i+1) + '.yaml'
        file_path = os.path.abspath(os.path.join(this_directory, '..', 'layout', net_name, file_name))
        generate_layout(file_path, layout_policy, arch_name, problem)

def python_call():
    this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
    this_directory = os.path.dirname(this_file_path)
    create_folder(this_directory)

    sys.path.append(this_directory)
    # construct problem shapes for each layer
    for arch_name in arch_name_list:
        layout_policy = layout_policy_list_dict[arch_name]
        for net_id, gemm_layers in enumerate(net_dim_list):
            for i in range(0, len(gemm_layers)):
                problem = gemm_layers[i]
                net_name = net_name_list[net_id]
                if "SIGMA" in arch_name:
                    file_name = "SIGMA_" + layout_policy + '_' + str(i+1) + '.yaml'
                else:
                    file_name = arch_name + "_" + layout_policy + '_' + str(i+1) + '.yaml'
                file_path = os.path.abspath(os.path.join(this_directory, '..', 'layout', net_name, file_name))
                generate_layout(file_path, layout_policy, arch_name, problem)


if __name__=="__main__":
    # cli_launch() # if wanna use from command line.
    python_call()