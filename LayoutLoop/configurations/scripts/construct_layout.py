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
from cnn_layers import *

def prod (l):
    return functools.reduce(lambda x, y: x*y, l)

# We use P/Q in the timeloop to represent W/H.
layout_policy_constraints_dict = {
    "SIGMA_HWC_C32":  [1,0,0],  # SIGMA
    "SIGMA_HWC_W32":  [0,1,0],  # SIGMA
    "SIGMA_HWC_H32":  [0,0,1],  # SIGMA
    "SIGMA_HWC_C4W8":  [1,1,0],  # SIGMA
    "SIGMA_HWC_C4H8":  [1,0,1],  # SIGMA
    "SIGMA_HWC_W4H8":  [0,1,1],  # SIGMA
    "SIGMA_HWC_C4W4H2":  [1,1,1],  # SIGMA
    "SIGMA_HWC_W32_W2": [0,1,0],  # SIGMA
    "SIGMA_HWC_H32_H2": [0,0,1],  # SIGMA
    "TPU_like": [1,0,0],
    "simple_output_stationary": [0,1,0],
    "MTIA_like": [1,0,0], 
    "Medusa_like": [1,0,0], 
    "eyeriss_like": [1,0,0], 
    "simba_like": [1,0,0], 
}

layout_permutation_constraints_dict = {
    "SIGMA_HWC_C32":  "SR CQP MN", #R=S=1
    "SIGMA_HWC_W32":  "SR QPC MN", #R=S=1
    "SIGMA_HWC_H32":  "SR PQC MN", #R=S=1
    "SIGMA_HWC_C4W8":  "SR CQP MN", #R=S=1 each spatial=sqrt(spatial_block_size) 8&4
    "SIGMA_HWC_C4H8":  "SR CPQ MN", #R=S=1 each spatial=sqrt(spatial_block_size) 8&4
    "SIGMA_HWC_W4H8":  "SR QPC MN", #R=S=1 each spatial=sqrt(spatial_block_size) 8&4
    "SIGMA_HWC_C4W4H2":  "SR QPC MN", #R=S=1 4&4&2
    "SIGMA_HWC_W32_W2": "QSRPC MN",
    "SIGMA_HWC_H32_H2": "PSRQC MN",
    "TPU_like": "SR CQP MN", #R=S=1
    "simple_output_stationary": "SR QPC MN", #R=S=1
    "MTIA_like": "SR CQP MN", #R=S=1
    "Medusa_like": "SR CQP MN", #R=S=1
    "eyeriss_like": "SR CQP MN", #R=S=1
    "simba_like": "SR CQP MN", #R=S=1
}


# num_line, line_size, reg_line_size
blocksize_dict = {
    "SIGMA_HWC_C32":             [8, 32, 1],
    "SIGMA_HWC_W32":             [8, 32, 1],
    "SIGMA_HWC_H32":             [8, 32, 1],
    "SIGMA_HWC_C4W8":            [8, 32, 1],
    "SIGMA_HWC_C4H8":            [8, 32, 1],
    "SIGMA_HWC_W4H8":            [8, 32, 1],
    "SIGMA_HWC_C4W4H2":          [8, 32, 1],
    "SIGMA_HWC_W32_W2":          [8, 32, 1],
    "SIGMA_HWC_H32_H2":          [8, 32, 1],
    "TPU_like":                  [8, 32, 1],
    "simple_output_stationary":  [8, 32, 1],
    "MTIA_like":                 [8, 32, 1],
    "Medusa_like":               [8, 32, 1],
    "eyeriss_like":              [8, 32, 1],
    "simba_like":                [8, 32, 1],
}

layout_policy_list_dict = {
    "SIGMA_HWC_C32":             "HWC_C32",
    "SIGMA_HWC_W32":             "HWC_W32",
    "SIGMA_HWC_H32":             "HWC_H32",
    "SIGMA_HWC_C4W8":            "HWC_C4W8",
    "SIGMA_HWC_C4H8":            "HWC_C4H8",
    "SIGMA_HWC_W4H8":            "HWC_W4H8",
    "SIGMA_HWC_C4W4H2":          "HWC_C4W4H2",
    "SIGMA_HWC_W32_W2":          "HWC_W32_W2",
    "SIGMA_HWC_H32_H2":          "HWC_H32_H2",
    "TPU_like":                  "HWC_C32",
    "simple_output_stationary":  "HWC_C32",
    "MTIA_like":                 "HWC_C32",
    "Medusa_like":               "HWC_C32",
    "eyeriss_like":              "HWC_C32",
    "simba_like":                "HWC_C32",
}


use_specific_design = True
# FEATHER (arbitrary layout choice), SIGMA (arbitrary layout choice), SIGMA (off-chip reordering), MTIA-like (Transpose), TPU-like (Transpose + Shift), SIGMA-like (HWC_C4W8), SIGMA-like (HWC_C32), Medusa-like (Line Rotation), Eyeriss-like (HWC_C32), NVDLA-like (HWC_C32)
arch_name_list = ["SIGMA_HWC_C32", "SIGMA_HWC_W32", "SIGMA_HWC_H32", "SIGMA_HWC_C4W8", "SIGMA_HWC_C4H8", "SIGMA_HWC_W4H8", "SIGMA_HWC_C4W4H2", "SIGMA_HWC_W32_W2", "SIGMA_HWC_H32_H2", "TPU_like", "simple_output_stationary", "MTIA_like", "Medusa_like", "eyeriss_like", "simba_like"]
# arch_name_list = ["template", "eyeriss", "simba", "edge_128", "edge_256", "edge_512", "edge_256_zcu104", "cloud_1024", "cloud_2048_alveoU50", "cloud_4096", "simple_output_stationary"]

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('ERROR: Creating directory. ' + directory)
        sys.exit()

def generate_layout(file_path, layout_policy, arch_name, workload_bounds):
    if(use_specific_design): # specify a SotA accel instead of the general accel template
        dim_spatial = layout_policy_constraints_dict[arch_name]
        dim_permutation = layout_permutation_constraints_dict[arch_name]
        mem_block_size, line_size, reg_block_size = blocksize_dict[arch_name]

        w, h, c, n, m, s, r, wpad, hpad, wstride, hstride = workload_bounds
        q = int((w - s + 2 * wpad) / wstride) + 1
        p = int((h - r + 2 * hpad) / hstride) + 1
        
        # overall iAct size post-padding calculation.
        w_post_padding =  w + 2 * wpad
        h_post_padding =  h + 2 * hpad

        c_spatial, c_temporal = 0, 0
        w_spatial, w_temporal = 0, 0
        h_spatial, h_temporal = 0, 0
        r_spatial, r_temporal = 1, 1
        s_spatial, s_temporal = 1, 1
        c_spatial = line_size
        c_temporal =  np.ceil(c/c_spatial)
        h_spatial, w_spatial, h_temporal, w_temporal = 1, 1, h_post_padding, w_post_padding
    
    else:
        dim_spatial = layout_policy_constraints_dict[layout_policy]
        dim_permutation = layout_permutation_constraints_dict[layout_policy]
        mem_block_size, line_size, reg_block_size = blocksize_dict[arch_name]

        w, h, c, n, m, s, r, wpad, hpad, wstride, hstride = workload_bounds
        q = int((w - s + 2 * wpad) / wstride) + 1
        p = int((h - r + 2 * hpad) / hstride) + 1
        
        # overall iAct size post-padding calculation.
        w_post_padding =  w + 2 * wpad
        h_post_padding =  h + 2 * hpad

        c_spatial, c_temporal = 0, 0
        w_spatial, w_temporal = 0, 0
        h_spatial, h_temporal = 0, 0
        r_spatial, r_temporal = 1, 1
        s_spatial, s_temporal = 1, 1

        if(layout_policy=="HWC_C32"):
            c_spatial = line_size
            c_temporal = np.ceil(c/c_spatial)
            w_spatial, h_spatial, w_temporal, h_temporal = 1, 1, w_post_padding, h_post_padding
            
        elif(layout_policy=="HWC_W32"):
            w_spatial = line_size
            w_temporal = np.ceil(w_post_padding/w_spatial)
            c_spatial, h_spatial, c_temporal, h_temporal = 1, 1, c, h_post_padding
        
        elif(layout_policy=="HWC_H32"):
            h_spatial = line_size
            h_temporal = np.ceil(h_post_padding/h_spatial)
            c_spatial, w_spatial, c_temporal, w_temporal = 1, 1, c, w_post_padding

        elif(layout_policy=="HWC_C4W8"):
            c_spatial = 4

            w_spatial = np.floor(line_size/c_spatial)
            w_temporal = np.ceil(w_post_padding/w_spatial)
            h_spatial, c_temporal, h_temporal = 1, np.ceil(c/c_spatial), h_post_padding

        elif(layout_policy=="HWC_C4H8"):
            c_spatial = 4
            h_spatial = np.floor(line_size/c_spatial)
            h_temporal = np.ceil(h_post_padding/h_spatial)
            w_spatial, c_temporal, w_temporal = 1, np.ceil(c/c_spatial), w_post_padding

        elif(layout_policy=="HWC_W4H8"):
            w_spatial = 4
            h_spatial = np.floor(line_size/w_spatial)
            h_temporal = np.ceil(h_post_padding/h_spatial)
            c_spatial, c_temporal, w_temporal = 1, c, np.ceil(w_post_padding/w_spatial)

        elif(layout_policy=="HWC_C4W4H2"):
            c_spatial = 4
            w_spatial = 4
            h_spatial = np.floor(line_size/c_spatial/w_spatial)
            h_temporal = np.ceil(h_post_padding/h_spatial)
            c_temporal, w_temporal = np.ceil(c/c_spatial), np.ceil(w_post_padding/w_spatial)

        elif(layout_policy=="HWC_W32_W2"):
            s_spatial = s
            w_spatial = np.floor(line_size/s_spatial)
            w_temporal = np.ceil(w_post_padding/w_spatial)
            c_spatial, h_spatial, c_temporal, h_temporal = 1, 1, c, h_post_padding

        elif(layout_policy=="HWC_H32_H2"):
            r_spatial = r
            h_spatial = np.floor(line_size/r_spatial)
            h_temporal = np.ceil(h_post_padding/h_spatial)
            c_spatial, w_spatial, c_temporal, w_temporal = 1, 1, c, w_post_padding

    if use_specific_design:
        # Main Memory
        with open(file_path, "w") as f:
            f.write("layout:\n")
            f.write("  - target: MainMemory\n")
            f.write("    type: temporal\n")
            f.write(f"    factors: R={int(r)} S={int(s)} P={int(h_post_padding)} Q={int(w_post_padding)} C={int(c)} M={1} N={1}\n")
            f.write(f"    permutation: {dim_permutation}\n")
            f.write("\n")
            f.write("  - target: GlobalBuffer\n")
            f.write("    type: temporal\n")
            f.write(f"    factors: R={int(r_temporal)} S={int(s_temporal)} P={int(h_temporal)} Q={int(w_temporal)} C={int(c_temporal)} M={1} N={1}\n")
            f.write(f"    permutation: {dim_permutation}\n")
            f.write("\n")
            f.write("  - target: GlobalBuffer\n")
            f.write("    type: spatial\n")
            f.write(f"    factors: R={int(r_spatial)} S={int(s_spatial)} P={int(h_spatial)} Q={int(w_spatial)} C={int(c_spatial)} M={1} N={1}\n")
            f.write(f"    permutation: {dim_permutation}\n")
            f.write("\n")
            f.write("  - target: RegisterFile\n")
            f.write("    type: temporal\n")
            f.write(f"    factors: R={1} S={1} P={1} Q={1} C={1} M={1} N={1}\n")
            f.write(f"    permutation: {dim_permutation}\n")
            f.write("\n")      
    else:
        # Main Memory
        with open(file_path, "w") as f:
            f.write("layout:\n")
            f.write("  - target: MainMemory\n")
            f.write("    type: temporal\n")
            f.write(f"    factors: R={int(r)} S={int(s)} P={int(h_post_padding)} Q={int(w_post_padding)} C={int(c)} M={1} N={1}\n")
            f.write(f"    permutation: {dim_permutation}\n")
            f.write("\n")
            f.write("  - target: GlobalBuffer\n")
            f.write("    type: temporal\n")
            f.write(f"    factors: R={int(r_temporal)} S={int(s_temporal)} P={int(h_temporal)} Q={int(w_temporal)} C={int(c_temporal)} M={1} N={1}\n")
            f.write(f"    permutation: {dim_permutation}\n")
            f.write("\n")
            f.write("  - target: GlobalBuffer\n")
            f.write("    type: spatial\n")
            f.write(f"    factors: R={int(r_spatial)} S={int(s_spatial)} P={int(h_spatial)} Q={int(w_spatial)} C={int(c_spatial)} M={1} N={1}\n")
            f.write(f"    permutation: {dim_permutation}\n")
            f.write("\n")
            f.write("  - target: RegisterFile\n")
            f.write("    type: temporal\n")
            f.write(f"    factors: R={1} S={1} P={1} Q={1} C={1} M={1} N={1}\n")
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

    cnn_layers = net_dim_list[0]
    for net_name_id, net_name_str in net_dim_list:
        if net_name_str == net_name:
            cnn_layers = net_dim_list[net_name_id]

    # construct problem shapes for each layer
    for i in range(0, len(cnn_layers)):
        problem = cnn_layers[i]
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
        for net_id, cnn_layers in enumerate(net_dim_list):
            for i in range(0, len(cnn_layers)):
                problem = cnn_layers[i]
                net_name = net_name_list[net_id]
                if "SIGMA" in arch_name:
                    file_name = f"SIGMA_{layout_policy}_" + str(i+1) + '.yaml'
                else:
                    file_name = arch_name + f"_{layout_policy}_" + str(i+1) + '.yaml'
                file_path = os.path.abspath(os.path.join(this_directory, '..', 'layout', net_name, file_name))
                generate_layout(file_path, layout_policy, arch_name, problem)

if __name__=="__main__":
    # cli_launch() # if wanna use from command line.
    python_call()