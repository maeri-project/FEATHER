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

def prod (l):
    return functools.reduce(lambda x, y: x*y, l)


def rewrite_workload_bounds(src, dst, workload_bounds):
    print(workload_bounds)
    m, n, k = workload_bounds

    print('Workload Dimensions:')
    print('  K        =', k)
    print('  M        =', m)
    print('  N        =', n)
    print()

    with open(src, "r") as f:
        config = yaml.load(f, Loader = yaml.SafeLoader)

    config['problem']['instance']['K'] = k
    config['problem']['instance']['M'] = m
    config['problem']['instance']['N'] = n

    with open(dst, "w") as f:
        f.write(yaml.dump(config))

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('ERROR: Creating directory. ' + directory)
        sys.exit()

if __name__=="__main__":

    import os, inspect, sys
    this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
    this_directory = os.path.dirname(this_file_path)

    sys.path.append(this_directory)
    bert = [
        (512, 768, 768),
        (512, 768, 3072),
        (512, 3072, 768)]

    if len(sys.argv) < 2:
        print('Usage: python3 construct_workloads.py <dnn_model_name>')
        sys.exit(0)
    net_name = sys.argv[1]

    # construct appropriate folder and file paths
    create_folder(os.path.abspath(os.path.join(this_directory, '..', 'layer_shapes', net_name)))
    config_abspath = os.path.join(this_directory, 'gemm_sample.yaml')

    # just test that path points to a valid config file.
    with open(config_abspath, 'r') as f:
        config = yaml.load(f, Loader = yaml.SafeLoader)

    # construct problem shapes for each layer
    for i in range(0, len(bert)):
        file_name = net_name + '_' + 'layer' + str(i+1) + '.yaml'
        file_path = os.path.abspath(os.path.join(this_directory, '..', 'layer_shapes', net_name, file_name))
        rewrite_workload_bounds(config_abspath, file_path, bert[i])