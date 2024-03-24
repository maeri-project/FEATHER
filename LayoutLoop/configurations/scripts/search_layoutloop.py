import subprocess
import shutil
import numpy as np

import os, inspect, sys


########### Must Change
arch_name = "edge_256.yaml"
# arch_name = "gemmini.yaml"
arch_prefix = arch_name[:-5] 
mapper_policy = "mapper_original_timeloop.yaml"
########### Must Change





def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('ERROR: Creating directory. ' + directory)
        sys.exit()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python3 search_layout_timeloop.py <work_directory_name>')
        sys.exit(0)
    work_directory_name = sys.argv[1]

    this_file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
    this_directory = os.path.dirname(this_file_path)
    
    work_directory = os.path.abspath(os.path.join(this_directory, '..', work_directory_name))
    create_folder(work_directory)

    mapping_directory = os.path.abspath(os.path.join(work_directory, "mapping_search"))
    create_folder(mapping_directory)

    os.chdir(work_directory)

    slowdown_values_list = []
    utilization_list = []
    pj_per_compute_list = []
    cycles_list = []

    layout_policy_list = ["HWC_C32"]
    # layout_policy_list = ["HWC_C32","HWC_W32","HWC_H32","HWC_C4W8","HWC_C4H8","HWC_W4H8","HWC_C4W4H2","HWC_W32_W2","HWC_H32_H2"]
    
    layer_num = {
        "resnet50": 53,
        "mobv3": 62
    }

    model_name_list = ["resnet50", "mobv3"]

    for layout_policy in layout_policy_list:
        for model_name in model_name_list:
            for layer_id in range(1, layer_num[model_name]+1):
                # Run the command and capture its output
                # command_output = subprocess.check_output([f"timeloop-mapper ../arch_designs/3levelspatial.arch.yaml ../mapper/mapper_test.yaml ../layer_shapes/{model_name}/{model_name}_layer{layer_id}.yaml ../layout/{model_name}/{layout_policy}_{layer_id}.yaml"], shell=True, universal_newlines=True)
                command_output = subprocess.check_output([f"timeloop-mapper ../arch_designs/{arch_name} ../mapper/{mapper_policy} ../layer_shapes/{model_name}/{model_name}_layer{layer_id}.yaml ../layout/{model_name}/{arch_prefix}_{layout_policy}_{layer_id}.yaml"], shell=True, universal_newlines=True)
                # absolute path
                src_path = os.path.join(work_directory, 'timeloop-mapper.map.yaml')
                dst_path = os.path.join(mapping_directory, f"{model_name}_{layout_policy}_{layer_id}.yaml")
                shutil.move(src_path, dst_path)
                # Split the output into individual lines
                output_lines = command_output.strip().split('\n')
                # Extract the values you're interested in
                slowdown_values = 0
                utilization=0
                pj_per_compute=0
                cycles=0
                for line in output_lines[-6:]:
                    if 'stats_.slowdown' in line  and "GlobalBuffer" in line:
                        key, value = line.split('\t')
                        slowdown_values = float(value.split(': ')[-1])
                    elif 'Utilization' in line:
                        utilization = float(line.split('Utilization = ')[-1].split(' |')[0])
                        pj_per_compute = float(line.split('| pJ/Compute = ')[-1].split(' |')[0])
                        cycles = int(line.split('| Cycles = ')[-1])

                # Print the values to verify that they were extracted correctly
                slowdown_values_list.append(slowdown_values)
                utilization_list.append(utilization)
                pj_per_compute_list.append(pj_per_compute)
                cycles_list.append(cycles)

    print(slowdown_values_list)
    print(utilization_list)
    print(pj_per_compute_list)
    print(cycles_list)

    slowdown_array = np.array(slowdown_values_list)
    utilization_array = np.array(utilization_list)
    pj_commpute_array = np.array(pj_per_compute_list)
    cycle_array       = np.array(cycles_list)

    total_layer_num = 0
    for model_name in model_name_list:
        total_layer_num += layer_num[model_name]

    slowdown_array = np.reshape(slowdown_array, (len(layout_policy_list), -1)).transpose()
    utilization_array = np.reshape(utilization_array, (len(layout_policy_list), -1)).transpose()
    pj_commpute_array = np.reshape(pj_commpute_array, (len(layout_policy_list), -1)).transpose()
    cycle_array =  np.reshape(cycle_array, (len(layout_policy_list), -1)).transpose()

    # slowdown_array = reorder_to_given_policy_list(policy_mapping, slowdown_array)
    # utilization_array = reorder_to_given_policy_list(policy_mapping, utilization_array)
    # pj_commpute_array = reorder_to_given_policy_list(policy_mapping, pj_commpute_array)
    # cycle_array = reorder_to_given_policy_list(policy_mapping, cycle_array)
    # print(cycle_array)

    np.savetxt(os.path.join(work_directory, "slowdown.csv"), slowdown_array, delimiter=',', fmt='%.2f')
    np.savetxt(os.path.join(work_directory, "utilization.csv"), utilization_array, delimiter=',', fmt='%.2f')
    np.savetxt(os.path.join(work_directory, "pj_commpute.csv"), pj_commpute_array, delimiter=',', fmt='%.2f')
    np.savetxt(os.path.join(work_directory, "cycle.csv"), cycle_array, delimiter=',', fmt='%.2f')

    interleave_overall_array = np.zeros([cycle_array.shape[0], 4*len(layout_policy_list)])
    for policy_id in range(len(layout_policy_list)):
        interleave_overall_array[:, 4*policy_id  ] = slowdown_array[:, policy_id]
        interleave_overall_array[:, 4*policy_id+1] = utilization_array[:, policy_id]
        interleave_overall_array[:, 4*policy_id+2] = pj_commpute_array[:, policy_id]
        interleave_overall_array[:, 4*policy_id+3] = cycle_array[:, policy_id]

    np.savetxt(os.path.join(work_directory, "interleave_layoutloop_search.csv"), interleave_overall_array, delimiter=',', fmt='%.2f')
