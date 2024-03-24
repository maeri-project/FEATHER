import subprocess
import shutil
import numpy as np

import os, inspect, sys

import FEATHER.LayoutLoop.configurations.scripts.utils as utils



########### Must Change
arch_name = "edge_256.yaml"
mapping_directory = "mapping_global_ideal_layout"
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

    work_direcotry = os.path.abspath(os.path.join(this_directory, '..', work_directory_name))
    create_folder(work_direcotry)
    os.chdir(work_direcotry)

    slowdown_values_list = []
    utilization_list = []
    pj_per_compute_list = []
    cycles_list = []

    layer_num = {
        "resnet50": 53,
        "mobv3": 62
    }

    model_name_list = ["resnet50", "mobv3"]

    for model_name in model_name_list:
        for layer_id in range(1, layer_num[model_name]+1):
            # Run the command and capture its output
            command_output = subprocess.check_output([f"timeloop-model ../arch_designs/{arch_name} ../{mapping_directory}/{model_name}_global_{layer_id}.yaml ../layer_shapes/{model_name}/{model_name}_layer{layer_id}.yaml"], shell=True, universal_newlines=True)
            # Split the output into individual lines
            output_lines = command_output.strip().split('\n')
            # Extract the values you're interested in
            utilization=0
            pj_per_compute=0
            cycles=0
            utilization = float(output_lines[-1].split('Utilization = ')[-1].split(' |')[0])
            pj_per_compute = float(output_lines[-1].split('| pJ/Compute = ')[-1].split(' |')[0])
            cycles = int(output_lines[-1].split('| Cycles = ')[-1])

            # Print the values to verify that they were extracted correctly
            utilization_list.append(utilization)
            pj_per_compute_list.append(pj_per_compute)
            cycles_list.append(cycles)

    print(utilization_list)
    print(pj_per_compute_list)
    print(cycles_list)

    utilization_array = np.array(utilization_list)
    pj_commpute_array = np.array(pj_per_compute_list)
    cycle_array       = np.array(cycles_list)

    np.savetxt(os.path.join(work_direcotry, "utilization.csv"), utilization_array, delimiter=',', fmt='%.2f')
    np.savetxt(os.path.join(work_direcotry, "pj_commpute.csv"), pj_commpute_array, delimiter=',', fmt='%.2f')
    np.savetxt(os.path.join(work_direcotry, "cycle.csv"), cycle_array, delimiter=',', fmt='%.2f')

    interleave_overall_array = np.zeros([cycle_array.shape[0], 3])
    interleave_overall_array[:, 0] = utilization_array
    interleave_overall_array[:, 1] = pj_commpute_array
    interleave_overall_array[:, 2] = cycle_array

    np.savetxt(os.path.join(work_direcotry, "interleaved_timeloop_evaluation.csv"), interleave_overall_array, delimiter=',', fmt='%.2f')
