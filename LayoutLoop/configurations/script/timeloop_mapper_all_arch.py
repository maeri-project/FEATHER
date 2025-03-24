import subprocess
import shutil
import numpy as np

import os, inspect, sys

########### Must Change
arch_prefix = "eyeriss"
# arch_prefix = "vector_256"
# arch_prefix = "medusa"
# arch_prefix = "simba"
# arch_prefix = "sigma"
########### Must Change

map_policy_dict = {
    "gemmini":  "../mapper/mapper_original_timeloop.yaml",
    "eyeriss": "../mapper/mapper_original_timeloop.yaml",
    "simba": "../mapper/mapper_original_timeloop.yaml",
    "medusa": "../mapper/mapper_original_timeloop.yaml",
    "systolic_array": "../mapper/mapper_original_timeloop.yaml",
}

map_constraint_dict = {
    "gemmini": "../arch_designs/gemmini_like/mapspace.yaml",
    "eyeriss": "../arch_designs/eyeriss_like/constraints/*",
    "simba": "../arch_designs/simba_like/constraints/*",
    "medusa": " ",
    "systolic_array": "../arch_designs/systolic_constraint/mapspace_XY_OS.yaml",
}

gemm_map_constraint_dict = {
    "gemmini": "",
    "eyeriss": "../arch_designs/eyeriss_like/constraints_gemm/*",
    "simba": "../arch_designs/simba_like/constraints_gemm/*",
    "medusa": " ",
    "systolic_array": "",
}

arch_dict = {
    "gemmini": "gemmini.yaml",
    "eyeriss": "eyeriss_like/arch/eyeriss_like.yaml ../arch_designs/eyeriss_like/arch/components/*",
    "simba": "simba_like/arch/simba_like.yaml ../arch_designs/simba_like/arch/components/*",
    "medusa": "medusa.yaml",
    "systolic_array": "../arch_designs/systolic_array.yaml",
}

model_name_list = ["resnet50", "mobv3", "bert"]

layout_policy_list = [
    "SRCQPMNHW_Cx32",
    "SRCQPMNHW_Hx32",
    "SRCQPMNHW_Mx32",
    "SRCQPMNHW_Wx32",
    "SRCQPMNHW_Cx4Hx8",
    "SRCQPMNHW_Cx8Hx4",
    "SRCQPMNHW_Cx8Wx4",
    "SRCQPMNHW_Wx8Hx4",
    "SRCQPMNHW_Cx8Wx2Hx2",
]

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

    utilization_list = []
    pj_per_compute_list = []
    cycles_list = []


    layer_num = {
        "resnet50": 53,
        "mobv3": 62,
        "bert": 3
    }

    model_name_dict= {
        "mobv3": "mobilenet_v3_large",
        "resnet50": "resnet50",
        "bert": "bert",
    }
    
    for model_name in model_name_list:
        for layer_id in range(1, layer_num[model_name]+1):
            # Run the command and capture its output
            if model_name == "bert":
              command_output = subprocess.run([f"source ~/.setup.sh && timeloop-mapper ../arch_designs/{arch_dict[arch_prefix]} {map_policy_dict[arch_prefix]} {gemm_map_constraint_dict[arch_prefix]} ../layer_shapes/{model_name}/{model_name_dict[model_name]}_{layer_id}.yaml"], shell=True, check=True, capture_output=True, text=True, executable="/bin/bash") # Ensure using bash if needed
            else:
              command_output = subprocess.run([f"source ~/.setup.sh && timeloop-mapper ../arch_designs/{arch_dict[arch_prefix]} {map_policy_dict[arch_prefix]} {map_constraint_dict[arch_prefix]} ../layer_shapes/{model_name}/{model_name_dict[model_name]}_{layer_id}.yaml"], shell=True, check=True, capture_output=True, text=True, executable="/bin/bash") # Ensure using bash if needed
            # absolute path
            src_path = os.path.join(work_directory, 'timeloop-mapper.map.yaml')
            dst_path = os.path.join(mapping_directory, f"{model_name}_{layer_id}.yaml")
            shutil.move(src_path, dst_path)
            # Split the output into individual lines
            output_lines = command_output.stdout.strip().split('\n')
            # Extract the values you're interested in
            utilization=0
            pj_per_compute=0
            cycles=0
            for line in output_lines[-6:]:
                if 'Utilization' in line:
                    utilization = float(line.split('Utilization = ')[-1].split(' |')[0])
                    pj_per_compute = float(line.split('| pJ/Compute = ')[-1].split(' |')[0])
                    cycles = int(line.split('| Cycles = ')[-1])

            # Print the values to verify that they were extracted correctly
            utilization_list.append(utilization)
            pj_per_compute_list.append(pj_per_compute)
            cycles_list.append(cycles)

    print(utilization_list)
    print(pj_per_compute_list)
    print(cycles_list)

    utilization_array = np.array(utilization_list).transpose()
    pj_commpute_array = np.array(pj_per_compute_list).transpose()
    cycle_array       = np.array(cycles_list).transpose()

    total_layer_num = 0
    for model_name in model_name_list:
        total_layer_num += layer_num[model_name]


    np.savetxt(os.path.join(work_directory, "utilization.csv"), utilization_array, delimiter=',', fmt='%.2f')
    np.savetxt(os.path.join(work_directory, "pj_commpute.csv"), pj_commpute_array, delimiter=',', fmt='%.2f')
    np.savetxt(os.path.join(work_directory, "cycle.csv"), cycle_array, delimiter=',', fmt='%.2f')

    try:
        interleave_overall_array = np.zeros([cycle_array.shape[0], 3])
        interleave_overall_array[:, 0] = utilization_array
        interleave_overall_array[:, 1] = pj_commpute_array
        interleave_overall_array[:, 2] = cycle_array

        np.savetxt(os.path.join(work_directory, "interleave_layoutloop_search.csv"), interleave_overall_array, delimiter=',', fmt='%.2f')
    except:
        print("need debug")