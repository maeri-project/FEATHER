import subprocess
import shutil
import numpy as np

import os, inspect, sys

########### Must Change
# arch_prefix = "eyeriss" # victory
# arch_prefix = "simba"
# arch_prefix = "sigma"
arch_prefix = "systolic_array"
########### Must Change

map_policy_dict = {
    "gemmini":  "../mapper/mapper.yaml",
    "eyeriss": "../mapper/mapper_eyeriss.yaml",
    "simba": "../mapper/mapper.yaml",
    "sigma": "../mapper/mapper_sigma.yaml",
    "systolic_array": "../mapper/mapper_systolic_array.yaml",
}

map_constraint_dict = {
    "gemmini": "../arch_designs/gemmini_like/mapspace.yaml",
    "eyeriss": "../arch_designs/eyeriss_like/constraints/*",
    "simba": "../arch_designs/simba_like/constraints/*",
    "sigma": "",
    "systolic_array": "../arch_designs/systolic_constraint/mapspace_XY_OS.yaml",
}

gemm_map_constraint_dict = {
    "gemmini": "",
    "eyeriss": "../arch_designs/eyeriss_like/constraints_gemm/*",
    "simba": "../arch_designs/simba_like/constraints_gemm/*",
    "sigma": "",
    "systolic_array": "",
}

depthwise_map_constraint_dict = {
    "gemmini": "",
    "eyeriss": "../arch_designs/eyeriss_like/constraints_depthwise/*",
    "simba": "../arch_designs/simba_like/constraints_depthwise/*",
    "sigma": "",
    "systolic_array": "",
}

arch_dict = {
    "gemmini": "../arch_designs/gemmini.yaml",
    "eyeriss": "../arch_designs/eyeriss_like/arch/eyeriss_like.yaml ../arch_designs/eyeriss_like/arch/components/*",
    "simba": "../arch_designs/simba_like/arch/simba_like.yaml ../arch_designs/simba_like/arch/components/*",
    "sigma": "../arch_designs/vector_256.yaml",
    "systolic_array": "../arch_designs/vector_256.yaml",
}

model_name_list = ["resnet18", "mobv3", "bert"]


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
    energy_list = []
    cycles_list = []


    layer_num = {
        "resnet18": 21,
        "mobv3": 62,
        "bert": 3
    }

    depthwise_layer_num = {
        "resnet18": [],
        "mobv3": [2, 5, 8, 11, 16, 21, 26, 29, 32, 35, 38, 43, 48, 53, 58],
        "bert": [],
    }

    model_name_dict= {
        "mobv3": "mobilenet_v3_large",
        "resnet18": "resnet18",
        "bert": "bert",
    }

    for model_name in model_name_list:
        for layer_id in range(1, layer_num[model_name]+1):
            # Run the command and display its output
            if model_name == "bert":
              command_output = subprocess.run([f"source ~/.setup.sh && timeloop-mapper {arch_dict[arch_prefix]} {map_policy_dict[arch_prefix]} {gemm_map_constraint_dict[arch_prefix]} ../layer_shapes/{model_name}/{model_name_dict[model_name]}_{layer_id}.yaml"], shell=True, check=True, executable="/bin/bash") # Ensure using bash if needed
            else:
              if model_name == "mobv3" and layer_id in depthwise_layer_num[model_name]:
                command_output = subprocess.run([f"source ~/.setup.sh && timeloop-mapper {arch_dict[arch_prefix]} {map_policy_dict[arch_prefix]} {depthwise_map_constraint_dict[arch_prefix]} ../layer_shapes/{model_name}/{model_name_dict[model_name]}_{layer_id}.yaml"], shell=True, check=True, executable="/bin/bash") # Ensure using bash if needed
              else:
                command_output = subprocess.run([f"source ~/.setup.sh && timeloop-mapper {arch_dict[arch_prefix]} {map_policy_dict[arch_prefix]} {map_constraint_dict[arch_prefix]} ../layer_shapes/{model_name}/{model_name_dict[model_name]}_{layer_id}.yaml"], shell=True, check=True, executable="/bin/bash") # Ensure using bash if needed
            # absolute path
            src_path = os.path.join(work_directory, 'timeloop-mapper.map.yaml')
            dst_path = os.path.join(mapping_directory, f"{model_name}_{layer_id}.mapping.yaml")
            shutil.move(src_path, dst_path)
            src_layout_path = os.path.join(work_directory, 'timeloop-mapper.layout.yaml')
            dst_layout_path = os.path.join(mapping_directory, f"{model_name}_{layer_id}.layout.yaml")
            shutil.move(src_layout_path, dst_layout_path)
            src_layout_path = os.path.join(work_directory, 'timeloop-mapper.stats.txt')
            dst_layout_path = os.path.join(mapping_directory, f"{model_name}_{layer_id}.stats.txt")
            shutil.move(src_layout_path, dst_layout_path)

            # Read Utilization, Cycles, and Energy directly from stats file
            utilization = 0.0
            energy_uj = 0.0
            cycles = 0
            with open(dst_layout_path, 'r') as stats_file:
                for line in stats_file:
                    line_stripped = line.strip()
                    if line_stripped.startswith('Utilization:'):
                        # e.g., "Utilization: 18.75%"
                        value_str = line_stripped.split(':', 1)[1].strip()
                        if value_str.endswith('%'):
                            value_str = value_str[:-1]
                        utilization = float(value_str)
                    elif line_stripped.startswith('Cycles:'):
                        # e.g., "Cycles: 25165824"
                        value_str = line_stripped.split(':', 1)[1].strip()
                        cycles = int(value_str)
                    elif line_stripped.startswith('Energy:'):
                        # e.g., "Energy: 152765.82 uJ"
                        value_str = line_stripped.split(':', 1)[1].strip().split()[0]
                        energy_uj = float(value_str)

            utilization_list.append(utilization)
            energy_list.append(energy_uj)
            cycles_list.append(cycles)

    print("utilization:", utilization_list)
    print("energy:", energy_list)
    print("cycles:", cycles_list)

    utilization_array = np.array(utilization_list).transpose()
    pj_commpute_array = np.array(energy_list).transpose()
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
