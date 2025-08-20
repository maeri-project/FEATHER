import subprocess
import shutil
import numpy as np

import os, inspect, sys

########### Must Change
# arch_prefix = "eyeriss" # victory
# arch_prefix = "simba"
# arch_prefix = "sigma"
# arch_prefix = "vpuv4"
# arch_prefix = "vpuv5e"
# arch_prefix = "mxu"
# arch_prefix = "mxu_inf_off_chip"
# arch_prefix = "systolic_array"
arch_dict_list = ["mxu_128"]#, "mxu_inf_off_chip"]
########### Must Change

map_policy_dict = {
    "gemmini":  "../mapper/mapper.yaml",
    "eyeriss": "../mapper/mapper_eyeriss.yaml",
    "simba": "../mapper/mapper.yaml",
    "sigma": "../mapper/mapper_sigma.yaml",
    "systolic_array": "../mapper/mapper_systolic_array.yaml",
    "vpuv4": "../mapper/mapper_vpu.yaml",
    "vpuv6e": "../mapper/mapper_vpu.yaml",
    "vpuv6e_inf_off_chip": "../mapper/mapper_vpu.yaml",
    "mxu_128": "../mapper/mapper_mxu.yaml",
    "mxu_inf_off_chip": "../mapper/mapper_mxu.yaml",
}

map_constraint_dict = {
    "gemmini": "../arch_designs/gemmini_like/mapspace.yaml",
    "eyeriss": "../arch_designs/eyeriss_like/constraints/*",
    "simba": "../arch_designs/simba_like/constraints/*",
    "sigma": "",
    "systolic_array": "../arch_designs/systolic_constraint/mapspace_XY_OS.yaml",
    "vpuv4": "",
    "vpuv6e": "",
    "vpuv6e_inf_off_chip": "",
    "mxu": "",
    "mxu_inf_off_chip": "",
}

gemm_map_constraint_dict = {
    "gemmini": "",
    "eyeriss": "../arch_designs/eyeriss_like/constraints_gemm/*",
    "simba": "../arch_designs/simba_like/constraints_gemm/*",
    "sigma": "",
    "systolic_array": "",
    "vpuv4": "",
    "vpuv6e": "",
    "vpuv6e_inf_off_chip": "",
    "mxu_128": "",
    # "mxu": "../arch_designs/tpuv6e_like/mxu/constraints_gemm/eyeriss_like_arch_constraints.yaml ../arch_designs/tpuv6e_like/mxu/constraints_gemm/eyeriss_like_map_constraints.yaml",
    "mxu_inf_off_chip": "",
}

depthwise_map_constraint_dict = {
    "gemmini": "",
    "eyeriss": "../arch_designs/eyeriss_like/constraints_depthwise/*",
    "simba": "../arch_designs/simba_like/constraints_depthwise/*",
    "sigma": "",
    "systolic_array": "",
    "vpuv4": "",
    "vpuv6e": "",
    "vpuv6e_inf_off_chip": "",
    "mxu_128": "",
    "mxu_inf_off_chip": "",
}

arch_dict = {
    "gemmini": "../arch_designs/gemmini.yaml",
    "eyeriss": "../arch_designs/eyeriss_like/arch/eyeriss_like.yaml ../arch_designs/eyeriss_like/arch/components/*",
    "simba": "../arch_designs/simba_like/arch/simba_like.yaml ../arch_designs/simba_like/arch/components/*",
    "sigma": "../arch_designs/vector_256.yaml",
    "systolic_array": "../arch_designs/vector_256.yaml",
    "vpuv4": "../arch_designs/tpuv4_like/vpu/vpu_like.yaml",
    "vpuv6e": "../arch_designs/tpuv6e_like/vpu/vpu_like.yaml",
    "vpuv6e_inf_off_chip": "../arch_designs/tpuv6e_like/vpu/vpu_inf_off_chip.yaml",
    "mxu_128": "../arch_designs/tpuv4_like/mxu/mxu_like.yaml",
    "mxu_inf_off_chip": "../arch_designs/tpuv4_like/mxu/mxu_inf_off_chip.yaml",
}


model_name_list = ["gemm"]
# model_name_list = ["resnet18", "mobv3", "bert", "vectorized", "gemm"]


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
        "bert": 3,
        "vectorized": 11,
        "gemm": 4,
    }

    depthwise_layer_num = {
        "resnet18": [],
        "mobv3": [2, 5, 8, 11, 16, 21, 26, 29, 32, 35, 38, 43, 48, 53, 58],
        "bert": [],
        "vectorized": [],
        "gemm": [],
    }

    model_name_dict= {
        "mobv3": "mobilenet_v3_large",
        "resnet18": "resnet18",
        "bert": "bert",
        "vectorized": ["/home/ubuntu/FEATHER/LayoutLoop/configurations/layer_shapes/vectorized/vectorized_M16384_K1_uint64.yaml",
                       "/home/ubuntu/FEATHER/LayoutLoop/configurations/layer_shapes/vectorized/vectorized_M4096_K32_uint64.yaml",
                       "/home/ubuntu/FEATHER/LayoutLoop/configurations/layer_shapes/vectorized/vectorized_M16384_K32_bool.yaml",
                       "/home/ubuntu/FEATHER/LayoutLoop/configurations/layer_shapes/vectorized/vectorized_M8192_K1_uint64.yaml",
                       "/home/ubuntu/FEATHER/LayoutLoop/configurations/layer_shapes/vectorized/vectorized_M16384_K32_uint64.yaml",
                       "/home/ubuntu/FEATHER/LayoutLoop/configurations/layer_shapes/vectorized/vectorized_M8192_K32_bool.yaml",
                       "/home/ubuntu/FEATHER/LayoutLoop/configurations/layer_shapes/vectorized/vectorized_M16384_K33_uint64.yaml",
                       "/home/ubuntu/FEATHER/LayoutLoop/configurations/layer_shapes/vectorized/vectorized_M8192_K32_uint64.yaml",
                       "/home/ubuntu/FEATHER/LayoutLoop/configurations/layer_shapes/vectorized/vectorized_M4096_K32_bool.yaml",
                       "/home/ubuntu/FEATHER/LayoutLoop/configurations/layer_shapes/vectorized/vectorized_M8192_K33_uint64.yaml",
                       "/home/ubuntu/FEATHER/LayoutLoop/configurations/layer_shapes/vectorized/vectorized_M4096_K32_uint32.yaml"],
         "gemm":      ["/home/ubuntu/FEATHER/LayoutLoop/configurations/layer_shapes/gemm/gemm_M16384_N132_K128.yaml",
                       "/home/ubuntu/FEATHER/LayoutLoop/configurations/layer_shapes/gemm/gemm_M16384_N32_K1.yaml",
                       "/home/ubuntu/FEATHER/LayoutLoop/configurations/layer_shapes/gemm/gemm_M8192_N132_K128.yaml",
                       "/home/ubuntu/FEATHER/LayoutLoop/configurations/layer_shapes/gemm/gemm_M8192_N32_K1.yaml"],
    }
    for arch_prefix in arch_dict_list:
      for model_name in model_name_list:
          for layer_id in range(0, layer_num[model_name]-1):
              # Run the command and capture its output
              if model_name == "gemm":
                print(f"source ~/.setup.sh && timeloop-mapper {arch_dict[arch_prefix]} {map_policy_dict[arch_prefix]} {gemm_map_constraint_dict[arch_prefix]} {model_name_dict[model_name][layer_id]}")
                command_output = subprocess.run([f"source ~/.setup.sh && timeloop-mapper {arch_dict[arch_prefix]} {map_policy_dict[arch_prefix]} {gemm_map_constraint_dict[arch_prefix]} {model_name_dict[model_name][layer_id]}"], shell=True, check=True, capture_output=True, text=True, executable="/bin/bash") # Ensure using bash if needed
              elif model_name == "vectorized":
                print(f"source ~/.setup.sh && timeloop-mapper {arch_dict[arch_prefix]} {map_policy_dict[arch_prefix]} {model_name_dict[model_name][layer_id]}")
                command_output = subprocess.run([f"source ~/.setup.sh && timeloop-mapper {arch_dict[arch_prefix]} {map_policy_dict[arch_prefix]} {model_name_dict[model_name][layer_id]}"], shell=True, check=True, capture_output=True, text=True, executable="/bin/bash") # Ensure using bash if needed
              elif model_name == "bert":
                command_output = subprocess.run([f"source ~/.setup.sh && timeloop-mapper {arch_dict[arch_prefix]} {map_policy_dict[arch_prefix]} {gemm_map_constraint_dict[arch_prefix]} ../layer_shapes/{model_name}/{model_name_dict[model_name]}_{layer_id}.yaml"], shell=True, check=True, capture_output=True, text=True, executable="/bin/bash") # Ensure using bash if needed
              else:
                if model_name == "mobv3" and layer_id in depthwise_layer_num[model_name]:
                  command_output = subprocess.run([f"source ~/.setup.sh && timeloop-mapper {arch_dict[arch_prefix]} {map_policy_dict[arch_prefix]} {depthwise_map_constraint_dict[arch_prefix]} ../layer_shapes/{model_name}/{model_name_dict[model_name]}_{layer_id}.yaml"], shell=True, check=True, capture_output=True, text=True, executable="/bin/bash") # Ensure using bash if needed
                else:
                  command_output = subprocess.run([f"source ~/.setup.sh && timeloop-mapper {arch_dict[arch_prefix]} {map_policy_dict[arch_prefix]} {map_constraint_dict[arch_prefix]} ../layer_shapes/{model_name}/{model_name_dict[model_name]}_{layer_id}.yaml"], shell=True, check=True, capture_output=True, text=True, executable="/bin/bash") # Ensure using bash if needed
              # absolute path
              src_path = os.path.join(work_directory, 'timeloop-mapper.map.yaml')
              dst_path = os.path.join(mapping_directory, f"{model_name}_{arch_prefix}_{layer_id}.mapping.yaml")
              shutil.move(src_path, dst_path)
              src_layout_path = os.path.join(work_directory, 'timeloop-mapper.layout.yaml')
              dst_layout_path = os.path.join(mapping_directory, f"{model_name}_{arch_prefix}_{layer_id}.layout.yaml")
              shutil.move(src_layout_path, dst_layout_path) 
              src_layout_path = os.path.join(work_directory, 'timeloop-mapper.stats.txt')
              dst_layout_path = os.path.join(mapping_directory, f"{model_name}_{arch_prefix}_{layer_id}.stats.txt")
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
