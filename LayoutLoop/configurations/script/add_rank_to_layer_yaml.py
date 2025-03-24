import os
import glob
import yaml


def update_yaml_file(file_path):
    """
    Load a YAML file with the new structure, add a 'ranks' definition to each data_space
    under problem/shape/data_spaces based on the data_space name, and write it back.
    """
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Ensure we have the expected top-level keys.
    if not isinstance(data, dict) or "problem" not in data:
        print(f"File {file_path} does not have a 'problem' key.")
        return
    
    problem = data["problem"]
    if "shape" not in problem:
        print(f"File {file_path} does not have a 'shape' key under 'problem'.")
        return
    shape = problem["shape"]
    if "data_spaces" not in shape:
        print(f"File {file_path} does not have a 'data_spaces' key under 'problem/shape'.")
        return
    
    data_spaces = shape["data_spaces"]
    
    # Mapping from data_space name to desired ranks.
    mapping = {
        "Weights": ["C", "M", "R", "S"],
        "Inputs": ["N", "C", "H", "W"],
        "Outputs": ["N", "M", "Q", "P"]
    }
    
    # Process each data space and add the "ranks" key if the name matches.
    for ds in data_spaces:
        name = ds.get("name")
        if name in mapping:
            ds["ranks"] = mapping[name]
    
    # Write the updated YAML content back to the file.
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"Updated file: {file_path}")

def main():
    # Set the directory containing the YAML files.
    # Replace '<path>' with the actual directory path.
    path = "/home/ubuntu/squareloop/benchmarks/layer_shapes/CONV"
    all_sub_dir = os.listdir(path)
    # Find all YAML files in the specified directory.
    for sub_dir in all_sub_dir:
      yaml_files = glob.glob(os.path.join(path, sub_dir, "*.yaml"))
      if not yaml_files:
          print("No YAML files found in the specified directory.")
          return

      for file_path in yaml_files:
          update_yaml_file(file_path)
    
    print("Done updating YAML files.")

if __name__ == '__main__':
    main()
