import numpy as np
def find_value(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()

    values_at_0_217 = []
    for i in range(len(lines)):
        #if "COL_WISE_O_DATA_BUS" in lines[i]:
        if "LAMBDA_GENVAR_DPE_INST_" in lines[i]:
            # Assuming the next line has values as shown in the example
            # Split the next line into a list of values
            values = lines[i+1].split()
            
            # Assuming the value at "0.217" location is always the fourth value
            value = values[0]
            values_at_0_217.append(float(value))

    return values_at_0_217

file_name = "../../8x8/reports/lambda_top_area.rpt"
values = find_value(file_name)
print(values)
print(np.sum(values))
