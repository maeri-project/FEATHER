import os

def alter(file,old_str,new_str):
  """
  replace the string in a file
  :param file:file name
  :param old_str: old string that will be replaced
  :param new_str: new string
  :return:
  """
  file_data = ""
  with open(file, "r", encoding="utf-8") as f:
    for line in f:
      if old_str in line:
        line = new_str
  #      line = line.replace(old_str,new_str)
      file_data += line
  with open(file,"w",encoding="utf-8") as f:
    f.write(file_data)

# step 1 -- set all parameters in need of test
# User needs to modify
parameter_value = [[32, 64, 128, 256, 512], [32, 64, 128, 256, 512], [4, 8, 16, 32, 64], [4, 8, 16, 32, 64], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6] ] # for 1k and 4k PE
# parameter_value = [[32,64,128], [32,64,128], [4,8,16], [4,8,16], [2,3,4], [2,3,4] ] # for number of input data
parameter_name = ["WEIGHTS_SRAM_DATA_WIDTH", "IACTS_SRAM_DATA_WIDTH", "DPE_COL_NUM", "DPE_ROW_NUM", "LOG2_DPE_COL_NUM", "LOG2_DPE_ROW_NUM"]
top_module = "lambda_top"
file_name = "lambda_top.v"

# step 2 -- loop all parameters, start syn and rename the result tar bar.
for i in range(len(parameter_value[0])):
  os.system("make all")
  for j in range(len(parameter_value)):
    alter("../RTL/" + file_name, "parameter " + parameter_name[j], "	parameter " + parameter_name[j] + " = " + str(parameter_value[j][i]) + ",\n")
  #subprocess.run("source ./top.sh")
  #sh.sh('-c', './top.sh')
  os.system("make run") 
  os.system("make comp")
  os.system("mv ../rpt.tar.gz ../" + top_module + str(parameter_name[2]) + str(parameter_value[2][i]) +".tar.gz")
