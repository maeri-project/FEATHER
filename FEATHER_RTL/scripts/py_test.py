import os

os.system("touch ../rpt.tar.gz")
os.system("mv ../rpt.tar.gz ../" + top_module + str(parameter_value[i]) +".tar.gz")
