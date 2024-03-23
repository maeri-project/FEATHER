set TOP_DESIGN feather_top

	set source_dir ../RTL
set lib_setup_dir {../lib_setup}

echo "\nSet Library\n"
#set Library "rtl";

set MY_LIB WORK
define_design_lib $MY_LIB -path ./work

# Update the dc_setup.tcl files if you want to choose a different library
source -verbose -echo ${lib_setup_dir}/dc_setup_svt.tcl

analyze [glob ${source_dir}/*v] -format sverilog -work $MY_LIB 

read_file ${source_dir}/${TOP_DESIGN}.v
analyze -work WORK -f sverilog ${source_dir}/${TOP_DESIGN}.v

set current_design $TOP_DESIGN

# convert to technology independent logic (GTECH)
elaborate ${TOP_DESIGN}

# link design
link


set compile_seqmap_propagate_constants false
set compile_seqmap_propagate_high_effort false

# source constraints
#source ../cons/design_cons_multi_clk.tcl
source ../cons/design_cons.tcl
source ../cons/defaults.tcl

set_fix_multiple_port_nets -all -buffer_constants [get_designs *]
set_dont_touch [get_cells -hier UI_*] true
# get_dont_touch_cells
# remove_attribute [get_cells -hier *UI_*] dont_touch
# compile design
#compile_ultra
#compile_ultra -no_autoungroup
compile_ultra -no_autoungroup -no_boundary_optimization

change_names -rules verilog -verbose -hier

# write output files
write_file -hierarchy -format verilog -output ../outputs/${TOP_DESIGN}.g.v
write_file -hierarchy -format ddc -output ../outputs/${TOP_DESIGN}.ddc
write_sdf ../outputs/${TOP_DESIGN}.sdf
write_sdc ../outputs/${TOP_DESIGN}.sdc

# write reports
report_timing > ../reports/${TOP_DESIGN}_timing.rpt
report_area -designware > ../reports/${TOP_DESIGN}_dw_area.rpt
report_area -hierarchy > ../reports/${TOP_DESIGN}_area.rpt
report_power -hierarchy > ../reports/${TOP_DESIGN}_power.rpt
exit
# do this to remove unconnected ports from netlist - DC does some optimization and removes some ports, nets
# remove_unconnected_ports [get_cells *]
