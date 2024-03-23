# Apply default drive strengths and typical loads
# for I/O ports
set_load 0.1 [all_outputs]

# check ot see if cell is available in the library
set_driving_cell -lib_cell BUFFD2BWP30P140LVT [all_inputs]
#set_driving_cell -lib_cell BUFFD2BWP30P140 [all_inputs]

# If real clock, set infinite drive strength
if {[sizeof_collection [get_ports clk]] > 0} {
   set_drive 0 clk
}

# Apply default timing constraints for modules
set inputPorts [remove_from_collection [all_inputs] {clk}]
set_input_delay [expr 0.2*$clk_period] $inputPorts -clock $clk_name -add_delay
set_output_delay [expr 0.2*$clk_period] [all_outputs] -clock $clk_name -add_delay
set_clock_uncertainty -setup 0.150 $clk_name

#set_output_delay [exp 0.9*$clk_period] [get_ports pmos_sel_final*] -clock $clk_name

# Set operating conditions
# set_operating_conditions WCCOM 
