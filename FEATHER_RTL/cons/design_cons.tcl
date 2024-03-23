set clk_name clk
set clk_period 2
set clk_port clk
set half_clk_period [expr $clk_period*0.5]
create_clock -name $clk_name -add -period $clk_period [get_ports $clk_port]
##create_clock -name $clk_name -add -period $clk_period -waveform {0 4} [get_ports $clk_port]
#set timing_enable_multiple_clocks_per_reg "true"
#create_clock -name sclk -add -period 16 -waveform {0 8} [get_ports sclk]
#set_clock_groups -name clockRelation -group {clk} -group {sclk} -asynchronous
##set_multicycle_path -setup 2 -from [get_ports cfg*] -to [get_clocks  clk]
##set_multicycle_path -hold 1 -from [get_ports cfg*] -to [get_clocks clk]
#
set_false_path -from [get_ports {rst_n}]
#set_multicycle_path 2 -setup -to [get_pins u_spi/config30_reg_*_/D] -end 
#set_multicycle_path 1 -hold -to [get_pins u_spi/config30_reg_*_/D] -end 
#set_multicycle_path 2 -setup -from [get_pins u_spi/config*_reg_*_/CK] -start
#set_multicycle_path 2 -hold -from [get_pins u_spi/config*_reg_*_/CK] -start
#
#set_output_delay [expr 0.8*$clk_period] [get_ports pmos_sel_final[*]] -clock $clk_name -add_delay

