set clk_period 2
set clk_name clk
set timing_enable_multiple_clocks_per_reg "true"
create_clock -name $clk_name -add -period ${clk_period} [get_ports {clk}]
create_clock -name ex_clk -add -period 20.1 -waveform {0 10.05} [get_ports ex_clk]

set_clock_groups -asynchronous -name g1 -group {ex_clk} -group {clk}




