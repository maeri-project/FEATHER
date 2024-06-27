/******************************************************************************
Copyright (c) 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*******************************************************************************/
/*
    Top Module:  birrd_simple_cmd_flow_seq
    Data:        Only data width matters.
    Format:      keeping the input format unchange
    Timing:      Sequential Logic
    Reset:       Asynchronized Reset [Low Reset]
    Pipeline:
                 [full pipeline]every stage is a pipeline stage
                                Total latency = # stages (cycle)
                 [2 stage pipeline] 0~LEVEL is the first pipeline stage.
    Dummy Data:  {DATA_WIDTH{1'b0}}

    Function:    Arbitrary Reordering + Arbitrary Reduction

                          1) for DATA PATH

        i_data_bus[0*DATA_WIDTH+:DATA_WIDTH]  -->|¯¯¯|------->|¯¯¯|-------->|¯¯¯|-------->|¯¯¯|-------->|¯¯¯|------->|¯¯¯|-->
        i_data_bus[1*DATA_WIDTH+:DATA_WIDTH]  -->|___|--\ /-->|___|-\    /->|___|-\    /->|___|-\    /->|___|--\ /-->|___|-->
                                        ID:        0     X      4    \  /     8    \  /     12   \  /     16    X      20
        i_data_bus[2*DATA_WIDTH+:DATA_WIDTH]  -->|¯¯¯|--/ \-->|¯¯¯|---\/--->|¯¯¯|---\/--->|¯¯¯|---\/--->|¯¯¯|--/ \-->|¯¯¯|-->
        i_data_bus[3*DATA_WIDTH+:DATA_WIDTH]  -->|___|------->|___|-\ /\ /->|___|-\ /\ /->|___|-\ /\ /->|___|------->|___|-->
                                        ID:        1            5    X  X     9    X  X     13   X  X     17           21
        i_data_bus[4*DATA_WIDTH+:DATA_WIDTH]  -->|¯¯¯|------->|¯¯¯|-/ \/ \->|¯¯¯|-/ \/ \->|¯¯¯|-/ \/ \->|¯¯¯|------->|¯¯¯|-->
        i_data_bus[5*DATA_WIDTH+:DATA_WIDTH]  -->|___|--\ /-->|___|---/\--->|___|---/\--->|___|---/\--->|___|--\ /-->|___|-->
                                        ID:        2     X      6    /  \     10   /  \     14   /  \     18    X      22
        i_data_bus[6*DATA_WIDTH+:DATA_WIDTH]  -->|¯¯¯|--/ \-->|¯¯¯|-/    \->|¯¯¯|-/    \->|¯¯¯|-/    \->|¯¯¯|--/ \-->|¯¯¯|-->
        i_data_bus[7*DATA_WIDTH+:DATA_WIDTH]  -->|___|------->|___|-------->|___|-------->|___|-------->|___|------->|___|-->
                                        ID:        3            7             11           15            19            23
        CONNECTION FUNCTION                          BitReverse    BitReverse    BitReverse   BitReverse    BitReverse             
        CONNECTION GROUP SIZE                            4             8             8             8            4              
  
                          2) for Configuration transmission
                 Note: (1) configurtion also traverse the BENES network to keep pace with data.
                       (2) This is for pre-generated configuration that are generated offline and used online.
                       (3) The o_cmd is for other design which is also configured in the cmd_flow manner.

              COMMAND_WIDTH is the command length for a single switch.
              ROW_COMMAND_WIDTH is specified by the input

        i_cmd[0*ROW_COMMAND_WIDTH+:ROW_COMMAND_WIDTH]  -->|¯¯¯|------>|¯¯¯|------>|¯¯¯|------>|¯¯¯|------>|¯¯¯|------>|¯¯¯|
        i_cmd[1*ROW_COMMAND_WIDTH+:ROW_COMMAND_WIDTH]  -->|___|------>|___|------>|___|------>|___|------>|___|------>|___|
                                                            0           4           8           12          16          20 
        i_cmd[2*ROW_COMMAND_WIDTH+:ROW_COMMAND_WIDTH]  -->|¯¯¯|------>|¯¯¯|------>|¯¯¯|------>|¯¯¯|------>|¯¯¯|------>|¯¯¯|
        i_cmd[3*ROW_COMMAND_WIDTH+:ROW_COMMAND_WIDTH]  -->|___|------>|___|------>|___|------>|___|------>|___|------>|___|
                                                            1           5           9           13          17          21 
        i_cmd[4*ROW_COMMAND_WIDTH+:ROW_COMMAND_WIDTH]  -->|¯¯¯|------>|¯¯¯|------>|¯¯¯|------>|¯¯¯|------>|¯¯¯|------>|¯¯¯|
        i_cmd[5*ROW_COMMAND_WIDTH+:ROW_COMMAND_WIDTH]  -->|___|------>|___|------>|___|------>|___|------>|___|------>|___|
                                                            2           6           10          14          18          22 
        i_cmd[6*ROW_COMMAND_WIDTH+:ROW_COMMAND_WIDTH]  -->|¯¯¯|------>|¯¯¯|------>|¯¯¯|------>|¯¯¯|------>|¯¯¯|------>|¯¯¯|
        i_cmd[7*ROW_COMMAND_WIDTH+:ROW_COMMAND_WIDTH]  -->|___|------>|___|------>|___|------>|___|------>|___|------>|___|
                                                            3           7           11          15          19          23      
    Configuration:
         The command lay out is shown below: Note: switch ID is specified in the diagram above.
         !!!! Note: the command layout is different from BENES_simple.

         i_cmd: MSB [ --------------------------------------------------------------------------------------------------------- ] LSB
          cmd for   SW3 SW7 SW11 SW15 SW19 SW23|SW2 SW6 SW10 SW14 SW18 SW22|SW1 SW5 SW9 SW13 SW17 SW21|SW0 SW4 SW8 SW12 SW16 SW20
                                               |                           |                          |
                                row4           |           row3            |           row2           |        row1
                                               |                           |                          |

    Author:      Jianming Tong (jianming.tong@gatech.edu)
*/


module birrd_simple_cmd_flow_seq#(
    parameter DATA_WIDTH = 32,       // could be arbitrary number
    parameter COMMAND_WIDTH  = 2,    // 2 when using simple distribute_2x2; 3 when using complex distribute_2x2;
	// parameter NUM_INPUT_DATA = 16,
	// parameter IN_COMMAND_WIDTH = 16
    parameter NUM_INPUT_DATA = 8,
	parameter IN_COMMAND_WIDTH = 12
)(
    // timeing signals
    clk,
    rst_n,

    // data signals
    i_valid,        // valid input data signal
    i_data_bus,     // input data bus coming into distribute switch

    o_valid,        // output valid
    o_data_bus,     // output data

    // control signals
    i_en,           // distribute switch enable
    i_cmd          // input command
);

    //parameter
    localparam [31:0]                   MAX_DATAWITDH_CMDWIDTH = ($clog2(DATA_WIDTH) > $clog2(IN_COMMAND_WIDTH))?$clog2(DATA_WIDTH):$clog2(IN_COMMAND_WIDTH);
    localparam [31:0]                   PARAMETER_BITWIDTH = $clog2(NUM_INPUT_DATA)+MAX_DATAWITDH_CMDWIDTH+2; // clog2 did flooring, +2 to avoid truncation precision loss. 

    localparam [PARAMETER_BITWIDTH-1:0] NUM_SWITCH_IN = NUM_INPUT_DATA >> 1;

    localparam [PARAMETER_BITWIDTH-1:0] LEVEL = $clog2(NUM_INPUT_DATA);
    localparam [PARAMETER_BITWIDTH-1:0] TOTAL_STAGE = 2*LEVEL;

    localparam [PARAMETER_BITWIDTH-1:0] TOTAL_COMMAND = NUM_SWITCH_IN*IN_COMMAND_WIDTH;

    localparam [PARAMETER_BITWIDTH-1:0] WIDTH_INPUT_DATA = NUM_INPUT_DATA*DATA_WIDTH;

    // interface
    input                                        clk;
    input                                        rst_n;

    input  [NUM_INPUT_DATA-1:0]                  i_valid;
    input  [WIDTH_INPUT_DATA-1:0]                i_data_bus;

    output [NUM_INPUT_DATA-1:0]                  o_valid;
    output [WIDTH_INPUT_DATA-1:0]                o_data_bus; //{o_data_a, o_data_b}

    input                                        i_en;
    input  [TOTAL_COMMAND-1:0]                   i_cmd;
        // 00 --> Pass
        // 11 --> Swap
        // 01 --> Add-Left
        // 10 --> Add-Right
                                    
    // inner logic
    wire   [DATA_WIDTH-1:0]                      connection[0:TOTAL_STAGE-1][0:NUM_INPUT_DATA-1];
    wire                                         connection_valid[0:TOTAL_STAGE-1][0:NUM_INPUT_DATA-1];

    function automatic [LEVEL-1:0] reverse_bits(input [LEVEL-1:0] value, input [LEVEL-1:0] reverse_bitwidth);
        integer i;
        begin
            for (i = 0; i < reverse_bitwidth; i = i + 1) begin
                reverse_bits[i] = value[reverse_bitwidth-1-i];
            end
            for (i = reverse_bitwidth; i < LEVEL; i=i+1) begin
                reverse_bits[i] = value[i];
            end
        end
    endfunction

    genvar i,j,s;
    generate
        for(s=0; s<TOTAL_STAGE; s=s+1)
        begin:cmd_stage
            localparam [31:0] IN_COMMAND_WIDTH_STAGE = IN_COMMAND_WIDTH - COMMAND_WIDTH*s ;
            localparam [31:0] TOTAL_IN_COMMAND_WIDTH_STAGE = IN_COMMAND_WIDTH_STAGE * NUM_SWITCH_IN ;
            wire [TOTAL_IN_COMMAND_WIDTH_STAGE-1:0]  inner_cmd_wire;
        end

        assign cmd_stage[0].inner_cmd_wire = i_cmd;

        for(j=0; j<NUM_INPUT_DATA; j=j+1)
        begin:valid
            assign connection_valid[0][j] = i_valid[j];
            // assign connection[0][j] = i_data_bus[j*DATA_WIDTH+:DATA_WIDTH];
        end
        
        for(j=0; j<NUM_INPUT_DATA; j=j+1)
        begin:data
            assign connection[0][j] = i_data_bus[j*DATA_WIDTH+:DATA_WIDTH];
        end

        // first stage -> las stage
        // shuffle function          [loop right shift]:  output of i-th stage    -> input of (i+1)-th stage
        // inverse shuffle function  [loop left shift]:   input of (i+1)-th stage -> output of i-th stage
        for(s=0;s<(TOTAL_STAGE-1);s=s+1)
        begin:stage_id
            
            localparam [31:0] IN_COMMAND_WIDTH_STAGE = IN_COMMAND_WIDTH - COMMAND_WIDTH*s;
            localparam [31:0] OUT_COMMAND_WIDTH_STAGE = IN_COMMAND_WIDTH_STAGE - COMMAND_WIDTH;
            localparam [31:0] TOTAL_IN_COMMAND_WIDTH_STAGE = IN_COMMAND_WIDTH_STAGE * NUM_SWITCH_IN;

            for(i=0; i<NUM_SWITCH_IN; i=i+1)
            begin:sw_id
                // iteration parameter
                localparam [31:0]                    group_bit_first_half = 2 + s;
                localparam [31:0]                    group_bit_max = LEVEL;
                localparam [31:0]                    group_bit_second_half = (LEVEL << 1) - s;
                
                // The following logic picks the minimal value among three value above
                localparam [31:0]                    group_bit_first_cmp = (group_bit_first_half > group_bit_max)? group_bit_max : group_bit_first_half;
                localparam [31:0]                    group_bit = (group_bit_first_cmp > group_bit_second_half)? group_bit_second_half : group_bit_first_cmp;

                localparam [LEVEL-1:0]               port_idx = (i << 1);
                localparam [LEVEL-1:0]               port_idx_add_1 = port_idx + 1;
                localparam [LEVEL-1:0]               output_port_1_idx = reverse_bits(port_idx, group_bit);
                localparam [LEVEL-1:0]               output_port_2_idx = reverse_bits(port_idx_add_1, group_bit);

                birrd_2x2_simple_cmd_flow_seq #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .COMMAND_WIDTH(COMMAND_WIDTH),
                    .IN_COMMAND_WIDTH(IN_COMMAND_WIDTH_STAGE)
                ) egg(
                    .clk(clk),
                    .rst_n(rst_n),
                    .i_valid({connection_valid[s][port_idx_add_1], connection_valid[s][port_idx]}),
                    .i_data_bus({connection[s][port_idx_add_1], connection[s][port_idx]}),
                    .o_valid({connection_valid[s+1][output_port_2_idx], connection_valid[s+1][output_port_1_idx]}),
                    .o_data_bus({connection[s+1][output_port_2_idx], connection[s+1][output_port_1_idx]}),
                    .i_en(i_en),
                    .i_cmd(cmd_stage[s].inner_cmd_wire[i*IN_COMMAND_WIDTH_STAGE+:IN_COMMAND_WIDTH_STAGE]),
                    .o_cmd(cmd_stage[s+1].inner_cmd_wire[i*OUT_COMMAND_WIDTH_STAGE+:OUT_COMMAND_WIDTH_STAGE])
                );
            end
        end
       
        // last stage
        // shuffle function         [loop left shift]:   output of i-th stage    -> input of (i+1)-th stage
        // inverse shuffle function [loop right shift]:  input of (i+1)-th stage -> output of i-th stage
        for(i=0; i<NUM_SWITCH_IN; i=i+1)
        begin:last_stage
            // iteration parameter
            localparam [LEVEL-1:0]               group_bit = 2;

            localparam [LEVEL-1:0]               port_idx = (i << 1);
            localparam [LEVEL-1:0]               port_idx_add_1 = port_idx + 1;

            localparam [LEVEL-1:0]               IN_COMMAND_WIDTH_STAGE = COMMAND_WIDTH * NUM_SWITCH_IN;

            birrd_2x2_simple_last_cmd_flow_seq #(
                .DATA_WIDTH(DATA_WIDTH),
                .COMMAND_WIDTH(COMMAND_WIDTH),
                .IN_COMMAND_WIDTH(IN_COMMAND_WIDTH_STAGE)
            ) egg(
                .clk(clk),
                .rst_n(rst_n),
                .i_valid({connection_valid[TOTAL_STAGE-1][port_idx_add_1], connection_valid[TOTAL_STAGE-1][port_idx]}),
                .i_data_bus({connection[TOTAL_STAGE-1][port_idx_add_1], connection[TOTAL_STAGE-1][port_idx]}),
                .o_valid({o_valid[port_idx_add_1], o_valid[port_idx]}),
                .o_data_bus({o_data_bus[port_idx_add_1*DATA_WIDTH+:DATA_WIDTH], o_data_bus[port_idx*DATA_WIDTH+:DATA_WIDTH]}),
                .i_en(i_en),
                .i_cmd(cmd_stage[TOTAL_STAGE-1].inner_cmd_wire[i*COMMAND_WIDTH+:COMMAND_WIDTH])
            );
        end
    endgenerate
        
endmodule
