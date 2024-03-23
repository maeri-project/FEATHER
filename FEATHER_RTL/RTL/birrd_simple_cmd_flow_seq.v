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
    Format:      keeping the input format unchanged
    Timing:      Sequential Logic
    Reset:       Asynchronized Reset [Low Reset]
    Pipeline:
                 [full pipeline]every stage is a pipeline stage
                                Total latency = # stages (cycle)
                 [2 stage pipeline] 0~LEVEL is the first pipeline stage.
    Dummy Data:  {DATA_WIDTH{1'b0}}

    Function:    Unicast  or  Multicast

*/

`timescale 1ns / 1ps

module birrd_simple_cmd_flow_seq#(
    parameter DATA_WIDTH = 32,       // could be arbitrary number
    parameter COMMAND_WIDTH  = 2,    // 2 when using simple distribute_2x2; 3 when using complex distribute_2x2;
	parameter NUM_INPUT_DATA = 16,
	parameter IN_COMMAND_WIDTH = 14
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
    localparam [31:0] NUM_SWITCH_IN = NUM_INPUT_DATA >> 1;

    localparam [31:0] LEVEL = $clog2(NUM_INPUT_DATA);
    localparam [31:0] TOTAL_STAGE = 2*LEVEL-1;

    localparam [31:0] TOTAL_COMMAND = NUM_SWITCH_IN*IN_COMMAND_WIDTH;

    localparam [31:0] WIDTH_INPUT_DATA = NUM_INPUT_DATA*DATA_WIDTH;

    // interface
    input                                        clk;
    input                                        rst_n;

    input  [NUM_INPUT_DATA-1:0]                  i_valid;
    input  [WIDTH_INPUT_DATA-1:0]                i_data_bus;

    output [NUM_INPUT_DATA-1:0]                  o_valid;
    output [WIDTH_INPUT_DATA-1:0]                o_data_bus; //{o_data_a, o_data_b}

    input                                        i_en;
    input  [TOTAL_COMMAND-1:0]                   i_cmd;
        // 00 --> Through
        // 11 --> Switch
        // 01 --> Add-Left
        // 10 --> Add-Right
                                    
    // inner logic
    wire   [DATA_WIDTH-1:0]                       connection[0:TOTAL_STAGE-2][0:NUM_INPUT_DATA-1];
    wire                                          connection_valid[0:TOTAL_STAGE-2][0:NUM_INPUT_DATA-1];

    genvar i,j,k,s,p;
    generate
        for(i=0; i<TOTAL_STAGE-1; i=i+1)
        begin:cmd_stage
            localparam [31:0] IN_COMMAND_WIDTH_STAGE = IN_COMMAND_WIDTH - COMMAND_WIDTH*(i+1) ;
            localparam [31:0] TOTAL_IN_COMMAND_WIDTH_STAGE = IN_COMMAND_WIDTH_STAGE * NUM_SWITCH_IN ;
            wire [TOTAL_IN_COMMAND_WIDTH_STAGE-1:0]  inner_cmd_wire;
        end

        // first stage
        for(i=0; i<NUM_SWITCH_IN; i=i+1)
        begin:first_stage_switch
            localparam [31:0] IN_COMMAND_WIDTH_STAGE = IN_COMMAND_WIDTH - COMMAND_WIDTH;

            birrd_2x2_simple_cmd_flow_seq #(
                .DATA_WIDTH(DATA_WIDTH),
                .COMMAND_WIDTH(COMMAND_WIDTH),
                .IN_COMMAND_WIDTH(IN_COMMAND_WIDTH)
            ) first_stage(
                .clk(clk),
                .rst_n(rst_n),
                .i_valid(i_valid[2*i+:2]),
                .i_data_bus(i_data_bus[i*2*DATA_WIDTH+:2*DATA_WIDTH]),
                .o_valid({connection_valid[0][2*i+1], connection_valid[0][2*i]}),
                .o_data_bus({connection[0][2*i+1], connection[0][2*i]}),
                .i_en(i_en),
                .i_cmd(i_cmd[i*IN_COMMAND_WIDTH+:IN_COMMAND_WIDTH]),
                .o_cmd(cmd_stage[0].inner_cmd_wire[i*IN_COMMAND_WIDTH_STAGE+:IN_COMMAND_WIDTH_STAGE])
            );
        end

        // first stage -> middle stage
        // shuffle function          [loop right shift]:  output of i-th stage    -> input of (i+1)-th stage
        // inverse shuffle function  [loop left shift]:   input of (i+1)-th stage -> output of i-th stage
        for(s=0;s<(LEVEL-1);s=s+1)
        begin:first_half
            // iteration parameter
            localparam [31:0]      num_group = LEVEL - 2 - s;
            localparam [31:0]      NUM_GROUP_SEC_HALF = 1 << num_group;

            // stage command width parameter
            localparam                                IN_COMMAND_WIDTH_STAGE = IN_COMMAND_WIDTH - COMMAND_WIDTH * (s + 1);
            localparam                                OUT_COMMAND_WIDTH_STAGE = IN_COMMAND_WIDTH - COMMAND_WIDTH * (s + 2);

            for(k=0; k<NUM_GROUP_SEC_HALF; k=k+1)
            begin:group_sec_half
                localparam  NUM_SWITCH_IN_GROUP = NUM_SWITCH_IN >> num_group;
                for(i=0; i<NUM_SWITCH_IN_GROUP; i=i+1)
                begin:switch_sec_half
                    // For low input [Loop right Shift (2*i)]
                    // localparam [$clog2(NUM_INPUT_DATA)-1-num_group:0] idx = i[$clog2(NUM_INPUT_DATA)-1-num_group:0];
                    localparam [31:0]            group_switch_offset = k*(NUM_SWITCH_IN>>num_group);
                    localparam [31:0]            group_offset = k*(NUM_INPUT_DATA>>num_group);
                    localparam [31:0]            MASK =  (1 << ($clog2(NUM_INPUT_DATA)-num_group)) - 1;

                    localparam [31:0]            l_idx = (i << 1) & MASK;
                    localparam [31:0]            l_idx_right_shift = (l_idx >> 1) & MASK;
                    localparam [31:0]            l_idx_LSB_is_1 = l_idx&1'b1;
                    localparam [31:0]            l_idx_LSB_loop_shift = l_idx_LSB_is_1 <<  ($clog2(NUM_INPUT_DATA)-1-num_group);
                    localparam [31:0]            l_idx_loop_right_shift = l_idx_LSB_loop_shift + l_idx_right_shift;
                    localparam [31:0]            l_idx_loop_right_shift_group = l_idx_loop_right_shift + group_offset;

                    // For high input [Loop right Shift (2*i+1)]
                    localparam [31:0]            h_idx = ((i << 1) + 1) & MASK;
                    localparam [31:0]            h_idx_right_shift = (h_idx >> 1) & MASK;
                    localparam [31:0]            h_idx_LSB_is_1 = h_idx&1'b1;
                    localparam [31:0]            h_idx_LSB_loop_shift = h_idx_LSB_is_1 <<  ($clog2(NUM_INPUT_DATA)-1-num_group);
                    localparam [31:0]            h_idx_loop_right_shift = h_idx_LSB_loop_shift + h_idx_right_shift;
                    localparam [31:0]            h_idx_loop_right_shift_group = h_idx_loop_right_shift + group_offset;

                    birrd_2x2_simple_cmd_flow_seq #(
                        .DATA_WIDTH(DATA_WIDTH),
                        .COMMAND_WIDTH(COMMAND_WIDTH),
                        .IN_COMMAND_WIDTH(IN_COMMAND_WIDTH_STAGE)
                    ) second_stage(
                        .clk(clk),
                        .rst_n(rst_n),
                        .i_valid({connection_valid[s][h_idx_loop_right_shift_group], connection_valid[s][l_idx_loop_right_shift_group]}),
                        .i_data_bus({connection[s][h_idx_loop_right_shift_group], connection[s][l_idx_loop_right_shift_group]}),
                        .o_valid({connection_valid[s+1][2*(i+group_switch_offset)+1], connection_valid[s+1][2*(i+group_switch_offset)]}),
                        .o_data_bus({connection[s+1][2*(i+group_switch_offset)+1], connection[s+1][2*(i+group_switch_offset)]}),
                        .i_en(i_en),
                        .i_cmd(cmd_stage[s].inner_cmd_wire[(k*NUM_SWITCH_IN_GROUP+i)*IN_COMMAND_WIDTH_STAGE+:IN_COMMAND_WIDTH_STAGE]),
                        .o_cmd(cmd_stage[s+1].inner_cmd_wire[(k*NUM_SWITCH_IN_GROUP+i)*OUT_COMMAND_WIDTH_STAGE+:OUT_COMMAND_WIDTH_STAGE])
                    );
                end
            end
        end


        // middle stage -> second last stage
        // shuffle function         [loop left shift]:   output of i-th stage    -> input of (i+1)-th stage
        // inverse shuffle function [loop right shift]:  input of (i+1)-th stage -> output of i-th stage
        // for(s=0;s<(LEVEL-1);s=s+1)
        // begin:first_half_stages
        for(s=(LEVEL-1);s<(TOTAL_STAGE-2);s=s+1)
        begin:second_half
            // iteration parameter
            localparam [31:0]    stage_idx = (s-LEVEL+1);
            localparam [31:0]    NUM_GROUP = 1 << stage_idx;
            localparam [31:0]    NUM_SWITCH_GROUP = NUM_SWITCH_IN >> stage_idx;
            localparam [31:0]    LEN_GROUP = $clog2(NUM_INPUT_DATA) - 1 - stage_idx;

            // stage command width parameter
            localparam                                IN_COMMAND_WIDTH_STAGE = IN_COMMAND_WIDTH - COMMAND_WIDTH * (s + 1);
            localparam                                OUT_COMMAND_WIDTH_STAGE = IN_COMMAND_WIDTH - COMMAND_WIDTH * (s + 2);

            for(k=0;k<NUM_GROUP;k=k+1)
            begin:group_first_half
                for(i=0;i<NUM_SWITCH_GROUP;i=i+1)
                begin:switch_first_half
                    localparam [31:0]    group_switch_offset = k*(NUM_SWITCH_IN>>stage_idx);
                    localparam [31:0]    group_offset = k*(NUM_INPUT_DATA>>stage_idx);
                    localparam [31:0]    MASK =  (1 << ($clog2(NUM_INPUT_DATA)-stage_idx)) - 1;

                    // For low input [Loop left Shift (2*i)]
                    localparam [31:0]    l_idx = (i << 1) & MASK;
                    localparam [31:0]    l_idx_left_shift = (l_idx << 1) & MASK;
                    localparam [31:0]    l_idx_MSB_1 = (1'b1 << LEN_GROUP) & MASK;
                    localparam [31:0]    l_idx_MSB_is_1 = l_idx & l_idx_MSB_1;
                    localparam [31:0]    l_idx_MSB_loop_shift = l_idx_MSB_is_1 >> LEN_GROUP;
                    localparam [31:0]    l_idx_loop_left_shift = l_idx_MSB_loop_shift + l_idx_left_shift;
                    localparam [31:0]    l_idx_loop_left_shift_group = l_idx_loop_left_shift + group_offset;

                    // For high input [Loop left Shift (2*i)+1]
                    localparam [31:0]    h_idx = ((i << 1) + 1) & MASK;
                    localparam [31:0]    h_idx_left_shift = (h_idx << 1) & MASK;
                    localparam [31:0]    h_idx_MSB_1 = (1'b1 << LEN_GROUP) & MASK;
                    localparam [31:0]    h_idx_MSB_is_1 =  h_idx&h_idx_MSB_1;
                    localparam [31:0]    h_idx_MSB_loop_shift = h_idx_MSB_is_1 >> LEN_GROUP;
                    localparam [31:0]    h_idx_loop_left_shift = h_idx_MSB_loop_shift + h_idx_left_shift;
                    localparam [31:0]    h_idx_loop_left_shift_group = h_idx_loop_left_shift + group_offset;

                    birrd_2x2_simple_cmd_flow_seq #(
                        .DATA_WIDTH(DATA_WIDTH),
                        .COMMAND_WIDTH(COMMAND_WIDTH),
                        .IN_COMMAND_WIDTH(IN_COMMAND_WIDTH_STAGE)
                    ) third_stage(
                        .clk(clk),
                        .rst_n(rst_n),
                        .i_valid({connection_valid[s][h_idx_loop_left_shift_group], connection_valid[s][l_idx_loop_left_shift_group]}),
                        .i_data_bus({connection[s][h_idx_loop_left_shift_group], connection[s][l_idx_loop_left_shift_group]}),
                        .o_valid({connection_valid[s+1][2*(i+group_switch_offset)+1], connection_valid[s+1][2*(i+group_switch_offset)]}),
                        .o_data_bus({connection[s+1][2*(i+group_switch_offset)+1], connection[s+1][2*(i+group_switch_offset)]}),
                        .i_en(i_en),
                        .i_cmd(cmd_stage[s].inner_cmd_wire[(k*NUM_SWITCH_GROUP+i)*IN_COMMAND_WIDTH_STAGE+:IN_COMMAND_WIDTH_STAGE]),
                        .o_cmd(cmd_stage[s+1].inner_cmd_wire[(k*NUM_SWITCH_GROUP+i)*OUT_COMMAND_WIDTH_STAGE+:OUT_COMMAND_WIDTH_STAGE])
                    );
                end
            end
        end

       
        // last stage
        // shuffle function         [loop left shift]:   output of i-th stage    -> input of (i+1)-th stage
        // inverse shuffle function [loop right shift]:  input of (i+1)-th stage -> output of i-th stage
        for(s=(TOTAL_STAGE-2);s<(TOTAL_STAGE-1);s=s+1)
        begin:last_stage
            // iteration parameter
            localparam [31:0]    stage_idx = (s-LEVEL+1);
            localparam [31:0]    NUM_GROUP = 1<<stage_idx;
            localparam [31:0]    NUM_SWITCH_GROUP = NUM_SWITCH_IN>>stage_idx;
            localparam [31:0]    LEN_GROUP = $clog2(NUM_INPUT_DATA)-1-stage_idx;

            // stage command width parameter
            localparam [31:0]    IN_COMMAND_WIDTH_STAGE = COMMAND_WIDTH;

            for(k=0;k<NUM_GROUP;k=k+1)
            begin:group_first_half
                for(i=0;i<NUM_SWITCH_GROUP;i=i+1)
                begin:switch_first_half
                    localparam [31:0]    group_switch_offset = k*(NUM_SWITCH_IN>>stage_idx);
                    localparam [31:0]    group_offset = k*(NUM_INPUT_DATA>>stage_idx);
                    localparam [31:0]    MASK =  (1 << ($clog2(NUM_INPUT_DATA)-stage_idx)) - 1;

                    // For low input [Loop left Shift (2*i)]
                    localparam [31:0]    l_idx = (i << 1) & MASK;
                    localparam [31:0]    l_idx_left_shift = (l_idx << 1) & MASK;
                    localparam [31:0]    l_idx_MSB_1 = (1'b1 << LEN_GROUP) & MASK;
                    localparam [31:0]    l_idx_MSB_is_1 = l_idx & l_idx_MSB_1;
                    localparam [31:0]    l_idx_MSB_loop_shift = l_idx_MSB_is_1 >> LEN_GROUP;
                    localparam [31:0]    l_idx_loop_left_shift = l_idx_MSB_loop_shift + l_idx_left_shift;
                    localparam [31:0]    l_idx_loop_left_shift_group = l_idx_loop_left_shift + group_offset;

                    // For high input [Loop left Shift (2*i)+1]
                    localparam [31:0]    h_idx = ((i << 1) + 1) & MASK;
                    localparam [31:0]    h_idx_left_shift = (h_idx << 1) & MASK;
                    localparam [31:0]    h_idx_MSB_1 = (1'b1 << LEN_GROUP) & MASK;
                    localparam [31:0]    h_idx_MSB_is_1 =  h_idx&h_idx_MSB_1;
                    localparam [31:0]    h_idx_MSB_loop_shift = h_idx_MSB_is_1 >> LEN_GROUP;
                    localparam [31:0]    h_idx_loop_left_shift = h_idx_MSB_loop_shift + h_idx_left_shift;
                    localparam [31:0]    h_idx_loop_left_shift_group = h_idx_loop_left_shift + group_offset;

                    birrd_2x2_simple_last_cmd_flow_seq #(
                        .DATA_WIDTH(DATA_WIDTH),
                        .COMMAND_WIDTH(COMMAND_WIDTH),
                        .IN_COMMAND_WIDTH(IN_COMMAND_WIDTH_STAGE)
                    ) last_stage(
                        .clk(clk),
                        .rst_n(rst_n),
                        .i_valid({connection_valid[s][h_idx_loop_left_shift_group], connection_valid[s][l_idx_loop_left_shift_group]}),
                        .i_data_bus({connection[s][h_idx_loop_left_shift_group], connection[s][l_idx_loop_left_shift_group]}),
                        .o_valid({o_valid[2*(i+group_switch_offset)+1+:1], o_valid[2*(i+group_switch_offset)+:1]}),
                        .o_data_bus({o_data_bus[(2*(i+group_switch_offset)+1)*DATA_WIDTH+:DATA_WIDTH], o_data_bus[2*(i+group_switch_offset)*DATA_WIDTH+:DATA_WIDTH]}),
                        .i_en(i_en),
                        .i_cmd(cmd_stage[s].inner_cmd_wire[(k*NUM_SWITCH_GROUP+i)*IN_COMMAND_WIDTH_STAGE+:IN_COMMAND_WIDTH_STAGE])
                    );
                end
            end
        end
    endgenerate

endmodule
