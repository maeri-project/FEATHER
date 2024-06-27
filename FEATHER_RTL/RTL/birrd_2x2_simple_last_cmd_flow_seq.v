`timescale 1ns / 1ps
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
    Top Module:  birrd_2x2_simple_last_cmd_flow_seq
    Data:        Only data width matters.
    Format:      keeping the input format unchange
    Timing:      Sequential Logic
    Reset:       Asynchronized Reset [Low Reset]
    Latency:     2 cycle for Complex; 1 cycle for Simple
    Dummy Data:  {DATA_WIDTH{1'b0}}

    Unicast Function:
                               Pass                                          Swap

            i_data_bus(high)          i_data_bus(low)      i_data_bus(high)          i_data_bus(low)
       [DATA_WIDTH+:DATA_WIDTH]    [DATA_WIDTH-1:0]    [DATA_WIDTH+:DATA_WIDTH]    [DATA_WIDTH-1:0]
                              \     /                                       \     /
                               v   v                                         v   v
                               |¯¯¯| <--i_valid=2'b11                        |¯¯¯| <--i_valid=2'b11
                               |___| <--i_cmd=2'b00 (MS 2 bits)              |___| <--i_cmd=2'b11 (MS 2 bits)
                              /     \                                       /     \
                             v       v                                     v       v
                     o_data_high   o_data_low                      o_data_low     o_data_high


    Multicast Function:      
                             Add-Left                                      Add-Right                

              i_data_bus(high)        i_data_bus(low)      i_data_bus(high)          i_data_bus(low)  
       [DATA_WIDTH+:DATA_WIDTH]    [DATA_WIDTH-1:0]    [DATA_WIDTH+:DATA_WIDTH]    [DATA_WIDTH-1:0] 
                              \     /                                       \     /                 
                               v   v                                         v   v                  
                               |¯+¯| <--i_valid=2'b11                        |¯+¯| <--i_valid=2'b11 
                               |___| <--i_cmd=2'b10 (MS 2 bits)              |___| <--i_cmd=2'b01  (MS 2 bits)
                              /     \                                       /     \                  
                             v       v                                     v       v        
                     accumulated   i_data_bus(low)              i_data_bus(high)  accumulated

      The i_cmd other than most significant bits -> forward to the following modules.

*/

/*
    parameter illustration:
    1. COMMAND_WIDTH means the length of command for a single switch, so each switch need COMMAND_WIDTH because of two inputs.
    2. IN_COMMAND_WIDTH means total length of command fed into the design. It is used to calculate the length of the output command.
    3. DATA_WIDTH means the length of a single data, arbitrary number is supported in the design.
*/

module birrd_2x2_simple_last_cmd_flow_seq#(
    parameter DATA_WIDTH = 32,
    parameter COMMAND_WIDTH  = 2,
    parameter IN_COMMAND_WIDTH  = 8
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
    i_cmd           // input command
);

    // localparam
    localparam OUT_DATA_WIDTH = 2*DATA_WIDTH;

    // interface
    input                              clk;
    input                              rst_n;

    input  [1:0]                       i_valid;
    input  [2*DATA_WIDTH-1:0]          i_data_bus;

    output [1:0]                       o_valid;
    output [2*DATA_WIDTH-1:0]          o_data_bus; //{o_data_a, o_data_b}

    input                              i_en;
    input  [1:0]                       i_cmd;
        // 11 --> Swap
        // 00 --> Pass
        // 10 --> Add-Left
        // 01 --> Add-Right

    // output register
    reg    [OUT_DATA_WIDTH-1:0]        o_data_bus_inner;
    reg    [1:0]                       o_valid_inner;

    // Intermediate Results
    wire   [DATA_WIDTH-1:0]            i_data_high = i_data_bus[DATA_WIDTH +:DATA_WIDTH];
    wire   [DATA_WIDTH-1:0]            i_data_low = i_data_bus[0 +:DATA_WIDTH];
    
    wire   [DATA_WIDTH-1:0]            accumulated_res_inner = i_data_high + i_data_low;
    wire                               accumulated_valid = &i_valid;

    always@(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin
            o_data_bus_inner <= {OUT_DATA_WIDTH{1'b0}};
        end
        else if(i_en)
        begin
            case({i_cmd})
                2'b00:
                begin
                    o_data_bus_inner <= i_data_bus;
                end
                2'b01:
                begin
                    o_data_bus_inner <= {i_data_high, accumulated_res_inner};
                end
                2'b10:
                begin
                    o_data_bus_inner <= {accumulated_res_inner, i_data_low};
                end
                2'b11:
                begin
                    o_data_bus_inner <= {i_data_low, i_data_high};
                end
            endcase
        end
        else
        begin
            o_data_bus_inner <= {OUT_DATA_WIDTH{1'b0}};
        end
    end

    always@(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin
            o_valid_inner <= 2'b0;
        end
        else if(i_en)
        begin
            case({i_cmd})
                2'b00:
                begin
                    o_valid_inner <= i_valid;
                end
                2'b01:
                begin
                    o_valid_inner <= {i_valid[1], accumulated_valid};
                end
                2'b10:
                begin
                    o_valid_inner <= {accumulated_valid, i_valid[0]};
                end
                2'b11:
                begin
                    o_valid_inner <= {i_valid[0], i_valid[1]};
                end
            endcase
        end
        else
        begin
            o_valid_inner <= 2'b0;
        end
    end

    assign o_data_bus = o_data_bus_inner;
    assign o_valid = o_valid_inner;

endmodule
