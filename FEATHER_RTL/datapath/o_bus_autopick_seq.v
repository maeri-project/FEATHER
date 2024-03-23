`timescale 1ns / 1ps
/*
    Top Module:  o_bus_autopick_seq
    Data:        Only data width matters.
    Format:      keeping the input format unchange
    Timing:      Sequential Logic
    Reset:       Asynchronized Reset [Low Reset]
    Latency:     1
    Dummy Data:  {DATA_WIDTH{1'b0}}

    Parameter:   NUM_INPUT_DATA could be arbitrary integer

    Function:   A big one-hot controlled mux, which select one 

      i_data_bus
       |   |       |   |       |   |       |   |
       v   v       v   v  ...  v   v       v   v
       _________________________________________
       \                                       /
        \                                     /
         \___________________________________/                   
                           |
                           v
                      o_data_bus(only pick 1 valid from all input data)

    Author:      Jianming Tong (jianming.tong@gatech.edu)
*/


module o_bus_autopick_seq#(
    parameter NUM_INPUT_DATA = 300,
    parameter DATA_WIDTH = 16
)(
    // timing signals
    clk,            // clock 
    rst_n,          // Negative edge reset

    // data signals
    i_valid,        // valid input data signal
    i_data_bus,     // input data bus coming into distribute switch

    o_valid,        // output valid
    o_data_bus,     // output data

    // control signals
    i_en            // distribute switch enable
);
    // timing signals
    input                                        clk;
    input                                        rst_n;

    // interface
    input  [NUM_INPUT_DATA-1:0]                  i_valid;
    input  [NUM_INPUT_DATA*DATA_WIDTH-1:0]       i_data_bus;

    output                                       o_valid;
    output [DATA_WIDTH-1:0]                      o_data_bus; //{o_data_a, o_data_b}

    input                                        i_en;

    reg                                          o_valid_inner;
    reg    [DATA_WIDTH-1:0]                      o_data_bus_inner; //{o_data_a, o_data_b}

    // Muxing Data Function
    function [DATA_WIDTH-1:0] sel_data;
        input [NUM_INPUT_DATA*DATA_WIDTH-1:0] data_bus;
        input [NUM_INPUT_DATA-1:0] en;
        integer i;
        begin
            sel_data = {DATA_WIDTH{1'b0}};
            for(i=0; i<NUM_INPUT_DATA; i=i+1) begin
                if(en[i] == 1'b1) begin
                    sel_data = data_bus[i*DATA_WIDTH +: DATA_WIDTH];
                end
            end
        end
    endfunction

    // Muxing Logic
    always @(negedge rst_n or posedge clk) begin
        if (~rst_n) begin
            o_data_bus_inner <= {DATA_WIDTH{1'b0}};
        end else begin
            o_data_bus_inner <= sel_data(i_data_bus, i_valid);
        end
    end

    // Output Valid Logic
    always @(negedge rst_n or posedge clk) begin
        if (~rst_n) begin
            o_valid_inner <= 1'b0;
        end else begin
            o_valid_inner <= |i_valid && i_en;
        end
    end

    // output logic
    assign o_data_bus = o_data_bus_inner;
    assign o_valid = o_valid_inner;

endmodule
