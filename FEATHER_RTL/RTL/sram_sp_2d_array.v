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
    Top Module:  Generic Single Port SRAM 
    Data:        Only data width matters.
    Format:      keeping the input format unchanged
    Timing:      Sequential Logic
    Reset:       Asynchronized Reset [Low Reset]

    Function:    Array of SRAM BANKs
*/

`timescale 1ns / 1ps

module sram_sp_2d_array #(
    parameter SRAM_BANK_DATA_WIDTH  = 8  ,                      //
    parameter SRAM_BANK_ADDR_WIDTH  = 10 ,                      //
    parameter SRAM_BANK_DEPTH       = 2**SRAM_BANK_ADDR_WIDTH,  //
    parameter NUM_BANK              = 4                         //
)(
    clk,
    rst_n,

    // Port A
    i_sram_a_wr_data,
    i_sram_a_wr_addr,
    i_sram_a_wr_en,

    // Port B
    i_sram_b_rd_addr,
    i_sram_b_rd_en,
    o_sram_b_rd_data 
);

    localparam SRAM_DATA_WIDTH  =   NUM_BANK*SRAM_BANK_DATA_WIDTH;
    localparam SRAM_ADDR_WIDTH  =   NUM_BANK*SRAM_BANK_ADDR_WIDTH;

    /*
        ports
    */
    input                               clk                 ;
    input                               rst_n               ;

    input   [SRAM_DATA_WIDTH    -1: 0]  i_sram_a_wr_data    ;
    input   [SRAM_ADDR_WIDTH    -1: 0]  i_sram_a_wr_addr    ;
    input   [NUM_BANK           -1: 0]  i_sram_a_wr_en      ;

    input   [SRAM_ADDR_WIDTH    -1: 0]  i_sram_b_rd_addr    ;
    input   [NUM_BANK           -1: 0]  i_sram_b_rd_en      ;
    output  [SRAM_DATA_WIDTH    -1: 0]  o_sram_b_rd_data    ;


    genvar BANK_ITER;

    wire                                w_i_rd_wr_en        [0  : NUM_BANK-1];
    wire [SRAM_BANK_ADDR_WIDTH  -1: 0]  w_i_addr            [0  : NUM_BANK-1];
    wire [SRAM_BANK_ADDR_WIDTH  -1: 0]  w_i_bank_a_addr     [0  : NUM_BANK-1];
    wire                                w_i_bank_a_wr_en    [0  : NUM_BANK-1];
    wire [SRAM_BANK_DATA_WIDTH  -1: 0]  w_i_bank_a_wr_data  [0  : NUM_BANK-1];
    wire [SRAM_BANK_ADDR_WIDTH  -1: 0]  w_i_bank_b_rd_addr  [0  : NUM_BANK-1];
    wire                                w_i_bank_b_rd_en    [0  : NUM_BANK-1];
    wire [SRAM_BANK_DATA_WIDTH  -1: 0]  w_o_bank_b_rd_data  [0  : NUM_BANK-1];

    generate
        for(BANK_ITER=0; BANK_ITER < NUM_BANK; BANK_ITER=BANK_ITER+1)
        begin:SP_SRAM_BANKS

            assign w_i_bank_a_wr_data   [BANK_ITER]    =   i_sram_a_wr_data [(SRAM_BANK_DATA_WIDTH*BANK_ITER)   +:  SRAM_BANK_DATA_WIDTH];
            assign w_i_bank_a_addr      [BANK_ITER]    =   i_sram_a_wr_addr [(SRAM_BANK_ADDR_WIDTH*BANK_ITER)   +:  SRAM_BANK_ADDR_WIDTH];
            assign w_i_bank_a_wr_en     [BANK_ITER]    =   i_sram_a_wr_en   [BANK_ITER];

            assign w_i_bank_b_rd_addr   [BANK_ITER]    =   i_sram_b_rd_addr [(SRAM_BANK_ADDR_WIDTH*BANK_ITER)   +:  SRAM_BANK_ADDR_WIDTH];
            assign w_i_bank_b_rd_en     [BANK_ITER]    =   i_sram_b_rd_en   [BANK_ITER];

            assign o_sram_b_rd_data     [(SRAM_BANK_DATA_WIDTH*BANK_ITER)   +:  SRAM_BANK_DATA_WIDTH]   =   w_o_bank_b_rd_data [BANK_ITER];


            sram_bank_sp#(
                .SRAM_BANK_DATA_WIDTH   (SRAM_BANK_DATA_WIDTH           ),
                .SRAM_BANK_ADDR_WIDTH   (SRAM_BANK_ADDR_WIDTH           ),
                .SRAM_BANK_DEPTH        (SRAM_BANK_DEPTH                )
            ) sram_bank_sp_inst(
                .clk                    (clk                            ),
                .rst_n                  (rst_n                          ),
                .i_rd_wr_en             (w_i_rd_wr_en       [BANK_ITER] ),
                .i_addr                 (w_i_addr           [BANK_ITER] ),
                .i_wr_data              (w_i_bank_a_wr_data [BANK_ITER] ),
                .o_rd_data              (w_o_bank_b_rd_data [BANK_ITER] )
            );

            assign  w_i_rd_wr_en[BANK_ITER] =   (w_i_bank_a_wr_en[BANK_ITER] == 1)  ?   1
                                                                                    :   0;
            assign  w_i_addr    [BANK_ITER] =   (w_i_bank_a_wr_en[BANK_ITER] == 1)  ?   w_i_bank_a_addr    [BANK_ITER]
                                                                                    :   w_i_bank_b_rd_addr [BANK_ITER];
        end

    endgenerate




endmodule