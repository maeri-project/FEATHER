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
    Top Module:  Generic Single Port SRAM Bank
    Data:        Only data width matters.
    Format:      keeping the input format unchanged
    Timing:      Sequential Logic
    Reset:       Asynchronized Reset [Low Reset]

    Function:    Single, Common RD/WR enable to control read and write
*/

`timescale 1ns / 1ps

module sram_bank_sp #(
    parameter SRAM_BANK_DATA_WIDTH      = 8  ,                      //
    parameter SRAM_BANK_ADDR_WIDTH      = 10 ,                      //
    parameter SRAM_BANK_DEPTH           = 2**SRAM_BANK_ADDR_WIDTH   //

)(
    clk,
    rst_n,

    // Single Port
    i_rd_wr_en,
    i_addr,
    i_wr_data,
    o_rd_data
);


    /*
        ports
    */
    input                                   clk;
    input                                   rst_n;

    input                                   i_rd_wr_en;
    input   [SRAM_BANK_ADDR_WIDTH   -1: 0]  i_addr;
    input   [SRAM_BANK_DATA_WIDTH   -1: 0]  i_wr_data;
    output  [SRAM_BANK_DATA_WIDTH   -1: 0]  o_rd_data;

    /*
        inner logics
    */
    reg     [SRAM_BANK_DATA_WIDTH   -1: 0]  r_sram_bank [0  :   SRAM_BANK_DEPTH-1];
    reg     [SRAM_BANK_DATA_WIDTH   -1: 0]  r_o_rd_data;

    integer i;


    /*
        Dual Port SRAM
        ->  i_rd_wr_en == 1 => Write , else Read
    */
    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin
            for (i=0; i<SRAM_BANK_DEPTH; i=i+1)
            begin
                r_sram_bank[i]   <=  0;
            end
        end
        else
        begin
            if(i_rd_wr_en==1)
            begin
                r_sram_bank[$unsigned(i_addr)]  <=  i_wr_data;
            end
        end
    end


    always @(posedge clk or negedge rst_n)
    begin
        if(!rst_n)
        begin
            r_o_rd_data        <=  0;
        end
        else
        begin
            if(i_rd_wr_en==0)
            begin
                r_o_rd_data    <=  r_sram_bank[$unsigned(i_addr)];
            end
        end
    end

    assign  o_rd_data    =  r_o_rd_data;

endmodule