// -----------------------------------------------------------------------------
// Module Name: LUTNN (Testbench)
// -----------------------------------------------------------------------------

`timescale 1ns/1ps

module tb_NET;

    // Parameters (replace with include if you already have Globals.svh)
    parameter int NET_INPUTS = 400;
    parameter int NET_OUTPUT = 10;

    // DUT Signals
    logic [NET_INPUTS-1:0] NET_I;
    logic [NET_OUTPUT-1:0] NET_O;

    // DUT Instantiation
    top DUT (
        .NET_I(NET_I),
        .NET_O(NET_O)
    );

    // Test Inputs (binarized and reversed 20x20 MNIST images)
    localparam logic [NET_INPUTS-1:0] MNIST_seven = 400'h003C0007C0007C00078000F8000F0001F0001E0003C0003800078000F0000F0001E0001E0003C0003FF803FFFC3FFFC000FC;
    localparam logic [NET_INPUTS-1:0] MNIST_two   = 400'h03FFCFFFFEFFFFEFFBFE0001E0003E0003E0007C00078000F8001F0001E0003E0007C0007860078F007BF007FE007FC003F8;
    localparam logic [NET_INPUTS-1:0] MNIST_one   = 400'h003C0003C000780007800070000700007000070000E0000E0001E0001C0001C0001C00038000380003800030000700007000;
    localparam logic [NET_INPUTS-1:0] MNIST_zero  = 400'h00FE000FF807FFC07FFC07FFC0FC1C1F81C3F81C3E01C3E01C3E07C3E0FC3F3FC0FFF807FF007FE003FE000FC000F8000F80;
    localparam logic [NET_INPUTS-1:0] MNIST_four  = 400'h03000070000700007000070000F000073E00FFF80FFF80F03C0E01C0E01C1E0383C0383C070380F0380E0380E0180C0180C0;

    // Stimulus
    initial begin
        NET_I = MNIST_seven; #20;
        NET_I = MNIST_two;   #20;
        NET_I = MNIST_one;   #20;
        NET_I = MNIST_zero;  #20;
        NET_I = MNIST_four;

        #20;
        $finish;
    end

endmodule