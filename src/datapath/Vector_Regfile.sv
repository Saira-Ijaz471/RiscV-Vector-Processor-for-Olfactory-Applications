module Vector_Regfile 
#(
    parameter VLEN = 4,           // number of elements per vector
    parameter SEW = 32            // element width
)(
    input  logic       clk,
    input  logic       rst,

    // Vector register selects
    input  logic [4:0] rs1,   // read vector for operand1
    input  logic [4:0] rs2,   // read vector for operand2
    input  logic [4:0] wr_reg,    // destination vector

    // ALU outputs
    input  logic [SEW-1:0] write_data0,
    input  logic [SEW-1:0] write_data1,
    input  logic [SEW-1:0] write_data2,
    input  logic [SEW-1:0] write_data3,

    input  logic reg_write,       // global write enable

    // Read outputs for ALUs
    output logic [SEW-1:0] rd1_0,
    output logic [SEW-1:0] rd1_1,
    output logic [SEW-1:0] rd1_2,
    output logic [SEW-1:0] rd1_3,

    output logic [SEW-1:0] rd2_0,
    output logic [SEW-1:0] rd2_1,
    output logic [SEW-1:0] rd2_2,
    output logic [SEW-1:0] rd2_3
);

// 32 vector registers × 4 elements each
logic [SEW-1:0] regs [31:0][VLEN-1:0];

integer i;

// Read logic
always_comb begin
    rd1_0 = regs[rs1][0];
    rd1_1 = regs[rs1][1];
    rd1_2 = regs[rs1][2];
    rd1_3 = regs[rs1][3];

    rd2_0 = regs[rs2][0];
    rd2_1 = regs[rs2][1];
    rd2_2 = regs[rs2][2];
    rd2_3 = regs[rs2][3];
end 

// Write logic
always_ff @(posedge clk) begin
    if (rst) begin
        for (i = 0; i < 32; i = i+1)
            regs[i][0] <= '0;
            regs[i][1] <= '0;
            regs[i][2] <= '0;
            regs[i][3] <= '0;
    end 

    else if (reg_write) begin
        regs[wr_reg][0] <= write_data0;
        regs[wr_reg][1] <= write_data1;
        regs[wr_reg][2] <= write_data2;
        regs[wr_reg][3] <= write_data3;
    end
end
endmodule
