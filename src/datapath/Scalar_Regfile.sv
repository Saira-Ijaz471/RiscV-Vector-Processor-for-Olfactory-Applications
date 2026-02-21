module Scalar_Regfile
(
   input  logic [ 4:0] rs1,
   input  logic [ 4:0] rs2,
   input  logic [ 4:0] wr_reg,
   input  logic [31:0] write_data_sca,
   input  logic        clk,
   input  logic        rst,
   input  logic        reg_write,
   output logic [31:0] rd1,
   output logic [31:0] rd2
);

logic [31:0] registers [31:0];

always_comb begin
   rd1 = registers[rs1];
   rd2 = registers[rs2];
end

always_ff @(posedge clk or negedge rst) begin

   if (!rst) begin
      for (i = 0; i < 32; i++) begin
            registers[i] <= '0;
      end
   end

   else if (wr_reg != '0) begin
      if (reg_write) begin
         registers[wr_reg] <= write_data_sca;
      end
   end

   registers[0] <= '0; 
end
endmodule