module Instruction_Memory
(
   input  logic [31:0] out_address,
   output logic [31:0] instruction
);

logic [31:0] memory [1000:0];  

   always_comb begin
      instruction = memory[out_address/4]; // Assuming addr is word-aligned
   end
endmodule