module Mux
(
   input  logic [31:0] a,
   input  logic [31:0] b,
   input  logic        sel,
   output logic [31:0] c
);

always_comb begin

   if (sel) begin
      c = b; 
   end 

   else begin
      c = a; 
   end

end
endmodule