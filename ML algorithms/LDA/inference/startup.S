# RISC-V Bare-Metal Startup Code
# Optimized for 32-bit Fixed-Point LDA Inference
# Production version - initializes processor and jumps to main()

.section .text.init
.globl _start
.type _start, @function

_start:
    # ==========================================
    # Disable all interrupts
    # ==========================================
    csrw mie, zero
    csrw mip, zero
    
    # ==========================================
    # Initialize global pointer (for small data access)
    # ==========================================
.option push
.option norelax
    la gp, __global_pointer$
.option pop

    # ==========================================
    # Initialize stack pointer
    # Stack grows downward from top of RAM
    # ==========================================
    la sp, _stack_top

    # ==========================================
    # Clear BSS section (uninitialized data)
    # All static/global variables start at zero
    # ==========================================
    la t0, _bss_start
    la t1, _bss_end
    bgeu t0, t1, bss_done      # Skip if no BSS

bss_loop:
    sw zero, 0(t0)             # Write zero to current address
    addi t0, t0, 4             # Move to next word
    bltu t0, t1, bss_loop      # Continue until end

bss_done:

    # ==========================================
    # Copy .data section from ROM to RAM
    # Initialized global variables need to be copied
    # from their ROM location to RAM
    # ==========================================
    la t0, _data_load          # Source address (in ROM)
    la t1, _data_start         # Destination address (in RAM)
    la t2, _data_end           # End of .data section
    bgeu t1, t2, data_done     # Skip if no initialized data

data_loop:
    lw t3, 0(t0)               # Load word from ROM
    sw t3, 0(t1)               # Store word to RAM
    addi t0, t0, 4             # Increment source
    addi t1, t1, 4             # Increment destination
    bltu t1, t2, data_loop     # Continue until end

data_done:

    # ==========================================
    # Copy .sdata section from ROM to RAM
    # Small data section (accessed via gp)
    # ==========================================
    la t0, _sdata_load         # Source address (in ROM)
    la t1, _sdata_start        # Destination address (in RAM)
    la t2, _sdata_end          # End of .sdata section
    bgeu t1, t2, sdata_done    # Skip if no small data

sdata_loop:
    lw t3, 0(t0)               # Load word from ROM
    sw t3, 0(t1)               # Store word to RAM
    addi t0, t0, 4             # Increment source
    addi t1, t1, 4             # Increment destination
    bltu t1, t2, sdata_loop    # Continue until end

sdata_done:

    # ==========================================
    # Call main function
    # Should never return, but handle it anyway
    # ==========================================
    call main

    # ==========================================
    # If main returns, infinite loop
    # ==========================================
main_returned:
    j main_returned

.size _start, .-_start

# ==========================================
# Default trap handler (catches exceptions)
# ==========================================
.section .text
.weak trap_handler
.type trap_handler, @function
trap_handler:
    # In production, this catches any faults
    # For now, just loop infinitely
    j trap_handler
.size trap_handler, .-trap_handler