/* RISC-V Linker Script for 32-bit Fixed-Point Vector Processor
 * Bare-metal Wine Classifier
 * 
 * Memory Layout:
 *   ROM: 0x80000000 - 0x80100000 (1 MB)  - Code + Constants
 *   RAM: 0x80100000 - 0x80180000 (512 KB) - Data + Stack
 */

OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY
{
    /* Adjust these addresses to match YOUR hardware */
    ROM (rx)  : ORIGIN = 0x80000000, LENGTH = 1M
    RAM (rwx) : ORIGIN = 0x80100000, LENGTH = 512K
}

SECTIONS
{
    /* ===================================
       .text section - Code and constants
       Contains all executable code and read-only data
       =================================== */
    .text : {
        /* Startup code MUST come first (entry point) */
        *(.text.init)
        *(.text.startup)
        KEEP(*(.text.init))
        KEEP(*(.text.startup))
        
        /* Regular code sections */
        *(.text)
        *(.text*)
        *(.text.unlikely)
        *(.text.hot)
        
        /* C++ specific sections */
        *(.gnu.linkonce.t.*)
        
        /* Ensure alignment for constants */
        . = ALIGN(8);
        
        /* Read-only data (model parameters, constants, strings) */
        *(.rodata)
        *(.rodata*)
        *(.srodata)
        *(.srodata*)
        *(.gnu.linkonce.r.*)
        
        /* End marker for text section */
        . = ALIGN(8);
        _etext = .;
        PROVIDE(etext = .);
    } > ROM

    /* ===================================
       C++ Constructor/Destructor Tables
       For bare-metal, usually empty
       =================================== */
    .preinit_array : {
        PROVIDE_HIDDEN (__preinit_array_start = .);
        KEEP (*(.preinit_array))
        PROVIDE_HIDDEN (__preinit_array_end = .);
    } > ROM

    .init_array : {
        PROVIDE_HIDDEN (__init_array_start = .);
        KEEP (*(SORT_BY_INIT_PRIORITY(.init_array.*)))
        KEEP (*(SORT_BY_INIT_PRIORITY(.ctors.*)))
        KEEP (*(.init_array EXCLUDE_FILE (*crtbegin.o *crtbegin?.o *crtend.o *crtend?.o ) .ctors))
        PROVIDE_HIDDEN (__init_array_end = .);
    } > ROM

    .fini_array : {
        PROVIDE_HIDDEN (__fini_array_start = .);
        KEEP (*(SORT_BY_INIT_PRIORITY(.fini_array.*)))
        KEEP (*(SORT_BY_INIT_PRIORITY(.dtors.*)))
        KEEP (*(.fini_array EXCLUDE_FILE (*crtbegin.o *crtbegin?.o *crtend.o *crtend?.o ) .dtors))
        PROVIDE_HIDDEN (__fini_array_end = .);
    } > ROM

    /* ===================================
       .data section - Initialized data
       Copied from ROM to RAM at startup
       =================================== */
    .data : {
        . = ALIGN(8);
        _data_start = .;
        __data_start = .;
        PROVIDE(data_start = .);
        
        /* Initialized data sections */
        *(.data)
        *(.data*)
        *(.gnu.linkonce.d.*)
        
        . = ALIGN(8);
        _data_end = .;
        __data_end = .;
        PROVIDE(data_end = .);
    } > RAM AT > ROM
    
    /* Load address (where data is stored in ROM) */
    _data_load = LOADADDR(.data);
    _data_load_start = LOADADDR(.data);

    /* ===================================
       .sdata section - Small data
       Accessed via global pointer (gp)
       =================================== */
    .sdata : {
        . = ALIGN(8);
        _sdata_start = .;
        __sdata_start = .;
        
        /* Global pointer points to middle of sdata */
        /* Allows +/- 2KB access range */
        __global_pointer$ = . + 0x800;
        
        *(.sdata)
        *(.sdata*)
        *(.sdata2)
        *(.sdata2*)
        *(.gnu.linkonce.s.*)
        
        . = ALIGN(8);
        _sdata_end = .;
        __sdata_end = .;
    } > RAM AT > ROM
    
    _sdata_load = LOADADDR(.sdata);

    /* ===================================
       .got and .got.plt - Global Offset Table
       For position-independent code (usually empty)
       =================================== */
    .got : {
        . = ALIGN(8);
        *(.got)
        *(.got.plt)
        . = ALIGN(8);
    } > ROM

    /* ===================================
       .bss section - Uninitialized data
       Zeroed by startup code
       =================================== */
    .bss : {
        . = ALIGN(8);
        _bss_start = .;
        __bss_start = .;
        PROVIDE(bss_start = .);
        
        /* Uninitialized data sections */
        *(.bss)
        *(.bss*)
        *(.sbss)
        *(.sbss*)
        *(.gnu.linkonce.b.*)
        *(COMMON)
        
        . = ALIGN(8);
        _bss_end = .;
        __bss_end = .;
        PROVIDE(bss_end = .);
    } > RAM
    
    /* ===================================
       Heap and Stack allocation
       =================================== */
    
    /* Heap (if malloc is used - usually not in bare-metal) */
    . = ALIGN(16);
    _heap_start = .;
    __heap_start = .;
    PROVIDE(heap_start = .);
    
    /* Stack size: 16KB (0x4000 bytes)
     * Adjust if your code needs more/less stack */
    _stack_size = 0x4000;
    
    /* Stack location: Top of RAM */
    _stack_start = ORIGIN(RAM) + LENGTH(RAM) - _stack_size;
    _stack_top = ORIGIN(RAM) + LENGTH(RAM);
    __stack_top = _stack_top;
    PROVIDE(stack_top = _stack_top);
    
    /* Heap ends where stack begins */
    _heap_end = _stack_start;
    __heap_end = _heap_end;
    PROVIDE(heap_end = _heap_end);
    
    /* ===================================
       Memory region symbols
       =================================== */
    _rom_start = ORIGIN(ROM);
    _rom_end = ORIGIN(ROM) + LENGTH(ROM);
    _ram_start = ORIGIN(RAM);
    _ram_end = ORIGIN(RAM) + LENGTH(RAM);
    
    /* ===================================
       Discard unnecessary sections
       =================================== */
    /DISCARD/ : {
        *(.note*)
        *(.comment)
        *(.eh_frame*)
        *(.eh_frame_hdr)
        *(.ARM.exidx*)
        *(.ARM.extab*)
    }
}

/* ===================================
   Build-time assertions
   Catch configuration errors at link time
   =================================== */

/* Ensure stack and heap don't overlap */
ASSERT(_stack_top > _heap_start, "ERROR: Stack and heap overlap!")

/* Ensure BSS doesn't extend into stack */
ASSERT(_bss_end < _stack_start, "ERROR: BSS section extends into stack region!")

/* Ensure code fits in ROM */
ASSERT(_etext < ORIGIN(ROM) + LENGTH(ROM), "ERROR: Code section too large for ROM!")

/* Ensure data+bss fit in RAM (excluding stack) */
ASSERT(_bss_end < _stack_start, "ERROR: Data sections too large for available RAM!")

/* Provide minimum heap size warning (optional) */
/* ASSERT(_heap_end - _heap_start >= 0x1000, "WARNING: Heap size less than 4KB") */