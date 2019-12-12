+++
title = "Evaluating the Performance Implications of Physical Addressing"
extra.author = "Drew Zagieboylo"
extra.bio = """
  [Drew Zagieboylo](https://www.cs.cornell.edu/~dzag/) is a 3rd year PhD student researching Security, Hardware Design, and Programming Languages. He enjoys rock climbing and gaming in his free time.
"""
+++


# Intro to Virtual Addressing

Modern processors use [_virtual addressing_](https://en.wikipedia.org/wiki/Virtual_address_space) to access _actual_ memory locations through a translation layer. Only highly privileged software, such as the operating system (OS), has access to physical memory addresses while all other processes can only refer to memory via these virtual addresses. When a process requests memory (e.g., via `malloc`), the OS will allocate physical memory in fixed size chunks, called pages, and then map them into the process' virtual address space. This allows the OS to allocate whichever regions of physical memory happen to be free despite the fact that the process may have requested a large, contiguous allocation.

Virtual addressing provides a few key abstractions for user-level software:
 1. A fully contiguous address space.
 2. A unique address space not shared by any other process.

The former enables software to easily calculate relative memory addresses; accessing any element in an array requires only one or two instructions to add the offset of the base pointer and then load from memory. Similarly, locations on the program stack are computed relative to the current stack pointer. Neither of these "pointer arithmetic" operations would be valid if executed on the physical addresses. The latter is a useful security primitive that enables strong process memory isolation "for free," since there is no way for a process to even reference memory owned by another process (unless the OS maps some physical location into both address spaces).


## The Case Against Virtual Addressing

