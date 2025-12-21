# Affinity

This example demonstrates how to discover and analyze GPU–device topology using
[libhwloc](https://github.com/open-mpi/hwloc). In particular, it illustrates
how each GPU can be associated with multiple Elastic Fabric Adapter (EFA)
devices—for example, a single GPU may be adjacent to four EFAs, all of which
can be leveraged for high-performance RDMA-based data transfers.

Understanding hardware topology is essential for performance analysis in
GPU-based systems. GPU execution frequently interacts with host memory,
and factors such as NUMA locality can significantly impact memory access latency.
For example, pinning a process to a remote NUMA node may increase overhead when
data is transferred between GPU and CPU memory.

The example prints the system hardware topology, exposing the affinity relationships
among GPUs, NUMA nodes, and EFA devices to support analysis, debugging, and
inspection of device connectivity and locality.
