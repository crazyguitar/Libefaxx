# AWS EFA Benchmark for GPU/CPU


## Direct libfabric SEND/RECV/WRITE Benchmarks

Amazon Elastic Fabric Adapter (EFA) provides high-bandwidth, low-latency
networking for HPC and machine learning workloads on AWS. Understanding raw EFA
performance characteristics is essential for optimizing distributed training
pipelines, particularly for large language models (LLMs) where inter-node
communication often becomes the bottleneck. This section presents EFA bandwidth
and latency measurements using libfabric [SEND/RECV](sendrecv) and [WRITE](write) operations on
[Amazon SageMaker HyperPod](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-slurm.html)
p5.48xlarge instances, which feature 8 NVIDIA H100 GPUs and 32 EFA devices
(4 EFAs per GPU).

**Two-Sided Communication (SEND/RECV):** Two-sided operations require both
sender and receiver to participate in the transfer, introducing synchronization
overhead but providing explicit flow control. Our benchmarks show that
round-trip operations (SEND followed by RECV) achieve approximately **48 Gbps**
effective bandwidth. Both device memory (DMA buffers) and pinned host memory
saturate EFA bandwidth at sufficiently large message sizes, indicating that
memory type does not limit peak throughput for bulk transfers.

**One-Sided Communication (WRITE):** One-sided RDMA WRITE operations bypass the
remote CPU entirely, allowing the initiator to write directly to pre-registered
remote memory. This eliminates receiver-side overhead and is particularly
beneficial for latency-sensitive workloads. Single-channel RDMA WRITE operations
reach approximately **97 Gbps**, approaching the theoretical **100 Gbps** EFA
peak. Both pinned host memory (`cudaHostRegister`) and device memory achieve
similar throughput, demonstrating that GPUDirect RDMA effectively eliminates
CPU-mediated copies.

**Multi-EFA Scaling:** Modern AWS GPU instances provide multiple EFA devices to
maximize aggregate bandwidth. By partitioning device memory across all available
EFAs (e.g., splitting a 256 MB buffer into 4 × 64 MB chunks on p5 instances with
4 EFAs per GPU), we achieve approximately **380 Gbps** aggregate bandwidth,
corresponding to **~95%** bus utilization. This near-linear scaling demonstrates
that EFA hardware does not introduce significant contention when properly
load-balanced.

![bandwidth](imgs/bandwidth.png)

```
 Two-Sided (SEND/RECV)                    |    One-Sided (WRITE)
                                          |
  Node A           Node B                 |   Node A           Node B
    |                 |                   |     |                 |
    |----SEND[0]----->|                   |     |----WRITE[0]---->|
    |<---RECV[0]------|                   |     |----WRITE[1]---->|
    |----SEND[1]----->|                   |     |----WRITE[2]---->|
    |<---RECV[1]------|                   |     |      ...        |
    |      ...        |                   |     v                 v
    v                 v                   |
                                          |
  - Round-trip latency measured           |   - One-way transfer
  - ~48 Gbps effective bandwidth          |   - ~97 Gbps bandwidth (single EFA)
                                          |   - ~380 Gbps aggregate (4 EFAs)
```

## All-to-All Collective Communication Benchmark

All-to-all communication is a fundamental collective operation where every
process exchanges data with every other process. This pattern appears frequently
in distributed deep learning, particularly in expert parallelism (MoE models),
tensor parallelism, and sequence parallelism where activations must be
redistributed across devices. The [Alltoall](all2all) benchmark evaluates
collective communication performance over EFA, measuring how bandwidth scales
with EFA count and cluster size—critical factors for understanding real-world
distributed training performance.

**EFA Scaling (8 nodes):** Device memory (DMA buffers) scales linearly with EFA
count—using 4 EFAs achieves approximately **4× throughput** compared to a single
EFA. This confirms that the EFA hardware and libfabric software stack can
effectively parallelize transfers across multiple network interfaces. However,
pinned host memory exhibits degraded performance with multiple EFAs, performing
worse than single-EFA configurations. This degradation likely stems from
increased PCIe contention when multiple EFAs compete for host memory bandwidth.

**Node Scaling:** Bandwidth decreases proportionally as node count increases.
Doubling the number of nodes halves the observed per-node bandwidth, as expected
since each process must exchange data with a larger number of peers during
all-to-all operations. This O(N²) communication pattern makes all-to-all
particularly sensitive to cluster size, highlighting the importance of
algorithmic optimizations (e.g., hierarchical collectives) for large-scale
deployments.

**Memory Type Comparison:** Device memory consistently outperforms pinned host
memory when scaling across multiple EFAs. Partitioning buffers across 4 EFAs
achieves nearly **4× performance** versus single-EFA configurations. This
advantage stems from GPUDirect RDMA's ability to transfer data directly between
GPU memory and the network, avoiding the PCIe round-trip through host memory.

![all2all](imgs/all2all.png)

```
 SingleDMA / SinglePin                    |    MultiDMA / MultiPin
 (Single EFA Channel)                     |    (Split Across All EFAs)
                                          |
  Buffer ──────► EFA 0 ──────► Network    |    Buffer ─┬─► EFA 0 ─► chunk[0] ──┐
                                          |            ├─► EFA 1 ─► chunk[1] ──┼──► Network
  - All data through one channel          |            ├─► EFA 2 ─► chunk[2] ──┤
  - ~97 Gbps max (single EFA limit)       |            └─► EFA 3 ─► chunk[3] ──┘
  - Simple, no coordination needed        |
                                          |    - Data split across all channels
                                          |    - ~380 Gbps aggregate (4× scaling)
                                          |    - Requires buffer partitioning
```

## Multi-EFA Data Distribution: Round-Robin vs Split Strategies

When utilizing multiple EFA devices, the data distribution strategy
significantly impacts both throughput and latency. Choosing the optimal strategy
depends on the workload characteristics: whether transferring multiple
independent buffers or a single large buffer. This section compares two
strategies for distributing data across multiple EFAs: **round-robin
assignment** and **data splitting**.

**Round-Robin:** Each buffer is assigned to a different EFA in rotation (buffer
A → EFA 0, buffer B → EFA 1, etc.), enabling concurrent independent transfers to
overlap. This approach maximizes aggregate throughput when multiple buffers are
in flight simultaneously, as each EFA operates independently without
coordination overhead. Round-robin is optimal for workloads with multiple
independent buffers targeting different endpoints, such as parameter server
updates or gradient aggregation across multiple peers.

**Data Splitting:** A single buffer is partitioned across all EFAs, with each
EFA transferring a portion in parallel (e.g., a 1 GB buffer split into 4 × 256
MB chunks). This approach minimizes latency for individual large transfers by
parallelizing the data path, but requires coordination to reassemble the buffer
at the destination. Data splitting is optimal for large single-buffer transfers
where minimizing latency is critical, such as broadcasting model weights or
collecting large activation tensors.

**Key Finding:** Round-robin achieves higher aggregate bandwidth for multiple
independent buffers due to transfer overlap. However, for large single-buffer
transfers (e.g., 1 GB), round-robin provides no latency benefit since the entire
buffer still traverses a single channel. Splitting across EFAs (4 × 256 MB)
significantly reduces latency by parallelizing the transfer path.

```
 Round-Robin (per-buffer assignment)      |    Data Splitting (per-chunk assignment)
                                          |
  Buffer A ──► EFA 0 ──────► Endpoint 0   |    Buffer ─┬─► EFA 0 ─► chunk[0] ──┐
  Buffer B ──► EFA 1 ──────► Endpoint 1   |            ├─► EFA 1 ─► chunk[1] ──┼──► Endpoint
  Buffer C ──► EFA 2 ──────► Endpoint 2   |            ├─► EFA 2 ─► chunk[2] ──┤
  Buffer D ──► EFA 3 ──────► Endpoint 3   |            └─► EFA 3 ─► chunk[3] ──┘
                                          |
  - Each buffer uses one channel          |    - Single buffer split across channels
  - Concurrent transfers overlap          |    - Parallel transfer reduces latency
  - Best for multiple independent bufs    |    - Best for single large buffer
```

**Recommendation:** Use round-robin for workloads with multiple independent
buffers targeting different endpoints. Use data splitting for single large
buffer transfers to minimize latency.

![round-robin](imgs/round_robin.png)

## GPU-Initiated Communication via CPU Proxy

GPU-Initiated Networking (GIN) is an emerging paradigm that allows GPUs to
trigger network operations directly, reducing CPU involvement and enabling
tighter integration between computation and communication. This approach is
particularly valuable for Mixture-of-Experts (MoE) models, where fine-grained
all-to-all communication patterns benefit from GPU-driven scheduling. A common
implementation uses a CPU proxy thread in a multi-producer, single-consumer
(MPSC) pattern: GPU threads push RDMA commands to a shared queue, and a
dedicated CPU thread polls the queue and issues the actual network operations.
This pattern is employed by production libraries including
[MSCCL++](https://github.com/microsoft/mscclpp) and
[UCCL](https://github.com/uccl-project/uccl).

Existing benchmarks typically measure GIN performance alongside collective
communication algorithms, making it difficult to isolate whether bottlenecks
originate from the RDMA protocol, network hardware, or CUDA kernel overhead.
This [benchmark](proxy) isolates GIN throughput from collective communication
overhead, measuring raw GPU-initiated EFA WRITE performance to identify
optimization opportunities.

**Baseline Performance:** CPU-initiated transfers (SingleDMA, MultiDMA) achieve
over **95%** of the theoretical 100 Gbps EFA bandwidth, establishing the
performance ceiling for GPU-initiated approaches.

**Synchronization Overhead:** The `quiet` operation (analogous to `nvshmem_quiet`
or memory fence) that flushes pending RDMA requests and ensures completion
introduces substantial overhead. Issuing `quiet` after each RDMA request
increases latency dramatically and reduces throughput by **>20%** for large
transfers and **>40%** for small writes (8 MiB). This overhead stems from the
round-trip synchronization between GPU and CPU proxy, creating pipeline bubbles.

**Non-Blocking Interface:** Using non-blocking commands (similar to
`nvshmem_put_nbi`) enables pipelined GPU-to-CPU requests, maintaining low
latency by allowing multiple RDMA operations to be in flight simultaneously.
Minimizing `quiet` frequency—batching multiple operations before
synchronization—allows performance to match CPU-initiated baselines for
transfers exceeding 64 MiB.

Source code: [proxy.cuh](https://github.com/crazyguitar/Libefaxx/blob/main/src/include/bench/modules/proxy.cuh)

![proxy](imgs/proxy.png)

```
 Blocking Interface (each Quiet)          |    NBI (Non-Blocking Interface)
                                          |
   GPU      CPU Proxy     Network         |    GPU      CPU Proxy    Network
    |           |            |            |     |           |            |
    |--Push[0]->|            |            |     |--Push[0]->|            |
    |           |--RDMA[0]-->|            |     |--Push[1]->|--RDMA[0]-->|
    |<-Complete-|            |            |     |--Push[2]->|--RDMA[1]-->|
    | (Quiet)   |            |            |     |    ...    |--RDMA[2]-->|
    |           |            |            |     | (Quiet)   |     ...    |
    |--Push[1]->|            | <- bubble  |     |<-Complete-|            |
    |           |--RDMA[1]-->|            |     v           v            v
    |<-Complete-|            |            |
    | (Quiet)   |            |            |
    v           v            v            |
```

### Command Queue Implementation Comparison

This section compares three queue implementations for GPU-CPU communication:
`cudaMallocManaged` (unified memory), `cudaHostAlloc + cudaHostGetDevicePointer`
(pinned host memory), and `GDRCopy` (GPU memory with BAR1 mapping). The command
queue serves as the communication channel between GPU threads and the CPU proxy,
where GPU kernels push RDMA commands and the CPU thread polls and executes them.
Choosing the right queue implementation significantly impacts end-to-end latency
and throughput.

**Unified Memory Latency Growth:** `cudaMallocManaged` exhibits latency that
scales with message size due to CUDA's page migration mechanism. In blocking
mode, each `Quiet` operation forces a synchronization point where the GPU must
wait for the CPU to acknowledge completion. During this wait, unified memory
pages containing the command queue are migrated between GPU and CPU memory on
each push-pop cycle. As payload size increases, more pages must be migrated per
operation, directly increasing latency. In contrast, `PinnedQueue` (host memory
with GPU device pointer) and `GdrQueue` (GPU memory with CPU BAR1 mapping)
maintain fixed memory locations—no page migration occurs regardless of payload
size, resulting in stable latency.

**NBI Amortization:** With non-blocking interface (NBI), the GPU pipelines
multiple push operations before issuing a single `Quiet` at the end. This
amortizes the page migration cost across many operations: pages migrate once
per batch rather than once per operation. The pipelining also allows the unified
memory driver to optimize page placement, reducing thrashing. This explains why
`cudaMallocManaged` with NBI shows minimal latency growth and can outperform
other implementations in some cases—the batched access pattern aligns well with
unified memory's design for bulk transfers rather than fine-grained
synchronization.

![queue](imgs/queue.png)

### Queue Performance with RDMA Operations

When combined with RDMA operations through the Proxy thread, queue
implementation differences become more pronounced. The figures below compare
queue performance in both blocking mode (SingleBlocking, MultiBlocking) and
non-blocking mode (SingleNBI, MultiNBI) across different message sizes.

**Blocking Mode:** `cudaMallocManaged` exhibits worse performance for small
writes compared to `PinnedQueue` and `GdrQueue`. The performance degradation
stems from page migration overhead amplified by RDMA operations—each `Quiet`
not only triggers page migration for the command queue, but also serializes
the RDMA completion path. For small writes, the page migration latency
dominates the total transfer time, making `cudaMallocManaged` significantly
slower than alternatives with fixed memory locations.

**NBI Mode:** SingleNBI shows no obvious difference across implementations
because the single-EFA configuration is already bottlenecked by network latency
rather than queue access. However, MultiNBI reveals that `cudaMallocManaged`
still underperforms compared to other queue implementations. When four EFAs
operate in parallel, the aggregate command latency decreases, and the page
migration overhead of `cudaMallocManaged` becomes the limiting factor,
preventing utilization of the available network bandwidth.

![proxy_queue](imgs/proxy_queue.png)

## NVLink GPU-to-GPU Communication Performance

NVLink is NVIDIA's high-bandwidth interconnect for direct GPU-to-GPU
communication, providing up to **3600 Gbps** (450 Gbps) theoretical bandwidth on
H100 GPUs—approximately 36× faster than PCIe Gen5. NVLink bypasses the CPU
entirely, enabling peer-to-peer memory access between GPUs within the same node.
This capability is essential for intra-node communication in tensor parallelism
and pipeline parallelism, where GPUs frequently exchange activations and
gradients.

Using CUDA IPC (Inter-Process Communication), processes can share GPU memory
across process boundaries. The exporting process calls `cudaIpcGetMemHandle` to
create a shareable handle, and the importing process calls `cudaIpcOpenMemHandle`
to map the remote memory into its address space. Once mapped, CUDA kernels can
read from and write to remote GPU memory directly over NVLink, enabling
zero-copy data exchange.

**Parallelism Impact:** Unlike EFA where bandwidth is primarily limited by
network hardware, NVLink throughput depends heavily on CUDA kernel parallelism.
Increasing grid dimensions (number of thread blocks) significantly improves
bandwidth by generating more concurrent memory transactions, while block
dimensions (threads per block) have minimal impact. This behavior reflects
NVLink's ability to handle many small transactions efficiently when sufficient
parallelism is available.

**Benchmark Results ([IPC](ipc)):**
- 1×256 grid: ~350 Gbps (9% peak) — insufficient parallelism to saturate NVLink
- 128×256 grid: ~2971 Gbps (78% peak) — near-optimal utilization

**Scaling Behavior:** Small and large grid configurations achieve similar
performance at smaller data sizes where NVLink is not saturated. Throughput
scales with message size until saturation. At 1 GB with 16 grid dimensions,
performance degradation suggests potential contention under full NVLink load,
possibly due to memory controller saturation or TLB pressure.

Note: Current implementation achieves ~78% peak due to `cudaStreamSynchronize`
overhead after each round. Asynchronous completion tracking could improve
utilization.

```
 NVLink IPC Write (GPU-to-GPU Direct Memory Access)

   Process A (GPU 0)                      Process B (GPU 1)
  ┌─────────────────┐                    ┌─────────────────┐
  │ cudaIpcGetMemHandle()                │ cudaIpcOpenMemHandle()
  │        │        |                    │        │        │
  │        └──── IPC Handle ─────────────│────────┘        │
  │                 |                    │                 │
  │  CUDA Kernel    │                    │  Remote Memory  │
  │  <<<grid,block>>│                    │                 │
  │                 │      NVLink        │                 │
  │  Thread 0 ──────│────────────────────│──► remote[0]    │
  │  Thread 1 ──────│────────────────────│──► remote[1]    │
  │  Thread 2 ──────│────────────────────│──► remote[2]    │
  │      ...        │                    │       ...       │
  │  Thread N ──────│────────────────────│──► remote[N]    │
  │                 │                    │                 │
  │  __threadfence_system()              │                 │
  └─────────────────┘                    └─────────────────┘

  Grid Parallelism Impact:
  - 1×256:   ~350 Gbps  (9% peak)   - insufficient parallelism
  - 128×256: ~2971 Gbps (78% peak)  - near-optimal utilization
```

![ipc](imgs/ipc.png)
