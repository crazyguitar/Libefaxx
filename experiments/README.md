# Libefaxx (AWS EFA Benchmark for GPU/CPU)

## Evaluation

### Direct libfabric SEND/RECV/WRITE Benchmarks

The figure below presents representative [SEND\/RECV](https://github.com/crazyguitar/Libefaxx/tree/main/experiments/sendrecv) benchmark results obtained on
[Amazon SageMaker HyperPod](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-slurm.html)
using p5.48xlarge instances. The results show that the two-sided round-trip
communication pattern (where one operation consists of a SEND followed by a RECV)
achieves an effective bandwidth of approximately 48 Gbps. We further evaluated
RDMA performance using both device memory and host memory, observing that both
memory types are capable of saturating EFA bandwidth once the message size
becomes sufficiently large.

In a separate [WRITE](https://github.com/crazyguitar/Libefaxx/tree/main/experiments/write) experiment, we benchmarked one-sided communication (WRITE) over
EFA. Using a single communication channel, both pinned host memory (registered via `cudaRegisterHost`)
and direct device memory access (DMA buffers) achieved bandwidths of approximately **97 Gbps**,
approaching the theoretical EFA peak bandwidth of **100 Gbps**.

We also evaluated scalability across multiple EFAs. By partitioning device
memory across all available EFAs—for example, leveraging four EFAs per GPU on p5
instance and splitting a 256 MB buffer into 4 chunks (each 64 MB), with each EFA writing a
portion of the data—we achieved an aggregate bandwidth of approximately [**380 Gbps**](data/p5/write.csv),
corresponding to **~95%** of the available bus bandwidth.

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

The following figures present results from a simple [Alltoall](all2all)
collective communication benchmark over EFA per GPU. In the first experiment,
which uses 8 nodes, the results show that device memory (DMA buffers) scales
effectively with the number of EFAs: leveraging four EFAs achieves approximately
4× higher performance compared to using a single EFA. In contrast, when using
pinned host memory with four EFAs, performance degrades and is worse than
the single-EFA configuration, indicating that pinned memory does not scale
efficiently for this Alltoall workload in this setup.

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

The second figure examines the relationship between Alltoall bandwidth and nodes
size. As the number of nodes doubles, the observed bandwidth decreases by
approximately 2×, which is expected since each process must exchange data with a
larger number of peers during each Alltoall operation.

In addition, results from Figures 3 and 4 show that using pinned host memory does
not yield performance improvements when scaling to all EFAs per GPU. In contrast,
device memory (DMA buffers) scales effectively: by partitioning the buffer into
four chunks and distributing them across four EFAs, the benchmark achieves nearly
4× higher performance compared to a single-EFA configuration.

![all2all](imgs/all2all.png)

### Multi-EFA Data Distribution: Round-Robin vs Split Strategies

We evaluated two strategies for distributing data across multiple EFAs: round-robin
assignment and data splitting. In round-robin mode, each buffer is assigned to a
different EFA in rotation, allowing independent transfers to overlap. In split mode,
a single buffer is partitioned across all EFAs, with each EFA transferring a portion
of the data in parallel.

Our results show that round-robin achieves higher bandwidth when transferring multiple
independent buffers, as concurrent transfers can fully overlap. However, for large
single-buffer transfers (e.g., 1 GB), round-robin provides no latency benefit since
the entire buffer still traverses a single channel. In contrast, splitting the buffer
across all EFAs—where each EFA transfers only 256 MB—significantly reduces latency
by parallelizing the transfer.

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

**Recommendation**: Use round-robin for workloads with multiple independent buffers
targeting different endpoints. Use data splitting for single large buffer transfers
to minimize latency.

![round-robin](imgs/round_robin.png)

### GPU-Initiated via CPU Proxy

GPU-Initiated Networking (GIN) is becoming essential for accelerating inter-node
GPU communication. A widely adopted technique uses a CPU proxy thread, as seen
in libraries like [MSCCL++](https://github.com/crazyguitar/mscclpp) and [UCCL](https://github.com/uccl-project/uccl/tree/main).
This approach implements a multi-producer, single-consumer (MPSC) pattern: GPUs
issue RDMA commands to a CPU thread, which then handles the actual data transfer.
However, existing benchmarks typically measure GIN performance alongside
collective communication operations, making it difficult to identify whether
bottlenecks originate from the RDMA protocol, RDMA hardware, or the CUDA kernel
itself. In this section, we benchmark GPU-initiated throughput using a proxy
thread with direct EFA WRITE operations, independent of any collective
communication algorithms.

Our testing reveals that CPU-initiated transfers (SingleDMA and MultiDMA)
reach over 95% of the theoretical 100 Gbps bandwidth limit. GPU-initiated
performance shows strong dependency on CUDA kernel execution patterns.
Critically, we identified that `quiet` operations (comparable to
`nvshmem_quiet`) that flush pending RDMA operations impose substantial overhead.
The first figure illustrates how latency increases dramatically when commands
are sent with `quiet` synchronization for each RDMA request to the CPU Proxy
thread. Conversely, non-blocking commands (similar to NVSHMEM's `nvshmem_int_nbi`
interface) maintain low latency by enabling pipelined GPU-to-CPU requests. The
second and third plots confirm that pairing each GPU-initiated operation with
`quiet` reduces performance by over 20% for large transfers and over 40% for
small writes (8MiB). Minimizing `quiet` call frequency allows performance to
match baseline levels (SingleDMA and MultiDMA) for large writes exceeding 64MiB.
The benchmark source code is available [here](https://github.com/crazyguitar/Libefaxx/blob/main/src/include/bench/modules/proxy.cuh).

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

### NVLink GPU-to-GPU Communication Performance

NVLink enables direct GPU-to-GPU data transfers with up to **3600 Gbps** theoretical
bandwidth, bypassing the CPU entirely. Our benchmarks reveal that NVLink throughput
is highly dependent on CUDA kernel parallelism. Specifically, increasing the grid
dimension significantly improves bandwidth, while block dimension has minimal impact
on performance. Additionally, NVLink bandwidth remains stable across message sizes,
unlike EFA where small writes underperform compared to large transfers.

```
 NVLink IPC Write (GPU-to-GPU Direct Memory Access)

   GPU 0 (Writer)                         GPU 1 (Target)
  ┌─────────────────┐                    ┌─────────────────┐
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
  - 1×256:   ~350 GB/s  (9% peak)   - insufficient parallelism
  - 128×256: ~2971 GB/s (78% peak)  - near-optimal utilization
```

In the following figure, large and small grid configurations show similar performance
at smaller data sizes because the data volume is insufficient to saturate all NVLink
connections. Throughput scales with increasing message size until NVLink saturation.
At 1 GB with 16 grid dimensions, we observed unexpected performance degradation,
suggesting potential contention when NVLink is fully saturated with large transfers.

![ipc](imgs/ipc.png)
