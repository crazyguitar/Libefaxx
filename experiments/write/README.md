# Write Benchmark

This experiment benchmarks libfabric RDMA WRITE operations over EFA, measuring
point-to-point bandwidth from rank 0 to all other ranks (`rank0 -> rank_k`,
k=1..N-1, results averaged across all pairs).

RDMA WRITE is a one-sided communication model where the sender directly writes
to the receiver's memory without receiver CPU involvement. The receiver is notified
via immediate data which triggers a CQE on the receive completion queue. This
enables zero-copy transfers with minimal receiver overhead.

The benchmark tests four configurations:
- `SinglePin`: Pinned host memory, single EFA channel
- `SingleDMA`: GPU memory with DMABUF, single EFA channel
- `MultiDMA`: GPU memory with DMABUF, all EFA channels (buffer split across channels)
- `RoundRobinDMA`: GPU memory with DMABUF, round-robin channel assignment per target

```
    Writer (Rank 0)                              Target (Rank 1)
    ===============                              ================

    ┌─────────────┐                              ┌─────────────┐
    │ Application │                              │ Application │
    └──────┬──────┘                              └──────┬──────┘
           │                                            │
           │ co_await Write()                           │ co_await WaitImmdata()
           ▼                                            ▼
    ┌─────────────┐                              ┌─────────────┐
    │ fi_writemsg │                              │  (waiting)  │
    │  (post WQE) │                              │             │
    └──────┬──────┘                              └──────┬──────┘
           │                                            │
           │ ┌────────────────────────────────────────┐ │
           │ │            EFA Network                 │ │
           ▼ │                                        │ ▼
    ┌────────┴───┐   RDMA Write (bypass recv CPU) ┌───┴────────┐
    │  Write WQE │ ══════════════════════════════►│ Remote MR  │
    │            │        + imm_data ────────────►│  (RX CQ)   │
    └──────┬─────┘                                └─────┬──────┘
           │                                            │
           │ CQE (write complete)                       │ CQE (imm_data arrived)
           ▼                                            ▼
    ┌─────────────┐                              ┌─────────────┐
    │   Resume    │                              │   Resume    │
    │  Coroutine  │                              │  Coroutine  │
    └─────────────┘                              └─────────────┘
```

## Results

Benchmark results on p5.48xlarge (4 nodes, 1 rank per node):

| Size |     Count |  SinglePin | BusBW (%) | Lat (us)  |  SingleDMA | BusBW (%) | Lat (us)  |   MultiDMA | BusBW (%) | Lat (us)  | RoundRobinDMA | BusBW (%) | Lat (us)  |
|-----:|----------:|-----------:|----------:|----------:|-----------:|----------:|----------:|-----------:|----------:|----------:|--------------:|----------:|----------:|
| 256K |     65536 |      31.58 |      31.6 |     66.54 |      32.56 |      32.6 |     64.42 |      32.76 |       8.2 |     64.11 |         90.54 |      22.6 |     69.49 |
| 512K |    131072 |      48.04 |      48.0 |     87.31 |      47.93 |      47.9 |     87.50 |      66.36 |      16.6 |     63.25 |        133.70 |      33.4 |     94.11 |
|   1M |    262144 |      63.27 |      63.3 |    132.63 |      64.23 |      64.2 |    130.61 |     114.82 |      28.7 |     73.06 |        181.24 |      45.3 |    138.85 |
|   2M |    524288 |      77.28 |      77.3 |    217.17 |      78.18 |      78.2 |    214.60 |     167.37 |      41.8 |    100.34 |        223.30 |      55.8 |    225.40 |
|   4M |   1048576 |      86.24 |      86.2 |    389.09 |      85.92 |      85.9 |    390.54 |     231.22 |      57.8 |    145.17 |        252.65 |      63.2 |    398.44 |
|   8M |   2097152 |      91.36 |      91.4 |    734.55 |      91.11 |      91.1 |    736.58 |     289.74 |      72.4 |    231.62 |        268.17 |      67.0 |    750.75 |
|  16M |   4194304 |      94.03 |      94.0 |   1427.45 |      93.99 |      94.0 |   1428.08 |     329.95 |      82.5 |    406.80 |        277.87 |      69.5 |   1449.08 |
|  32M |   8388608 |      95.75 |      95.8 |   2803.46 |      95.65 |      95.6 |   2806.56 |     354.61 |      88.7 |    757.00 |        282.07 |      70.5 |   2855.01 |
|  64M |  16777216 |      96.61 |      96.6 |   5556.89 |      96.59 |      96.6 |   5557.97 |     366.64 |      91.7 |   1464.32 |        284.74 |      71.2 |   5656.40 |
| 128M |  33554432 |      97.22 |      97.2 |  11044.12 |      97.29 |      97.3 |  11036.08 |     372.88 |      93.2 |   2879.58 |        286.64 |      71.7 |  11238.06 |
| 256M |  67108864 |      97.45 |      97.4 |  22037.31 |      97.42 |      97.4 |  22042.66 |     376.66 |      94.2 |   5701.45 |        287.61 |      71.9 |  22400.05 |
| 512M | 134217728 |      97.54 |      97.5 |  44032.13 |      97.54 |      97.5 |  44034.21 |     378.46 |      94.6 |  11348.69 |        288.13 |      72.0 |  44718.84 |
|   1G | 268435456 |      97.68 |      97.7 |  87936.29 |      97.71 |      97.7 |  87911.73 |     380.00 |      95.0 |  22604.91 |        288.58 |      72.1 |  89297.71 |

SinglePin and SingleDMA achieve ~97% bus bandwidth utilization on a single EFA channel,
demonstrating near-optimal performance for single-channel RDMA writes. MultiDMA scales
to ~380 Gbps (95% of 4x100G aggregate) by splitting buffers across all 4 EFA channels.
RoundRobinDMA reaches ~288 Gbps (~72%) because with 4 nodes (rank 0 → ranks 1,2,3),
only 3 targets exist, so only 3 channels are utilized. Each target is assigned a
different channel via round-robin `(peer + rank) % num_channels`, and writes to all
3 targets are overlapped concurrently via coroutines, achieving ~3x single-channel bandwidth.
