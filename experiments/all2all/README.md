# All-to-All Benchmark

This experiment benchmarks libfabric RDMA WRITE operations in an all-to-all
communication pattern, where every rank sends data to every other rank simultaneously.
This pattern is common in distributed training for gradient synchronization and
MoE (Mixture of Experts) all-to-all exchanges.

Each rank writes to all other ranks concurrently using RDMA WRITE with immediate
data for completion notification. The benchmark measures aggregate bandwidth
across all participating ranks.

The benchmark tests five configurations:
- `SingleDMA`: GPU memory with DMABUF, single EFA channel for all peers
- `MultiDMA`: GPU memory with DMABUF, buffer split across all EFA channels per peer
- `RoundRobinDMA`: GPU memory with DMABUF, round-robin channel assignment `(peer + rank) % num_channels`
- `SinglePin`: Pinned host memory, single EFA channel
- `MultiPin`: Pinned host memory, buffer split across all EFA channels

```
All-to-all communication pattern (4 ranks):

   Rank 0          Rank 1          Rank 2          Rank 3
      │               │               │               │
      │ write[1]      │               │               │
      │──────────────>│               │               │
      │ write[2]      │               │               │
      │──────────────────────────────>│               │
      │ write[3]      │               │               │
      │──────────────────────────────────────────────>│
      │               │ write[0]      │               │
      │<──────────────│               │               │
      │               │ write[2]      │               │
      │               │──────────────>│               │
      │               │ write[3]      │               │
      │               │──────────────────────────────>│
      │               │               │ write[0]      │
      │<──────────────────────────────│               │
      │               │               │ write[1]      │
      │               │<──────────────│               │
      │               │               │ write[3]      │
      │               │               │──────────────>│
      │               │               │               │ write[0]
      │<──────────────────────────────────────────────│
      │               │               │               │ write[1]
      │               │<──────────────────────────────│
      │               │               │               │ write[2]
      │               │               │<──────────────│
      ▼               ▼               ▼               ▼
```

## Results

Benchmark results on p5.48xlarge (4 nodes, 1 rank per node):

| Size |    Count | SingleDMA | BusBW (%) | Lat (us)  | MultiDMA | BusBW (%) | Lat (us) | RoundRobinDMA | BusBW (%) | Lat (us)  | SinglePin | BusBW (%) | Lat (us)  | MultiPin | BusBW (%) | Lat (us)  |
|-----:|---------:|----------:|----------:|----------:|---------:|----------:|---------:|--------------:|----------:|----------:|----------:|----------:|----------:|---------:|----------:|----------:|
|   4M |  1048576 |     12.53 |      12.5 |   2677.27 |    40.49 |      10.1 |   828.78 |         39.64 |       9.9 |    846.58 |     12.57 |      12.6 |   2669.65 |     7.27 |       1.8 |   4618.26 |
|   8M |  2097152 |     12.84 |      12.8 |   5228.39 |    44.77 |      11.2 |  1499.11 |         41.33 |      10.3 |   1623.75 |     13.06 |      13.1 |   5138.83 |     9.13 |       2.3 |   7350.10 |
|  16M |  4194304 |     12.70 |      12.7 |  10564.54 |    46.86 |      11.7 |  2864.34 |         43.13 |      10.8 |   3112.11 |     12.97 |      13.0 |  10349.76 |    11.48 |       2.9 |  11695.65 |
|  32M |  8388608 |     11.97 |      12.0 |  22427.68 |    48.53 |      12.1 |  5531.00 |         42.21 |      10.6 |   6359.63 |     12.19 |      12.2 |  22012.09 |    11.98 |       3.0 |  22409.94 |
|  64M | 16777216 |     12.64 |      12.6 |  42488.40 |    48.97 |      12.2 | 10962.23 |         42.38 |      10.6 |  12668.68 |     12.69 |      12.7 |  42298.21 |    12.18 |       3.0 |  44074.23 |
| 128M | 33554432 |     12.73 |      12.7 |  84351.53 |    46.73 |      11.7 | 22976.40 |         39.95 |      10.0 |  26878.22 |     13.08 |      13.1 |  82064.63 |    12.98 |       3.2 |  82696.61 |

SingleDMA and SinglePin achieve ~12-13% bus bandwidth on a single channel, limited by
the all-to-all pattern where each rank must send to N-1 peers sequentially on one channel.
MultiDMA reaches ~47-49 Gbps by splitting each peer's buffer across all 4 EFA channels.
RoundRobinDMA achieves ~40-43 Gbps with round-robin channel assignment per peer, where
writes to different peers are overlapped concurrently via coroutines. MultiPin was
expected to perform similarly to MultiDMA, but experiments show no performance gain
and even worse results (~3% BusBW), suggesting pinned memory does not benefit from
multi-channel parallelism in this workload.
