# Proxy Write Benchmark

This experiment benchmarks GPU-initiated RDMA writes via a CPU proxy, demonstrating
a GPU-Initiated Networking (GIN) pattern. The GPU kernel pushes write requests to
a queue, and a CPU coroutine polls the queue and executes RDMA operations on behalf
of the GPU.

This pattern enables GPU-driven communication without requiring GPU-side networking
support, useful for implementing custom collective operations or MoE all-to-all
exchanges where the GPU controls the communication schedule.

The benchmark tests four configurations:
- `ProxySingle`: Single EFA channel, blocking (Quiet after each write)
- `ProxyMulti`: All EFA channels, blocking (Quiet after each write)
- `ProxySingleNBI`: Single EFA channel, non-blocking interface (Quiet at end)
- `ProxyMultiNBI`: All EFA channels, non-blocking interface (Quiet at end)

```
Blocking (Quiet per write):              NBI (Non-Blocking Interface):

   GPU       CPU Proxy    Network           GPU       CPU Proxy    Network
    │            │           │               │            │           │
    │──Push[0]──>│           │               │──Push[0]──>│           │
    │            │──RDMA[0]─>│               │──Push[1]──>│──RDMA[0]─>│
    │<─Complete──│           │               │──Push[2]──>│──RDMA[1]─>│
    │  (Quiet)   │           │               │    ...     │──RDMA[2]─>│
    │            │           │               │  (Quiet)   │    ...    │
    │──Push[1]──>│           │               │<─Complete──│           │
    │            │──RDMA[1]─>│               ▼            ▼           ▼
    │<─Complete──│           │
    │  (Quiet)   │           │           NBI eliminates per-operation
    ▼            ▼           ▼           wait bubbles by overlapping
                                         push and RDMA operations.
    ^ GPU idle waiting for
      each completion
```

## Results

Benchmark results on p5.48xlarge (4 nodes, 1 rank per node):

| Size |     Count | ProxySingle | BusBW (%) | Lat (us)  | ProxyMulti | BusBW (%) | Lat (us)  | ProxySingleNBI | BusBW (%) | Lat (us)  | ProxyMultiNBI | BusBW (%) | Lat (us)  |
|-----:|----------:|------------:|----------:|----------:|-----------:|----------:|----------:|---------------:|----------:|----------:|--------------:|----------:|----------:|
|   8M |   2097152 |       52.50 |      52.5 |   1279.14 |      93.58 |      23.4 |    718.64 |          82.78 |      82.8 |    811.19 |        214.07 |      53.5 |    314.83 |
|  16M |   4194304 |       61.90 |      61.9 |   2168.69 |     118.36 |      29.6 |   1134.28 |          89.26 |      89.3 |   1503.80 |        271.29 |      67.8 |    495.28 |
|  32M |   8388608 |       67.20 |      67.2 |   3994.87 |     142.82 |      35.7 |   1879.80 |          92.62 |      92.6 |   2898.32 |        277.77 |      69.4 |    970.55 |
|  64M |  16777216 |       71.81 |      71.8 |   7476.35 |     160.29 |      40.1 |   3349.52 |          95.50 |      95.5 |   5621.97 |        329.24 |      82.3 |   1630.89 |
| 128M |  33554432 |       74.21 |      74.2 |  14469.47 |     172.48 |      43.1 |   6225.55 |          96.54 |      96.5 |  11122.86 |        345.69 |      86.4 |   3106.84 |
| 256M |  67108864 |       75.93 |      75.9 |  28281.26 |     180.23 |      45.1 |  11914.96 |          97.21 |      97.2 |  22090.19 |        363.97 |      91.0 |   5900.22 |
| 512M | 134217728 |       76.90 |      76.9 |  55853.67 |     185.22 |      46.3 |  23188.72 |          97.43 |      97.4 |  44084.66 |        374.16 |      93.5 |  11478.94 |
|   1G | 268435456 |       77.42 |      77.4 | 110953.88 |     188.63 |      47.2 |  45539.67 |          97.60 |      97.6 |  88009.27 |        379.08 |      94.8 |  22660.21 |

ProxySingleNBI achieves ~97% single-channel bandwidth at large sizes, demonstrating that
the NBI pipelining effectively hides the GPU-CPU queue latency. ProxyMultiNBI scales to
~379 Gbps (~95% of 4x100G) by combining NBI pipelining with multi-channel parallelism.
Blocking modes (ProxySingle/ProxyMulti) show lower efficiency due to GPU idle time
waiting for each RDMA completion before pushing the next request.
