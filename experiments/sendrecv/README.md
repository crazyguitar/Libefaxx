# Send/Recv Benchmark

This experiment benchmarks libfabric SEND/RECV operations over EFA, measuring
point-to-point bandwidth between rank 0 and all other ranks in a ping-pong pattern
(`rank0 <-> rank_k`, k=1..N-1, results averaged across all pairs).

SEND/RECV is a two-sided communication model where both sender and receiver must
actively participate. The receiver must pre-post a buffer via `fi_recvmsg` before
the sender's message arrives. Once each operation completes, the completion queue
(CQ) receives an event (`FI_SEND` or `FI_RECV`) to notify the process, similar to
TCP send/recv I/O operations.

The benchmark tests two buffer types:
- `SingleDevice`: GPU memory with DMABUF registration for zero-copy RDMA
- `SingleHost`: Host pinned memory registered with libfabric

```
    Sender (Rank 0)                              Receiver (Rank 1)
    ===============                              =================

    ┌─────────────┐                              ┌─────────────┐
    │ Application │                              │ Application │
    └──────┬──────┘                              └──────┬──────┘
           │                                            │
           │ co_await Send()                            │ co_await Recv()
           ▼                                            ▼
    ┌─────────────┐                              ┌─────────────┐
    │ fi_sendmsg  │                              │ fi_recvmsg  │
    │  (post WQE) │                              │  (post WQE) │
    └──────┬──────┘                              └──────┬──────┘
           │                                            │
           │ ┌────────────────────────────────────────┐ │
           │ │            EFA Network                 │ │
           ▼ │                                        │ ▼
    ┌────────┴───┐    RDMA Send (data + header)  ┌───┴────────┐
    │  Send WQE  │ ─────────────────────────────►│  Recv WQE  │
    │  (TX CQ)   │                               │  (RX CQ)   │
    └──────┬─────┘                               └─────┬──────┘
           │                                           │
           │ CQE (send complete)                       │ CQE (recv complete)
           ▼                                           ▼
    ┌─────────────┐                              ┌─────────────┐
    │   Resume    │                              │   Resume    │
    │  Coroutine  │                              │  Coroutine  │
    └─────────────┘                              └─────────────┘
```

## Results

Benchmark results on p5.48xlarge (2 nodes, 1 rank per node, single EFA channel):

| Size | Count    | Device (Gbps) | BusBW (%) | Lat (us)  | Host (Gbps) | BusBW (%) | Lat (us)  |
|-----:|---------:|--------------:|----------:|----------:|------------:|----------:|----------:|
| 128K |    32768 |          9.75 |       9.8 |    107.52 |        7.14 |       7.1 |    146.79 |
| 256K |    65536 |         14.51 |      14.5 |    144.61 |       12.05 |      12.1 |    174.06 |
| 512K |   131072 |         20.71 |      20.7 |    202.57 |       14.22 |      14.2 |    295.07 |
|   1M |   262144 |         30.57 |      30.6 |    274.44 |       26.83 |      26.8 |    312.70 |
|   2M |   524288 |         25.69 |      25.7 |    653.01 |       34.63 |      34.6 |    484.50 |
|   4M |  1048576 |         41.57 |      41.6 |    807.16 |       40.23 |      40.2 |    834.11 |
|   8M |  2097152 |         44.98 |      45.0 |   1491.89 |       43.70 |      43.7 |   1535.60 |
|  16M |  4194304 |         46.19 |      46.2 |   2906.00 |       45.90 |      45.9 |   2924.09 |
|  32M |  8388608 |         47.33 |      47.3 |   5671.67 |       47.00 |      47.0 |   5711.56 |
|  64M | 16777216 |         47.97 |      48.0 |  11191.56 |       47.71 |      47.7 |  11252.39 |
| 128M | 33554432 |         48.28 |      48.3 |  22237.97 |       48.23 |      48.2 |  22262.47 |

Both Device (GPU DMABUF) and Host (pinned memory) achieve ~48 Gbps at large message
sizes, approaching the theoretical 100 Gbps EFA link bandwidth (~48% utilization).
At smaller sizes, Device memory shows slightly better bandwidth due to lower
registration overhead. The performance gap narrows as message size increases,
where transfer time dominates over setup costs.
