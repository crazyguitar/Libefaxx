# Libefaxx (AWS EFA Benchmark for GPU/CPU)

High-performance inter-node communication over AWS Elastic Fabric Adapter (EFA)
is a key enabler for scaling large-language-model (LLM) training efficiently.
Existing benchmarking tools primarily focus on collective communication libraries
such as [NCCL](https://github.com/NVIDIA/nccl) or [NVSHMEM](https://github.com/NVIDIA/nvshmem),
making it difficult to isolate and understand the raw performance characteristics
of EFA itself. At the same time, [GPU-Initiated Networking](https://arxiv.org/pdf/2511.15076) (GIN)
has gained significant attention following the release of [Deep-EP](https://github.com/deepseek-ai/DeepEP),
which demonstrated substantial MoE performance gains by enabling GPU-driven
communication.

This repository provides a focused benchmarking framework for EFA, designed to
analyze low-level inter-node communication performances. It complements existing tools
such as [nccl-tests](https://github.com/NVIDIA/nccl-tests) by enabling direct
measurement of EFA latency, bandwidth, and GIN behavior, helping engineers and
researchers optimize distributed training pipelines on AWS.

## Development

The following snippets demonstrate how to build the source code for a simple test.
To save time on environment setup and dependency management, this repository provides
a [Dockerfile](Dockerfile) that can be used to build the project in a consistent
and reproducible environment.

```bash
# build a Docker image
docker build -f Dockerfile -t cuda:latest .

# build examples
make build
```

If [enroot](https://github.com/NVIDIA/enroot) is available in your environment,
you can launch the experiment using the following commands:

```bash
# build an enroot sqush file
make sqush

# launch an interactive enroot environment
enroot create --name cuda cuda+latest.sqsh
enroot start --mount /fsx:/fsx cuda /bin/bash

# run a test via enroot on a Slurm cluster
srun -N 1 \
  --container-image "${PWD}/cuda+latest.sqsh"  \
  --container-mounts /fsx:/fsx \
  --container-name cuda \
  --mpi=pmix \
  --ntasks-per-node=1 \
  "${PWD}/build/experiments/affinity/affinity"
```

## Example

When implementing custom algorithms directly over EFA, developers often face the
complexity of asynchronous RDMA APIs and event-driven scheduling. To simplify
this workflow, this repository includes a coroutine-based scheduler built on
[C++20 coroutine](https://en.cppreference.com/w/cpp/language/coroutines.html),
enabling a more straightforward programming model without manual callback management.
The example below shows how to build a PoC using pure [libfabric](https://github.com/ofiwg/libfabric/) and [MPI](https://www.open-mpi.org/).

```cpp
#include <io/runner.h>
#include <rdma/fabric/memory.h>
#include <bench/mpi/fabric.cuh>

struct PairBench {
  int target;
  template <typename T>
  void operator()(FabricBench& peer, FabricBench::Buffers<T>& send, FabricBench::Buffers<T>& recv) {
    for (auto& efa : peer.efas) IO::Get().Join<FabricSelector>(efa);
    Run([&]() -> Coro<> {
      size_t channel = 0;
      if (peer.mpi.GetWorldRank() == 0) {
        co_await send[target]->Sendall(channel);
        co_await recv[target]->Recvall(channel);
      } else if (peer.mpi.GetWorldRank() == target) {
        co_await recv[0]->Recvall(channel);
        co_await send[0]->Sendall(channel);
      }
      for (auto& efa : peer.efas) IO::Get().Quit<FabricSelector>(efa);
    }());
  }
};

template <typename BufType>
struct Test {
  static BenchResult Run(size_t size) {
    FabricBench peer;
    peer.Exchange();
    peer.Connect();
    int rank = peer.mpi.GetWorldRank();
    int world = peer.mpi.GetWorldSize();
    auto send = peer.Alloc<BufType>(size, rank);
    auto recv = peer.Alloc<BufType>(size, -1);
    auto noop = [](auto&, auto&) {};
    std::vector<BenchResult> res;
    for (int t = 1; t < world; ++t) res.emplace_back(peer.Bench(send, recv, PairBench{t}, noop, 100));
    return res;
  }
};

using DeviceTest = Test<SymmetricDMAMemory>;

// mpirun -np 2 --npernode 1 example

int main(int argc, char *argv[]) {
  size_t bufsize = 128 << 10; // 128k
  auto results = DeviceTest::Run(bufsize);
  return 0;
}
```

To learn how to use the library provided in this repository, please refer to the
following example experiments, which illustrate common usage patterns and benchmarking scenarios:

* [Affinity](experiments/affinity): Demonstrates how to query and enumerate GPU device information.
* [EFA](experiments/efa): Shows how to discover and inspect available EFA devices.
* [Echo](experiments/echo): Implements a simple TCP echo server/client to illustrate usage of the coroutine-based scheduler.
* [Bootstrap](experiments/bootstrap): Illustrates exchanging RDMA details via MPI communication.
* [Send\/Recv](experiments/sendrecv): Benchmarks libfabric SEND/RECV operations over EFA.
* [Write](experiments/write): Benchmarks libfabric WRITE operations over EFA.
* [Alltoall](experiments/all2all): Benchmarks a simple all-to-all communication pattern over EFA.
* [Queue](experiments/queue): Benchmarks a multi-producer, single-consumer (MPSC) queue between GPU and CPU.
* [Proxy](experiments/proxy): Benchmarks GPU-initiated RDMA writes via a CPU proxy coroutine.

## Citation

See [CITATION.cff](CITATION.cff) for machine-readable citation information.

### BibTeX
```bibtex
@software{tsai2025aws_efa_gpu_benchmark,
  title = {AWS EFA GPU Benchmark},
  author = {Tsai, Chang-Ning},
  year = {2025},
  month = {12},
  url = {https://github.com/crazyguitar/Libefaxx},
  version = {0.2.4},
  abstract = {High-performance RDMA communication experiments using CUDA and Amazon Elastic Fabric Adapter (EFA)},
  keywords = {RDMA, CUDA, EFA, High-Performance Computing, GPU Communication, Amazon EFA, Fabric, MPI}
}
```

### APA Style
Tsai, C.-N. (2025). *AWS EFA GPU Benchmark* (Version 0.2.4) [Computer software]. https://github.com/crazyguitar/Libefaxx

## References

1. Q. Le, "Libfabric EFA Series," 2024. [\[link\]](https://le.qun.ch/en/blog/2024/12/25/libfabric-efa-0-intro/)
2. K. Punniyamurthy et al., "Optimizing Distributed ML Communication," arXiv:2305.06942, 2023. [\[link\]](https://arxiv.org/pdf/2305.06942)
3. S. Liu et al., "GPU-Initiated Networking," arXiv:2511.15076, 2025. [\[link\]](https://arxiv.org/abs/2511.15076)
4. Netcan, "asyncio: C++20 coroutine library," GitHub. [\[link\]](https://github.com/netcan/asyncio)
5. UCCL Project, "UCCL: User-space Collective Communication Library," GitHub. [\[link\]](https://github.com/uccl-project/uccl)
6. Microsoft, "MSCCL++: Multi-Scale Collective Communication Library," GitHub. [\[link\]](https://github.com/microsoft/mscclpp)
7. DeepSeek-AI, "DeepEP: Expert parallelism with GPU-initiated communication," GitHub. [\[link\]](https://github.com/deepseek-ai/DeepEP)
