#!/bin/bash
#
# Test hybrid CUDA IPC (intra-node) and EFA RDMA (inter-node) communication
#

set -exo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT="${DIR}/../.."
sqsh="${ROOT}/cuda+latest.sqsh"
mount="/fsx:/fsx"
binary="${ROOT}/build/experiments/shmem/shmem"

echo "=== Test 1: Pure Intra-node Communication (IPC only) ==="
echo "Running 8 ranks on 1 node (8 GPUs, all local communication via CUDA IPC)"
srun -N 1 --ntasks-per-node=8 \
  --container-image "${sqsh}" \
  --container-mounts "${mount}" \
  --container-name efa \
  --mpi=pmix \
  "${binary}"

echo ""
echo "=== Test 2: Pure Inter-node Communication (RDMA only) ==="
echo "Running 4 ranks on 4 nodes (1 per node, all remote communication via EFA RDMA)"
srun -N 4 --ntasks-per-node=1 \
  --container-image "${sqsh}" \
  --container-mounts "${mount}" \
  --container-name efa \
  --mpi=pmix \
  "${binary}"

echo ""
echo "=== Test 3: Mixed Communication (IPC + RDMA) ==="
echo "Running 32 ranks on 4 nodes (8 per node, mix of IPC and RDMA)"
srun -N 4 --ntasks-per-node=8 \
  --container-image "${sqsh}" \
  --container-mounts "${mount}" \
  --container-name efa \
  --mpi=pmix \
  "${binary}"
