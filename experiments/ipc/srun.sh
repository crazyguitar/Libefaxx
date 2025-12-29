#!/bin/bash

set -exo pipefail

DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT="${DIR}/../.."
sqsh="${ROOT}/cuda+latest.sqsh"
mount="/fsx:/fsx"
binary="${ROOT}/build/experiments/ipc/ipc"

srun -N 1 --container-image "${sqsh}" \
  --container-mounts "${mount}" \
  --container-name efa \
  --mpi=pmix \
  --ntasks-per-node=8 \
  "${binary}" -b 256K
