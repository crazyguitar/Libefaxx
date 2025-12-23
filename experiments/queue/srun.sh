#!/bin/bash

set -exo pipefail

DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT="${DIR}/../.."
sqsh="${ROOT}/cuda+latest.sqsh"
mount="/fsx:/fsx"
binary="${ROOT}/build/experiments/queue/queue"

srun -N 1 \
  --container-image "${sqsh}" \
  --container-mounts "${mount}" \
  --container-name cuda \
  --ntasks-per-node=1 \
  "${binary}" -b 8 -e 1024 -r 65536 -w 8
