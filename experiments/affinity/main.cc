#include <affinity/affinity.h>

#include <device/common.cuh>
#include <iostream>

/**
 * @brief Verify GPU affinity mapping by comparing CUDA device PCI info with GPUloc output
 *
 * This example demonstrates that the GPUloc affinity vector is correctly indexed
 * by CUDA device index. For each CUDA device i, affinity[i] should contain the
 * same PCI bus/device as reported by cudaGetDeviceProperties(prop, i).
 *
 * The GPUloc operator<< now prints both the topology summary and CUDA device
 * verification, so we only need to print loc to get the full output.
 */
int main(int argc, char *argv[]) {
  GPUloc loc;
  std::cout << loc;

  // Print detailed info for first device (original behavior)
  int deviceCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  if (deviceCount > 0) {
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("=== Detailed Device 0 Info ===\n");
    printf("Device %d: \"%s\"\n", device, prop.name);
    printf("  Compute capability:           %d.%d\n", prop.major, prop.minor);
    printf("  Total global memory:          %zu bytes (%.2f GB)\n", (size_t)prop.totalGlobalMem, (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Shared memory per block:      %zu bytes\n", (size_t)prop.sharedMemPerBlock);
    printf("  Registers per block:          %d\n", prop.regsPerBlock);
    printf("  Warp size:                    %d\n", prop.warpSize);
    printf("  Max threads per block:        %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads dim:              (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max grid size:                (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Clock rate:                   %.2f MHz\n", prop.clockRate / 1000.0);
    printf("  Memory clock rate:            %.2f MHz\n", prop.memoryClockRate / 1000.0);
    printf("  Memory bus width:             %d bits\n", prop.memoryBusWidth);
    printf("  L2 cache size:                %d bytes\n", prop.l2CacheSize);
    printf("  Multiprocessor count:         %d\n", prop.multiProcessorCount);
    printf("  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Unified addressing:           %s\n", prop.unifiedAddressing ? "Yes" : "No");
    printf("  Concurrent kernels:           %s\n", prop.concurrentKernels ? "Yes" : "No");
    printf("  Concurrent copy and kernel:   %s\n", (prop.deviceOverlap ? "Yes" : "No"));
    printf("  ECC enabled:                  %s\n", prop.ECCEnabled ? "Yes" : "No");
    printf("  Can map host memory:          %s\n", prop.canMapHostMemory ? "Yes" : "No");
    printf("  Integrated GPU:               %s\n", prop.integrated ? "Yes" : "No");
    printf("  Compute mode:                 %d\n", prop.computeMode);
    printf("  PCI Domain:Bus:Device:        %04x:%02x:%02x\n", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    printf("  TCC driver:                   %s\n", prop.tccDriver ? "Yes" : "No");
  }

  return 0;
}
