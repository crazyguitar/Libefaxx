/**
 * @file affinity.h
 * @brief Hardware topology discovery and GPU affinity management
 */
#pragma once

#include <cuda_runtime.h>
#include <hwloc.h>
#include <io/common.h>
#include <nvml.h>
#include <spdlog/spdlog.h>
#include <string.h>

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/**
 * @brief Error checking macro that throws runtime_error on failure
 * @param exp Expression to evaluate
 */
#define GPULOC_CHECK(exp)                                               \
  do {                                                                  \
    if ((exp)) {                                                        \
      auto msg = fmt::format(#exp " fail. error: {}", strerror(errno)); \
      SPDLOG_ERROR(msg);                                                \
      throw std::runtime_error(msg);                                    \
    }                                                                   \
  } while (0)

/**
 * @brief Assertion macro that throws runtime_error on failure
 * @param exp Expression to evaluate
 */
#define GPULOC_ASSERT(exp)                             \
  do {                                                 \
    if (!(exp)) {                                      \
      auto msg = fmt::format(#exp " assertion fail."); \
      SPDLOG_ERROR(msg);                               \
      throw std::runtime_error(msg);                   \
    }                                                  \
  } while (0)

/**
 * @brief NVML error checking macro
 * @param exp NVML function call to check
 */
#define NVML_CHECK(exp)                                     \
  do {                                                      \
    nvmlReturn_t res = exp;                                 \
    if (res != NVML_SUCCESS) {                              \
      const char* err = nvmlErrorString(res);               \
      auto msg = fmt::format(#exp " fail. error: {}", err); \
      SPDLOG_ERROR(msg);                                    \
      throw std::runtime_error(msg);                        \
    }                                                       \
  } while (0)

/**
 * @brief CUDA error checking macro
 * @param exp CUDA function call to check
 */
#define GPULOC_CUDA_CHECK(exp)                                                  \
  do {                                                                          \
    cudaError_t err = (exp);                                                    \
    if (err != cudaSuccess) {                                                   \
      auto msg = fmt::format(#exp " fail. error: {}", cudaGetErrorString(err)); \
      SPDLOG_ERROR(msg);                                                        \
      throw std::runtime_error(msg);                                            \
    }                                                                           \
  } while (0)

/** @brief NVIDIA PCI vendor ID */
static constexpr uint16_t NVIDIA_VENDOR_ID = 0x10de;
/** @brief AMD PCI vendor ID */
static constexpr uint16_t AMD_VENDOR_ID = 0x1002;

using pci_type = std::unordered_set<hwloc_obj_t>;

/**
 * @brief Represents a NUMA node with its associated cores and PCI bridges
 */
struct Numanode {
  hwloc_obj_t numanode;                              ///< NUMA node object
  std::unordered_set<hwloc_obj_t> cores;             ///< CPU cores in this NUMA node
  std::unordered_map<hwloc_obj_t, pci_type> bridge;  ///< PCI bridges and their devices
};

/**
 * @brief Hardware locality wrapper class for topology discovery
 */
class Hwloc : private NoCopy {
 public:
  /**
   * @brief Constructor - initializes hwloc topology and discovers hardware
   */
  Hwloc() {
    GPULOC_CHECK(hwloc_topology_init(&topology_));
    GPULOC_CHECK(hwloc_topology_set_all_types_filter(topology_, HWLOC_TYPE_FILTER_KEEP_ALL));
    GPULOC_CHECK(hwloc_topology_set_io_types_filter(topology_, HWLOC_TYPE_FILTER_KEEP_IMPORTANT));
    GPULOC_CHECK(hwloc_topology_set_flags(topology_, HWLOC_TOPOLOGY_FLAG_IMPORT_SUPPORT));
    GPULOC_CHECK(hwloc_topology_load(topology_));
    Traverse(hwloc_get_root_obj(topology_), nullptr, numanodes_);
  }

  /**
   * @brief Destructor - cleans up hwloc topology
   */
  ~Hwloc() { hwloc_topology_destroy(topology_); }

  /**
   * @brief Get discovered NUMA nodes
   * @return Reference to vector of NUMA nodes
   */
  const std::vector<Numanode>& GetNumaNodes() const noexcept { return numanodes_; }

  /**
   * @brief Check if object is a CPU package
   * @param l hwloc object to check
   * @return true if object is a package
   */
  inline static bool IsPackage(hwloc_obj_t l) { return l->type == HWLOC_OBJ_PACKAGE; }

  /**
   * @brief Check if object is a NUMA node
   * @param l hwloc object to check
   * @return true if object is a NUMA node
   */
  inline static bool IsNumaNode(hwloc_obj_t l) { return l->type == HWLOC_OBJ_NUMANODE; }

  /**
   * @brief Check if object is a CPU core
   * @param l hwloc object to check
   * @return true if object is a core
   */
  inline static bool IsCore(hwloc_obj_t l) { return l->type == HWLOC_OBJ_CORE; }

  /**
   * @brief Check if object is a PCI device
   * @param l hwloc object to check
   * @return true if object is a PCI device
   */
  inline static bool IsPCI(hwloc_obj_t l) { return l->type == HWLOC_OBJ_PCI_DEVICE; }

  /**
   * @brief Check if object is a host bridge
   * @param l hwloc object to check
   * @return true if object is a host bridge
   */
  inline static bool IsHostBridge(hwloc_obj_t l) {
    if (l->type != HWLOC_OBJ_BRIDGE) return false;
    return l->attr->bridge.upstream_type != HWLOC_OBJ_BRIDGE_PCI;
  }

  /**
   * @brief Check if PCI device is an EFA adapter
   * @param l hwloc object to check
   * @return true if object is an EFA device
   */
  inline static bool IsEFA(hwloc_obj_t l) {
    if (l->type != HWLOC_OBJ_PCI_DEVICE) return false;
    return IsOSDevType(HWLOC_OBJ_OSDEV_OPENFABRICS, l);
  }

  /**
   * @brief Check if PCI device is an NVIDIA GPU
   * @param l hwloc object to check
   * @return true if object is an NVIDIA GPU
   */
  inline static bool IsGPU(hwloc_obj_t l) {
    if (l->type != HWLOC_OBJ_PCI_DEVICE) return false;
    auto class_id = l->attr->pcidev.class_id >> 8;
    if (class_id != 0x03) return false;
    auto vendor_id = l->attr->pcidev.vendor_id;
    if (vendor_id != NVIDIA_VENDOR_ID) return false;
    return true;
  }

  /**
   * @brief Check if object has OS device of specified type
   * @param type OS device type to check for
   * @param l hwloc object to check
   * @return true if object has the specified OS device type
   */
  static bool IsOSDevType(hwloc_obj_osdev_type_e type, hwloc_obj_t l) {
    if (!l) return false;
    if (l->attr->osdev.type == type) return true;
    for (hwloc_obj_t child = l->memory_first_child; !!child; child = child->next_sibling) {
      if (child->type != HWLOC_OBJ_PU and IsOSDevType(type, child)) return true;
    }
    for (hwloc_obj_t child = l->first_child; !!child; child = child->next_sibling) {
      if (child->type != HWLOC_OBJ_PU and IsOSDevType(type, child)) return true;
    }
    for (hwloc_obj_t child = l->io_first_child; !!child; child = child->next_sibling) {
      if (IsOSDevType(type, child)) return true;
    }
    for (hwloc_obj_t child = l->misc_first_child; !!child; child = child->next_sibling) {
      if (IsOSDevType(type, child)) return true;
    }
    return false;
  }

  /**
   * @brief Recursively traverse hwloc topology to build NUMA node structure
   * @param l Current hwloc object
   * @param bridge Current PCI bridge
   * @param numanodes Vector to populate with discovered NUMA nodes
   */
  static void Traverse(hwloc_obj_t l, hwloc_obj_t bridge, std::vector<Numanode>& numanodes) {
    if (IsPackage(l)) {
      numanodes.emplace_back(Numanode{});
    } else if (IsNumaNode(l)) {
      auto& numa = numanodes.back();
      numa.numanode = l;
    } else if (IsHostBridge(l)) {
      auto& numa = numanodes.back();
      numa.bridge.emplace(l, pci_type{});
      bridge = l;
    } else if (IsCore(l)) {
      auto& numa = numanodes.back();
      numa.cores.emplace(l);
    } else if (IsPCI(l)) {
      assert(!!bridge);
      auto& numa = numanodes.back();
      numa.bridge[bridge].emplace(l);
    }

    for (hwloc_obj_t child = l->memory_first_child; !!child; child = child->next_sibling) {
      if (child->type != HWLOC_OBJ_PU) Traverse(child, bridge, numanodes);
    }
    for (hwloc_obj_t child = l->first_child; !!child; child = child->next_sibling) {
      if (child->type != HWLOC_OBJ_PU) Traverse(child, bridge, numanodes);
    }
    for (hwloc_obj_t child = l->io_first_child; !!child; child = child->next_sibling) {
      Traverse(child, bridge, numanodes);
    }
    for (hwloc_obj_t child = l->misc_first_child; !!child; child = child->next_sibling) {
      Traverse(child, bridge, numanodes);
    }
  }

 private:
  hwloc_topology_t topology_;
  std::vector<Numanode> numanodes_;
};

/**
 * @brief GPU affinity information including associated NUMA node, cores, and EFA devices
 *
 * When the affinity vector is indexed by CUDA device index, entries that could not be
 * matched to hwloc topology will have null pointers for gpu and numanode fields, and
 * empty vectors for cores and efas. Consumers should check for null gpu pointer before
 * accessing hwloc-related fields.
 *
 * The prop field is always populated with CUDA runtime properties for the device,
 * regardless of whether the device was matched to hwloc topology. This allows consumers
 * to access device information even when hwloc matching fails.
 */
struct GPUAffinity {
  hwloc_obj_t gpu = nullptr;       ///< GPU device object (nullptr if not matched)
  hwloc_obj_t numanode = nullptr;  ///< Associated NUMA node (nullptr if not matched)
  std::vector<hwloc_obj_t> cores;  ///< CPU cores in the same NUMA node
  std::vector<hwloc_obj_t> efas;   ///< EFA devices on the same PCI bridge
  cudaDeviceProp prop = {};        ///< CUDA device properties (always populated)
};

/**
 * @brief Stream output operator for GPUAffinity
 * @param os Output stream
 * @param affinity GPUAffinity object to output
 * @return Reference to output stream
 *
 * Prints GPU affinity information including:
 * - CUDA PCI address (domain:bus:device from prop)
 * - hwloc PCI address and match status (if gpu is non-null)
 * - NUMA node index (if numanode is non-null)
 * - Core range (if cores is non-empty)
 * - EFA device information (if efas is non-empty)
 *
 * Note: The device name (from prop.name) is typically printed by the caller
 * (e.g., GPUloc::operator<<) as part of the header line.
 *
 * This operator relies solely on the stored cudaDeviceProp and does not
 * call any CUDA APIs.
 */
inline std::ostream& operator<<(std::ostream& os, const GPUAffinity& affinity) {
  // Print CUDA PCI address from stored device properties
  os << fmt::format("  CUDA PCI:     {:04x}:{:02x}:{:02x}\n", affinity.prop.pciDomainID, affinity.prop.pciBusID, affinity.prop.pciDeviceID);

  if (affinity.gpu) {
    // Print hwloc PCI address
    os << fmt::format(
        "  GPUloc PCI:   {:04x}:{:02x}:{:02x}.{:x}\n", affinity.gpu->attr->pcidev.domain, affinity.gpu->attr->pcidev.bus,
        affinity.gpu->attr->pcidev.dev, affinity.gpu->attr->pcidev.func
    );

    // Verify PCI match between CUDA and hwloc
    bool match =
        (static_cast<int>(affinity.gpu->attr->pcidev.domain) == affinity.prop.pciDomainID &&
         static_cast<int>(affinity.gpu->attr->pcidev.bus) == affinity.prop.pciBusID &&
         static_cast<int>(affinity.gpu->attr->pcidev.dev) == affinity.prop.pciDeviceID);
    os << fmt::format("  PCI Match:    {}\n", match ? "YES" : "NO - MISMATCH!");

    // Print NUMA node index
    if (affinity.numanode) {
      os << fmt::format("  NUMA Node:    {}\n", affinity.numanode->logical_index);
    }

    // Print core range
    if (!affinity.cores.empty()) {
      os << fmt::format("  Core Range:   {}-{}\n", affinity.cores.front()->logical_index, affinity.cores.back()->logical_index);
    }

    // Print EFA devices
    if (!affinity.efas.empty()) {
      os << fmt::format("  EFA Devices:  {}\n", affinity.efas.size());
      for (const auto& efa : affinity.efas) {
        os << fmt::format("    EFA: {:02x}:{:02x}.{:x}\n", efa->attr->pcidev.bus, efa->attr->pcidev.dev, efa->attr->pcidev.func);
      }
    }
  } else {
    os << "  GPUloc PCI:   (no hwloc match)\n";
  }

  return os;
}

/**
 * @brief GPU locality analyzer that maps GPUs to their optimal CPU and network resources
 */
class GPUloc : private NoCopy {
 public:
  using affinity_type = std::vector<GPUAffinity>;

  /**
   * @brief Constructor - discovers hardware topology and builds GPU affinity map
   */
  GPUloc() : hwloc_{Hwloc()} {
    NVML_CHECK(nvmlInit());
    affinity_ = GetAffinity(hwloc_);
  }

  ~GPUloc() { nvmlShutdown(); }

  /**
   * @brief Get GPU affinity mapping
   * @return Reference to GPU affinity map
   */
  const affinity_type& GetGPUAffinity() const noexcept { return affinity_; }

  inline static const GPUloc& Get() {
    static GPUloc loc;
    return loc;
  }

 private:
  /**
   * @brief Build GPU affinity mapping from hardware topology
   * @param hwloc Hardware topology object
   * @return GPU affinity mapping indexed by CUDA device index
   *
   * The affinity vector is indexed by CUDA device index, so affinity[i]
   * corresponds to the GPU accessible via cudaSetDevice(i). This ensures
   * reliable mapping between CUDA device indices and hardware topology.
   */
  static affinity_type GetAffinity(Hwloc& hwloc) {
    std::unordered_map<hwloc_obj_t, GPUAffinity> gpuloc;
    for (auto& numa : hwloc.GetNumaNodes()) {
      for (auto& bridge : numa.bridge) {
        std::vector<hwloc_obj_t> gpus;
        std::vector<hwloc_obj_t> efas;
        for (auto pci : bridge.second) {
          if (Hwloc::IsGPU(pci)) {
            gpus.emplace_back(pci);
          } else if (Hwloc::IsEFA(pci)) {
            efas.emplace_back(pci);
          }
        }
        std::vector<hwloc_obj_t> cores(numa.cores.begin(), numa.cores.end());
        std::sort(cores.begin(), cores.end(), [](auto&& x, auto&& y) { return x->logical_index < y->logical_index; });
        for (auto& gpu : gpus) gpuloc[gpu] = GPUAffinity{gpu, numa.numanode, cores, efas};
      }
    }

    // Get CUDA device count for affinity vector sizing
    int cudaCount = 0;
    GPULOC_CUDA_CHECK(cudaGetDeviceCount(&cudaCount));

    // Log warning if CUDA device count differs from hwloc GPU count
    if (static_cast<size_t>(cudaCount) != gpuloc.size()) {
      SPDLOG_WARN("CUDA device count ({}) does not match hwloc NVIDIA GPU count ({})", cudaCount, gpuloc.size());
    }

    // Create affinity vector indexed by CUDA device index
    affinity_type affinity(static_cast<size_t>(cudaCount));
    size_t matchedCount = 0;

    for (int dev = 0; dev < cudaCount; ++dev) {
      cudaDeviceProp prop;
      GPULOC_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

      // Extract PCI identifiers from CUDA device properties
      // Note: cudaDeviceProp provides pciBusID, pciDeviceID, and pciDomainID
      int cudaDomain = prop.pciDomainID;
      int cudaBus = prop.pciBusID;
      int cudaDev = prop.pciDeviceID;

      bool matched = false;
      for (auto& [gpu, loc] : gpuloc) {
        // Match on (domain, bus, dev) triple - don't assume func == 0
        if (static_cast<int>(gpu->attr->pcidev.domain) == cudaDomain && static_cast<int>(gpu->attr->pcidev.bus) == cudaBus &&
            static_cast<int>(gpu->attr->pcidev.dev) == cudaDev) {
          affinity[static_cast<size_t>(dev)] = loc;
          // Store the CUDA device properties in the affinity entry
          affinity[static_cast<size_t>(dev)].prop = prop;
          matched = true;
          matchedCount++;
          break;
        }
      }

      if (!matched) {
        SPDLOG_WARN("Could not find hwloc match for CUDA device {} (PCI {:04x}:{:02x}:{:02x})", dev, cudaDomain, cudaBus, cudaDev);
        // Even for unmatched devices, store the device properties
        // The affinity entry already has default values (nullptr, empty vectors)
        affinity[static_cast<size_t>(dev)].prop = prop;
      }
    }

    // Ensure at least one GPU was successfully matched
    GPULOC_ASSERT(matchedCount > 0);
    return affinity;
  }

  /**
   * @brief Stream output operator for GPUloc
   * @param os Output stream
   * @param loc GPUloc object to output
   * @return Reference to output stream
   *
   * Output format shows GPU index (corresponding to CUDA device index),
   * device name, and delegates detailed affinity printing to the
   * GPUAffinity operator<<.
   *
   * The high-level flow:
   * - Query device count with cudaGetDeviceCount
   * - Loop over devices, call cudaGetDeviceProperties to verify CUDA is functioning
   * - For each device, print a header line and delegate to GPUAffinity operator<<
   */
  friend std::ostream& operator<<(std::ostream& os, const GPUloc& loc) {
    // Print topology summary
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
      os << fmt::format("CUDA not available: {}\n", cudaGetErrorString(err));
      return os;
    }

    os << fmt::format("Found {} CUDA device(s)\n\n", deviceCount);

    for (int device = 0; device < deviceCount; ++device) {
      cudaDeviceProp prop;
      err = cudaGetDeviceProperties(&prop, device);
      if (err != cudaSuccess) {
        os << fmt::format("CUDA Device {}: Error getting properties: {}\n\n", device, cudaGetErrorString(err));
        continue;
      }

      // Print device header with name
      os << fmt::format("CUDA Device {}: \"{}\"\n", device, prop.name);

      // Delegate detailed printing to GPUAffinity operator<<
      if (static_cast<size_t>(device) < loc.affinity_.size()) {
        os << loc.affinity_[static_cast<size_t>(device)];
      } else {
        os << "  GPUloc:       (device not in affinity vector)\n";
      }
      os << "\n";
    }

    return os;
  }

 private:
  Hwloc hwloc_;
  affinity_type affinity_;
};
