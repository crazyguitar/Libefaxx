/**
 * @file efa.h
 * @brief EFA device discovery using ibverbs (libibverbs)
 *
 * ibverbs API call flow:
 *
 *   ┌─────────────────────────┐
 *   │  ibv_get_device_list()  │  ← Get all RDMA devices
 *   └───────────┬─────────────┘
 *               │
 *               ▼
 *   ┌─────────────────────────┐
 *   │    ibv_open_device()    │  ← Open device context
 *   └───────────┬─────────────┘
 *               │
 *       ┌───────┴───────┬────────────────┐
 *       ▼               ▼                ▼
 *  ┌──────────┐  ┌─────────────┐  ┌─────────────┐
 *  │query_    │  │query_port() │  │query_gid()  │
 *  │device()  │  │             │  │             │
 *  └──────────┘  └─────────────┘  └─────────────┘
 *       │               │                │
 *       ▼               ▼                ▼
 *  ┌──────────┐  ┌─────────────┐  ┌─────────────┐
 *  │max_qp,   │  │port_state,  │  │GID for      │
 *  │max_cq,   │  │max_mtu,     │  │routing      │
 *  │max_mr    │  │active_mtu   │  │             │
 *  └──────────┘  └─────────────┘  └─────────────┘
 *
 *   (cleanup)
 *   ┌─────────────────────────┐
 *   │   ibv_close_device()    │
 *   └───────────┬─────────────┘
 *               │
 *               ▼
 *   ┌─────────────────────────┐
 *   │ ibv_free_device_list()  │
 *   └─────────────────────────┘
 */
#pragma once

#include <hwloc.h>
#include <infiniband/verbs.h>
#include <io/common.h>
#include <spdlog/spdlog.h>

#include <cstring>
#include <vector>

/**
 * @brief Check ibverbs call result and throw on error
 * @param exp Expression to evaluate
 * @throws std::runtime_error if expression is false/null
 */
#define IBV_CHECK(exp)                                           \
  do {                                                           \
    if (!(exp)) {                                                \
      auto msg = fmt::format(#exp " fail: {}", strerror(errno)); \
      SPDLOG_ERROR(msg);                                         \
      throw std::runtime_error(msg);                             \
    }                                                            \
  } while (0)

/**
 * @brief RDMA device list manager via ibverbs (singleton)
 *
 * Queries ibverbs for available RDMA devices and filters EFA devices
 * by name prefix (rdmap*, efa*) or Amazon vendor ID (0x1d0f).
 */
class IBDeviceList : private NoCopy {
 public:
  /**
   * @brief Get singleton instance
   * @return Reference to device list manager
   */
  static IBDeviceList& Get() {
    static IBDeviceList instance;
    return instance;
  }

  /** @brief Get all RDMA devices */
  [[nodiscard]] const std::vector<ibv_device*>& All() const { return all_; }
  /** @brief Get EFA devices only */
  [[nodiscard]] const std::vector<ibv_device*>& EFA() const { return efa_; }
  /** @brief Get total RDMA device count */
  [[nodiscard]] size_t Count() const { return all_.size(); }
  /** @brief Get EFA device count */
  [[nodiscard]] size_t EFACount() const { return efa_.size(); }

  /**
   * @brief Find ibv_device matching hwloc PCI address
   * @param efa hwloc object representing the EFA device
   * @return Matching ibv_device or nullptr
   *
   * Matches by comparing PCI domain, bus, device, and function IDs
   */
  [[nodiscard]] ibv_device* Find(hwloc_obj_t efa) const {
    if (!efa || !efa->attr) return nullptr;
    auto hw = efa->attr->pcidev;
    for (auto* dev : efa_) {
      unsigned domain, bus, slot, func;
      if (ParsePCI(dev, domain, bus, slot, func)) {
        if (domain == hw.domain && bus == hw.bus && slot == hw.dev && func == hw.func) {
          return dev;
        }
      }
    }
    return nullptr;
  }

 private:
  /**
   * @brief Parse PCI address from ibv_device sysfs path
   * @return true if successfully parsed
   */
  static bool ParsePCI(ibv_device* dev, unsigned& domain, unsigned& bus, unsigned& slot, unsigned& func) {
    char sym_path[PATH_MAX], real_path[PATH_MAX];
    snprintf(sym_path, sizeof(sym_path), "%s/device", dev->ibdev_path);
    if (!realpath(sym_path, real_path)) return false;
    const char* dbdf = strrchr(real_path, '/');
    if (!dbdf) return false;
    dbdf++;
    return sscanf(dbdf, "%x:%x:%x.%x", &domain, &bus, &slot, &func) == 4;
  }

 private:
  /**
   * @brief Check if device is EFA by name or vendor ID
   * @param dev Device to check
   * @return true if EFA device
   */
  static bool IsEFA(ibv_device* dev) {
    const char* name = ibv_get_device_name(dev);
    if (std::strncmp(name, "rdmap", 5) == 0 || std::strncmp(name, "efa", 3) == 0) return true;
    auto* ctx = ibv_open_device(dev);
    if (!ctx) return false;
    ibv_device_attr attr{};
    bool is_efa = (ibv_query_device(ctx, &attr) == 0 && attr.vendor_id == 0x1d0f);
    ibv_close_device(ctx);
    return is_efa;
  }

  IBDeviceList() {
    int num = 0;
    list_ = ibv_get_device_list(&num);
    if (!list_) {
      SPDLOG_ERROR("ibv_get_device_list fail");
      return;
    }
    for (int i = 0; i < num; ++i) {
      all_.push_back(list_[i]);
      if (IsEFA(list_[i])) efa_.push_back(list_[i]);
    }
  }

  ~IBDeviceList() {
    if (list_) ibv_free_device_list(list_);
  }

 private:
  ibv_device** list_ = nullptr;
  std::vector<ibv_device*> all_;
  std::vector<ibv_device*> efa_;
};

/**
 * @brief Single RDMA device context via ibverbs
 *
 * Opens device and queries attributes, port info, and GID.
 * Supports move semantics for efficient resource transfer.
 */
class IBDevice : private NoCopy {
 public:
  IBDevice() = delete;

  /**
   * @brief Construct device context from ibv_device
   * @param dev Device from ibv_get_device_list
   */
  explicit IBDevice(ibv_device* dev) : dev_{dev} {
    IBV_CHECK(dev_);
    IBV_CHECK(ctx_ = ibv_open_device(dev));
    IBV_CHECK(ibv_query_device(ctx_, &attr_) == 0);
    IBV_CHECK(ibv_query_port(ctx_, 1, &port_) == 0);
    IBV_CHECK(ibv_query_gid(ctx_, 1, 0, &gid_) == 0);
  }

  /**
   * @brief Construct device context from hwloc object
   * @param efa hwloc object representing the EFA device
   */
  explicit IBDevice(hwloc_obj_t efa) : IBDevice(IBDeviceList::Get().Find(efa)) {}

  /**
   * @brief Move constructor
   * @param o Source device to move from
   */
  IBDevice(IBDevice&& o) noexcept
      : dev_{std::exchange(o.dev_, nullptr)}, ctx_{std::exchange(o.ctx_, nullptr)}, attr_{o.attr_}, port_{o.port_}, gid_{o.gid_} {}

  /**
   * @brief Destructor - closes device context
   */
  ~IBDevice() {
    if (ctx_) ibv_close_device(ctx_);
  }

  /** @brief Get device name */
  [[nodiscard]] const char* Name() const { return ibv_get_device_name(dev_); }
  /** @brief Get device context handle */
  [[nodiscard]] ibv_context* Ctx() const { return ctx_; }
  /** @brief Get device attributes (max_qp, max_cq, max_mr, etc.) */
  [[nodiscard]] const ibv_device_attr& Attr() const { return attr_; }
  /** @brief Get port attributes (state, mtu, etc.) */
  [[nodiscard]] const ibv_port_attr& Port() const { return port_; }
  /** @brief Get GID for routing */
  [[nodiscard]] const ibv_gid& GID() const { return gid_; }

 private:
  ibv_device* dev_ = nullptr;
  ibv_context* ctx_ = nullptr;
  ibv_device_attr attr_{};
  ibv_port_attr port_{};
  ibv_gid gid_{};
};
