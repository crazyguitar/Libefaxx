/**
 * @file efa.h
 * @brief EFA (Elastic Fabric Adapter) fabric initialization and management
 */
#pragma once

#include <hwloc.h>
#include <io/common.h>
#include <io/event.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <spdlog/spdlog.h>

/** @brief Maximum buffer size for endpoint addresses */
static constexpr size_t kMaxAddrSize = 64;
/** @brief Actual size of EFA endpoint addresses */
static constexpr size_t kAddrSize = 32;

/**
 * @brief Check libfabric call result and throw on error
 * @param exp Libfabric expression to evaluate
 * @throws std::runtime_error if expression returns non-zero
 */
#define FI_CHECK(exp)                                                            \
  do {                                                                           \
    auto rc = exp;                                                               \
    if (rc) {                                                                    \
      auto msg = fmt::format(#exp " fail. error({}): {}", rc, fi_strerror(-rc)); \
      SPDLOG_ERROR(msg);                                                         \
      throw std::runtime_error(msg);                                             \
    }                                                                            \
  } while (0)

/**
 * @brief Check libfabric call result matches expected value
 * @param exp Libfabric expression to evaluate
 * @param expect Expected return value
 * @throws std::runtime_error if expression result != expect
 */
#define FI_EXPECT(exp, expect)                                                   \
  do {                                                                           \
    auto rc = (exp);                                                             \
    if (rc != expect) {                                                          \
      auto msg = fmt::format(#exp " fail. error({}): {}", rc, fi_strerror(-rc)); \
      SPDLOG_ERROR(msg);                                                         \
      throw std::runtime_error(msg);                                             \
    }                                                                            \
  } while (0)

/**
 * @brief Singleton for EFA provider information discovery
 *
 * Queries libfabric for available EFA providers with required capabilities
 * (FI_MSG, FI_RMA, FI_HMEM). Thread-safe singleton pattern.
 */
class EFAInfo : private NoCopy {
 public:
  /**
   * @brief Get singleton fi_info for EFA providers
   * @return Linked list of fi_info structures for available EFA devices
   */
  static const struct fi_info* Get() {
    static EFAInfo instance;
    return instance.info_;
  }

 private:
  EFAInfo() : info_{New()} {}
  EFAInfo(EFAInfo&&) = delete;
  EFAInfo& operator=(EFAInfo&&) = delete;

  ~EFAInfo() {
    if (info_) {
      fi_freeinfo(info_);
      info_ = nullptr;
    }
  }

 private:
  inline static struct fi_info* New() {
    int rc = 0;
    struct fi_info* hints = nullptr;
    struct fi_info* info = nullptr;
    hints = fi_allocinfo();
    if (!hints) {
      SPDLOG_ERROR("fi_allocinfo fail.");
      goto end;
    }

    hints->caps = FI_MSG | FI_RMA | FI_HMEM | FI_LOCAL_COMM | FI_REMOTE_COMM;
    hints->ep_attr->type = FI_EP_RDM;
    hints->fabric_attr->prov_name = strdup("efa");
    hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_HMEM | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
    hints->domain_attr->threading = FI_THREAD_SAFE;

    rc = fi_getinfo(FI_VERSION(1, 20), NULL, NULL, 0, hints, &info);
    if (rc != 0) {
      SPDLOG_ERROR("fi_getinfo fail. error({}): {}", rc, fi_strerror(-rc));
      goto error;
    } else {
      goto end;
    }

  error:
    if (info) {
      fi_freeinfo(info);
      info = nullptr;
    }

  end:
    if (hints) {
      fi_freeinfo(hints);
      hints = nullptr;
    }
    return info;
  }

 private:
  /**
   * @brief Stream output operator for EFA information
   * @param os Output stream
   * @param efa EFA instance to output
   * @return Reference to output stream
   */
  friend std::ostream& operator<<(std::ostream& os, const EFAInfo& efa) {
    for (auto cur = efa.info_; !!cur; cur = cur->next) {
      os << fmt::format("provider: {}\n", cur->fabric_attr->prov_name);
      os << fmt::format("  fabric: {}\n", cur->fabric_attr->name);
      os << fmt::format("  domain: {}\n", cur->domain_attr->name);
      os << fmt::format("  version: {}.{}\n", FI_MAJOR(cur->fabric_attr->prov_version), FI_MINOR(cur->fabric_attr->prov_version));
      os << fmt::format("  type: {}\n", fi_tostr(&cur->ep_attr->type, FI_TYPE_EP_TYPE));
      os << fmt::format("  protocol: {}\n", fi_tostr(&cur->ep_attr->protocol, FI_TYPE_PROTOCOL));
    }
    return os;
  }

 private:
  struct fi_info* info_ = nullptr;
};

/**
 * @brief EFA (Elastic Fabric Adapter) wrapper for libfabric operations
 *
 * Manages libfabric resources including fabric, domain, endpoint, completion queue,
 * and address vector. Supports move semantics for efficient resource transfer.
 */
class EFA : private NoCopy {
 public:
  EFA() = delete;

  /**
   * @brief Move constructor
   * @param other Source EFA object to move from
   */
  EFA(EFA&& other) noexcept
      : efa_{std::exchange(other.efa_, nullptr)},
        fabric_{std::exchange(other.fabric_, nullptr)},
        domain_{std::exchange(other.domain_, nullptr)},
        ep_{std::exchange(other.ep_, nullptr)},
        cq_{std::exchange(other.cq_, nullptr)},
        av_{std::exchange(other.av_, nullptr)} {
    std::memcpy(addr_, other.addr_, sizeof(addr_));
    std::memset(other.addr_, 0, sizeof(other.addr_));
  }

  /**
   * @brief Move assignment operator
   * @param other Source EFA object to move from
   * @return Reference to this object
   */
  EFA& operator=(EFA&& other) noexcept {
    if (this != &other) {
      efa_ = std::exchange(other.efa_, nullptr);
      fabric_ = std::exchange(other.fabric_, nullptr);
      domain_ = std::exchange(other.domain_, nullptr);
      ep_ = std::exchange(other.ep_, nullptr);
      cq_ = std::exchange(other.cq_, nullptr);
      av_ = std::exchange(other.av_, nullptr);
      std::memcpy(addr_, other.addr_, sizeof(addr_));
      std::memset(other.addr_, 0, sizeof(other.addr_));
    }
    return *this;
  }

  /**
   * @brief Construct EFA instance for specific hardware device
   * @param efa hwloc object representing the EFA device
   *
   * Finds matching libfabric provider info by comparing PCI bus addresses
   * and initializes all libfabric resources (fabric, domain, endpoint, etc.)
   */
  EFA(hwloc_obj_t efa) {
    efa_ = Get(efa);
    ASSERT(!!efa_);
    Open(efa_);
  }

  /**
   * @brief Destructor - closes all libfabric resources
   *
   * Closes resources in reverse order: completion queue, address vector,
   * endpoint, domain, and fabric. Safe for moved-from objects.
   */
  ~EFA() noexcept {
    if (cq_) {
      fi_close((fid_t)cq_);
      cq_ = nullptr;
    }
    if (av_) {
      fi_close((fid_t)av_);
      av_ = nullptr;
    }
    if (ep_) {
      fi_close((fid_t)ep_);
      ep_ = nullptr;
    }
    if (domain_) {
      fi_close((fid_t)domain_);
      domain_ = nullptr;
    }
    if (fabric_) {
      fi_close((fid_t)fabric_);
      fabric_ = nullptr;
    }
  }

  /**
   * @brief Get local endpoint address
   * @return Local address buffer
   */
  const char* GetAddr() const noexcept { return addr_; }

  /** @brief Get completion queue handle */
  [[nodiscard]] struct fid_cq* GetCQ() noexcept { return cq_; }
  /** @brief Get address vector handle */
  [[nodiscard]] struct fid_av* GetAV() noexcept { return av_; }
  /** @brief Get domain handle */
  [[nodiscard]] struct fid_domain* GetDomain() noexcept { return domain_; }
  /** @brief Get endpoint handle */
  [[nodiscard]] struct fid_ep* GetEP() noexcept { return ep_; }
  /** @brief Get libfabric provider info for this EFA */
  [[nodiscard]] const struct fi_info* GetInfo() const noexcept { return efa_; }

  /**
   * @brief Convert binary address to hex string
   * @param addr Binary address buffer
   * @return Hex string representation
   */
  inline static std::string Addr2Str(const char* addr) {
    std::string out;
    for (size_t i = 0; i < kAddrSize; ++i) out += fmt::format("{:02x}", addr[i]);
    return out;
  }

  /**
   * @brief Convert hex string to binary address
   * @param addr Hex string address
   * @param bytes Output binary buffer
   */
  inline static void Str2Addr(const std::string& addr, char* bytes) {
    for (size_t i = 0; i < kAddrSize; ++i) sscanf(addr.c_str() + 2 * i, "%02hhx", &bytes[i]);
  }

 private:
  /**
   * @brief Find libfabric provider info matching hardware EFA device
   * @param efa hwloc object representing the EFA device
   * @return Pointer to matching fi_info, or nullptr if not found
   *
   * Matches by comparing PCI domain, bus, device, and function IDs
   */
  struct fi_info* Get(hwloc_obj_t efa) {
    auto* info = EFAInfo::Get();
    for (auto p = info; !!p; p = p->next) {
      ASSERT(!!p->nic);
      ASSERT(p->nic->bus_attr and p->nic->bus_attr->bus_type == FI_BUS_PCI);
      auto fi = p->nic->bus_attr->attr.pci;
      auto hw = efa->attr->pcidev;
      if (fi.domain_id == hw.domain and fi.bus_id == hw.bus and fi.device_id == hw.dev and fi.function_id == hw.func) {
        return const_cast<struct fi_info*>(p);
      }
    }
    return nullptr;
  }

  /**
   * @brief Initialize libfabric resources for the EFA device
   * @param info Libfabric provider info for the device
   *
   * Opens fabric, domain, completion queue, address vector, and endpoint.
   * Binds endpoint to CQ and AV, enables endpoint, and retrieves local address.
   */
  void Open(struct fi_info* info) {
    struct fi_av_attr av_attr{};
    struct fi_cq_attr cq_attr{};
    FI_CHECK(fi_fabric(info->fabric_attr, &fabric_, nullptr));
    FI_CHECK(fi_domain(fabric_, info, &domain_, nullptr));

    cq_attr.format = FI_CQ_FORMAT_DATA;
    FI_CHECK(fi_cq_open(domain_, &cq_attr, &cq_, nullptr));
    FI_CHECK(fi_av_open(domain_, &av_attr, &av_, nullptr));
    FI_CHECK(fi_endpoint(domain_, info, &ep_, nullptr));
    FI_CHECK(fi_ep_bind(ep_, &cq_->fid, FI_SEND | FI_RECV));
    FI_CHECK(fi_ep_bind(ep_, &av_->fid, 0));
    FI_CHECK(fi_enable(ep_));

    size_t len = sizeof(addr_);
    FI_CHECK(fi_getname(&ep_->fid, addr_, &len));
  }

 private:
  friend std::ostream& operator<<(std::ostream& os, const EFA& efa) {
    os << fmt::format("provider: {}\n", efa.efa_->fabric_attr->prov_name);
    os << fmt::format("  fabric: {}\n", efa.efa_->fabric_attr->name);
    os << fmt::format("  domain: {}\n", efa.efa_->domain_attr->name);
    os << fmt::format("  version: {}.{}\n", FI_MAJOR(efa.efa_->fabric_attr->prov_version), FI_MINOR(efa.efa_->fabric_attr->prov_version));
    os << fmt::format("  type: {}\n", fi_tostr(&efa.efa_->ep_attr->type, FI_TYPE_EP_TYPE));
    os << fmt::format("  protocol: {}\n", fi_tostr(&efa.efa_->ep_attr->protocol, FI_TYPE_PROTOCOL));
    return os;
  }

 private:
  struct fi_info* efa_ = nullptr;
  struct fid_fabric* fabric_ = nullptr;
  struct fid_domain* domain_ = nullptr;
  struct fid_ep* ep_ = nullptr;
  struct fid_cq* cq_ = nullptr;
  struct fid_av* av_ = nullptr;
  char addr_[kMaxAddrSize] = {0};
};
