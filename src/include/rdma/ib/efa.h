/**
 * @file efa.h
 * @brief EFA (Elastic Fabric Adapter) ibverbs initialization and management
 *
 * This is a copy of rdma/fabric/efa.h but uses ibverbs-based ib_* functions
 * instead of libfabric fi_* functions.
 */
#pragma once

#include <hwloc.h>
#include <io/common.h>
#include <rdma/ib/ib.h>
#include <spdlog/spdlog.h>

namespace ib {

/**
 * @brief Singleton for EFA device information discovery via ibverbs
 */
class EFAInfo : private NoCopy {
 public:
  static std::vector<ib_info>& Get() {
    static EFAInfo instance;
    return instance.infos_;
  }

 private:
  EFAInfo() {
    int num = 0;
    ibv_device** list = ibv_get_device_list(&num);
    if (!list) return;

    for (int i = 0; i < num; ++i) {
      const char* name = ibv_get_device_name(list[i]);
      bool is_efa = (strncmp(name, "rdmap", 5) == 0 || strncmp(name, "efa", 3) == 0);
      if (!is_efa) {
        auto* ctx = ibv_open_device(list[i]);
        if (ctx) {
          ibv_device_attr attr{};
          is_efa = (ibv_query_device(ctx, &attr) == 0 && attr.vendor_id == 0x1d0f);
          ibv_close_device(ctx);
        }
      }
      if (is_efa) {
        ib_info info{};
        info.dev = list[i];
        info.ctx = ibv_open_device(list[i]);
        if (info.ctx) {
          ibv_query_device(info.ctx, &info.attr);
          ibv_query_port(info.ctx, 1, &info.port);
          ibv_query_gid(info.ctx, 1, 0, &info.gid);
          infos_.push_back(info);
        }
      }
    }
    list_ = list;
  }

  ~EFAInfo() {
    for (auto& info : infos_) {
      if (info.ctx) ibv_close_device(info.ctx);
    }
    if (list_) ibv_free_device_list(list_);
  }

  static bool ParsePCI(ibv_device* dev, unsigned& domain, unsigned& bus, unsigned& slot, unsigned& func) {
    char sym[PATH_MAX], real[PATH_MAX];
    snprintf(sym, sizeof(sym), "%s/device", dev->ibdev_path);
    if (!realpath(sym, real)) return false;
    const char* p = strrchr(real, '/');
    return p && sscanf(++p, "%x:%x:%x.%x", &domain, &bus, &slot, &func) == 4;
  }

  ibv_device** list_ = nullptr;
  std::vector<ib_info> infos_;

 public:
  friend std::ostream& operator<<(std::ostream& os, const EFAInfo& efa) {
    for (auto& info : efa.infos_) {
      os << fmt::format("device: {}\n", ibv_get_device_name(info.dev));
      os << fmt::format("  vendor_id: 0x{:04x}\n", info.attr.vendor_id);
      os << fmt::format("  max_qp: {}\n", info.attr.max_qp);
      os << fmt::format("  max_cq: {}\n", info.attr.max_cq);
    }
    return os;
  }
};

/**
 * @brief EFA (Elastic Fabric Adapter) wrapper for ibverbs operations
 *
 * Manages ibverbs resources including domain, endpoint, completion queue,
 * and address vector. Supports move semantics for efficient resource transfer.
 */
class EFA : private NoCopy {
 public:
  EFA() = delete;

  EFA(EFA&& other) noexcept
      : efa_{other.efa_},
        domain_{std::exchange(other.domain_, nullptr)},
        ep_{std::exchange(other.ep_, nullptr)},
        cq_{std::exchange(other.cq_, nullptr)},
        av_{std::exchange(other.av_, nullptr)} {
    std::memcpy(addr_, other.addr_, sizeof(addr_));
    std::memset(other.addr_, 0, sizeof(other.addr_));
  }

  EFA& operator=(EFA&& other) noexcept {
    if (this != &other) {
      efa_ = other.efa_;
      domain_ = std::exchange(other.domain_, nullptr);
      ep_ = std::exchange(other.ep_, nullptr);
      cq_ = std::exchange(other.cq_, nullptr);
      av_ = std::exchange(other.av_, nullptr);
      std::memcpy(addr_, other.addr_, sizeof(addr_));
      std::memset(other.addr_, 0, sizeof(other.addr_));
    }
    return *this;
  }

  EFA(hwloc_obj_t efa) {
    efa_ = Get(efa);
    ASSERT(efa_ != nullptr);
    Open(efa_);
  }

  ~EFA() noexcept {
    if (cq_) {
      ib_cq_close(cq_);
      cq_ = nullptr;
    }
    if (av_) {
      ib_av_close(av_);
      av_ = nullptr;
    }
    if (ep_) {
      ib_ep_close(ep_);
      ep_ = nullptr;
    }
    if (domain_) {
      ib_domain_close(domain_);
      domain_ = nullptr;
    }
  }

  const char* GetAddr() const noexcept { return addr_; }

  [[nodiscard]] ib_cq* GetCQ() noexcept { return cq_; }
  [[nodiscard]] ib_av* GetAV() noexcept { return av_; }
  [[nodiscard]] ib_domain* GetDomain() noexcept { return domain_; }
  [[nodiscard]] ib_ep* GetEP() noexcept { return ep_; }
  [[nodiscard]] ib_info* GetInfo() noexcept { return efa_; }

  static std::string Addr2Str(const char* addr) {
    std::string out;
    for (size_t i = 0; i < kAddrSize; ++i) out += fmt::format("{:02x}", static_cast<uint8_t>(addr[i]));
    return out;
  }

  static void Str2Addr(const std::string& addr, char* bytes) noexcept {
    for (size_t i = 0; i < kAddrSize; ++i) sscanf(addr.c_str() + 2 * i, "%02hhx", &bytes[i]);
  }

 private:
  static bool ParsePCI(ibv_device* dev, unsigned& domain, unsigned& bus, unsigned& slot, unsigned& func) {
    char sym[PATH_MAX], real[PATH_MAX];
    snprintf(sym, sizeof(sym), "%s/device", dev->ibdev_path);
    if (!realpath(sym, real)) return false;
    const char* p = strrchr(real, '/');
    return p && sscanf(++p, "%x:%x:%x.%x", &domain, &bus, &slot, &func) == 4;
  }

  ib_info* Get(hwloc_obj_t efa) {
    auto& infos = EFAInfo::Get();
    for (auto& info : infos) {
      unsigned domain, bus, slot, func;
      if (ParsePCI(info.dev, domain, bus, slot, func)) {
        auto hw = efa->attr->pcidev;
        if (domain == hw.domain && bus == hw.bus && slot == hw.dev && func == hw.func) {
          return &info;
        }
      }
    }
    return nullptr;
  }

  void Open(ib_info* info) {
    IB_CHECK(ib_domain_open(info, &domain_) == 0);
    IB_CHECK(ib_cq_open(domain_, &cq_) == 0);
    IB_CHECK(ib_av_open(domain_, &av_) == 0);
    IB_CHECK(ib_endpoint(domain_, info, &ep_, cq_) == 0);
    ib_ep_bind(ep_, cq_, 0);
    ib_ep_bind(ep_, av_, 0);
    IB_CHECK(ib_enable(ep_) == 0);

    size_t len = sizeof(addr_);
    IB_CHECK(ib_getname(ep_, domain_, addr_, &len) == 0);
  }

 private:
  friend std::ostream& operator<<(std::ostream& os, const EFA& efa) {
    os << fmt::format("device: {}\n", ibv_get_device_name(efa.efa_->dev));
    os << fmt::format("  vendor_id: 0x{:04x}\n", efa.efa_->attr.vendor_id);
    return os;
  }

 private:
  ib_info* efa_ = nullptr;
  ib_domain* domain_ = nullptr;
  ib_ep* ep_ = nullptr;
  ib_cq* cq_ = nullptr;
  ib_av* av_ = nullptr;
  char addr_[kMaxAddrSize] = {0};
};

}  // namespace ib
