/**
 * @file ib/efa.h
 * @brief IB (ibverbs) EFA device management and Backend traits
 */
#pragma once

#include <hwloc.h>
#include <io/common.h>
#include <rdma/efa.h>
#include <rdma/ib/context.h>
#include <rdma/ib/ib.h>

#include <climits>
#include <vector>

namespace ib {

// Forward declarations
class EFA;
class IBSelector;

// Use IB-specific context types directly (defined in rdma/ib/context.h)
// Note: ib::Context and ib::ImmContext are non-templated for nvcc compatibility

/**
 * @brief IB Backend traits for unified Channel/Buffer/Memory
 */
struct Backend {
  using efa_type = EFA;
  using mr_type = ib_mr*;
  using cq_type = ib_cq*;
  using cq_entry_type = ib_cq_data_entry;
  using rma_iov_type = ib_rma_iov;
  using context_type = Context;
  using imm_context_type = ImmContext;
  using selector_type = IBSelector;
  using key_type = uint32_t;
  using remote_addr_type = struct {
    ibv_ah* ah;
    uint32_t qpn;
    uint32_t qkey;
  };

  static constexpr remote_addr_type kInvalidAddr = {nullptr, 0, 0};
  static constexpr bool kNeedsRmaForSend = true;

  // MR operations
  static uint32_t GetRkey(mr_type mr) { return ib_mr_rkey(mr); }
  static uint32_t GetLkey(mr_type mr) { return ib_mr_lkey(mr); }
  static void CloseMR(mr_type mr) { ib_mr_close(mr); }

  // RMA IOV
  static rma_iov_type MakeRmaIov(uint64_t addr, size_t size, mr_type mr) { return {addr, size, ib_mr_rkey(mr)}; }

  // Connection
  static remote_addr_type Connect(EFA* efa, const char* addr);
  static void Disconnect(EFA* efa, remote_addr_type& remote);

  // Post operations
  static ssize_t
  PostWrite(EFA* efa, void* data, size_t len, mr_type mr, uint64_t addr, key_type key, uint64_t imm, remote_addr_type remote, Context* ctx);
  static ssize_t
  PostSend(EFA* efa, void* data, size_t len, mr_type mr, uint64_t addr, key_type key, uint64_t imm, remote_addr_type remote, Context* ctx);
  static ssize_t PostRecv(EFA* efa, void* data, size_t len, mr_type mr, Context* ctx) { return -ENOTSUP; }  // IB uses imm_data wait

  // Completion length
  static ssize_t GetWriteLen(const Context& ctx, size_t req_size) { return static_cast<ssize_t>(req_size); }
  static ssize_t GetSendLen(const Context& ctx, size_t req_size) { return static_cast<ssize_t>(req_size); }
  static ssize_t GetRecvLen(const Context& ctx) { return ctx.entry.len; }
};

/**
 * @brief Singleton for IB EFA device enumeration
 *
 * Discovers and caches ibverbs EFA devices at startup.
 * EFA devices may be named "efa*" or "rdmap*" depending on driver version.
 */
class EFAInfo : private NoCopy {
 public:
  /** @brief Get cached list of EFA device info */
  static std::vector<ib_info>& Get() {
    static EFAInfo instance;
    return instance.infos_;
  }

 private:
  EFAInfo() {
    int num_devices = 0;
    list_ = ibv_get_device_list(&num_devices);
    if (list_) {
      for (int i = 0; i < num_devices; ++i) {
        const char* name = ibv_get_device_name(list_[i]);
        // EFA devices can be named "efa*" or "rdmap*" depending on driver
        if (strstr(name, "efa") == nullptr && strstr(name, "rdmap") == nullptr) continue;
        ib_info info{};
        info.dev = list_[i];
        info.ctx = ibv_open_device(list_[i]);
        if (info.ctx) {
          ibv_query_device(info.ctx, &info.attr);
          ibv_query_port(info.ctx, 1, &info.port);
          ibv_query_gid(info.ctx, 1, 0, &info.gid);
          infos_.push_back(info);
        }
      }
    }
  }

  ~EFAInfo() {
    for (auto& info : infos_) {
      if (info.ctx) ibv_close_device(info.ctx);
    }
    if (list_) ibv_free_device_list(list_);
  }

  /** @brief Parse PCI address from ibverbs device sysfs path */
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
  friend class EFA;
};

/**
 * @brief IB EFA device wrapper
 */
class EFA : private NoCopy {
 public:
  EFA() = delete;

  EFA(hwloc_obj_t efa) {
    efa_ = Get(efa);
    if (!efa_) throw std::runtime_error("IB EFA device not found for hwloc object");
    Open(efa_);
  }

  EFA(EFA&& other) noexcept
      : efa_{std::exchange(other.efa_, nullptr)},
        domain_{std::exchange(other.domain_, nullptr)},
        cq_{std::exchange(other.cq_, nullptr)},
        av_{std::exchange(other.av_, nullptr)},
        ep_{std::exchange(other.ep_, nullptr)} {
    std::memcpy(addr_, other.addr_, sizeof(addr_));
    std::memset(other.addr_, 0, sizeof(other.addr_));
  }

  ~EFA() noexcept {
    if (ep_) {
      ib_ep_close(ep_);
      ep_ = nullptr;
    }
    if (av_) {
      ib_av_close(av_);
      av_ = nullptr;
    }
    if (cq_) {
      ib_cq_close(cq_);
      cq_ = nullptr;
    }
    if (domain_) {
      ib_domain_close(domain_);
      domain_ = nullptr;
    }
  }

  const char* GetAddr() const noexcept { return addr_; }
  ib_cq* GetCQ() noexcept { return cq_; }
  ib_av* GetAV() noexcept { return av_; }
  ib_domain* GetDomain() noexcept { return domain_; }
  ib_ep* GetEP() noexcept { return ep_; }

  static std::string Addr2Str(const char* addr) { return rdma::Addr2Str(addr); }

 private:
  /**
   * @brief Find IB device matching hwloc PCI object
   * @param efa hwloc object representing an EFA device
   * @return Pointer to matching ib_info or nullptr if not found
   */
  ib_info* Get(hwloc_obj_t efa) {
    auto& infos = EFAInfo::Get();
    auto hw = efa->attr->pcidev;
    for (auto& info : infos) {
      unsigned domain, bus, slot, func;
      if (EFAInfo::ParsePCI(info.dev, domain, bus, slot, func)) {
        if (domain == hw.domain && bus == hw.bus && slot == hw.dev && func == hw.func) {
          return &info;
        }
      }
    }
    return nullptr;
  }

  /** @brief Initialize IB resources (domain, CQ, AV, endpoint) */
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
  ib_info* efa_ = nullptr;
  ib_domain* domain_ = nullptr;
  ib_cq* cq_ = nullptr;
  ib_av* av_ = nullptr;
  ib_ep* ep_ = nullptr;
  char addr_[kMaxAddrSize] = {};
};

// Backend implementation
inline Backend::remote_addr_type Backend::Connect(EFA* efa, const char* addr) {
  auto* av = efa->GetAV();
  ibv_ah* ah = ib_av_insert(av, addr);
  ASSERT(ah);
  return {ah, ib_addr_qpn(addr), ib_addr_qkey(addr)};
}

inline void Backend::Disconnect(EFA* efa, remote_addr_type& remote) {
  if (remote.ah) {
    ibv_destroy_ah(remote.ah);
    remote.ah = nullptr;
  }
}

inline ssize_t
Backend::PostWrite(EFA* efa, void* data, size_t len, mr_type mr, uint64_t addr, key_type key, uint64_t imm, remote_addr_type remote, Context* ctx) {
  iovec iov{data, len};
  uint32_t lkey = ib_mr_lkey(mr);
  ib_rma_iov rma_iov{addr, len, key};
  ib_msg_rma msg{&iov, &lkey, 1, &rma_iov, 1, remote.ah, remote.qpn, remote.qkey, ctx, imm};
  uint64_t flags = imm ? IB_REMOTE_CQ_DATA : 0;
  return ib_writemsg(efa->GetEP(), &msg, flags);
}

inline ssize_t
Backend::PostSend(EFA* efa, void* data, size_t len, mr_type mr, uint64_t addr, key_type key, uint64_t imm, remote_addr_type remote, Context* ctx) {
  iovec iov{data, len};
  uint32_t lkey = ib_mr_lkey(mr);
  ib_rma_iov rma_iov{addr, len, key};
  ib_msg_rma msg{&iov, &lkey, 1, &rma_iov, 1, remote.ah, remote.qpn, remote.qkey, ctx, imm};
  return ib_sendmsg(efa->GetEP(), &msg, 0);
}

}  // namespace ib
