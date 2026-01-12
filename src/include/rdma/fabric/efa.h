/**
 * @file fabric/efa.h
 * @brief Fabric (libfabric) EFA device management and Backend traits
 */
#pragma once

#include <hwloc.h>
#include <io/common.h>
#include <rdma/efa.h>
#include <rdma/fabric.h>
#include <rdma/fabric/context.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <spdlog/spdlog.h>

namespace fi {

// Forward declarations
class EFA;
class FabricSelector;

// Context types defined in rdma/fabric/context.h

#define FI_CHECK(exp)                                                            \
  do {                                                                           \
    auto rc = exp;                                                               \
    if (rc) {                                                                    \
      auto msg = fmt::format(#exp " fail. error({}): {}", rc, fi_strerror(-rc)); \
      SPDLOG_ERROR(msg);                                                         \
      throw std::runtime_error(msg);                                             \
    }                                                                            \
  } while (0)

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
 * @brief Fabric Backend traits for unified Channel/Buffer/Memory
 */
struct Backend {
  using efa_type = EFA;
  using mr_type = struct fid_mr*;
  using cq_type = struct fid_cq*;
  using cq_entry_type = struct fi_cq_data_entry;
  using rma_iov_type = struct fi_rma_iov;
  using context_type = Context;
  using imm_context_type = ImmContext;
  using selector_type = FabricSelector;
  using key_type = uint64_t;
  using remote_addr_type = fi_addr_t;

  static constexpr remote_addr_type kInvalidAddr = FI_ADDR_UNSPEC;
  static constexpr bool kNeedsRmaForSend = false;

  // MR operations
  static uint64_t GetKey(mr_type mr) { return mr->key; }
  static void* GetDesc(mr_type mr) { return mr->mem_desc; }
  static void CloseMR(mr_type mr) { fi_close((fid_t)mr); }

  // RMA IOV
  static rma_iov_type MakeRmaIov(uint64_t addr, size_t size, mr_type mr) { return {addr, size, mr->key}; }

  // Connection
  static remote_addr_type Connect(EFA* efa, const char* addr);
  static void Disconnect(EFA* efa, remote_addr_type& remote);

  // Post operations
  static ssize_t
  PostWrite(EFA* efa, void* data, size_t len, mr_type mr, uint64_t addr, key_type key, uint64_t imm, remote_addr_type remote, Context* ctx);
  static ssize_t
  PostSend(EFA* efa, void* data, size_t len, mr_type mr, uint64_t addr, key_type key, uint64_t imm, remote_addr_type remote, Context* ctx);
  static ssize_t PostRecv(EFA* efa, void* data, size_t len, mr_type mr, Context* ctx);

  // Completion length
  static ssize_t GetWriteLen(const Context& ctx, size_t req_size) { return ctx.entry.len; }
  static ssize_t GetSendLen(const Context& ctx, size_t req_size) { return ctx.entry.len; }
  static ssize_t GetRecvLen(const Context& ctx) { return ctx.entry.len; }
};

/**
 * @brief Singleton for Fabric EFA provider discovery
 */
class EFAInfo : private NoCopy {
 public:
  static const struct fi_info* Get() {
    static EFAInfo instance;
    return instance.info_;
  }

 private:
  EFAInfo() : info_{New()} {}
  ~EFAInfo() {
    if (info_) fi_freeinfo(info_);
  }

  static struct fi_info* New() {
    struct fi_info* hints = fi_allocinfo();
    if (!hints) return nullptr;

    hints->caps = FI_MSG | FI_RMA | FI_HMEM | FI_LOCAL_COMM | FI_REMOTE_COMM;
    hints->ep_attr->type = FI_EP_RDM;
    hints->fabric_attr->prov_name = strdup("efa");
    hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_HMEM | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
    hints->domain_attr->threading = FI_THREAD_SAFE;

    struct fi_info* info = nullptr;
    int rc = fi_getinfo(FI_VERSION(1, 20), NULL, NULL, 0, hints, &info);
    fi_freeinfo(hints);
    if (rc != 0) {
      SPDLOG_ERROR("fi_getinfo fail. error({}): {}", rc, fi_strerror(-rc));
      return nullptr;
    }
    return info;
  }

  struct fi_info* info_ = nullptr;
};

/**
 * @brief Fabric EFA device wrapper
 */
class EFA : private NoCopy {
 public:
  EFA() = delete;

  EFA(hwloc_obj_t efa) {
    efa_ = Get(efa);
    ASSERT(!!efa_);
    Open(efa_);
  }

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

  ~EFA() noexcept {
    // Close in reverse order of creation: ep depends on cq/av, which depend on domain, which depends on fabric
    if (ep_) {
      fi_close((fid_t)ep_);
      ep_ = nullptr;
    }
    if (av_) {
      fi_close((fid_t)av_);
      av_ = nullptr;
    }
    if (cq_) {
      fi_close((fid_t)cq_);
      cq_ = nullptr;
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

  const char* GetAddr() const noexcept { return addr_; }
  struct fid_cq* GetCQ() noexcept { return cq_; }
  struct fid_av* GetAV() noexcept { return av_; }
  struct fid_domain* GetDomain() noexcept { return domain_; }
  struct fid_ep* GetEP() noexcept { return ep_; }
  const struct fi_info* GetInfo() const noexcept { return efa_; }

  static std::string Addr2Str(const char* addr) { return rdma::Addr2Str(addr); }

 private:
  struct fi_info* Get(hwloc_obj_t efa) {
    auto* info = EFAInfo::Get();
    for (auto p = info; !!p; p = p->next) {
      ASSERT(!!p->nic && p->nic->bus_attr && p->nic->bus_attr->bus_type == FI_BUS_PCI);
      auto fi = p->nic->bus_attr->attr.pci;
      auto hw = efa->attr->pcidev;
      if (fi.domain_id == hw.domain && fi.bus_id == hw.bus && fi.device_id == hw.dev && fi.function_id == hw.func) {
        return const_cast<struct fi_info*>(p);
      }
    }
    return nullptr;
  }

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
  struct fi_info* efa_ = nullptr;
  struct fid_fabric* fabric_ = nullptr;
  struct fid_domain* domain_ = nullptr;
  struct fid_ep* ep_ = nullptr;
  struct fid_cq* cq_ = nullptr;
  struct fid_av* av_ = nullptr;
  char addr_[rdma::kMaxAddrSize] = {};
};

// Backend implementation
inline Backend::remote_addr_type Backend::Connect(EFA* efa, const char* addr) {
  fi_addr_t remote = FI_ADDR_UNSPEC;
  FI_EXPECT(fi_av_insert(efa->GetAV(), addr, 1, &remote, 0, nullptr), 1);
  return remote;
}

inline void Backend::Disconnect(EFA* efa, remote_addr_type& remote) {
  if (remote != FI_ADDR_UNSPEC && efa) {
    fi_av_remove(efa->GetAV(), &remote, 1, 0);
    remote = FI_ADDR_UNSPEC;
  }
}

inline ssize_t
Backend::PostWrite(EFA* efa, void* data, size_t len, mr_type mr, uint64_t addr, key_type key, uint64_t imm, remote_addr_type remote, Context* ctx) {
  struct iovec iov{data, len};
  struct fi_rma_iov rma_iov{addr, len, key};
  struct fi_msg_rma msg{&iov, &mr->mem_desc, 1, remote, &rma_iov, 1, ctx, imm};
  uint64_t flags = imm ? FI_REMOTE_CQ_DATA : 0;
  return fi_writemsg(efa->GetEP(), &msg, flags);
}

inline ssize_t
Backend::PostSend(EFA* efa, void* data, size_t len, mr_type mr, uint64_t addr, key_type key, uint64_t imm, remote_addr_type remote, Context* ctx) {
  struct iovec iov{data, len};
  struct fi_msg msg{&iov, &mr->mem_desc, 1, remote, ctx, 0};
  return fi_sendmsg(efa->GetEP(), &msg, 0);
}

inline ssize_t Backend::PostRecv(EFA* efa, void* data, size_t len, mr_type mr, Context* ctx) {
  struct iovec iov{data, len};
  struct fi_msg msg{&iov, &mr->mem_desc, 1, FI_ADDR_UNSPEC, ctx, 0};
  return fi_recvmsg(efa->GetEP(), &msg, 0);
}

}  // namespace fi
