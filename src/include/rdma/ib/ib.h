/**
 * @file ib.h
 * @brief ibverbs utility functions equivalent to libfabric fi_* functions
 *
 * This provides ib_domain, ib_cq_open, ib_av_open, ib_endpoint, ib_ep_bind,
 * ib_enable, ib_getname as ibverbs-based equivalents to libfabric.
 *
 * Function mapping (libfabric -> ibverbs):
 *
 *   ┌─────────────────┬─────────────────────────────────────────────────┐
 *   │ libfabric       │ ibverbs equivalent                              │
 *   ├─────────────────┼─────────────────────────────────────────────────┤
 *   │ fi_domain()     │ ib_domain_open() -> ibv_alloc_pd()              │
 *   │ fi_cq_open()    │ ib_cq_open()     -> ibv_create_cq_ex()          │
 *   │ fi_av_open()    │ ib_av_open()     -> (stores PD for AH creation) │
 *   │ fi_endpoint()   │ ib_endpoint()    -> efadv_create_qp_ex()        │
 *   │ fi_ep_bind()    │ ib_ep_bind()     -> (CQ bound at QP creation)   │
 *   │ fi_enable()     │ ib_enable()      -> ibv_modify_qp() x3          │
 *   │ fi_getname()    │ ib_getname()     -> (GID + QPN + QKEY)          │
 *   │ fi_av_insert()  │ ib_av_insert()   -> ibv_create_ah()             │
 *   └─────────────────┴─────────────────────────────────────────────────┘
 *
 * Initialization flow:
 *
 *   ┌──────────────────────┐
 *   │  ib_domain_open()    │  Allocate protection domain
 *   │  ibv_alloc_pd()      │
 *   └──────────┬───────────┘
 *              │
 *              ▼
 *   ┌──────────────────────┐
 *   │  ib_cq_open()        │  Create completion queue
 *   │  ibv_create_cq_ex()  │
 *   └──────────┬───────────┘
 *              │
 *              ▼
 *   ┌──────────────────────┐
 *   │  ib_av_open()        │  Initialize address vector (stores PD)
 *   └──────────┬───────────┘
 *              │
 *              ▼
 *   ┌──────────────────────┐
 *   │  ib_endpoint()       │  Create SRD queue pair
 *   │  efadv_create_qp_ex()│  (EFA-specific QP type)
 *   └──────────┬───────────┘
 *              │
 *              ▼
 *   ┌──────────────────────┐
 *   │  ib_enable()         │  Transition QP state
 *   │  ibv_modify_qp()     │  RESET -> INIT -> RTR -> RTS
 *   └──────────┬───────────┘
 *              │
 *              ▼
 *   ┌──────────────────────┐
 *   │  ib_getname()        │  Get local address
 *   │  (GID + QPN + QKEY)  │  (32 bytes total)
 *   └──────────────────────┘
 *
 * QP state transition in ib_enable():
 *
 *   ┌───────┐  IBV_QP_STATE |   ┌───────┐  IBV_QP_STATE   ┌───────┐  IBV_QP_STATE |
 *   │ RESET │  PKEY | PORT  │   │ INIT  │  ────────────►  │  RTR  │  SQ_PSN       │
 *   └───┬───┘  QKEY         │   └───┬───┘                 └───┬───┘  ────────────►│
 *       │  ─────────────────►       │                         │                   │
 *       │                           │                         │              ┌────┴───┐
 *       │                           │                         │              │  RTS   │
 *       │                           │                         │              │ (Ready)│
 *       │                           │                         │              └────────┘
 *
 * Address format (32 bytes):
 *
 *   ┌────────────────────────────────────────────────────────────────┐
 *   │  GID (16 bytes)  │  QPN (4 bytes)  │  QKEY (4 bytes)  │ pad(8) │
 *   └────────────────────────────────────────────────────────────────┘
 *    0                 16                20                 24       32
 */
#pragma once

#include <hwloc.h>
#include <infiniband/efadv.h>
#include <infiniband/verbs.h>
#include <io/common.h>
#include <spdlog/spdlog.h>

#include <cstring>
#include <vector>

namespace ib {

/** @brief Maximum buffer size for endpoint addresses */
static constexpr size_t kMaxAddrSize = 64;
/** @brief Actual size of EFA endpoint addresses */
static constexpr size_t kAddrSize = 32;

/**
 * @brief Check ibverbs call result and throw on error
 * @param exp Expression to evaluate
 * @throws std::runtime_error if expression is false/null
 */
#define IB_CHECK(exp)                                            \
  do {                                                           \
    if (!(exp)) {                                                \
      auto msg = fmt::format(#exp " fail: {}", strerror(errno)); \
      SPDLOG_ERROR(msg);                                         \
      throw std::runtime_error(msg);                             \
    }                                                            \
  } while (0)

/** @brief Domain handle (equivalent to fid_domain) */
struct ib_domain {
  ibv_context* ctx; /**< Device context */
  ibv_pd* pd;       /**< Protection domain */
  ibv_gid gid;      /**< Global identifier */
};

/** @brief Completion queue handle (equivalent to fid_cq) */
struct ib_cq {
  ibv_cq_ex* cq; /**< Extended completion queue */
};

/** @brief Address vector handle (equivalent to fid_av) */
struct ib_av {
  ibv_pd* pd; /**< Protection domain for AH creation */
};

/** @brief Endpoint handle (equivalent to fid_ep) */
struct ib_ep {
  ibv_qp_ex* qp; /**< Extended queue pair */
  ib_cq* cq;     /**< Bound completion queue */
  ib_av* av;     /**< Bound address vector */
  uint32_t qkey; /**< Queue key for UD/SRD */
};

/** @brief EFA device info (equivalent to fi_info) */
struct ib_info {
  ibv_device* dev;      /**< Device handle */
  ibv_context* ctx;     /**< Device context */
  ibv_device_attr attr; /**< Device attributes */
  ibv_port_attr port;   /**< Port attributes */
  ibv_gid gid;          /**< Global identifier */
};

/**
 * @brief Open domain (equivalent to fi_domain)
 *
 * Allocates a protection domain for the device context.
 *
 * @param info Device info containing context
 * @param domain Output domain handle
 * @return 0 on success, negative on error
 */
inline int ib_domain_open(ib_info* info, ib_domain** domain) {
  *domain = new ib_domain{};
  (*domain)->ctx = info->ctx;
  (*domain)->pd = ibv_alloc_pd(info->ctx);
  if (!(*domain)->pd) {
    delete *domain;
    *domain = nullptr;
    return -1;
  }
  (*domain)->gid = info->gid;
  return 0;
}

/**
 * @brief Close domain
 * @param domain Domain to close
 * @return 0 on success
 */
inline int ib_domain_close(ib_domain* domain) {
  if (domain) {
    if (domain->pd) ibv_dealloc_pd(domain->pd);
    delete domain;
  }
  return 0;
}

/**
 * @brief Open completion queue (equivalent to fi_cq_open)
 *
 * Creates an extended completion queue with byte length and immediate data.
 *
 * @param domain Domain handle
 * @param cq Output CQ handle
 * @return 0 on success, negative on error
 */
inline int ib_cq_open(ib_domain* domain, ib_cq** cq) {
  *cq = new ib_cq{};
  ibv_cq_init_attr_ex cq_attr{};
  cq_attr.cqe = 1024;
  cq_attr.wc_flags = IBV_WC_EX_WITH_BYTE_LEN | IBV_WC_EX_WITH_IMM;
  (*cq)->cq = ibv_create_cq_ex(domain->ctx, &cq_attr);
  if (!(*cq)->cq) {
    delete *cq;
    *cq = nullptr;
    return -1;
  }
  return 0;
}

/**
 * @brief Close completion queue
 * @param cq CQ to close
 * @return 0 on success
 */
inline int ib_cq_close(ib_cq* cq) {
  if (cq) {
    if (cq->cq) ibv_destroy_cq(ibv_cq_ex_to_cq(cq->cq));
    delete cq;
  }
  return 0;
}

/**
 * @brief Open address vector (equivalent to fi_av_open)
 *
 * Stores the protection domain for later AH creation.
 *
 * @param domain Domain handle
 * @param av Output AV handle
 * @return 0 on success
 */
inline int ib_av_open(ib_domain* domain, ib_av** av) {
  *av = new ib_av{};
  (*av)->pd = domain->pd;
  return 0;
}

/**
 * @brief Close address vector
 * @param av AV to close
 * @return 0 on success
 */
inline int ib_av_close(ib_av* av) {
  if (av) delete av;
  return 0;
}

/**
 * @brief Create endpoint (equivalent to fi_endpoint)
 *
 * Creates an EFA SRD queue pair with the specified CQ bound.
 *
 * @param domain Domain handle
 * @param info Device info
 * @param ep Output endpoint handle
 * @param cq Completion queue to bind
 * @return 0 on success, negative on error
 */
inline int ib_endpoint(ib_domain* domain, ib_info* info, ib_ep** ep, ib_cq* cq) {
  *ep = new ib_ep{};
  (*ep)->qkey = 0x11111111;

  ibv_qp_init_attr_ex qp_attr{};
  qp_attr.send_cq = ibv_cq_ex_to_cq(cq->cq);
  qp_attr.recv_cq = ibv_cq_ex_to_cq(cq->cq);
  qp_attr.cap.max_send_wr = 512;
  qp_attr.cap.max_recv_wr = 512;
  qp_attr.cap.max_send_sge = 1;
  qp_attr.cap.max_recv_sge = 1;
  qp_attr.qp_type = IBV_QPT_DRIVER;
  qp_attr.pd = domain->pd;
  qp_attr.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
  qp_attr.send_ops_flags = IBV_QP_EX_WITH_SEND | IBV_QP_EX_WITH_SEND_WITH_IMM;

  efadv_qp_init_attr efa_attr{};
  efa_attr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;
  auto* qp = efadv_create_qp_ex(domain->ctx, &qp_attr, &efa_attr, sizeof(efa_attr));
  if (!qp) {
    SPDLOG_ERROR("efadv_create_qp_ex failed: errno={} ({})", errno, strerror(errno));
    delete *ep;
    *ep = nullptr;
    return -errno;
  }
  (*ep)->qp = ibv_qp_to_qp_ex(qp);
  (*ep)->cq = cq;
  return 0;
}

/**
 * @brief Bind endpoint to CQ/AV (equivalent to fi_ep_bind)
 *
 * Note: CQ is already bound at QP creation time. This stores AV reference.
 *
 * @param ep Endpoint handle
 * @param res Resource to bind (ib_cq* or ib_av*)
 * @param flags Binding flags (unused)
 * @return 0 on success
 */
inline int ib_ep_bind(ib_ep* ep, void* res, uint64_t flags) {
  if (auto* cq = static_cast<ib_cq*>(res); cq && cq->cq) {
    ep->cq = cq;
  }
  if (auto* av = static_cast<ib_av*>(res); av && av->pd) {
    ep->av = av;
  }
  return 0;
}

/**
 * @brief Enable endpoint (equivalent to fi_enable)
 *
 * Transitions QP state: RESET -> INIT -> RTR -> RTS
 *
 * @param ep Endpoint handle
 * @return 0 on success, negative on error
 */
inline int ib_enable(ib_ep* ep) {
  ibv_qp* qp = &ep->qp->qp_base;

  // RESET -> INIT
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = 1;
  attr.qkey = ep->qkey;
  if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY)) return -1;

  // INIT -> RTR
  attr = {};
  attr.qp_state = IBV_QPS_RTR;
  if (ibv_modify_qp(qp, &attr, IBV_QP_STATE)) return -1;

  // RTR -> RTS
  attr = {};
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = 0;
  if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN)) return -1;

  return 0;
}

/**
 * @brief Close endpoint
 * @param ep Endpoint to close
 * @return 0 on success
 */
inline int ib_ep_close(ib_ep* ep) {
  if (ep) {
    if (ep->qp) ibv_destroy_qp(&ep->qp->qp_base);
    delete ep;
  }
  return 0;
}

/**
 * @brief Get endpoint name/address (equivalent to fi_getname)
 *
 * Returns 32-byte address: GID (16) + QPN (4) + QKEY (4) + padding (8)
 *
 * @param ep Endpoint handle
 * @param domain Domain handle (for GID)
 * @param addr Output address buffer
 * @param addrlen Input/output address length
 * @return 0 on success, negative on error
 */
inline int ib_getname(ib_ep* ep, ib_domain* domain, void* addr, size_t* addrlen) {
  if (*addrlen < kAddrSize) return -1;
  char* p = static_cast<char*>(addr);
  std::memcpy(p, &domain->gid, sizeof(ibv_gid));  // 16 bytes
  uint32_t qpn = ep->qp->qp_base.qp_num;
  std::memcpy(p + 16, &qpn, sizeof(uint32_t));       // 4 bytes
  std::memcpy(p + 20, &ep->qkey, sizeof(uint32_t));  // 4 bytes
  std::memset(p + 24, 0, kAddrSize - 24);            // padding to 32 bytes
  *addrlen = kAddrSize;
  return 0;
}

/**
 * @brief Insert address into AV (equivalent to fi_av_insert)
 *
 * Creates an address handle for the remote peer.
 *
 * @param av Address vector handle
 * @param addr Remote address (32 bytes)
 * @return ibv_ah* on success, nullptr on error
 */
inline ibv_ah* ib_av_insert(ib_av* av, const void* addr) {
  const char* p = static_cast<const char*>(addr);
  ibv_gid gid;
  std::memcpy(&gid, p, sizeof(ibv_gid));

  ibv_ah_attr ah_attr{};
  ah_attr.is_global = 1;
  ah_attr.grh.dgid = gid;
  ah_attr.port_num = 1;
  return ibv_create_ah(av->pd, &ah_attr);
}

/**
 * @brief Get QPN from address
 * @param addr Address buffer
 * @return Queue pair number
 */
inline uint32_t ib_addr_qpn(const void* addr) {
  uint32_t qpn;
  std::memcpy(&qpn, static_cast<const char*>(addr) + 16, sizeof(uint32_t));
  return qpn;
}

/**
 * @brief Get QKEY from address
 * @param addr Address buffer
 * @return Queue key
 */
inline uint32_t ib_addr_qkey(const void* addr) {
  uint32_t qkey;
  std::memcpy(&qkey, static_cast<const char*>(addr) + 20, sizeof(uint32_t));
  return qkey;
}

}  // namespace ib
