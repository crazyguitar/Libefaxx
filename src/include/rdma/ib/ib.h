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

#include <arpa/inet.h>
#include <hwloc.h>
#include <infiniband/efadv.h>
#include <infiniband/verbs.h>
#include <io/common.h>
#include <spdlog/spdlog.h>
#include <sys/uio.h>

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
  struct ibv_qp* qp_handle; /**< Original queue pair pointer */
  ibv_qp_ex* qp;            /**< Extended queue pair */
  ib_cq* cq;                /**< Bound completion queue */
  ib_av* av;                /**< Bound address vector */
  uint32_t qkey;            /**< Queue key for UD/SRD */
  bool is_wr_started;       /**< Work request batch in progress */
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
 * Creates an extended completion queue with standard flags.
 *
 * Reference: libfabric/prov/efa/src/efa_cq.c:943 (wc_flags = IBV_WC_STANDARD_FLAGS)
 * Reference: libfabric/prov/efa/src/efa_cq.h:166-171 (efa_cq_open_ibv_cq_with_ibv_create_cq_ex)
 *
 * @param domain Domain handle
 * @param cq Output CQ handle
 * @return 0 on success, negative on error
 */
inline int ib_cq_open(ib_domain* domain, ib_cq** cq) {
  *cq = new ib_cq{};

  ibv_cq_init_attr_ex cq_attr{};
  cq_attr.cqe = 1024;
  cq_attr.wc_flags = IBV_WC_STANDARD_FLAGS;
  cq_attr.comp_mask = 0;

  // Use efadv_create_cq to enable unsolicited write recv support
  // Reference: libfabric/prov/efa/src/efa_cq.c:952-957
  efadv_cq_init_attr efa_cq_attr{};
  efa_cq_attr.comp_mask = 0;
  efa_cq_attr.wc_flags = EFADV_WC_EX_WITH_SGID | EFADV_WC_EX_WITH_IS_UNSOLICITED;

  (*cq)->cq = efadv_create_cq(domain->ctx, &cq_attr, &efa_cq_attr, sizeof(efa_cq_attr));
  if (!(*cq)->cq) {
    delete *cq;
    *cq = nullptr;
    return -errno;
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
 * Enables RDMA WRITE operations for RMA support.
 *
 * Reference: libfabric/prov/efa/src/efa_base_ep.c:223-270 (efa_qp_create)
 * Reference: libfabric/prov/efa/src/efa_base_ep.c:243-244 (IBV_QP_EX_WITH_RDMA_WRITE flags)
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
  (*ep)->is_wr_started = false;

  ibv_qp_init_attr_ex qp_attr{};
  qp_attr.send_cq = ibv_cq_ex_to_cq(cq->cq);
  qp_attr.recv_cq = ibv_cq_ex_to_cq(cq->cq);
  qp_attr.cap.max_send_wr = 512;
  qp_attr.cap.max_recv_wr = 512;
  qp_attr.cap.max_send_sge = 2;
  qp_attr.cap.max_recv_sge = 2;
  qp_attr.qp_type = IBV_QPT_DRIVER;
  qp_attr.pd = domain->pd;
  qp_attr.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
  qp_attr.send_ops_flags = IBV_QP_EX_WITH_SEND | IBV_QP_EX_WITH_SEND_WITH_IMM | IBV_QP_EX_WITH_RDMA_WRITE | IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM;

  efadv_qp_init_attr efa_attr{};
  efa_attr.driver_qp_type = EFADV_QP_DRIVER_TYPE_SRD;
  // Enable unsolicited write recv - allows receiving RDMA write with immediate
  // data without posting receive buffers (EFA feature)
  // Reference: libfabric/prov/efa/src/efa_base_ep.c:246-249
  efa_attr.flags = EFADV_QP_FLAGS_UNSOLICITED_WRITE_RECV;

  auto* qp = efadv_create_qp_ex(domain->ctx, &qp_attr, &efa_attr, sizeof(efa_attr));
  if (!qp) {
    delete *ep;
    *ep = nullptr;
    return -errno;
  }
  (*ep)->qp_handle = qp;  // Store original ibv_qp*
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
  struct ibv_qp* qp = ep->qp_handle;

  // RESET -> INIT
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = 1;
  attr.qkey = ep->qkey;
  int rc = ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY);
  if (rc) return -1;

  // INIT -> RTR
  attr = {};
  attr.qp_state = IBV_QPS_RTR;
  rc = ibv_modify_qp(qp, &attr, IBV_QP_STATE);
  if (rc) return -1;

  // RTR -> RTS
  attr = {};
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = 0;
  return ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN) ? -1 : 0;
}

/**
 * @brief Close endpoint
 * @param ep Endpoint to close
 * @return 0 on success
 */
inline int ib_ep_close(ib_ep* ep) {
  if (ep) {
    if (ep->qp_handle) ibv_destroy_qp(ep->qp_handle);
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
  uint32_t qpn = ep->qp_handle->qp_num;
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

// ============================================================================
// Memory Registration (equivalent to fi_mr_reg)
// ============================================================================

/** @brief Memory region handle (equivalent to fid_mr) */
struct ib_mr {
  ibv_mr* mr;     /**< ibverbs memory region */
  ib_domain* dom; /**< Associated domain */
};

/** @brief Access flags for memory registration */
constexpr int IB_MR_LOCAL_READ = IBV_ACCESS_LOCAL_WRITE;
constexpr int IB_MR_REMOTE_WRITE = IBV_ACCESS_REMOTE_WRITE;
constexpr int IB_MR_REMOTE_READ = IBV_ACCESS_REMOTE_READ;

/**
 * @brief Register memory region (equivalent to fi_mr_reg)
 *
 * @param domain Domain handle
 * @param buf Buffer to register
 * @param len Buffer length
 * @param access Access flags (IB_MR_LOCAL_READ | IB_MR_REMOTE_WRITE | ...)
 * @param mr Output MR handle
 * @return 0 on success, negative on error
 */
inline int ib_mr_reg(ib_domain* domain, void* buf, size_t len, int access, ib_mr** mr) {
  *mr = new ib_mr{};
  (*mr)->dom = domain;
  (*mr)->mr = ibv_reg_mr(domain->pd, buf, len, access);
  if (!(*mr)->mr) {
    delete *mr;
    *mr = nullptr;
    return -errno;
  }
  return 0;
}

/**
 * @brief Deregister memory region (equivalent to fi_close on MR)
 * @param mr MR to close
 * @return 0 on success
 */
inline int ib_mr_close(ib_mr* mr) {
  if (mr) {
    if (mr->mr) ibv_dereg_mr(mr->mr);
    delete mr;
  }
  return 0;
}

/**
 * @brief Get local key from MR (for use in SGE)
 * @param mr Memory region
 * @return Local key
 */
inline uint32_t ib_mr_lkey(ib_mr* mr) { return mr->mr->lkey; }

// ============================================================================
// Receive Operations (required for RDMA write with immediate data)
// ============================================================================

/**
 * @brief Post receive buffer (required for RDMA write with immediate data)
 *
 * EFA SRD requires posted receive buffers to receive RDMA write with
 * immediate data completions (IBV_WC_RECV_RDMA_WITH_IMM).
 *
 * @param ep Endpoint handle
 * @param context User context (returned in CQE)
 * @return 0 on success, negative errno on error
 */
inline int ib_post_recv(ib_ep* ep, void* context) {
  ibv_recv_wr wr{};
  wr.wr_id = reinterpret_cast<uintptr_t>(context);
  wr.next = nullptr;
  wr.sg_list = nullptr;
  wr.num_sge = 0;

  ibv_recv_wr* bad_wr = nullptr;
  int ret = ibv_post_recv(&ep->qp->qp_base, &wr, &bad_wr);
  return ret ? -ret : 0;
}

/**
 * @brief Get remote key from MR (for remote access)
 * @param mr Memory region
 * @return Remote key
 */
inline uint32_t ib_mr_rkey(ib_mr* mr) { return mr->mr->rkey; }

// ============================================================================
// RMA Operations (equivalent to fi_writemsg)
// ============================================================================

/** @brief RMA IOV structure (equivalent to fi_rma_iov) */
struct ib_rma_iov {
  uint64_t addr; /**< Remote address */
  size_t len;    /**< Length */
  uint32_t key;  /**< Remote key (rkey) */
};

/**
 * @brief RMA message structure (equivalent to fi_msg_rma)
 *
 * Describes an RDMA write operation including local buffers,
 * remote memory targets, and addressing information.
 */
struct ib_msg_rma {
  const iovec* msg_iov;      /**< Local scatter-gather array */
  uint32_t* lkeys;           /**< Local keys for each IOV (from ib_mr_lkey) */
  size_t iov_count;          /**< Number of local IOVs */
  const ib_rma_iov* rma_iov; /**< Remote IOV array */
  size_t rma_iov_count;      /**< Number of remote IOVs (typically 1) */
  ibv_ah* ah;                /**< Address handle for remote peer */
  uint32_t qpn;              /**< Remote queue pair number */
  uint32_t qkey;             /**< Remote queue key */
  void* context;             /**< User context (returned in CQE) */
  uint64_t data;             /**< Immediate data (if IB_REMOTE_CQ_DATA) */
};

/** @brief Flag to send immediate data with RDMA write */
constexpr uint64_t IB_REMOTE_CQ_DATA = 1ULL << 0;
/** @brief Flag to batch multiple work requests */
constexpr uint64_t IB_MORE = 1ULL << 1;

/**
 * @brief Post RDMA write message (equivalent to fi_writemsg)
 *
 * Posts an RDMA write operation to the send queue.
 *
 * Reference: libfabric/prov/efa/src/efa_rma.c:177-241 (efa_rma_post_write)
 * Reference: libfabric/prov/efa/src/efa_rma.c:244-255 (efa_rma_writemsg)
 *
 * @param ep Endpoint handle
 * @param msg RMA message descriptor
 * @param flags Operation flags (IB_REMOTE_CQ_DATA, IB_MORE)
 * @return 0 on success, negative errno on error
 */
inline int ib_writemsg(ib_ep* ep, const ib_msg_rma* msg, uint64_t flags) {
  ibv_qp_ex* qp = ep->qp;

  if (!ep->is_wr_started) {
    ibv_wr_start(qp);
    ep->is_wr_started = true;
  }

  qp->wr_id = reinterpret_cast<uintptr_t>(msg->context);
  qp->wr_flags = IBV_SEND_SIGNALED;

  // Note: immediate data must be in network byte order (big-endian)
  if (flags & IB_REMOTE_CQ_DATA) {
    ibv_wr_rdma_write_imm(qp, msg->rma_iov[0].key, msg->rma_iov[0].addr, htonl(static_cast<uint32_t>(msg->data)));
  } else {
    ibv_wr_rdma_write(qp, msg->rma_iov[0].key, msg->rma_iov[0].addr);
  }

  ibv_sge sge_list[msg->iov_count];
  for (size_t i = 0; i < msg->iov_count; ++i) {
    sge_list[i].addr = reinterpret_cast<uint64_t>(msg->msg_iov[i].iov_base);
    sge_list[i].length = static_cast<uint32_t>(msg->msg_iov[i].iov_len);
    sge_list[i].lkey = msg->lkeys[i];
  }
  ibv_wr_set_sge_list(qp, msg->iov_count, sge_list);
  ibv_wr_set_ud_addr(qp, msg->ah, msg->qpn, msg->qkey);

  if (!(flags & IB_MORE)) {
    int ret = ibv_wr_complete(qp);
    ep->is_wr_started = false;
    return ret ? -ret : 0;
  }
  return 0;
}

/**
 * @brief Send message using RDMA WRITE with immediate data (like UCCL)
 *
 * Implements send semantics using RDMA WRITE to avoid EFA MTU limitations.
 * Requires pre-exchanged remote memory info (addr, key).
 *
 * @param ep Endpoint handle
 * @param msg Send message descriptor (includes remote RMA info)
 * @param flags Operation flags
 * @return 0 on success, negative errno on error
 */
inline int ib_sendmsg(ib_ep* ep, const ib_msg_rma* msg, uint64_t flags) { return ib_writemsg(ep, msg, flags | IB_REMOTE_CQ_DATA); }

/**
 * @brief Simplified RDMA write (equivalent to fi_write)
 *
 * Convenience wrapper for single-buffer RDMA write operations.
 *
 * @param ep Endpoint handle
 * @param buf Local buffer to write from
 * @param len Number of bytes to write
 * @param lkey Local memory key
 * @param ah Address handle for remote peer
 * @param qpn Remote queue pair number
 * @param qkey Remote queue key
 * @param remote_addr Remote memory address
 * @param remote_key Remote memory key
 * @param context User context
 * @return 0 on success, negative errno on error
 */
inline int ib_write(
    ib_ep* ep,
    void* buf,
    size_t len,
    uint32_t lkey,
    ibv_ah* ah,
    uint32_t qpn,
    uint32_t qkey,
    uint64_t remote_addr,
    uint32_t remote_key,
    void* context
) {
  iovec iov{buf, len};
  ib_rma_iov rma_iov{remote_addr, len, remote_key};
  ib_msg_rma msg{&iov, &lkey, 1, &rma_iov, 1, ah, qpn, qkey, context, 0};
  return ib_writemsg(ep, &msg, 0);
}

/**
 * @brief RDMA write with immediate data (equivalent to fi_writedata)
 *
 * @param ep Endpoint handle
 * @param buf Local buffer to write from
 * @param len Number of bytes to write
 * @param lkey Local memory key
 * @param ah Address handle for remote peer
 * @param qpn Remote queue pair number
 * @param qkey Remote queue key
 * @param remote_addr Remote memory address
 * @param remote_key Remote memory key
 * @param imm_data Immediate data to send
 * @param context User context
 * @return 0 on success, negative errno on error
 */
inline int ib_writedata(
    ib_ep* ep,
    void* buf,
    size_t len,
    uint32_t lkey,
    ibv_ah* ah,
    uint32_t qpn,
    uint32_t qkey,
    uint64_t remote_addr,
    uint32_t remote_key,
    uint64_t imm_data,
    void* context
) {
  iovec iov{buf, len};
  ib_rma_iov rma_iov{remote_addr, len, remote_key};
  ib_msg_rma msg{&iov, &lkey, 1, &rma_iov, 1, ah, qpn, qkey, context, imm_data};
  return ib_writemsg(ep, &msg, IB_REMOTE_CQ_DATA);
}

}  // namespace ib
