/**
 * @file ib.h
 * @brief Base RDMA peer class for ibverbs communication
 *
 * ibverbs connection exchange flow:
 *
 *   Rank 0                          Rank 1
 *      │                               │
 *      │  ┌─────────────────────────┐  │
 *      ├──│ ibv_alloc_pd()          │──┤
 *      │  └─────────────────────────┘  │
 *      │  ┌─────────────────────────┐  │
 *      ├──│ ibv_create_cq()         │──┤
 *      │  └─────────────────────────┘  │
 *      │  ┌─────────────────────────┐  │
 *      ├──│ ibv_create_qp()         │──┤
 *      │  └─────────────────────────┘  │
 *      │                               │
 *      │      MPI_Allgather(GID)       │
 *      │◄─────────────────────────────►│
 *      │                               │
 *      │  ┌─────────────────────────┐  │
 *      ├──│ ibv_create_ah()         │──┤
 *      │  └─────────────────────────┘  │
 *      │                               │
 *      ▼                               ▼
 *    Ready                           Ready
 */
#pragma once

#include <affinity/affinity.h>
#include <bootstrap/mpi/mpi.h>
#include <rdma/ib/efa.h>

#include <cstring>
#include <vector>

namespace ib {

/** @brief Connection metadata for ibverbs peer exchange */
struct ConnInfo {
  uint32_t qpn; /**< Queue pair number */
  uint32_t psn; /**< Packet sequence number */
  ibv_gid gid;  /**< GID for routing */
};

/**
 * @brief Base RDMA peer for ibverbs communication
 *
 * Manages MPI, ibverbs resources (PD, CQ, QP), and connection setup.
 * Uses GPU affinity to select EFA devices on the same PCI bridge.
 */
class Peer : private NoCopy {
 public:
  MPI& mpi;
  const GPUloc& loc;
  int device = -1;
  std::vector<IBDevice> devices;
  std::vector<ibv_pd*> pds;
  std::vector<ibv_cq*> cqs;
  std::vector<ibv_qp*> qps;
  std::vector<std::vector<ConnInfo>> remote_info;  // [rank][device]
  std::vector<std::vector<ibv_ah*>> ahs;           // [rank][device]

  Peer() : mpi(MPI::Get()), loc(GPUloc::Get()) {
    const auto world_size = mpi.GetWorldSize();
    device = mpi.GetLocalRank();
    remote_info.resize(world_size);
    ahs.resize(world_size);

    // Use GPU affinity to find corresponding EFA devices
    auto& affinity = loc.GetGPUAffinity()[device];
    for (auto e : affinity.efas) {
      devices.emplace_back(IBDevice(e));
    }

    // Init PD, CQ, QP for each device
    for (auto& dev : devices) {
      auto* pd = ibv_alloc_pd(dev.Ctx());
      IBV_CHECK(pd);
      pds.push_back(pd);

      auto* cq = ibv_create_cq(dev.Ctx(), 128, nullptr, nullptr, 0);
      IBV_CHECK(cq);
      cqs.push_back(cq);

      ibv_qp_init_attr qp_attr{};
      qp_attr.send_cq = cq;
      qp_attr.recv_cq = cq;
      qp_attr.qp_type = IBV_QPT_UD;
      qp_attr.cap.max_send_wr = 64;
      qp_attr.cap.max_recv_wr = 64;
      qp_attr.cap.max_send_sge = 1;
      qp_attr.cap.max_recv_sge = 1;

      auto* qp = ibv_create_qp(pd, &qp_attr);
      IBV_CHECK(qp);
      qps.push_back(qp);

      ModifyQPToInit(qp);
      ModifyQPToRTR(qp);
      ModifyQPToRTS(qp);
    }
  }

  ~Peer() {
    for (auto& rank_ahs : ahs) {
      for (auto* ah : rank_ahs)
        if (ah) ibv_destroy_ah(ah);
    }
    for (auto* qp : qps)
      if (qp) ibv_destroy_qp(qp);
    for (auto* cq : cqs)
      if (cq) ibv_destroy_cq(cq);
    for (auto* pd : pds)
      if (pd) ibv_dealloc_pd(pd);
  }

  /** @brief Exchange connection info (GID, QPN) via MPI_Allgather */
  void Exchange() {
    const auto world_size = mpi.GetWorldSize();

    for (size_t d = 0; d < devices.size(); ++d) {
      ConnInfo local{};
      local.qpn = qps[d]->qp_num;
      local.psn = 0;
      local.gid = devices[d].GID();

      std::vector<ConnInfo> all_info(world_size);
      MPI_Allgather(&local, sizeof(ConnInfo), MPI_BYTE, all_info.data(), sizeof(ConnInfo), MPI_BYTE, MPI_COMM_WORLD);

      for (int r = 0; r < world_size; ++r) {
        remote_info[r].push_back(all_info[r]);
      }
    }
  }

  /** @brief Create address handles for all remote peers */
  void Connect() {
    const auto rank = mpi.GetWorldRank();
    const auto world_size = mpi.GetWorldSize();

    for (int r = 0; r < world_size; ++r) {
      if (r == rank) continue;
      for (size_t d = 0; d < devices.size(); ++d) {
        ibv_ah_attr ah_attr{};
        ah_attr.is_global = 1;
        ah_attr.port_num = 1;
        ah_attr.grh.dgid = remote_info[r][d].gid;
        ah_attr.grh.sgid_index = 0;
        ah_attr.grh.hop_limit = 255;

        auto* ah = ibv_create_ah(pds[d], &ah_attr);
        IBV_CHECK(ah);
        ahs[r].push_back(ah);
      }
    }
  }

 private:
  void ModifyQPToInit(ibv_qp* qp) {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = 1;
    attr.qkey = 0x11111111;
    IBV_CHECK(ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY) == 0);
  }

  void ModifyQPToRTR(ibv_qp* qp) {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTR;
    IBV_CHECK(ibv_modify_qp(qp, &attr, IBV_QP_STATE) == 0);
  }

  void ModifyQPToRTS(ibv_qp* qp) {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0;
    IBV_CHECK(ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN) == 0);
  }
};

}  // namespace ib
