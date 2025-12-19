#pragma once
#include <mpi.h>
#include <spdlog/spdlog.h>

#include <iostream>

/**
 * @brief Singleton wrapper for MPI initialization and process information
 */
class MPI {
 public:
  /**
   * @brief Get singleton MPI instance
   * @return Reference to the MPI singleton
   */
  [[nodiscard]] inline static MPI& Get() {
    static MPI mpi;
    return mpi;
  }

  MPI(const MPI&) = delete;
  MPI(MPI&&) = delete;
  MPI& operator=(const MPI&) = delete;
  MPI& operator=(MPI&&) = delete;

  /** @brief Get total number of MPI processes */
  [[nodiscard]] inline int GetWorldSize() const noexcept { return world_size_; }
  /** @brief Get current process rank in world communicator */
  [[nodiscard]] inline int GetWorldRank() const noexcept { return world_rank_; }
  /** @brief Get number of processes on local node */
  [[nodiscard]] inline int GetLocalSize() const noexcept { return local_size_; }
  /** @brief Get current process rank on local node */
  [[nodiscard]] inline int GetLocalRank() const noexcept { return local_rank_; }
  /** @brief Get total number of compute nodes */
  [[nodiscard]] inline int GetNumNodes() const noexcept { return num_nodes_; }
  /** @brief Get current node index */
  [[nodiscard]] inline int GetNodeIndex() const noexcept { return node_; };
  /** @brief Get processor name string */
  [[nodiscard]] const char* GetProcessName() const noexcept { return processor_name_; }

 private:
  MPI() {
    MPI_Init(nullptr, nullptr);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm_);
    MPI_Comm_rank(local_comm_, &local_rank_);
    MPI_Comm_size(local_comm_, &local_size_);

    int len;
    MPI_Get_processor_name(processor_name_, &len);
    num_nodes_ = world_size_ / local_size_;
    node_ = world_rank_ / local_size_;
  }

  ~MPI() noexcept { MPI_Finalize(); }

 private:
  friend std::ostream& operator<<(std::ostream& os, const MPI& mpi) {
    os << "world_size: " << mpi.GetWorldSize();
    os << " world_rank: " << mpi.GetWorldRank();
    os << " local_size: " << mpi.GetLocalSize();
    os << " local_rank: " << mpi.GetLocalRank();
    os << " num_nodes: " << mpi.GetNumNodes();
    os << " node_index: " << mpi.GetNodeIndex();
    os << " process_name: " << mpi.GetProcessName();
    return os;
  }

 private:
  int world_size_;
  int world_rank_;
  int local_size_;
  int local_rank_;
  int num_nodes_;
  int node_;
  char processor_name_[MPI_MAX_PROCESSOR_NAME] = {0};
  MPI_Comm local_comm_;
};
