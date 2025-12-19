/**
 * @file proxy.h
 * @brief Proxy selector for GPU-initiated networking (GIN)
 *
 * Implements a proxy pattern for polling GPU commands and initiating
 * inter-node RDMA communication directly from GPU memory, avoiding
 * the overhead of offloading data to CPU before transmission.
 */
#pragma once
#include <io/common.h>
#include <io/selector.h>
#include <rdma/fabric/selector.h>

#include <algorithm>
#include <memory>
#include <type_traits>
#include <vector>

/**
 * @brief Proxy selector for GPU-initiated inter-node communication
 *
 * Aggregates completion events from RDMA and GPU command queues,
 * enabling a proxy thread to poll GPU-issued commands and dispatch
 * RDMA operations without CPU-side memory copies.
 *
 * @tparam SelectorType The underlying selector type
 */
template <typename SelectorType>
class Proxy : public detail::Selector {
 public:
  Proxy() : main_selector_{std::make_unique<SelectorType>()} { selectors_.reserve(8); }

  /**
   * @brief Poll for events from RDMA and GPU command sources
   * @param duration Timeout duration for main selector
   * @return Combined vector of events from all sources
   */
  [[nodiscard]] std::vector<Event> Select(ms duration = ms{500}) override final {
    auto events = main_selector_->Select(duration);
    for (auto* __restrict s : selectors_) {
      auto x = s->Select(ms{0});
      for (auto& e : x) events.push_back(std::move(e));
    }
    return events;
  }

  /**
   * @brief Check if proxy has no registered sources
   * @return true if no completion queues are registered
   */
  [[nodiscard]] bool Stopped() const noexcept override final { return main_selector_->Stopped(); }

  /**
   * @brief Register an RDMA endpoint for completion monitoring
   * @tparam E Endpoint type (must have GetCQ() method)
   * @param e Endpoint to monitor
   */
  template <typename E, std::enable_if_t<!std::is_same_v<std::decay_t<E>, ImmContext>, int> = 0>
  void Join(E& e) noexcept {
    main_selector_->Join(e);
  }

  /**
   * @brief Unregister an RDMA endpoint
   * @tparam E Endpoint type (must have GetCQ() method)
   * @param e Endpoint to stop monitoring
   */
  template <typename E, std::enable_if_t<!std::is_same_v<std::decay_t<E>, ImmContext>, int> = 0>
  void Quit(E& e) noexcept {
    main_selector_->Quit(e);
  }

  /**
   * @brief Register for immediate data events (RDMA write notifications)
   * @param ctx Context to monitor for immediate data
   * @return true if completion already pending
   */
  bool Join(ImmContext& ctx) noexcept { return main_selector_->Join(ctx); }

  /**
   * @brief Unregister immediate data context
   * @param ctx Context to stop monitoring
   */
  void Quit(ImmContext& ctx) noexcept { main_selector_->Quit(ctx); }

  /**
   * @brief Add a sub-selector (e.g., GPU command queue poller)
   * @param s Selector to aggregate events from
   */
  void Join(detail::Selector* s) noexcept {
    ASSERT(s != nullptr);
    if (std::find(selectors_.begin(), selectors_.end(), s) == selectors_.end()) [[likely]] {
      selectors_.push_back(s);
    }
  }

  /**
   * @brief Remove a sub-selector
   * @param s Selector to remove
   */
  void Quit(detail::Selector* s) noexcept {
    ASSERT(s != nullptr);
    if (auto it = std::find(selectors_.begin(), selectors_.end(), s); it != selectors_.end()) [[likely]] {
      *it = selectors_.back();
      selectors_.pop_back();
    }
  }

 protected:
  std::unique_ptr<SelectorType> main_selector_;  ///< RDMA completion queue selector
  std::vector<detail::Selector*> selectors_;     ///< GPU command queue pollers
};

/// Type alias for libfabric-based proxy
using FabricProxy = Proxy<FabricSelector>;
