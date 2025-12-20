#pragma once

#include <array>
#include <cstdio>
#include <string>
#include <vector>

struct BenchResult;  // Forward declaration, defined in bench/mpi/fabric.cuh

class BenchPrinter {
 public:
  BenchPrinter(const char* title, int nranks, int warmup, int iters, double link_bw, const char* pattern, const std::vector<std::string>& columns)
      : title_(title), nranks_(nranks), warmup_(warmup), iters_(iters), link_bw_(link_bw), pattern_(pattern), columns_(columns) {}

  template <size_t N>
  void Print(const std::vector<std::array<BenchResult, N>>& results) {
    PrintHeader();
    for (const auto& r : results) PrintResult(r);
    PrintFooter();
  }

 private:
  const char* title_;
  int nranks_;
  int warmup_;
  int iters_;
  double link_bw_;
  const char* pattern_;
  std::vector<std::string> columns_;

  void PrintHeader() {
    printf("#\n# %s\n#\n", title_);
    printf("# nranks: %d\n", nranks_);
    printf("# warmup iters: %d\n", warmup_);
    printf("# bench iters: %d\n", iters_);
    printf("# link bandwidth: %.0f Gbps\n#\n", link_bw_);
    if (pattern_) printf("# Pattern: %s\n#\n", pattern_);
    printf("# BusBW: Percentage of theoretical link bandwidth achieved\n#\n");
    printf("%12s %12s", "size", "count");
    for (const auto& col : columns_) printf(" %14s %10s", col.c_str(), "BusBW(%)");
    printf("\n");
  }

  template <size_t N>
  void PrintResult(const std::array<BenchResult, N>& results) {
    printf("%12zu %12zu", results[0].size, results[0].size / sizeof(float));
    for (const auto& r : results) printf(" %14.2f %10.1f", r.bw_gbps, r.bus_bw);
    printf("\n");
  }

  void PrintFooter() { printf("#\n# Benchmark complete.\n"); }
};
