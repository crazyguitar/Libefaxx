/**
 * @file arguments.h
 * @brief Command-line argument parsing for benchmarks
 */
#pragma once

#include <getopt.h>

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Benchmark configuration options
 */
struct Options {
  size_t minbytes = 8 * 1024 * 1024;     ///< Minimum buffer size (default: 8M)
  size_t maxbytes = 1024 * 1024 * 1024;  ///< Maximum buffer size (default: 1G)
  size_t stepbytes = 2;                  ///< Step multiply factor (default: 2)
  int repeat = 1024;                     ///< Benchmark iterations (default: 1024)
  int warmup = 8;                        ///< Warmup iterations (default: 8)
};

/**
 * @brief Print usage information
 * @param prog Program name
 */
inline void usage(const char* prog) {
  std::cout << "Usage: " << prog << " [OPTIONS]\n"
            << "Options:\n"
            << "  -h, --help              Show this help message\n"
            << "  -b, --minbytes=SIZE     Minimum buffer size (default: 8M)\n"
            << "  -e, --maxbytes=SIZE     Maximum buffer size (default: 1G)\n"
            << "  -i, --stepbytes=FACTOR  Step factor (default: 2, multiply)\n"
            << "  -r, --repeat=N          Number of benchmark iterations (default: 1024)\n"
            << "  -w, --warmup=N          Number of warmup iterations (default: 8)\n";
}

/**
 * @brief Parse size string with K/M/G suffix
 * @param str Size string (e.g., "128K", "8M", "1G")
 * @return Size in bytes
 */
inline size_t parse_size(const char* str) {
  char* end;
  size_t val = std::strtoull(str, &end, 10);
  if (*end == 'K' || *end == 'k')
    val *= 1024;
  else if (*end == 'M' || *end == 'm')
    val *= 1024 * 1024;
  else if (*end == 'G' || *end == 'g')
    val *= 1024 * 1024 * 1024;
  return val;
}

/**
 * @brief Parse command-line arguments
 * @param argc Argument count
 * @param argv Argument values
 * @return Parsed options
 */
inline Options parse_args(int argc, char* argv[]) {
  Options opts;
  static struct option long_opts[] = {
      {"help", no_argument, nullptr, 'h'},
      {"minbytes", required_argument, nullptr, 'b'},
      {"maxbytes", required_argument, nullptr, 'e'},
      {"stepbytes", required_argument, nullptr, 'i'},
      {"repeat", required_argument, nullptr, 'r'},
      {"warmup", required_argument, nullptr, 'w'},
      {nullptr, 0, nullptr, 0}
  };
  int opt;
  while ((opt = getopt_long(argc, argv, "hb:e:i:r:w:", long_opts, nullptr)) != -1) {
    switch (opt) {
      case 'h':
        usage(argv[0]);
        exit(0);
      case 'b':
        opts.minbytes = parse_size(optarg);
        break;
      case 'e':
        opts.maxbytes = parse_size(optarg);
        break;
      case 'i':
        opts.stepbytes = parse_size(optarg);
        break;
      case 'r':
        opts.repeat = std::stoi(optarg);
        break;
      case 'w':
        opts.warmup = std::stoi(optarg);
        break;
      default:
        usage(argv[0]);
        exit(1);
    }
  }
  return opts;
}

/**
 * @brief Generate list of sizes from min to max with step factor
 * @param opts Benchmark options
 * @return Vector of buffer sizes to test
 */
inline std::vector<size_t> generate_sizes(const Options& opts) {
  std::vector<size_t> sizes;
  for (size_t s = opts.minbytes; s <= opts.maxbytes; s *= opts.stepbytes) sizes.push_back(s);
  if (sizes.empty() || sizes.back() < opts.maxbytes) sizes.push_back(opts.maxbytes);
  return sizes;
}
