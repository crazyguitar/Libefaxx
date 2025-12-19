#pragma once

#include <getopt.h>

#include <iostream>
#include <vector>

struct Options {
  size_t minbytes = 128 * 1024;         // 128K
  size_t maxbytes = 128 * 1024 * 1024;  // 128M
  size_t stepbytes = 2;                 // multiply factor
  int repeat = 1024;
  int warmup = 8;
};

inline void usage(const char* prog) {
  std::cout << "Usage: " << prog << " [OPTIONS]\n"
            << "Options:\n"
            << "  -h, --help              Show this help message\n"
            << "  -b, --minbytes=SIZE     Minimum buffer size (default: 128K)\n"
            << "  -e, --maxbytes=SIZE     Maximum buffer size (default: 128M)\n"
            << "  -i, --stepbytes=FACTOR  Step factor (default: 2, multiply)\n"
            << "  -r, --repeat=N          Number of benchmark iterations (default: 1024)\n"
            << "  -w, --warmup=N          Number of warmup iterations (default: 8)\n";
}

/// Parse size string with K/M/G suffix
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

/// Generate list of sizes from min to max with step factor
inline std::vector<size_t> generate_sizes(const Options& opts) {
  std::vector<size_t> sizes;
  for (size_t s = opts.minbytes; s <= opts.maxbytes; s *= opts.stepbytes) {
    sizes.push_back(s);
  }
  if (sizes.empty() || sizes.back() < opts.maxbytes) {
    sizes.push_back(opts.maxbytes);
  }
  return sizes;
}

/// Format size with appropriate unit
inline std::string format_size(size_t bytes) {
  char buf[32];
  if (bytes >= 1024 * 1024 * 1024)
    snprintf(buf, sizeof(buf), "%zuG", bytes / (1024 * 1024 * 1024));
  else if (bytes >= 1024 * 1024)
    snprintf(buf, sizeof(buf), "%zuM", bytes / (1024 * 1024));
  else if (bytes >= 1024)
    snprintf(buf, sizeof(buf), "%zuK", bytes / 1024);
  else
    snprintf(buf, sizeof(buf), "%zuB", bytes);
  return buf;
}

/// Print benchmark results table header
inline void print_table_header() {
  printf("\n%12s %20s %20s %20s %20s\n", "Size", "SingleDev(Gbps)", "MultiDev(Gbps)", "SingleHost(Gbps)", "MultiHost(Gbps)");
  printf("%s\n", std::string(100, '-').c_str());
}
