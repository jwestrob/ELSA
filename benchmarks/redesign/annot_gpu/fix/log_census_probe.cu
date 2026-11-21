// Diagnostic probe (not production): exhaustive census of host libm log versus CUDA device log over every
// positive finite binary32 input, at both the binary32-rounded level used by the bias filter
// (__double2float_rn(log((double)x))) and the binary64 level used by Forward/domain scoring.
// Derived from hmmer_gpu/oracle/bias_log_attestation.cu; host threads via std::thread.
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <gnu/libc-version.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

__global__ void device_logs(uint32_t first, uint32_t count, float *f_out, double *d_out) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const uint32_t raw = first + index;
  float x;
  memcpy(&x, &raw, sizeof(x));
  const double d = log(static_cast<double>(x));
  d_out[index] = d;
  f_out[index] = __double2float_rn(d);
}

int main(int argc, char **argv) {
  const int threads = argc > 1 ? atoi(argv[1]) : 1;
  constexpr uint32_t kLast = UINT32_C(0x7f7fffff);
  constexpr uint32_t kChunk = UINT32_C(1) << 24;
  using log_fn = double (*)(double);
  log_fn host_log = reinterpret_cast<log_fn>(dlsym(RTLD_DEFAULT, "log"));
  if (host_log == nullptr) { fprintf(stderr, "dlsym log failed\n"); return 2; }
  Dl_info info;
  dladdr(reinterpret_cast<void *>(host_log), &info);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("glibc=%s libm=%s gpu=%s cc=%d.%d threads=%d\n", gnu_get_libc_version(),
         info.dli_fname ? info.dli_fname : "?", prop.name, prop.major, prop.minor, threads);
  float *df = nullptr, *hf = nullptr;
  double *dd = nullptr, *hd = nullptr;
  cudaMalloc(&df, kChunk * sizeof(float));
  cudaMalloc(&dd, kChunk * sizeof(double));
  cudaMallocHost(&hf, kChunk * sizeof(float));
  cudaMallocHost(&hd, kChunk * sizeof(double));
  std::atomic<unsigned long long> float_mis{0}, double_mis{0};
  std::atomic<unsigned long long> max_dulp{0};
  std::mutex ex_mutex;
  std::vector<std::pair<uint32_t, uint64_t>> dexamples;
  std::vector<std::pair<uint32_t, uint32_t>> fexamples;
  const auto t0 = std::chrono::steady_clock::now();
  for (uint64_t first64 = 1; first64 <= kLast; first64 += kChunk) {
    const uint32_t first = static_cast<uint32_t>(first64);
    const uint32_t count = static_cast<uint32_t>(std::min<uint64_t>(kChunk, kLast - first64 + 1));
    device_logs<<<(count + 255) / 256, 256>>>(first, count, df, dd);
    if (cudaMemcpy(hf, df, count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(hd, dd, count * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
      fprintf(stderr, "CUDA failure\n");
      return 2;
    }
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
      pool.emplace_back([&, t]() {
        unsigned long long fm = 0, dm = 0, mu = 0;
        const uint32_t lo = static_cast<uint64_t>(count) * t / threads;
        const uint32_t hi = static_cast<uint64_t>(count) * (t + 1) / threads;
        for (uint32_t i = lo; i < hi; ++i) {
          const uint32_t raw = first + i;
          float x;
          memcpy(&x, &raw, sizeof(x));
          const double h = host_log(static_cast<double>(x));
          uint64_t hb, db;
          memcpy(&hb, &h, 8);
          memcpy(&db, &hd[i], 8);
          if (hb != db) {
            ++dm;
            const unsigned long long u = hb > db ? hb - db : db - hb;
            mu = std::max(mu, u);
            if (dm <= 4) {
              std::lock_guard<std::mutex> lock(ex_mutex);
              if (dexamples.size() < 16) dexamples.push_back({raw, u});
            }
          }
          const float hfv = static_cast<float>(h);
          uint32_t a, b;
          memcpy(&a, &hfv, 4);
          memcpy(&b, &hf[i], 4);
          if (a != b) {
            ++fm;
            std::lock_guard<std::mutex> lock(ex_mutex);
            if (fexamples.size() < 16) fexamples.push_back({raw, a ^ b});
          }
        }
        float_mis += fm;
        double_mis += dm;
        unsigned long long prev = max_dulp.load();
        while (mu > prev && !max_dulp.compare_exchange_weak(prev, mu)) {}
      });
    }
    for (auto &th : pool) th.join();
  }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("inputs=%u float_mismatches=%llu double_mismatches=%llu max_double_ulp=%llu seconds=%.3f\n", kLast,
         float_mis.load(), double_mis.load(), max_dulp.load(), secs);
  for (auto &e : fexamples) printf("float_mismatch input=0x%08x xor=0x%08x\n", e.first, e.second);
  for (auto &e : dexamples) printf("double_mismatch input=0x%08x ulp=%llu\n", e.first, (unsigned long long)e.second);
  return float_mis.load() == 0 ? 0 : 1;
}
