#pragma once
// CPU thread pool for the AArch64 kernels (stage 2: the CPU GEMV).
//
// Sized from hardware discovery (aegis/hw.h): one worker per physical core of
// the chosen class, each pinned to its core with sched affinity.
//   CoreClass::Performance  the performance class (the 10 Cortex-X925 on the Spark)
//   CoreClass::Efficiency   every other class, for background work (the 10
//                           Cortex-A725). A CPU with one class has no efficiency
//                           cores: the pool then uses that class too and says so
//                           (efficiency_fallback()).
// The submit path allocates nothing:
//   submit(fn, ctx)    a function pointer and a context pointer into a ring of
//                      fixed capacity (PoolOptions::queue_capacity, allocated
//                      once at construction); false when the ring is full
//   parallel_for(n, f) f(i) for every i in [0, n), split dynamically over the
//                      workers and the calling thread; blocks until all are done.
//                      The job lives on the caller's stack; one at a time.
// Never throws after construction; tasks must not throw.
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "aegis/hw.h"

namespace aegis::cpu {

enum class CoreClass { Performance, Efficiency };

struct PoolOptions {
  CoreClass cores = CoreClass::Performance;
  int max_threads = 0;          // 0: one per physical core of the class
  bool pin = true;              // sched_setaffinity each worker to its core
  size_t queue_capacity = 256;  // submit() ring
};

// The physical CPUs (one per core, CoreType::physical_cpus) of the class, sorted.
// Efficiency on a CPU with one class: that class, with *fallback = true.
std::vector<int> select_cpus(const hw::CpuInfo& cpu, CoreClass cls, bool* fallback = nullptr);

class ThreadPool {
 public:
  using TaskFn = void (*)(void* ctx);

  explicit ThreadPool(const PoolOptions& opt = {});  // discovers this machine's CPUs
  ThreadPool(const hw::CpuInfo& cpu, const PoolOptions& opt);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return int(workers_.size()); }
  std::span<const int> cpus() const { return cpus_; }  // worker i runs on cpus()[i]
  int pinned() const { return pinned_; }               // workers whose affinity was set
  bool efficiency_fallback() const { return fallback_; }
  // The worker index of the calling thread, or -1 if it is not one of this pool's.
  int worker_index() const;

  bool submit(TaskFn fn, void* ctx);
  void wait_idle();  // every submitted task has finished

  template <class F>
  void parallel_for(int n, F&& f) {
    auto thunk = [](void* obj, int i) { (*static_cast<std::remove_reference_t<F>*>(obj))(i); };
    run_job(n, thunk, const_cast<void*>(static_cast<const void*>(&f)));
  }

 private:
  struct Job {
    void (*fn)(void*, int) = nullptr;
    void* obj = nullptr;
    int n = 0;
    std::atomic<int> next{0};
    std::atomic<int> left{0};  // workers still inside the job
  };
  struct Task {
    TaskFn fn;
    void* ctx;
  };
  void start(const PoolOptions& opt);
  void worker(int index, bool pin);
  void run_job(int n, void (*fn)(void*, int), void* obj);
  static void drain(Job& j);

  std::vector<int> cpus_;
  std::vector<std::thread> workers_;
  int pinned_ = 0;
  bool fallback_ = false;

  std::mutex mu_;
  std::condition_variable wake_;  // workers: a task, a job or stop
  std::condition_variable done_;  // wait_idle / run_job: progress
  std::vector<Task> ring_;        // fixed capacity
  size_t head_ = 0, count_ = 0;
  size_t running_ = 0;            // tasks being executed
  Job* job_ = nullptr;
  unsigned long long job_gen_ = 0;
  bool stop_ = false;
  size_t started_ = 0;  // workers that have pinned themselves
  std::mutex job_mu_;  // one parallel_for at a time
};

}  // namespace aegis::cpu
