// CPU thread pool (thread_pool.h).
#include "thread_pool.h"

#include <pthread.h>
#include <sched.h>

#include <algorithm>

namespace aegis::cpu {

namespace {
thread_local const void* tl_pool = nullptr;
thread_local int tl_index = -1;
}  // namespace

std::vector<int> select_cpus(const hw::CpuInfo& cpu, CoreClass cls, bool* fallback) {
  std::vector<int> perf, eff;
  for (const hw::CoreType& t : cpu.core_types) {
    const std::vector<int>& p = t.physical_cpus.empty() ? t.cpus : t.physical_cpus;
    (t.performance ? perf : eff).insert((t.performance ? perf : eff).end(), p.begin(), p.end());
  }
  std::sort(perf.begin(), perf.end());
  std::sort(eff.begin(), eff.end());
  const bool fb = cls == CoreClass::Efficiency && eff.empty();
  if (fallback) *fallback = fb;
  if (cls == CoreClass::Performance || fb) return perf;
  return eff;
}

ThreadPool::ThreadPool(const PoolOptions& opt) {
  hw::DiscoverOptions d;
  d.cuda = false;
  d.nvml = false;
  d.video = false;
  cpus_ = select_cpus(hw::discover(d).cpu, opt.cores, &fallback_);
  start(opt);
}

ThreadPool::ThreadPool(const hw::CpuInfo& cpu, const PoolOptions& opt) {
  cpus_ = select_cpus(cpu, opt.cores, &fallback_);
  start(opt);
}

void ThreadPool::start(const PoolOptions& opt) {
  if (cpus_.empty()) {  // discovery found nothing: one unpinned worker per hardware thread
    for (unsigned i = 0, n = std::max(1u, std::thread::hardware_concurrency()); i < n; ++i) cpus_.push_back(-1);
  }
  if (opt.max_threads > 0 && size_t(opt.max_threads) < cpus_.size()) cpus_.resize(size_t(opt.max_threads));
  ring_.resize(std::max<size_t>(1, opt.queue_capacity));
  workers_.reserve(cpus_.size());
  for (size_t i = 0; i < cpus_.size(); ++i) {
    workers_.emplace_back([this, i, pin = opt.pin] { worker(int(i), pin); });
  }
  // Every worker has pinned itself (or failed to) before the constructor returns.
  std::unique_lock<std::mutex> l(mu_);
  done_.wait(l, [&] { return started_ == workers_.size(); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> l(mu_);
    stop_ = true;
  }
  wake_.notify_all();
  for (std::thread& t : workers_) t.join();
}

int ThreadPool::worker_index() const { return tl_pool == this ? tl_index : -1; }

bool ThreadPool::submit(TaskFn fn, void* ctx) {
  {
    std::lock_guard<std::mutex> l(mu_);
    if (count_ == ring_.size() || stop_) return false;
    ring_[(head_ + count_) % ring_.size()] = Task{fn, ctx};
    ++count_;
  }
  wake_.notify_one();
  return true;
}

void ThreadPool::wait_idle() {
  std::unique_lock<std::mutex> l(mu_);
  done_.wait(l, [&] { return count_ == 0 && running_ == 0; });
}

void ThreadPool::drain(Job& j) {
  for (int i; (i = j.next.fetch_add(1, std::memory_order_relaxed)) < j.n;) j.fn(j.obj, i);
}

void ThreadPool::run_job(int n, void (*fn)(void*, int), void* obj) {
  if (n <= 0) return;
  std::lock_guard<std::mutex> one(job_mu_);
  Job j;
  j.fn = fn;
  j.obj = obj;
  j.n = n;
  {
    std::lock_guard<std::mutex> l(mu_);
    j.left.store(int(workers_.size()), std::memory_order_relaxed);
    job_ = &j;
    ++job_gen_;
  }
  wake_.notify_all();
  drain(j);  // the caller works too
  std::unique_lock<std::mutex> l(mu_);
  done_.wait(l, [&] { return j.left.load(std::memory_order_acquire) == 0; });
  job_ = nullptr;
}

void ThreadPool::worker(int index, bool pin) {
  tl_pool = this;
  tl_index = index;
  // Pinned before it can run anything: every task and job index runs on its core.
  bool ok = false;
  if (pin && cpus_[size_t(index)] >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpus_[size_t(index)], &set);
    ok = pthread_setaffinity_np(pthread_self(), sizeof set, &set) == 0;
  }
  unsigned long long seen = 0;
  std::unique_lock<std::mutex> l(mu_);
  pinned_ += ok;
  ++started_;
  done_.notify_all();
  for (;;) {
    wake_.wait(l, [&] { return stop_ || count_ > 0 || (job_ && job_gen_ != seen); });
    if (job_ && job_gen_ != seen) {  // every worker enters every job once
      seen = job_gen_;
      Job* j = job_;
      l.unlock();
      drain(*j);
      l.lock();
      if (j->left.fetch_sub(1, std::memory_order_acq_rel) == 1) done_.notify_all();
      continue;
    }
    if (count_ > 0) {
      const Task t = ring_[head_];
      head_ = (head_ + 1) % ring_.size();
      --count_;
      ++running_;
      l.unlock();
      t.fn(t.ctx);
      l.lock();
      --running_;
      if (count_ == 0 && running_ == 0) done_.notify_all();
      continue;
    }
    if (stop_) return;
  }
}

}  // namespace aegis::cpu
