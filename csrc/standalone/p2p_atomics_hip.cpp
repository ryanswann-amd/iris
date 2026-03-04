// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// p2p_atomics_hip.cpp — Standalone HIP VMem P2P atomic test (PATH 2, BUGGY)
//
// Demonstrates the BROKEN approach for multi-GPU P2P atomic operations:
//   hipMemCreate -> CLR -> hsa_amd_vmem_handle_create (coarse-grained pool hardcoded)
//
// Stack:
//   hipMemCreate(&handle, size, &prop, 0)
//     -> CLR: SvmBuffer::malloc(ROCCLR_MEM_PHYMEM)
//     -> hsa_amd_vmem_handle_create(COARSE_GRAINED_POOL)   ← CLR hardcodes this
//     -> KFD: hsaKmtAllocMemory(CoarseGrain=1, NoAddress=1)
//   hipMemExportToShareableHandle -> DMA-BUF fd -> SCM_RIGHTS
//   hipMemImportFromShareableHandle -> hipMemMap + hipMemSetAccess
//   P2P atomic_add -> GPU PAGE FAULT (coarse-grained P2P atomics not supported)
//
// ┌-------------------------------------------------------------------------┐
// │ WARNING: P2P atomic operations (any scope) on coarse-grained memory     │
// │ trigger GPU page faults that send SIGSEGV to the process.  By default  │
// │ this program only runs a P2P non-atomic read to verify the mapping.    │
// │ Pass --atomics to also run P2P atomics — this WILL CRASH.              │
// └-------------------------------------------------------------------------┘
//
// Options:
//   --pinned     hipMemAllocationTypePinned (0x1) — default
//   --uncached   hipMemAllocationTypeUncached (0x40000000) — AMD extension;
//                accepted by hipMemCreate but CLR still uses coarse-grained pool
//   --atomics    Run P2P atomic kernels (WARNING: causes GPU page fault!)
//   --agent      Use agent-scope atomics (default when --atomics given)
//   --sys        Use system-scope atomics (slower, still faults on coarse-grained)
//   N            Number of iterations (default 200; auto-reduced to 20 for atomics)
//
// Build:
//   hipcc -o p2p_atomics_hip p2p_atomics_hip.cpp
//
// Run examples:
//   ./p2p_atomics_hip                          # safe: P2P non-atomic read only
//   ./p2p_atomics_hip --uncached               # probe uncached alloc type (safe)
//   ./p2p_atomics_hip --atomics                # agent-scope P2P atomics (WILL CRASH)
//   ./p2p_atomics_hip --atomics --sys          # sys-scope P2P atomics (WILL CRASH)
//   ./p2p_atomics_hip --uncached --atomics     # uncached + atomics (same crash)
//
// Compare with p2p_atomics_hsa.cpp (HSA fine-grained path) which passes all tests.

#include <hip/hip_runtime.h>

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

// ============================================================================
// Error-checking macro
// ============================================================================

#define HIP_CHECK(expr)                                                            \
  do {                                                                             \
    hipError_t _e = (expr);                                                        \
    if (_e != hipSuccess) {                                                        \
      fprintf(stderr, "[rank %d] HIP error at %s:%d — %s\n", g_rank, __FILE__,   \
              __LINE__, hipGetErrorString(_e));                                    \
      abort();                                                                     \
    }                                                                              \
  } while (0)

static int g_rank = -1;

// ============================================================================
// SCM_RIGHTS helpers
// ============================================================================

static void send_fd(int sock, int fd) {
  char            buf[1]                  = {0};
  struct iovec    iov                     = {buf, 1};
  char            cmsg_buf[CMSG_SPACE(sizeof(int))];
  memset(cmsg_buf, 0, sizeof(cmsg_buf));
  struct msghdr   msg                     = {};
  msg.msg_iov                             = &iov;
  msg.msg_iovlen                          = 1;
  msg.msg_control                         = cmsg_buf;
  msg.msg_controllen                      = sizeof(cmsg_buf);
  struct cmsghdr* cmsg                    = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level                        = SOL_SOCKET;
  cmsg->cmsg_type                         = SCM_RIGHTS;
  cmsg->cmsg_len                          = CMSG_LEN(sizeof(int));
  memcpy(CMSG_DATA(cmsg), &fd, sizeof(int));
  ssize_t n = sendmsg(sock, &msg, 0);
  assert(n == 1);
}

static int recv_fd(int sock) {
  char            buf[1];
  struct iovec    iov                     = {buf, 1};
  char            cmsg_buf[CMSG_SPACE(sizeof(int))];
  memset(cmsg_buf, 0, sizeof(cmsg_buf));
  struct msghdr   msg                     = {};
  msg.msg_iov                             = &iov;
  msg.msg_iovlen                          = 1;
  msg.msg_control                         = cmsg_buf;
  msg.msg_controllen                      = sizeof(cmsg_buf);
  ssize_t         n                       = recvmsg(sock, &msg, 0);
  assert(n == 1);
  struct cmsghdr* cmsg                    = CMSG_FIRSTHDR(&msg);
  assert(cmsg && cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS);
  int fd;
  memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));
  return fd;
}

static void barrier(int sock) {
  char c = 'b';
  write(sock, &c, 1);
  read(sock, &c, 1);
}

// ============================================================================
// Device kernels
// ============================================================================

// Agent-scope atomic add — fails on coarse-grained P2P (GPU page fault)
__global__ void k_atomic_add_agent(float* ptr) {
  __hip_atomic_fetch_add(ptr, 1.0f, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
}

// System-scope atomic add — also fails on coarse-grained P2P (GPU page fault)
__global__ void k_atomic_add_sys(float* ptr) {
  __hip_atomic_fetch_add(ptr, 1.0f, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Non-atomic write / read — safe on coarse-grained P2P
__global__ void k_write_value(float* ptr, float val) { *ptr = val; }

// Read with system-scope fence to see writes from remote GPUs
__global__ void k_read_to(float* dst, const float* src) {
  __threadfence_system();
  *dst = *src;
}
__global__ void k_zero(float* ptr) { *ptr = 0.0f; }

// ============================================================================
// Config passed through env vars for the self-exec'd rank 1
// ============================================================================

struct Config {
  int  alloc_type;   // 0x1 = pinned, 0x40000000 = uncached
  bool run_atomics;
  bool agent_scope;  // true = agent, false = sys
  int  n_iters;
};

static Config read_config_from_env() {
  Config cfg   = {};
  cfg.alloc_type  = 0x1;  // pinned default
  cfg.run_atomics = false;
  cfg.agent_scope = true;
  cfg.n_iters     = 200;

  const char* at = getenv("P2P_ALLOC_TYPE");
  if (at) cfg.alloc_type = (int)strtol(at, nullptr, 0);
  const char* ra = getenv("P2P_ATOMICS");
  if (ra) cfg.run_atomics = atoi(ra) != 0;
  const char* ag = getenv("P2P_AGENT_SCOPE");
  if (ag) cfg.agent_scope = atoi(ag) != 0;
  const char* ni = getenv("P2P_NITERS");
  if (ni) cfg.n_iters = atoi(ni);
  return cfg;
}

// ============================================================================
// Per-rank logic
// ============================================================================

static int run_rank(int rank, int sock, const Config& cfg) {
  g_rank     = rank;
  int gpu_id = rank;

  const char* alloc_name =
      (cfg.alloc_type == 0x40000000) ? "UNCACHED(0x40000000)" : "PINNED(0x1)";
  printf("[rank %d] GPU=%d  alloc_type=%s\n", rank, gpu_id, alloc_name);
  fflush(stdout);

  HIP_CHECK(hipSetDevice(gpu_id));

  // -- Query allocation granularity ----------------------------------------
  hipMemAllocationProp prop = {};
  prop.type                   = (hipMemAllocationType)cfg.alloc_type;
  prop.requestedHandleType    = hipMemHandleTypePosixFileDescriptor;
  prop.location.type          = hipMemLocationTypeDevice;
  prop.location.id            = gpu_id;

  size_t gran = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityRecommended));
  if (gran == 0) gran = 2u << 20;
  size_t alloc_size = gran;

  printf("[rank %d] granularity = %zu bytes\n", rank, gran);
  fflush(stdout);

  // -- Reserve virtual address space -------------------------------------------
  void* my_va = nullptr;
  HIP_CHECK(hipMemAddressReserve(&my_va, alloc_size, gran, /*addr=*/0, /*flags=*/0));

  // -- Create physical allocation --------------------------------------------
  // hipMemCreate with any hipMemAllocationType always allocates from the
  // COARSE-GRAINED GPU pool in HIP/CLR (SvmBuffer::malloc ROCCLR_MEM_PHYMEM).
  // Even hipMemAllocationTypeUncached (0x40000000) is silently ignored by CLR.
  hipMemGenericAllocationHandle_t my_handle = 0;
  hipError_t create_err = hipMemCreate(&my_handle, alloc_size, &prop, /*flags=*/0);
  if (create_err != hipSuccess) {
    fprintf(stderr, "[rank %d] hipMemCreate(%s) failed: %s\n", rank, alloc_name,
            hipGetErrorString(create_err));
    // Clean up and synchronize with peer so it doesn't hang
    (void)hipMemAddressFree(my_va, alloc_size);
    barrier(sock);
    return 0;  // Driver rejected this alloc type — not a test failure
  }
  printf("[rank %d] hipMemCreate(%s) succeeded\n", rank, alloc_name);
  fflush(stdout);

  // -- Map + set access ------------------------------------------------------
  HIP_CHECK(hipMemMap(my_va, alloc_size, /*offset=*/0, my_handle, /*flags=*/0));

  hipMemAccessDesc desc       = {};
  desc.location.type          = hipMemLocationTypeDevice;
  desc.location.id            = gpu_id;
  desc.flags                  = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(my_va, alloc_size, &desc, 1));

  // -- DMA-BUF Export ------------------------------------
  int my_fd = -1;
  HIP_CHECK(hipMemExportToShareableHandle(&my_fd, my_handle,
                                          hipMemHandleTypePosixFileDescriptor, /*flags=*/0));

  // -- Exchange fds with peer (rank 0 sends first) ----------
  int peer_fd = -1;
  if (rank == 0) {
    send_fd(sock, my_fd);
    peer_fd = recv_fd(sock);
  } else {
    peer_fd = recv_fd(sock);
    send_fd(sock, my_fd);
  }
  close(my_fd);

  // -- Import peer handle + map -----------------------------------------------
  hipMemGenericAllocationHandle_t peer_handle = 0;
  HIP_CHECK(hipMemImportFromShareableHandle(
      &peer_handle,
      (void*)(intptr_t)peer_fd,  // POSIX fd passed as void*
      hipMemHandleTypePosixFileDescriptor));
  close(peer_fd);

  void* peer_va = nullptr;
  HIP_CHECK(hipMemAddressReserve(&peer_va, alloc_size, gran, 0, 0));
  HIP_CHECK(hipMemMap(peer_va, alloc_size, 0, peer_handle, 0));

  hipMemAccessDesc peer_desc  = {};
  peer_desc.location.type     = hipMemLocationTypeDevice;
  peer_desc.location.id       = gpu_id;
  peer_desc.flags             = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(peer_va, alloc_size, &peer_desc, 1));

  printf("[rank %d] my_va=%p  peer_va=%p\n", rank, my_va, peer_va);
  fflush(stdout);

  float* my_ptr   = (float*)my_va;
  float* peer_ptr = (float*)peer_va;

  // Bounce buffer: hipMalloc VA guaranteed safe for hipMemcpy(D2H)
  float* bounce;
  HIP_CHECK(hipMalloc(&bounce, sizeof(float)));

  // Barrier: ensure both ranks finished VA setup before starting the test
  barrier(sock);

  // -- P2P non-atomic read/write (safe, runs regardless of --atomics flag) --
  // Kernel-based init avoids hipMemset on HIP VMem VA (may not be registered)
  k_zero<<<1, 1>>>(my_ptr);
  HIP_CHECK(hipDeviceSynchronize());
  barrier(sock);

  k_write_value<<<1, 1>>>(my_ptr, (float)(rank + 1));
  HIP_CHECK(hipDeviceSynchronize());
  barrier(sock);

  k_read_to<<<1, 1>>>(bounce, peer_ptr);
  HIP_CHECK(hipDeviceSynchronize());

  float peer_val = -1;
  HIP_CHECK(hipMemcpy(&peer_val, bounce, sizeof(float), hipMemcpyDeviceToHost));

  float expected = (float)(2 - rank);  // peer rank is (1-rank), wrote (peer_rank+1)
  int   read_ok  = (fabsf(peer_val - expected) < 0.1f);
  printf("[rank %d] P2P non-atomic read: got %g (expected %g) -> %s\n", rank, peer_val,
         expected, read_ok ? "PASS" : "FAIL");
  fflush(stdout);

  int fails = 0;

  // -- P2P atomic test (opt-in, WARNING: WILL CRASH on coarse-grained!) ----
  if (cfg.run_atomics) {
    const char* scope_name = cfg.agent_scope ? "agent" : "sys";
    printf("[rank %d] === P2P ATOMIC (%s-scope, alloc=%s) — EXPECT GPU PAGE FAULT! ===\n",
           rank, scope_name, alloc_name);
    fflush(stdout);

    for (int iter = 0; iter < cfg.n_iters; iter++) {
      k_zero<<<1, 1>>>(my_ptr);
      HIP_CHECK(hipDeviceSynchronize());
      barrier(sock);

      if (cfg.agent_scope) {
        k_atomic_add_agent<<<1, 1>>>(my_ptr);
        k_atomic_add_agent<<<1, 1>>>(peer_ptr);  // P2P: expect GPU page fault
      } else {
        k_atomic_add_sys<<<1, 1>>>(my_ptr);
        k_atomic_add_sys<<<1, 1>>>(peer_ptr);  // P2P: also faults on coarse-grained
      }
      // hipDeviceSynchronize surfaces the GPU fault (SIGSEGV to this process)
      HIP_CHECK(hipDeviceSynchronize());
      barrier(sock);

      k_read_to<<<1, 1>>>(bounce, my_ptr);
      HIP_CHECK(hipDeviceSynchronize());
      float val = 0;
      HIP_CHECK(hipMemcpy(&val, bounce, sizeof(float), hipMemcpyDeviceToHost));
      if (fabsf(val - 2.0f) > 0.1f) {
        if (++fails <= 5)
          printf("[rank %d] iter %d %s-scope: expected 2.0, got %g\n", rank, iter,
                 scope_name, val);
      }
    }
    printf("[rank %d] %s-scope P2P atomics: %d/%d failures\n", rank, scope_name, fails,
           cfg.n_iters);
    fflush(stdout);
  } else {
    printf("[rank %d] P2P atomics skipped (pass --atomics to enable, expect crash)\n",
           rank);
    fflush(stdout);
  }

  HIP_CHECK(hipFree(bounce));

  printf("[rank %d] %s\n", rank, (read_ok && fails == 0) ? "PASS" : "FAIL");
  fflush(stdout);

  // -- Cleanup ---------------------------------------------------------------
  barrier(sock);  // sync before cleanup to avoid one rank freeing while other maps

  (void)hipMemUnmap(peer_va, alloc_size);
  (void)hipMemAddressFree(peer_va, alloc_size);
  (void)hipMemRelease(peer_handle);

  (void)hipMemUnmap(my_va, alloc_size);
  (void)hipMemAddressFree(my_va, alloc_size);
  (void)hipMemRelease(my_handle);

  return (read_ok && fails == 0) ? 0 : 1;
}

// ============================================================================
// main -- self-exec trick (same rationale as p2p_atomics_hsa.cpp)
// ============================================================================

int main(int argc, char** argv) {
  // -- Re-exec path: invoked as rank 1 --
  if (getenv("P2P_RANK")) {
    int rank    = atoi(getenv("P2P_RANK"));
    int sock_fd = atoi(getenv("P2P_SOCK_FD"));
    Config cfg  = read_config_from_env();
    int rc      = run_rank(rank, sock_fd, cfg);
    close(sock_fd);
    return rc;
  }

  // -- Primary path: parse args, spawn rank 1, run rank 0 -------------------
  Config cfg      = {};
  cfg.alloc_type  = 0x1;  // pinned
  cfg.run_atomics = false;
  cfg.agent_scope = true;  // default scope when --atomics given
  cfg.n_iters     = 200;
  bool n_iters_set = false;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--uncached") == 0)     cfg.alloc_type  = 0x40000000;
    else if (strcmp(argv[i], "--pinned") == 0)  cfg.alloc_type  = 0x1;
    else if (strcmp(argv[i], "--atomics") == 0) cfg.run_atomics = true;
    else if (strcmp(argv[i], "--agent") == 0)   cfg.agent_scope = true;
    else if (strcmp(argv[i], "--sys") == 0)     cfg.agent_scope = false;
    else { int n = atoi(argv[i]); if (n > 0) { cfg.n_iters = n; n_iters_set = true; } }
  }
  if (cfg.run_atomics && !n_iters_set) cfg.n_iters = 20;

  int ndev = 0;
  (void)hipGetDeviceCount(&ndev);
  if (ndev < 2) {
    fprintf(stderr, "This example requires at least 2 GPU devices (found %d).\n", ndev);
    return 1;
  }

  const char* alloc_name = (cfg.alloc_type == 0x40000000) ? "UNCACHED" : "PINNED";
  printf("p2p_atomics_hip: PATH 2 — HIP VMem (coarse-grained, alloc=%s)\n", alloc_name);
  printf("  N_ITERS=%d  GPUs=%d  atomics=%s  scope=%s\n\n", cfg.n_iters, ndev,
         cfg.run_atomics ? "yes" : "no", cfg.agent_scope ? "agent" : "sys");

  if (cfg.run_atomics) {
    printf("WARNING: P2P atomics on coarse-grained memory cause GPU page faults (SIGSEGV).\n");
    printf("hipMemCreate ALWAYS uses the coarse-grained pool regardless of "
           "hipMemAllocationType\n(incl. UNCACHED). "
           "See p2p_atomics_hsa.cpp for the correct HSA fix.\n\n");
    fflush(stdout);
  }

  int sv[2];
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) { perror("socketpair"); return 1; }

  // Pass config to rank 1 via env vars
  char at_str[32], ra_str[4], ag_str[4], ni_str[32], sk_str[32];
  snprintf(at_str, sizeof(at_str), "0x%x", cfg.alloc_type);
  snprintf(ra_str, sizeof(ra_str), "%d", (int)cfg.run_atomics);
  snprintf(ag_str, sizeof(ag_str), "%d", (int)cfg.agent_scope);
  snprintf(ni_str, sizeof(ni_str), "%d", cfg.n_iters);
  snprintf(sk_str, sizeof(sk_str), "%d", sv[1]);
  setenv("P2P_RANK", "1", 1);
  setenv("P2P_SOCK_FD", sk_str, 1);
  setenv("P2P_ALLOC_TYPE", at_str, 1);
  setenv("P2P_ATOMICS", ra_str, 1);
  setenv("P2P_AGENT_SCOPE", ag_str, 1);
  setenv("P2P_NITERS", ni_str, 1);

  pid_t pid = fork();
  if (pid < 0) { perror("fork"); return 1; }
  if (pid == 0) {
    close(sv[0]);
    execl("/proc/self/exe", argv[0], NULL);
    perror("execl");
    _exit(127);
  }

  unsetenv("P2P_RANK");
  unsetenv("P2P_SOCK_FD");
  close(sv[1]);

  int rc0 = run_rank(0, sv[0], cfg);
  close(sv[0]);

  int status = 0;
  waitpid(pid, &status, 0);
  int rc1 = WIFEXITED(status) ? WEXITSTATUS(status) : 1;

  if (rc0 != 0 || rc1 != 0)
    printf("\nOverall: FAIL (rank0=%d rank1=%d)\n", rc0, rc1);
  else
    printf("\nOverall: PASS\n");

  return (rc0 != 0 || rc1 != 0) ? 1 : 0;
}
