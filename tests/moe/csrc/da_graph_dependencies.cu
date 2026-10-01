/*
 * Copyright (c) 2026 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0.
 */

#include <cstdio>
#include <cstdlib>

#include "flashinfer/fused_moe/da_moe.cuh"

// Standalone regression: compile with nvcc and the repository/CCCL include paths.
__global__ void DependencyProbe() {}

void Check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(1);
  }
}

void Require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
  }
}

int main() {
  cudaGraph_t graph;
  Check(cudaGraphCreate(&graph, 0));
  cudaKernelNodeParams params{};
  params.func = reinterpret_cast<void*>(DependencyProbe);
  params.gridDim = dim3(1);
  params.blockDim = dim3(1);
  cudaGraphNode_t ancestor, trigger_only, completed;
  Check(cudaGraphAddKernelNode(&ancestor, graph, nullptr, 0, &params));
  Check(cudaGraphAddKernelNode(&trigger_only, graph, nullptr, 0, &params));
  Check(cudaGraphAddKernelNode(&completed, graph, &ancestor, 1, &params));
  cudaGraphEdgeData edge{};
  edge.type = cudaGraphDependencyTypeProgrammatic;
  edge.from_port = cudaGraphKernelNodePortProgrammatic;
#if CUDART_VERSION >= 13000
  Check(cudaGraphAddDependencies(graph, &ancestor, &trigger_only, &edge, 1));
#else
  Check(cudaGraphAddDependencies_v2(graph, &ancestor, &trigger_only, &edge, 1));
#endif
  cudaGraphNode_t predecessor;
  size_t predecessor_count = 1;
#if CUDART_VERSION >= 13000
  cudaError_t lossy =
      cudaGraphNodeGetDependencies(trigger_only, &predecessor, nullptr, &predecessor_count);
#else
  cudaError_t lossy =
      cudaGraphNodeGetDependencies_v2(trigger_only, &predecessor, nullptr, &predecessor_count);
#endif
  Require(lossy == cudaErrorLossyQuery, "metadata-free query must reproduce the failure");
  std::vector<cudaGraphNode_t> dependencies;
  Check(flashinfer::da_moe::GetGraphNodeDependencies(trigger_only, &dependencies));
  Require(dependencies.size() == 1 && dependencies[0] == ancestor,
          "metadata-aware query must preserve a programmatic predecessor");
  Check(cudaGraphDestroy(graph));
  std::puts("PASS: metadata-aware graph dependencies");
}
