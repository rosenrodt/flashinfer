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
  bool depends_on = true;
  Check(flashinfer::da_moe::GraphNodeDependsOn(trigger_only, ancestor, &depends_on));
  Require(!depends_on, "programmatic trigger must not prove workspace serialization");
  Check(flashinfer::da_moe::GraphNodeDependsOn(completed, ancestor, &depends_on));
  Require(depends_on, "full-completion edge must prove workspace serialization");
  flashinfer::da_moe::ActiveCaptureContext context{};
  context.capture_id = 42;
  context.graph = graph;
  context.dependencies = {trigger_only};
  Check(flashinfer::da_moe::PrepareWorkspaceLaneSequence(&context, 42, ancestor, &depends_on));
  Require(depends_on && context.dependencies.size() == 2 && context.dependencies.back() == ancestor,
          "ordered PDL ancestry must acquire an explicit full-completion dependency");
  Check(flashinfer::da_moe::ValidateWorkspaceLaneSequence(context, 42, ancestor, &depends_on));
  Require(depends_on, "strengthened frontier must prove full-completion workspace ordering");
  cudaGraphNode_t unrelated;
  Check(cudaGraphAddKernelNode(&unrelated, graph, nullptr, 0, &params));
  context.dependencies = {unrelated};
  Check(flashinfer::da_moe::PrepareWorkspaceLaneSequence(&context, 42, ancestor, &depends_on));
  Require(!depends_on && context.dependencies.size() == 1,
          "unordered fork must remain rejected without frontier mutation");
  context.dependencies = {trigger_only};
  Check(flashinfer::da_moe::PrepareWorkspaceLaneSequence(&context, 43, ancestor, &depends_on));
  Require(!depends_on && context.dependencies.size() == 1,
          "cross-generation lane must remain rejected");
  Check(cudaGraphDestroy(graph));
  std::puts("PASS: metadata query and conservative PDL workspace ordering");
}
