// Copyright (c) YugabyteDB, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.  You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied.  See the License for the specific language governing permissions and limitations
// under the License.
//

// Compares the cosine metric of HNSWOptions::CreateMetric with the one usearch builds by itself,
// which dispatches to SimSIMD and which vector indexes used before #37160.

#include <benchmark/benchmark.h>

#include <map>
#include <random>
#include <vector>

#include "yb/gutil/casts.h"

#include "yb/vector_index/hnsw_options.h"
#include "yb/vector_index/usearch_include_wrapper_internal.h"

namespace yb::vector_index {
namespace {

// Few enough vectors to stay in cache, so that the benchmark measures the kernels.
constexpr size_t kNumVectors = 256;

const std::vector<float>& Vectors(size_t dimensions) {
  static std::map<size_t, std::vector<float>> vectors;
  auto& result = vectors[dimensions];
  if (result.empty()) {
    std::mt19937 rng(42);
    std::normal_distribution<float> distribution;
    result.resize(kNumVectors * dimensions);
    for (auto& coordinate : result) {
      coordinate = distribution(rng);
    }
  }
  return result;
}

void Dimensions(benchmark::Benchmark* benchmark) {
  for (int64_t dimensions : {32, 128, 384, 768, 1536}) {
    benchmark->Arg(dimensions);
  }
}

void Run(benchmark::State& state, const unum::usearch::metric_punned_t& metric) {
  const auto dimensions = static_cast<size_t>(state.range(0));
  const auto& vectors = Vectors(dimensions);
  const auto* query = pointer_cast<const unum::usearch::byte_t*>(vectors.data());
  size_t i = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(
        metric(query, pointer_cast<const unum::usearch::byte_t*>(&vectors[i * dimensions])));
    if (++i == kNumVectors) {
      i = 0;
    }
  }
  state.SetItemsProcessed(state.iterations());
}

void BM_CosineUsearchBuiltin(benchmark::State& state) {
  Run(state, unum::usearch::metric_punned_t(
      static_cast<size_t>(state.range(0)), unum::usearch::metric_kind_t::cos_k,
      unum::usearch::scalar_kind_t::f32_k));
}
BENCHMARK(BM_CosineUsearchBuiltin)->Apply(Dimensions);

void BM_CosineCreateMetric(benchmark::State& state) {
  HNSWOptions options {
    .dimensions = static_cast<size_t>(state.range(0)),
    .distance_kind = DistanceKind::kCosine,
  };
  Run(state, options.CreateMetric<FloatVector>());
}
BENCHMARK(BM_CosineCreateMetric)->Apply(Dimensions);

} // namespace
} // namespace yb::vector_index
