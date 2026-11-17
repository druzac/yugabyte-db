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

#include "yb/vector_index/hnsw_options.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include <boost/preprocessor/seq/for_each.hpp>

#include "yb/gutil/port.h"

#include "yb/util/tostring.h"

#include "yb/vector_index/usearch_include_wrapper_internal.h"

namespace yb::vector_index {

namespace {

using unum::usearch::metric_kind_t;
using unum::usearch::scalar_kind_t;

// Single precision sums are accurate when both squared norms are in this range. The partial sums of
// squares never exceed the final sum, and those of the dot product never exceed the mean of the
// squared norms, so nothing overflowed below 2^120. Each product that underflows is off by less
// than 2^-126, so with fewer than 2^16 coordinates the error stays below 2^-110, negligible next to
// squared norms above 2^-80.
constexpr float kMinSinglePrecisionSquaredNorm = 0x1p-80f;
constexpr float kMaxSinglePrecisionSquaredNorm = 0x1p+120f;

bool SinglePrecisionIsAccurate(float squared_norm) {
  return squared_norm >= kMinSinglePrecisionSquaredNorm &&
         squared_norm <= kMaxSinglePrecisionSquaredNorm;
}

float CosineDistanceFromSums(double ab, double a2, double b2) {
  // Same as usearch: zero vectors are at distance 0 from each other and 1 from other vectors.
  if (a2 == 0 || b2 == 0) {
    return a2 == b2 ? 0 : 1;
  }
  // Rounding could take the similarity slightly out of [-1, 1].
  return static_cast<float>(1 - std::clamp(ab / std::sqrt(a2 * b2), -1.0, 1.0));
}

// The SimSIMD kernels usearch picks for cosine sum in single precision, so the squared norm of a
// vector with coordinates of about 1e-23 underflows to 0, and that of a vector longer than about
// 1.8e19 overflows. The infinite or NaN distances such a vector gets then break the HNSW graph for
// every other vector too (#37160). This kernel falls back to double precision for such vectors,
// which holds the product of any two floats exactly.
//
// It is inlined into a copy for each instruction set, see SupportedCosineKernels. Every copy adds
// the same products in the same order, so all of them return the same distances.
ATTRIBUTE_ALWAYS_INLINE inline float CosineDistance(
    const float* lhs, const float* rhs, size_t dimensions) {
#if defined(__clang__)
  // A fused multiply-add rounds once instead of twice, so the AVX-512 copy, the only one that has
  // FMA, would return different distances.
#pragma clang fp contract(off)
#endif
  // A sum per lane lets the compiler vectorize the loop without reordering floating point
  // additions, which it would not do for a single sum.
  constexpr size_t kLanes = 16;
  float lane_ab[kLanes] = {};
  float lane_a2[kLanes] = {};
  float lane_b2[kLanes] = {};
  size_t i = 0;
  for (; i + kLanes <= dimensions; i += kLanes) {
    for (size_t lane = 0; lane != kLanes; ++lane) {
      float l = lhs[i + lane];
      float r = rhs[i + lane];
      lane_ab[lane] += l * r;
      lane_a2[lane] += l * l;
      lane_b2[lane] += r * r;
    }
  }
  // Pairwise, so that a sum takes log2(kLanes) dependent additions instead of kLanes.
  for (size_t width = kLanes / 2; width != 0; width /= 2) {
    for (size_t lane = 0; lane != width; ++lane) {
      lane_ab[lane] += lane_ab[lane + width];
      lane_a2[lane] += lane_a2[lane + width];
      lane_b2[lane] += lane_b2[lane + width];
    }
  }
  float ab = lane_ab[0];
  float a2 = lane_a2[0];
  float b2 = lane_b2[0];
  for (; i != dimensions; ++i) {
    ab += lhs[i] * rhs[i];
    a2 += lhs[i] * lhs[i];
    b2 += rhs[i] * rhs[i];
  }
  if (PREDICT_TRUE(SinglePrecisionIsAccurate(a2) && SinglePrecisionIsAccurate(b2))) {
    return CosineDistanceFromSums(ab, a2, b2);
  }

  double ab_exact = 0;
  double a2_exact = 0;
  double b2_exact = 0;
  for (i = 0; i != dimensions; ++i) {
    double l = lhs[i];
    double r = rhs[i];
    ab_exact += l * r;
    a2_exact += l * l;
    b2_exact += r * r;
  }
  return CosineDistanceFromSums(ab_exact, a2_exact, b2_exact);
}

// CosineDistance with the metric_punned_signature_t::array_array_size_k signature.
using PunnedDistanceFunction = unum::usearch::distance_punned_t (*)(size_t, size_t, size_t);

unum::usearch::distance_punned_t PunnedCosineDistance(
    size_t lhs, size_t rhs, size_t dimensions) {
  return CosineDistance(
      reinterpret_cast<const float*>(lhs), reinterpret_cast<const float*>(rhs), dimensions);
}

// The build targets SSE4.2 on x86, these copies use the wider vectors of newer CPUs. Only clang
// builds have them, since only clang turns off FMA contraction with the pragma in CosineDistance.
#if defined(__clang__) && defined(__x86_64__)
#define YB_HAVE_X86_COSINE_KERNELS 1

__attribute__((target("avx2")))
unum::usearch::distance_punned_t PunnedCosineDistanceAvx2(
    size_t lhs, size_t rhs, size_t dimensions) {
  return CosineDistance(
      reinterpret_cast<const float*>(lhs), reinterpret_cast<const float*>(rhs), dimensions);
}

__attribute__((target("avx512f")))
unum::usearch::distance_punned_t PunnedCosineDistanceAvx512(
    size_t lhs, size_t rhs, size_t dimensions) {
  return CosineDistance(
      reinterpret_cast<const float*>(lhs), reinterpret_cast<const float*>(rhs), dimensions);
}
#endif

struct CosineKernel {
  const char* instruction_set;
  PunnedDistanceFunction function;
};

// The copies of the cosine kernel that this host can run, the best one last.
std::vector<CosineKernel> SupportedCosineKernels() {
  std::vector<CosineKernel> result = {{"baseline", &PunnedCosineDistance}};
#if defined(YB_HAVE_X86_COSINE_KERNELS)
  __builtin_cpu_init();
  if (__builtin_cpu_supports("avx2")) {
    result.push_back({"avx2", &PunnedCosineDistanceAvx2});
  }
  if (__builtin_cpu_supports("avx512f")) {
    result.push_back({"avx512f", &PunnedCosineDistanceAvx512});
  }
#endif
  return result;
}

const CosineKernel& HostCosineKernel() {
  static const auto kernel = SupportedCosineKernels().back();
  return kernel;
}

unum::usearch::metric_punned_t CosineMetric(size_t dimensions, PunnedDistanceFunction function) {
  return unum::usearch::metric_punned_t::stateless(
      dimensions, reinterpret_cast<std::uintptr_t>(function),
      unum::usearch::metric_punned_signature_t::array_array_size_k, metric_kind_t::cos_k,
      scalar_kind_t::f32_k);
}

metric_kind_t MetricKindFromDistanceType(DistanceKind distance_kind) {
  switch (distance_kind) {
    case DistanceKind::kL2Squared:
      return metric_kind_t::l2sq_k;
    case DistanceKind::kInnerProduct:
      return metric_kind_t::ip_k;
    case DistanceKind::kCosine:
      return metric_kind_t::cos_k;
  }
  FATAL_INVALID_ENUM_VALUE(DistanceKind, distance_kind);
}

scalar_kind_t ConvertCoordinateKind(CoordinateKind coordinate_kind) {
  switch (coordinate_kind) {
#define YB_COORDINATE_KIND_TO_USEARCH_CASE(r, data, coordinate_info_tuple) \
    case CoordinateKind::YB_COORDINATE_ENUM_ELEMENT_NAME(coordinate_info_tuple): \
      return scalar_kind_t::BOOST_PP_CAT( \
          YB_EXTRACT_COORDINATE_TYPE_SHORT_NAME(coordinate_info_tuple), _k);
    BOOST_PP_SEQ_FOR_EACH(YB_COORDINATE_KIND_TO_USEARCH_CASE, _, YB_COORDINATE_TYPE_INFO)
#undef YB_COORDINATE_KIND_TO_USEARCH_CASE
  }
  FATAL_INVALID_ENUM_VALUE(CoordinateKind, coordinate_kind);
}

} // namespace

std::string HNSWOptions::ToString() const {
  return YB_STRUCT_TO_STRING(
      extend_candidates,
      keep_pruned_connections,
      num_neighbors_per_vertex,
      max_neighbors_per_vertex,
      num_neighbors_per_vertex_base,
      max_neighbors_per_vertex_base,
      ef_construction,
      robust_prune_alpha);
}

template <class Vector>
unum::usearch::metric_punned_t HNSWOptions::CreateMetric() const {
  if constexpr (std::is_same_v<typename Vector::value_type, float>) {
    if (distance_kind == DistanceKind::kCosine) {
      return CosineMetric(dimensions, HostCosineKernel().function);
    }
  }
  return unum::usearch::metric_punned_t(
      dimensions,
      MetricKindFromDistanceType(distance_kind),
      ConvertCoordinateKind(CoordinateTypeTraits<typename Vector::value_type>::kKind));
}

template
unum::usearch::metric_punned_t HNSWOptions::CreateMetric<FloatVector>() const;

const char* CosineKernelInstructionSet() {
  return HostCosineKernel().instruction_set;
}

std::vector<std::pair<const char*, unum::usearch::metric_punned_t>> TEST_CosineMetrics(
    size_t dimensions) {
  std::vector<std::pair<const char*, unum::usearch::metric_punned_t>> result;
  for (const auto& kernel : SupportedCosineKernels()) {
    result.emplace_back(kernel.instruction_set, CosineMetric(dimensions, kernel.function));
  }
  return result;
}

}  // namespace yb::vector_index
