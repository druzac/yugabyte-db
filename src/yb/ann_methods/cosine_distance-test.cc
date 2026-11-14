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

#include <algorithm>
#include <cmath>
#include <iterator>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

#include "yb/ann_methods/hnswlib_wrapper.h"
#include "yb/ann_methods/usearch_wrapper.h"

#include "yb/gutil/casts.h"

#include "yb/hnsw/hnsw_block_cache.h"
#include "yb/hnsw/vector_index_test_base.h"

#include "yb/util/test_util.h"
#include "yb/util/tostring.h"

#include "yb/vector_index/distance.h"
#include "yb/vector_index/hnsw_options.h"
#include "yb/vector_index/usearch_include_wrapper_internal.h"
#include "yb/vector_index/vector_index_if.h"

namespace yb::ann_methods {

using vector_index::DistanceKind;
using vector_index::FactoryMode;
using vector_index::HNSWOptions;
using vector_index::SearchOptions;
using vector_index::StoreVectorPayload;
using vector_index::VectorId;
using vector_index::VectorIndexIf;
using vector_index::VectorIndexTraitsPtr;

// The squared norm of a vector with coordinates of these magnitudes underflows or overflows float.
constexpr float kExtremeScales[] = {1e-40f, 1e-23f, 1e20f, 1e30f};

// Depending on the number of dimensions, the squared norm of a vector with coordinates of these
// magnitudes is on either side of the bounds of the range where the metric sums in single
// precision.
constexpr float kBoundaryScales[] = {1e-13f, 1e-12f, 1e17f, 1e18f};

class CosineDistanceTest : public hnsw::VectorIndexTestBase {
 protected:
  FloatVector GaussianVector(size_t dimensions, float scale = 1) {
    std::normal_distribution<float> distribution;
    FloatVector result(dimensions);
    for (auto& coordinate : result) {
      coordinate = scale * distribution(rng_);
    }
    return result;
  }

  static float Distance(
      const unum::usearch::metric_punned_t& metric, const FloatVector& lhs,
      const FloatVector& rhs) {
    return metric(
        pointer_cast<const unum::usearch::byte_t*>(lhs.data()),
        pointer_cast<const unum::usearch::byte_t*>(rhs.data()));
  }
};

// The metric of the indexes matches the double precision reference for vectors of any magnitude.
TEST_F(CosineDistanceTest, Magnitudes) {
  std::vector<float> scales = {1};
  scales.insert(scales.end(), std::begin(kExtremeScales), std::end(kExtremeScales));
  scales.insert(scales.end(), std::begin(kBoundaryScales), std::end(kBoundaryScales));
  // With and without coordinates left after the lanes of the single precision sums.
  constexpr size_t kDimensions[] = {1, 3, 16, 37, 100};
  for (auto dimensions : kDimensions) {
    HNSWOptions options {
      .dimensions = dimensions,
      .distance_kind = DistanceKind::kCosine,
    };
    auto metric = options.CreateMetric<FloatVector>();
    for (auto lhs_scale : scales) {
      for (auto rhs_scale : scales) {
        for (int i = 0; i != 10; ++i) {
          auto lhs = GaussianVector(dimensions, lhs_scale);
          auto rhs = GaussianVector(dimensions, rhs_scale);
          auto distance = Distance(metric, lhs, rhs);
          ASSERT_NEAR(distance, vector_index::distance::DistanceCosine(lhs, rhs), 1e-5)
              << "lhs: " << AsString(lhs) << ", rhs: " << AsString(rhs);
          ASSERT_GE(distance, 0);
          ASSERT_LE(distance, 2);
        }
      }
    }
  }
}

// Products of some coordinates underflow, while the squared norms stay in the single precision
// range.
TEST_F(CosineDistanceTest, UnderflowingCoordinates) {
  constexpr size_t kDimensions = 37;
  HNSWOptions options {
    .dimensions = kDimensions,
    .distance_kind = DistanceKind::kCosine,
  };
  auto metric = options.CreateMetric<FloatVector>();
  for (int i = 0; i != 100; ++i) {
    auto lhs = GaussianVector(kDimensions);
    auto rhs = GaussianVector(kDimensions);
    for (size_t j = 0; j < kDimensions; j += 2) {
      lhs[j] *= 1e-30f;
      rhs[j] *= 1e-40f;
    }
    ASSERT_NEAR(Distance(metric, lhs, rhs), vector_index::distance::DistanceCosine(lhs, rhs), 1e-5)
        << "lhs: " << AsString(lhs) << ", rhs: " << AsString(rhs);
  }
  // One vector in the single precision range and one outside of it.
  FloatVector lhs = {1, 1e-30f};
  FloatVector rhs = {1e-20f, 1e-30f};
  ASSERT_NEAR(Distance(metric, lhs, rhs), 0, 1e-6);
}

TEST_F(CosineDistanceTest, ExtremeMagnitudeDirections) {
  HNSWOptions options {
    .dimensions = 2,
    .distance_kind = DistanceKind::kCosine,
  };
  auto metric = options.CreateMetric<FloatVector>();
  auto distance = [&metric](const FloatVector& lhs, const FloatVector& rhs) {
    return Distance(metric, lhs, rhs);
  };
  for (auto scale : kExtremeScales) {
    FloatVector x = {scale, 0};
    FloatVector minus_x = {-scale, 0};
    FloatVector y = {0, scale};
    FloatVector diagonal = {scale, scale};
    ASSERT_NEAR(distance(x, x), 0, 1e-6) << scale;
    ASSERT_NEAR(distance(x, minus_x), 2, 1e-6) << scale;
    ASSERT_NEAR(distance(x, y), 1, 1e-6) << scale;
    ASSERT_NEAR(distance(x, diagonal), 1 - M_SQRT1_2, 1e-6) << scale;
    ASSERT_NEAR(distance(x, {1, 0}), 0, 1e-6) << scale;
    ASSERT_NEAR(distance({0, 1}, x), 1, 1e-6) << scale;
  }
  // Zero vectors are at distance 0 from each other and 1 from other vectors, as in usearch.
  ASSERT_EQ(distance({0, 0}, {0, 0}), 0.0f);
  ASSERT_EQ(distance({0, 0}, {1, 2}), 1.0f);
  ASSERT_EQ(distance({1e-40f, 0}, {0, 0}), 1.0f);
}

// Searches near ordinary vectors after vectors with extreme magnitudes were inserted, see #37160.
class CosineIndexTest : public CosineDistanceTest {
 protected:
  static constexpr size_t kDimensions = 32;
  static constexpr size_t kNumVectors = 3000;
  static constexpr size_t kNumQueries = 20;
  static constexpr size_t kMaxResults = 10;
  static constexpr size_t kEf = 40;

  static HNSWOptions Options() {
    return {
        .dimensions = kDimensions,
        .num_neighbors_per_vertex = 16,
        .num_neighbors_per_vertex_base = 32,
        .ef_construction = 64,
        .distance_kind = DistanceKind::kCosine,
    };
  }

  void TestExtremeMagnitudes(const VectorIndexTraitsPtr<FloatVector, float>& traits) {
    auto index = traits->Create(FactoryMode::kCreate, StoreVectorPayload::kFalse);
    ASSERT_OK(index->Reserve(
        kNumVectors + std::size(kExtremeScales), /* max_concurrent_inserts= */ 1,
        /* max_concurrent_reads= */ 1, rocksdb::Cache::ReservationMode::kAlways));
    for (size_t i = 0; i != kNumVectors; ++i) {
      const auto& [id, vector] = vectors_.emplace_back(
          VectorId::GenerateRandom(), GaussianVector(kDimensions));
      ASSERT_OK(index->Insert(id, vector, Slice()));
    }
    // Inserted after the ordinary vectors, which damages the graph the most.
    for (auto scale : kExtremeScales) {
      auto id = VectorId::GenerateRandom();
      extreme_ids_.insert(id);
      ASSERT_OK(index->Insert(id, GaussianVector(kDimensions, scale), Slice()));
    }
    ASSERT_NO_FATALS(CheckSearches(*index));

    auto saved = ASSERT_RESULT(index->SaveToFile(GetTestPath("chunk")));
    ASSERT_NE(saved, nullptr);
    ASSERT_NO_FATALS(CheckSearches(*saved));
  }

  // Queries are ordinary vectors with some noise, also scaled to extreme magnitudes. Vectors with
  // extreme magnitudes are left out of both the results and the expected results.
  void CheckSearches(const VectorIndexIf<FloatVector, float>& index) {
    constexpr float kQueryScales[] = {1, 1e-23f, 1e20f};
    size_t found = 0;
    size_t worst_query_found = kMaxResults;
    for (size_t i = 0; i != kNumQueries; ++i) {
      auto query = vectors_[i * (kNumVectors / kNumQueries)].second;
      auto noise = GaussianVector(kDimensions, 0.3f);
      for (size_t j = 0; j != kDimensions; ++j) {
        query[j] += noise[j];
      }
      for (auto scale : kQueryScales) {
        auto scaled_query = query;
        for (auto& coordinate : scaled_query) {
          coordinate *= scale;
        }
        auto results = ASSERT_RESULT(index.Search(scaled_query, SearchOptions {
          .max_num_results = kMaxResults + extreme_ids_.size(),
          .ef = kEf,
        }));
        auto expected = BruteForce(scaled_query);
        size_t query_found = 0;
        size_t num_ordinary = 0;
        for (const auto& result : results) {
          if (extreme_ids_.contains(result.vector_id)) {
            continue;
          }
          query_found += expected.contains(result.vector_id);
          if (++num_ordinary == kMaxResults) {
            break;
          }
        }
        found += query_found;
        worst_query_found = std::min(worst_query_found, query_found);
      }
    }
    constexpr size_t kExpected = kNumQueries * std::size(kQueryScales) * kMaxResults;
    LOG(INFO) << "Recall: " << found * 1.0 / kExpected
              << ", worst query: " << worst_query_found << " of " << kMaxResults;
    ASSERT_GE(found, kExpected * 0.9);
    ASSERT_GE(worst_query_found, kMaxResults / 2);
  }

  std::unordered_set<VectorId> BruteForce(const FloatVector& query) const {
    std::vector<std::pair<float, VectorId>> candidates;
    candidates.reserve(vectors_.size());
    for (const auto& [id, vector] : vectors_) {
      candidates.emplace_back(vector_index::distance::DistanceCosine(query, vector), id);
    }
    std::partial_sort(
        candidates.begin(), candidates.begin() + kMaxResults, candidates.end());
    std::unordered_set<VectorId> result;
    for (size_t i = 0; i != kMaxResults; ++i) {
      result.insert(candidates[i].second);
    }
    return result;
  }

  std::vector<std::pair<VectorId, FloatVector>> vectors_;
  std::unordered_set<VectorId> extreme_ids_;
};

TEST_F(CosineIndexTest, Hnswlib) {
  auto traits = ASSERT_RESULT((CreateHnswlibIndexTraits<FloatVector, float>(
      block_cache_, Options(), /* mem_tracker= */ nullptr, /* tenant_extractor= */ nullptr)));
  ASSERT_NO_FATALS(TestExtremeMagnitudes(traits));
}

TEST_F(CosineIndexTest, Usearch) {
  auto traits = ASSERT_RESULT((CreateUsearchIndexTraits<FloatVector, float>(
      block_cache_, Options(), /* mem_tracker= */ nullptr)));
  ASSERT_NO_FATALS(TestExtremeMagnitudes(traits));
}

}  // namespace yb::ann_methods
