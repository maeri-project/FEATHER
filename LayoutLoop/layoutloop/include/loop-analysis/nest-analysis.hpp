/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <unordered_map>
#include <map>
#include <unordered_set>

#include "mapping/nest.hpp"
#include "workload/util/per-problem-dimension.hpp"
#include "nest-analysis-tile-info.hpp"

namespace analysis
{

class NestAnalysis
{
 private:
  // Cached copy of loop nest under evaluation (used for speedup).
  loop::Nest cached_nest;
  loop::Nest layout_loop_nest;
  
  // Properties of the nest being analyzed (copied over during construction).
  std::vector<uint64_t> storage_tiling_boundaries_;

  // Live state.
  std::vector<analysis::LoopState> nest_state_;
  // Added by JT
  std::vector<analysis::LoopState> layout_nest_state_;
  // Done added by JT
  std::vector<int> indices_;
  std::uint64_t num_epochs_;
  
  // Identifies the spatial element
  // whose working set is currently being computed.
  // Dynamically updated by recursive calls.
  std::uint64_t spatial_id_;
  
  CompoundDataMovementNest working_sets_;
  std::map<std::vector<unsigned>, ComputeInfo> compute_info_;
  CompoundComputeNest compute_info_sets_;

  // Memoization structures to accelerate IndexToOperationPoint()
  std::vector<problem::OperationPoint> vector_strides_;
  std::vector<problem::OperationPoint> mold_low_;
  std::vector<problem::OperationPoint> mold_high_;
  std::vector<problem::OperationPoint> mold_high_residual_;
  problem::OperationPoint cur_transform_;

  // per-level properties.
  std::vector<uint64_t> num_spatial_elems_;
  std::vector<uint64_t> logical_fanouts_;

  // used to accelerate to IndexToOperationPoint computation
  // relevant only for master spatial levels.
  std::vector<uint64_t> logical_fanoutX_;
  std::vector<uint64_t> logical_fanoutY_;
  
  // Added by JT --- The following codes are for mapping. 
  std::vector<std::vector<std::vector<uint64_t> > > spatial_access_buffer_level; 
  std::vector<std::vector<uint64_t> >               temporal_access_below_loop_level; 
  std::vector<std::vector<uint64_t> >               spatial_access_loop_level; 
  std::vector<std::vector<uint64_t> >               total_access_loop_level; 
  std::vector<std::vector<std::vector<uint64_t> > > total_access_buffer_level; 
      // The outer vector contains number of loop dimension.
      // The inner vector contains number of data in need of access for all loop dimensions, initialized by 1.
  std::vector<std::vector<uint64_t > >              total_data_size_access_loop_level; 
  std::vector<std::vector<std::pair<uint64_t, uint64_t> > > data_related_dim; 
  std::vector<std::vector<std::vector<uint64_t> > > total_data_size_access_buffer_level; 
  std::vector<std::vector<uint64_t > >              read_length_loop_level;                     //   L in the iActs[x,x+L] Δx
  std::vector<std::vector<uint64_t > >              reading_start_index_step_loop_level;        //   L in the iActs[x,x+L] Δx
  std::vector<std::vector<std::vector<uint64_t> > > reading_start_index_step_buffer_level;        //   L in the iActs[x,x+L] Δx
  std::vector<std::vector<std::vector<uint64_t> > > read_length_buffer_level;           //  Δx in the iActs[x,x+L] Δx
  std::vector<unsigned> oActs_height_width_dim_id;
  std::vector<unsigned> weights_height_dim_id;
  std::vector<uint64_t> stride_list; // We assume stride for height and width to be exactly the same.
  std::vector<std::vector<uint64_t> > read_length_loop_level_related_dim_id;
  unsigned iacts_data_id;
  unsigned weights_data_id;
  unsigned oacts_data_id;
  // Done added by JT --- The following codes are for mapping. 
  
  // Added by JT --- The following codes are for layout
  std::vector<std::vector<uint64_t> >               layout_data_length_per_buf_loop_level_related_dim_id;
  std::vector<std::vector<uint64_t > >              layout_data_start_index_step_loop_level;    //  Δx in the iActs[x,x+L] Δx
  std::vector<std::vector<uint64_t > >              layout_data_length_per_buf_row_loop_level;  //   L in the iActs[x,x+L] Δx
  std::vector<std::vector<std::vector<uint64_t> > > layout_data_start_index_step_buffer_level;        //   L in the iActs[x,x+L] Δx
  std::vector<std::vector<std::vector<uint64_t> > > layout_data_length_per_buf_row_buffer_level;           //  Δx in the iActs[x,x+L] Δx
  std::vector<std::pair<uint64_t, uint64_t> >       storage_boundary_loop_index_map;           //  Δx in the iActs[x,x+L] Δx
  
  // Done added by JT --- The following codes are for layout

  // records if a level corresponds to the starting
  // point of a new storage tile.
  std::vector<bool> storage_boundary_level_;
  
  // architectural storage level corresponding to a given loop level.
  std::vector<unsigned> arch_storage_level_;

  // extrapolation may be disabled at certain levels.
  std::vector<bool> disable_temporal_extrapolation_;

  // any level which is at the transition point from temporal to
  // spatial nests is a master spatial level.
  // there should be one such level between each set of
  // consecutive physical storage levels.
  std::vector<bool> master_spatial_level_;
  
  // true if the spatial elements at a given master spatial
  // level are connected by on-chip links.
  std::vector<bool> linked_spatial_level_;

  // The following data structures are used for skew calculation. We can
  // possibly optimize the implementation by holding the data in a few
  // OperationPoints instead of these maps. At each storage tiling
  // boundary, we initiate a new loop gist that captures the information
  // for all the loops in that block (i.e., before the next-inner
  // storage tiling boundary).
  struct LoopGist
  {
    int index = 0;
    int bound = 1;
  };
  // Hold the gists in a vector instead of a map. This is because trivial
  // unit-loops are omitted from the loop nest, which means the gist may
  // not be complete by the time we arrive at the innermost
  // FillSpatialDeltas in a loop block. Using a vector (along with the
  // default values in the struct above) allows us to pre-initialize all
  // loops. Just be careful to expand them in the Reset() call.
  std::vector<LoopGist> loop_gists_temporal_;
  std::vector<LoopGist> loop_gists_spatial_;

  // Storage level to fanout map.
  std::map<unsigned, std::uint64_t> physical_fanoutX_; 
  std::map<unsigned, std::uint64_t> physical_fanoutY_; 

  std::unordered_map<unsigned, loop::Nest::SkewDescriptor> packed_skew_descriptors_; // per storage level.
  std::unordered_map<unsigned, loop::Nest::SkewDescriptor> skew_descriptors_; // per loop level.
  loop::Nest::SkewDescriptor* cur_skew_descriptor_ = nullptr;

  std::unordered_map<unsigned, problem::PerDataSpace<bool>> no_link_transfer_;
  std::unordered_map<unsigned, problem::PerDataSpace<bool>> no_multicast_;
  std::unordered_map<unsigned, problem::PerDataSpace<bool>> no_temporal_reuse_;

  // Other state.

  bool working_sets_computed_ = false;
  bool imperfectly_factorized_ = false;

  problem::Workload* workload_ = nullptr;

  std::vector<unsigned> time_stamp_;
  std::vector<unsigned> space_stamp_;

  // Internal helper methods.
  void ComputeWorkingSets();

  void InitNumSpatialAccessEveryBufferLevel();
  void InitDimAccessEveryBufferLevel();
  void InitRunLengthEveryBufferLevel();
  void InitStartIndexStepEveryBufferLevel();
  void InitLayoutNestEveryBufferLevel();
  void TestElementState();
  void DetectImperfectFactorization();
  void InitializeNestProperties();
  void InitNumSpatialElems();
  void InitStorageBoundaries();
  void InitSpatialFanouts();
  void InitPerLevelDimScales();

  void InitializeLiveState();
  void CollectWorkingSets();

  problem::OperationPoint IndexToOperationPoint_(const std::vector<int>& indices) const;
  bool IsLastGlobalIteration_(int level, problem::Shape::FlattenedDimensionID dim) const;
  problem::OperationSpace GetCurrentWorkingSet(std::vector<analysis::LoopState>::reverse_iterator cur);
  problem::PerDataSpace<Point> GetCurrentTranslationVectors(std::vector<analysis::LoopState>::reverse_iterator cur);

  problem::OperationSpace ComputeDeltas(std::vector<analysis::LoopState>::reverse_iterator cur);

  void ComputeTemporalWorkingSet(std::vector<analysis::LoopState>::reverse_iterator cur,
                                 analysis::ElementState& cur_state);
  void ComputeSpatialWorkingSet(std::vector<analysis::LoopState>::reverse_iterator cur);

  void FillSpatialDeltas(std::vector<analysis::LoopState>::reverse_iterator cur,
                         std::unordered_map<std::uint64_t, problem::OperationSpace>& spatial_deltas,
                         std::unordered_map<std::uint64_t, std::uint64_t>& skew_table,
                         std::uint64_t base_index,
                         int depth,
                         int extrapolation_stride,
                         std::vector<analysis::LoopState>::reverse_iterator extrapolation_level);

  std::uint64_t ApplySkew(std::uint64_t unskewed_index);

  void ComputeAccurateMulticastedAccesses(
      std::vector<analysis::LoopState>::reverse_iterator cur,
      const std::unordered_map<std::uint64_t, problem::OperationSpace>& spatial_deltas,
      problem::PerDataSpace<std::unordered_set<std::uint64_t>>& unaccounted_delta,
      problem::PerDataSpace<AccessStatMatrix>& access_stats);

  void ComputeNetworkLinkTransfers(
      std::vector<analysis::LoopState>::reverse_iterator cur,
      const std::unordered_map<std::uint64_t, problem::OperationSpace>& cur_spatial_deltas,
      problem::PerDataSpace<std::unordered_set<std::uint64_t>>& unaccounted_delta,
      problem::PerDataSpace<std::uint64_t>& link_transfers);
 
  void CompareSpatioTemporalDeltas(
    const std::unordered_map<std::uint64_t, problem::OperationSpace>& cur_spatial_deltas,
    const std::unordered_map<std::uint64_t, problem::OperationSpace>& prev_spatial_deltas,
    const std::uint64_t cur_spatial_index,
    const std::uint64_t prev_spatial_index,
    std::vector<problem::PerDataSpace<bool>>& inter_elem_reuse,
    const problem::PerDataSpace<bool>& ignore_dataspaces);
  
  void ComputeDataDensity();
  void PrintSpaceTimeStamp();

 public:  
  // API
  NestAnalysis();
  void Init(problem::Workload* wc, const loop::Nest* nest,
            std::map<unsigned, std::uint64_t> fanoutX_map,
            std::map<unsigned, std::uint64_t> fanoutY_map);

  void Init(problem::Workload* wc, const loop::Nest* layout_nest, const loop::Nest* nest,
            std::map<unsigned, std::uint64_t> fanoutX_map,
            std::map<unsigned, std::uint64_t> fanoutY_map);

  void Reset();
 
  std::vector<problem::PerDataSpace<std::size_t>> GetWorkingSetSizes_LTW() const;

  CompoundDataMovementNest GetWorkingSets();
  CompoundComputeNest GetComputeInfo();
  problem::Workload* GetWorkload();
  

  // Serialization.
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version=0) 
  {
    if(version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(nest_state_);
      ar& boost::serialization::make_nvp("work_sets_",boost::serialization::make_array(working_sets_.data(),working_sets_.size()));
      ar& BOOST_SERIALIZATION_NVP(working_sets_computed_);
      // ar& BOOST_SERIALIZATION_NVP(compute_cycles_);
    }
  }

  friend std::ostream& operator << (std::ostream& out, const NestAnalysis& n);  
};

} // namespace analysis
