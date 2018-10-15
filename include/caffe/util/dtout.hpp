/******************************************************************************

* Copyright 2018 The Apollo Authors. All Rights Reserved.

*

* Licensed under the Apache License, Version 2.0 (the License);

* you may not use this file except in compliance with the License.

* You may obtain a copy of the License at

*

* http://www.apache.org/licenses/LICENSE-2.0

*

* Unless required by applicable law or agreed to in writing, software

* distributed under the License is distributed on an AS IS BASIS,

* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

* See the License for the specific language governing permissions and

* limitations under the License.

*****************************************************************************/

#ifndef CAFFE_DTOUT_HPP_
#define CAFFE_DTOUT_HPP_

#include <vector>
#include "boost/scoped_ptr.hpp"
#include "caffe/syncedmem.hpp"

using caffe::SyncedMemory;

namespace caffe {

#ifndef CPU_ONLY
template <typename Dtype>
void rpn_cmp_conf_bbox_gpu(const int num_anchors, 
        const int map_height, const int map_width,
        const Dtype input_height, const Dtype input_width,
        const Dtype heat_map_a, const Dtype heat_map_b,
        const Dtype allow_border, const Dtype allow_border_ratio,
        const Dtype min_size_w, const Dtype min_size_h,
        const bool min_size_mode_and_else_or, const Dtype thr_obj,  
        const Dtype bsz01, const bool do_bbox_norm,
        const Dtype mean0, const Dtype mean1, 
        const Dtype mean2, const Dtype mean3,
        const Dtype std0, const Dtype std1,
        const Dtype std2, const Dtype std3,
        const bool refine_out_of_map_bbox, const Dtype* anc_data, 
        const Dtype* prob_data, const Dtype* tgt_data, 
        Dtype* conf_data, Dtype* bbox_data);

template <typename Dtype>
void rcnn_cmp_conf_bbox_gpu(const int num_rois, 
        const Dtype input_height, const Dtype input_width,
        const Dtype allow_border, const Dtype allow_border_ratio,
        const Dtype min_size_w, const Dtype min_size_h,
        const bool min_size_mode_and_else_or, const Dtype thr_obj,  
        const Dtype bsz01, const bool do_bbox_norm,
        const Dtype mean0, const Dtype mean1, 
        const Dtype mean2, const Dtype mean3,
        const Dtype std0, const Dtype std1,
        const Dtype std2, const Dtype std3,
        const bool refine_out_of_map_bbox, const bool regress_agnostic,
        const int num_class, const Dtype* thr_cls,
        const Dtype* rois_data, const Dtype* prob_data, 
        const Dtype* tgt_data, Dtype* conf_data, Dtype* bbox_data);

template <typename Dtype>
void apply_nms_gpu(const Dtype *bbox_data, const Dtype *conf_data,
        const int num_bboxes, const int bbox_step, const Dtype confidence_threshold,
        const int max_canditate_n, const int top_k, const Dtype nms_threshold, 
        const Dtype bsz01, std::vector<int> *indices,
        boost::shared_ptr<SyncedMemory> overlapped,
               boost::shared_ptr<SyncedMemory> idx_sm,
        const cudaStream_t &stream, std::vector<int> *idx_ptr = NULL,
        const int conf_step = 1, const int conf_idx = 0,
        const int nms_gpu_max_n_per_time = 1000000);
#endif

}  // namespace caffe

#endif  // CAFFE_DTOUT_HPP_
