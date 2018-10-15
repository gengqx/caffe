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

#include <string>
#include <vector>
#include <algorithm>
#include "caffe/layers/rpn_proposal_ssd_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/util_others.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/util/dtout.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe
{
template <typename Dtype>
void RPNProposalSSDLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top)//check
{
    ROIOutputSSDLayer<Dtype>::LayerSetUp(bottom, top);

    CHECK_EQ(1, this->heat_map_a_vec_.size());
    CHECK_EQ(1, this->heat_map_b_vec_.size());

    if (top.size() == 0) {
        CHECK_GT(this->num_class_, 0);
    }

    num_anchors_ = this->anchor_x1_vec_.size(); 
    CHECK_GE(num_anchors_, 1);

    rois_dim_ = this->rpn_proposal_output_score_?6:5;
  
#ifndef CPU_ONLY
    anc_.Reshape(num_anchors_, 4, 1, 1);
    Dtype* anc_data = anc_.mutable_cpu_data();
    Dtype bsz01 = this->bbox_size_add_one_ ? Dtype(1.0) : Dtype(0.0);
    for (int a = 0; a < num_anchors_; ++a) {
        Dtype anchor_width = this->anchor_x2_vec_[a] 
            - this->anchor_x1_vec_[a] + bsz01;
        Dtype anchor_height = this->anchor_y2_vec_[a] 
            - this->anchor_y1_vec_[a] + bsz01;
        Dtype anchor_ctr_x = this->anchor_x1_vec_[a] 
            + 0.5 * (anchor_width - bsz01);
        Dtype anchor_ctr_y = this->anchor_y1_vec_[a] 
            + 0.5 * (anchor_height - bsz01);
        anc_data[a * 4] = anchor_ctr_x;
        anc_data[a * 4 + 1] = anchor_ctr_y;
        anc_data[a * 4 + 2] = anchor_width;
        anc_data[a * 4 + 3] = anchor_height;
    }

    overlapped_.reset(new caffe::SyncedMemory(
                this->nms_gpu_max_n_per_time_ * 
                this->nms_gpu_max_n_per_time_ * sizeof(bool)));
    overlapped_->cpu_data();
    overlapped_->gpu_data();
    idx_sm_.reset(new caffe::SyncedMemory(
                this->nms_gpu_max_n_per_time_ * sizeof(int)));
    cudaStreamCreate(&stream_);
#endif

}

template <typename Dtype>
void RPNProposalSSDLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    //dummy reshape
    for (int i = 0; i < top.size(); i++) {
        if (top[i]->count() == 0) { 
            top[i]->Reshape(1, rois_dim_, 1, 1);
            Dtype* top_boxes_scores = top[i]->mutable_cpu_data();
            caffe_set(top[i]->count(), Dtype(0), top_boxes_scores); 
        }
    }
}

template <typename Dtype>
void RPNProposalSSDLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top)//check
{
    NOT_IMPLEMENTED;
}

template <typename Dtype>
void RPNProposalSSDLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, 
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

#ifdef CPU_ONLY
STUB_GPU(RPNProposalSSDLayer);
#endif

INSTANTIATE_CLASS(RPNProposalSSDLayer);
REGISTER_LAYER_CLASS(RPNProposalSSD);
}  // namespace caffe
