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
#include "caffe/layers/rcnn_proposal_layer.hpp"
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
void RCNNProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top)//check
{
    ROIOutputSSDLayer<Dtype>::LayerSetUp(bottom, top);

    CHECK_GT(this->num_class_, 0);

    rois_dim_ = this->rpn_proposal_output_score_ ?
                                                 (5 + this->num_class_ + 1) : 5;
  
#ifndef CPU_ONLY
    thr_cls_.Reshape(1, this->num_class_, 1, 1);
    Dtype* thr_cls_data = thr_cls_.mutable_cpu_data();
    for (int c = 0; c < this->num_class_; c++) {
        thr_cls_data[c] = this->threshold_[c];
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
void RCNNProposalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top) {
    // bottom[0] --> probs
    // bottom[1] --> coords
    // bottom[2] --> anchors

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
void RCNNProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top)//check
{
    NOT_IMPLEMENTED;
}

template <typename Dtype>
void RCNNProposalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, 
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) {}

#ifdef CPU_ONLY
STUB_GPU(RCNNProposalLayer);
#endif

INSTANTIATE_CLASS(RCNNProposalLayer);
REGISTER_LAYER_CLASS(RCNNProposal);
}  // namespace caffe
