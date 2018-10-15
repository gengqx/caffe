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

#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/dfmb_psroi_align_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {
template <typename Dtype>
void DFMBPSROIAlignLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    DFMBPSROIPoolingParameter dfmb_psroi_pooling_param =
        this->layer_param_.dfmb_psroi_pooling_param();
    heat_map_a_ = dfmb_psroi_pooling_param.heat_map_a();
    heat_map_b_ = dfmb_psroi_pooling_param.heat_map_b();
    pad_ratio_ = dfmb_psroi_pooling_param.pad_ratio();
    CHECK_GT(heat_map_a_, 0);
    CHECK_GE(heat_map_b_, 0);
    CHECK_GE(pad_ratio_, 0);
    output_dim_ = dfmb_psroi_pooling_param.output_dim();
    trans_std_ = dfmb_psroi_pooling_param.trans_std();
    sample_per_part_ = dfmb_psroi_pooling_param.sample_per_part();
    group_height_ = dfmb_psroi_pooling_param.group_height();
    group_width_ = dfmb_psroi_pooling_param.group_width();
    pooled_height_ = dfmb_psroi_pooling_param.pooled_height();
    pooled_width_ = dfmb_psroi_pooling_param.pooled_width();
    part_height_ = dfmb_psroi_pooling_param.part_height();
    part_width_ = dfmb_psroi_pooling_param.part_width();
    no_trans_ = (bottom.size() < 3);

    CHECK_GT(output_dim_, 0);
    CHECK_GT(sample_per_part_, 0);
    CHECK_GT(group_height_, 0);
    CHECK_GT(group_width_, 0);
    CHECK_GT(pooled_height_, 0);
    CHECK_GT(pooled_width_, 0);
    CHECK_GT(part_height_, 0);
    CHECK_GT(part_width_, 0);
}

template <typename Dtype>
void DFMBPSROIAlignLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();

    CHECK_EQ(channels_, output_dim_ * group_height_ * group_width_);
    CHECK_EQ(bottom[1]->channels(), 5);
    if (!no_trans_) {
        CHECK_EQ(bottom[2]->channels() % 2, 0);
        int num_classes = bottom[2]->channels() / 2;
        CHECK_EQ(output_dim_ % num_classes, 0);
        CHECK_EQ(part_height_, bottom[2]->height());
        CHECK_EQ(part_width_, bottom[2]->width());
    }

    top[0]->Reshape(bottom[1]->num(), output_dim_, pooled_height_,
                                                                pooled_width_);
    top_count_.Reshape(bottom[1]->num(), output_dim_, pooled_height_,
                                                                pooled_width_);
}

template <typename Dtype>
void DFMBPSROIAlignLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
}

template <typename Dtype>
void DFMBPSROIAlignLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(DFMBPSROIAlignLayer);
#endif

INSTANTIATE_CLASS(DFMBPSROIAlignLayer);
REGISTER_LAYER_CLASS(DFMBPSROIAlign);

}  // namespace caffe
