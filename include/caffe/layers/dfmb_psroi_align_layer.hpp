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

#ifndef CAFFE_DFMB_PSROI_ALIGN_LAYER_HPP_
#define CAFFE_DFMB_PSROI_ALIGN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class DFMBPSROIAlignLayer : public Layer<Dtype> {
    public:
        explicit DFMBPSROIAlignLayer(const LayerParameter& param)
            : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "DFMBPSROIAlign"; }

        virtual inline int MinBottomBlobs() const { return 2; }
        virtual inline int MaxBottomBlobs() const { return 3; }
        virtual inline int MinTopBlobs() const { return 1; }
        virtual inline int MaxTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down,
                const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                const vector<bool>& propagate_down,
                const vector<Blob<Dtype>*>& bottom);

        Dtype heat_map_a_;
        Dtype heat_map_b_;
        Dtype pad_ratio_;

        int output_dim_;
        bool no_trans_;
        Dtype trans_std_;
        int sample_per_part_;
        int group_height_;
        int group_width_;
        int pooled_height_;
        int pooled_width_;
        int part_height_;
        int part_width_;

        int channels_;
        int height_;
        int width_;

        Blob<Dtype> top_count_;

};

}  // namespace caffe

#endif  // CAFFE_DFMB_PSROI_ALIGN_LAYER_HPP_
