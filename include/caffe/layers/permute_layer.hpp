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

#ifndef CAFFE_PERMUTE_LAYER_HPP_
#define CAFFE_PERMUTE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Permute the input blob by changing the memory order of the data.
 */

// The main function which does the permute.
template<typename Dtype>
void Permute(const int count, Dtype *bottom_data, const bool forward,
             const int *permute_order, const int *old_steps,
             const int *new_steps,
             const int num_axes, Dtype *top_data);

template<typename Dtype>
class PermuteLayer : public Layer<Dtype> {
public:
    explicit PermuteLayer(const LayerParameter &param)
        : Layer<Dtype>(param) {}

    virtual void LayerSetUp(const std::vector<Blob<Dtype> *> &bottom,
                            const std::vector<Blob<Dtype> *> &top);

    virtual void Reshape(const std::vector<Blob<Dtype> *> &bottom,
                         const std::vector<Blob<Dtype> *> &top);

    virtual inline const char *type() const { return "Permute"; }

    virtual inline int ExactNumBottomBlobs() const { return 1; }

    virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
    virtual void Forward_cpu(const std::vector<Blob<Dtype> *> &bottom,
                             const std::vector<Blob<Dtype> *> &top);

    virtual void Forward_gpu(const std::vector<Blob<Dtype> *> &bottom,
                             const std::vector<Blob<Dtype> *> &top);

    virtual void Backward_cpu(const std::vector<Blob<Dtype> *> &top,
                              const std::vector<bool> &propagate_down,
                              const std::vector<Blob<Dtype> *> &bottom);

    virtual void Backward_gpu(const std::vector<Blob<Dtype> *> &top,
                              const std::vector<bool> &propagate_down,
                              const std::vector<Blob<Dtype> *> &bottom);

    int num_axes_;
    bool need_permute_;

    // Use Blob because it is convenient to be accessible in .cu file.
    Blob<int> permute_order_;
    Blob<int> old_steps_;
    Blob<int> new_steps_;
};

}  // namespace caffe

#endif  // CAFFE_PERMUTE_LAYER_HPP_
