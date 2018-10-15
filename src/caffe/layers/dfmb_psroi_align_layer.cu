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

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/dfmb_psroi_align_layer.hpp"
#include "caffe/util/gpu_util.cuh"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void DFMBPSROIAlignForward(
        const int nthreads,
        const Dtype* bottom_data,
        const Dtype heat_map_a,
        const Dtype heat_map_b,
        const Dtype pad_ratio,
        const int channels,
        const int height, const int width,
        const int pooled_height, const int pooled_width,
        const Dtype* bottom_rois, const Dtype* bottom_trans,
        const bool no_trans,
        const Dtype trans_std,
        const int sample_per_part,
        const int output_dim,
        const int group_height, const int group_width,
        const int part_height, const int part_width,
        const int num_classes,
        const int channels_each_class,
        Dtype* top_data,
        Dtype* top_count) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        // The output is in order (n, ctop, ph, pw)
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int ctop = (index / pooled_width / pooled_height) % output_dim;
        int n = index / pooled_width / pooled_height / output_dim;

        // [start, end) interval for spatial sampling
        const Dtype* offset_bottom_rois = bottom_rois + n * 5;
        int roi_batch_ind = offset_bottom_rois[0];

        Dtype pad_w = (offset_bottom_rois[3] -\
                                         offset_bottom_rois[1] + 1) * pad_ratio;
        Dtype pad_h = (offset_bottom_rois[4] -\
                                         offset_bottom_rois[2] + 1) * pad_ratio;
        Dtype roi_start_w = (offset_bottom_rois[1] -\
                                               pad_w - heat_map_b) / heat_map_a;
        Dtype roi_start_h = (offset_bottom_rois[2] -\
                                               pad_h - heat_map_b) / heat_map_a;
        Dtype roi_end_w = (offset_bottom_rois[3] +\
                                               pad_w - heat_map_b) / heat_map_a;
        Dtype roi_end_h = (offset_bottom_rois[4] +\
                                               pad_h - heat_map_b) / heat_map_a;
        // Force too small ROIs to be 1x1
        Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);
        Dtype roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0

        // Compute w and h at bottom
        Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
        Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

        Dtype sub_bin_size_h = bin_size_h / static_cast<Dtype>(sample_per_part);
        Dtype sub_bin_size_w = bin_size_w / static_cast<Dtype>(sample_per_part);

        int part_h = floor(
                          static_cast<Dtype>(ph) / pooled_height * part_height);
        int part_w = floor(static_cast<Dtype>(pw) / pooled_width * part_width);
        int class_id = ctop / channels_each_class;
        Dtype trans_x = no_trans ? static_cast<Dtype>(0) :
            bottom_trans[(((n * num_classes + class_id) * 2) * 
                    part_height + part_h) * part_width + part_w] * trans_std;
        Dtype trans_y = no_trans ? static_cast<Dtype>(0) :
            bottom_trans[(((n * num_classes + class_id) * 2 + 1) * 
                    part_height + part_h) * part_width + part_w] * trans_std;

        int hstart = static_cast<Dtype>(ph) * bin_size_h + 
            roi_start_h + trans_y * roi_height;
        int wstart =  static_cast<Dtype>(pw)* bin_size_w + 
            roi_start_w + trans_x * roi_width;

        Dtype sum = 0;
        int count = 0;
        int gh = floor(static_cast<Dtype>(ph)* group_height / pooled_height);
        int gw = floor(static_cast<Dtype>(pw) * group_width / pooled_width);
        gh = min(max(gh, 0), group_height - 1);
        gw = min(max(gw, 0), group_width - 1);

        const Dtype* offset_bottom_data = bottom_data + 
            (roi_batch_ind * channels) * height * width;
        for (int ih = 0; ih < sample_per_part; ih++) {
            for (int iw = 0; iw < sample_per_part; iw++) {
                Dtype w = wstart + (iw + 0.5) * sub_bin_size_w;
                Dtype h = hstart + (ih + 0.5) * sub_bin_size_h;
                // bilinear interpolation
                if (w <= -1 || w >= width || h <= -1 || h >= height) {
                    continue;
                }
                int c = (ctop * group_height + gh) * group_width + gw;
                int x1 = floor(w);
                int x2 = ceil(w);
                int y1 = floor(h);
                int y2 = ceil(h);
                Dtype dist_x = static_cast<Dtype>(w - x1);
                Dtype dist_y = static_cast<Dtype>(h - y1);
                const Dtype* data = offset_bottom_data + c * height * width;
                Dtype value11 = (x1 >= 0 && x1 < width && y1 >= 0 &&
                              y1 < height) ? data[y1 * width + x1] : Dtype(0.0);
                Dtype value12 = (x1 >= 0 && x1 < width && y2 >= 0 &&
                              y2 < height) ? data[y2 * width + x1] : Dtype(0.0);
                Dtype value21 = (x2 >= 0 && x2 < width && y1 >= 0 &&
                              y1 < height) ? data[y1 * width + x2] : Dtype(0.0);
                Dtype value22 = (x2 >= 0 && x2 < width && y2 >= 0 &&
                              y2 < height) ? data[y2 * width + x2] : Dtype(0.0);
                Dtype value = (1 - dist_x) * (1 - dist_y) * value11 
                    + (1 - dist_x) * dist_y * value12 
                    + dist_x * (1 - dist_y) * value21 
                    + dist_x * dist_y * value22;
                sum += value;
                count++;
            }
        }
        top_data[index] = count == 0 ? static_cast<Dtype>(0) : sum / count;
        top_count[index] = count;
    }
}

template <typename Dtype>
void DFMBPSROIAlignLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_rois = bottom[1]->gpu_data();
    const Dtype *bottom_trans = no_trans_ ? NULL : bottom[2]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    Dtype* top_count_data = top_count_.mutable_gpu_data();
    int count = top[0]->count();
    int num_classes = no_trans_ ? 1 : bottom[2]->channels() / 2; 
    int channels_each_class =
                           no_trans_ ? output_dim_ : output_dim_ / num_classes;

    caffe_gpu_set(count, Dtype(0), top_data);
    caffe_gpu_set(count, Dtype(0), top_count_data);

    // NOLINT_NEXT_LINE(whitespace/operators)
    DFMBPSROIAlignForward<Dtype> << <CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS >> >(count, bottom_data, 
                heat_map_a_, heat_map_b_, pad_ratio_, channels_, height_,
                width_, pooled_height_, pooled_width_, bottom_rois,
                bottom_trans, no_trans_, trans_std_, sample_per_part_,
                output_dim_, group_height_, group_width_, part_height_,
                part_width_, num_classes, channels_each_class, top_data,
                top_count_data);
    CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void DFMBPSROIAlignBackward(
        const int nthreads,
        const Dtype* top_diff,
        const Dtype* top_count,
        const int num_rois,
        const Dtype heat_map_a,
        const Dtype heat_map_b,
        const Dtype pad_ratio,
        const int channels,
        const int height, const int width,
        const int pooled_height, const int pooled_width,
        const int output_dim,
        Dtype* bottom_data_diff,
        Dtype* bottom_trans_diff,
        const Dtype* bottom_data,
        const Dtype* bottom_rois,
        const Dtype* bottom_trans,
        const bool no_trans,
        const Dtype trans_std,
        const int sample_per_part,
        const int group_height, const int group_width,
        const int part_height, const int part_width,
        const int num_classes, 
        const int channels_each_class) {
            CUDA_KERNEL_LOOP(index, nthreads) {
                if (top_count[index] <= 0) continue;

                // The output is in order (n, ctop, ph, pw)
                int pw = index % pooled_width;
                int ph = (index / pooled_width) % pooled_height;
                int ctop = (index / pooled_width / pooled_height) % output_dim;
                int n = index / pooled_width / pooled_height / output_dim;

                // [start, end) interval for spatial sampling
                const Dtype* offset_bottom_rois = bottom_rois + n * 5;
                int roi_batch_ind = offset_bottom_rois[0];

                Dtype pad_w = (offset_bottom_rois[3] -
                                         offset_bottom_rois[1] + 1) * pad_ratio;
                Dtype pad_h = (offset_bottom_rois[4] -
                                         offset_bottom_rois[2] + 1) * pad_ratio;
                Dtype roi_start_w = (offset_bottom_rois[1] -
                                               pad_w - heat_map_b) / heat_map_a;
                Dtype roi_start_h = (offset_bottom_rois[2] -
                                               pad_h - heat_map_b) / heat_map_a;
                Dtype roi_end_w = (offset_bottom_rois[3] +
                                               pad_w - heat_map_b) / heat_map_a;
                Dtype roi_end_h = (offset_bottom_rois[4] +
                                               pad_h - heat_map_b) / heat_map_a;
                // Force too small ROIs to be 1x1
                Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);
                Dtype roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0

                // Compute w and h at bottom
                Dtype bin_size_h = roi_height /
                                              static_cast<Dtype>(pooled_height);
                Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

                Dtype sub_bin_size_h = bin_size_h /
                                            static_cast<Dtype>(sample_per_part);
                Dtype sub_bin_size_w = bin_size_w /
                                            static_cast<Dtype>(sample_per_part);

                int part_h = floor(static_cast<Dtype>(ph) /
                                                   pooled_height * part_height);
                int part_w = floor(static_cast<Dtype>(pw) /
                                                     pooled_width * part_width);
                int class_id = ctop / channels_each_class;
                Dtype trans_x = no_trans ? static_cast<Dtype>(0) :
                    bottom_trans[(((n * num_classes + class_id) * 2) * 
                       part_height + part_h) * part_width + part_w] * trans_std;
                Dtype trans_y = no_trans ? static_cast<Dtype>(0) :
                    bottom_trans[(((n * num_classes + class_id) * 2 + 1) * 
                       part_height + part_h) * part_width + part_w] * trans_std;

                int hstart = static_cast<Dtype>(ph) * bin_size_h + 
                    roi_start_h + trans_y * roi_height;
                int wstart =  static_cast<Dtype>(pw)* bin_size_w + 
                    roi_start_w + trans_x * roi_width;

                Dtype diff_val = top_diff[index] / top_count[index];
                const Dtype* offset_bottom_data = bottom_data + 
                    roi_batch_ind * channels * height * width;
                Dtype* offset_bottom_data_diff = bottom_data_diff + 
                    roi_batch_ind * channels * height * width;
                int gh = floor(static_cast<Dtype>(ph)*
                                                  group_height / pooled_height);
                int gw = floor(static_cast<Dtype>(pw) *
                                                    group_width / pooled_width);
                gh = min(max(gh, 0), group_height - 1);
                gw = min(max(gw, 0), group_width - 1);

                for (int ih = 0; ih < sample_per_part; ih++) {
                    for (int iw = 0; iw < sample_per_part; iw++) {
                        Dtype w = wstart + (iw + 0.5) * sub_bin_size_w ;
                        Dtype h = hstart + (ih + 0.5) * sub_bin_size_h;
                        // bilinear interpolation
                        if (w <= -1 || w >= width || h <= -1 || h >= height) {
                            continue;
                        }
                        int c = (ctop*group_height + gh) * group_width + gw;
                        // backward on feature
                        int x0 = floor(w);
                        int x1 = ceil(w);
                        int y0 = floor(h);
                        int y1 = ceil(h);
                        Dtype dist_x = w - x0, dist_y = h - y0;
                        Dtype q00 = (1 - dist_x) * (1 - dist_y);
                        Dtype q01 = (1 - dist_x) * dist_y;
                        Dtype q10 = dist_x * (1 - dist_y);
                        Dtype q11 = dist_x * dist_y;
                        int bottom_index_base = c * height * width;
                        if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
                            caffe_gpu_atomic_add(q00 * diff_val, 
                                    offset_bottom_data_diff + 
                                           bottom_index_base + y0 * width + x0);
                        }
                        if (x0 >= 0 && x0 < width && y1 >= 0 && y1 < height) {
                            caffe_gpu_atomic_add(q01 * diff_val, 
                                    offset_bottom_data_diff +
                                           bottom_index_base + y1 * width + x0);
                        }
                        if (x1 >= 0 && x1 < width && y0 >= 0 && y0 < height) {
                            caffe_gpu_atomic_add(q10 * diff_val, 
                                    offset_bottom_data_diff +
                                           bottom_index_base + y0 * width + x1);
                        }
                        if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
                            caffe_gpu_atomic_add(q11 * diff_val, 
                                    offset_bottom_data_diff +
                                           bottom_index_base + y1 * width + x1);
                        }

                        if (no_trans) continue;

                        Dtype U00 = (x0 >= 0 && x0 < width && y0 >= 0 &&
                            y0 < height) ? offset_bottom_data[bottom_index_base
                                               + y0 * width + x0] : Dtype(0.0);
                        Dtype U01 = (x0 >= 0 && x0 < width && y1 >= 0 &&
                            y1 < height) ? offset_bottom_data[bottom_index_base
                                                + y1 * width + x0] : Dtype(0.0);
                        Dtype U10 = (x1 >= 0 && x1 < width && y0 >= 0 &&
                            y0 < height) ? offset_bottom_data[bottom_index_base
                                                + y0 * width + x1] : Dtype(0.0);
                        Dtype U11 = (x1 >= 0 && x1 < width && y1 >= 0 &&
                            y1 < height) ? offset_bottom_data[bottom_index_base
                                                + y1 * width + x1] : Dtype(0.0);
                        Dtype diff_x = (U11 * dist_y + U10 * (1 - dist_y) - 
                                U01 * dist_y -
                                     U00 * (1 - dist_y)) * trans_std * diff_val;
                        diff_x *= roi_width;
                        Dtype diff_y = (U11 * dist_x + U01 * (1 - dist_x) - 
                                U10 * dist_x -
                                     U00 * (1 - dist_x)) * trans_std * diff_val;
                        diff_y *= roi_height;
                        caffe_gpu_atomic_add(diff_x, bottom_trans_diff + 
                                (((n * num_classes + class_id) * 2) *
                                   part_height + part_h) * part_width + part_w);
                        caffe_gpu_atomic_add(diff_y, bottom_trans_diff + 
                                (((n * num_classes + class_id) * 2 + 1) *
                                   part_height + part_h) * part_width + part_w);
                    }
                }
            }
        }

template <typename Dtype>
void DFMBPSROIAlignLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0] && !propagate_down[2]) {
        return;
    }

    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_rois = bottom[1]->gpu_data();
    const Dtype* bottom_trans = no_trans_ ? NULL : bottom[2]->gpu_data();
    Dtype* bottom_data_diff = bottom[0]->mutable_gpu_diff();
    Dtype* bottom_rois_diff = bottom[1]->mutable_gpu_diff();
    Dtype* bottom_trans_diff = no_trans_ ? NULL : bottom[2]->mutable_gpu_diff();
    const Dtype* top_count_data = top_count_.gpu_data();
    const int count = top[0]->count();
    const int num_rois = bottom[1]->num();
    const int num_classes = no_trans_ ? 1 : bottom[2]->channels() / 2;
    const int channels_each_class =
                            no_trans_ ? output_dim_ : output_dim_ / num_classes;
    caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_data_diff);
    caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom_rois_diff);
    if (!no_trans_) {
        caffe_gpu_set(bottom[2]->count(), Dtype(0), bottom_trans_diff);
    }

    DFMBPSROIAlignBackward<Dtype> << <CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS >> >(count, top_diff, top_count_data, 
                num_rois, heat_map_a_, heat_map_b_, pad_ratio_, channels_, 
                height_, width_, pooled_height_, pooled_width_, output_dim_,
                bottom_data_diff, bottom_trans_diff, bottom_data, bottom_rois,
                bottom_trans, no_trans_, trans_std_, sample_per_part_, 
                group_height_, group_width_, part_height_, part_width_, 
                num_classes, channels_each_class);
    CUDA_POST_KERNEL_CHECK;
  }

  INSTANTIATE_LAYER_GPU_FUNCS(DFMBPSROIAlignLayer);

}  // namespace caffe
