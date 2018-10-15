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


#include "caffe/util/dtout.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "thrust/functional.h"
#include "thrust/sort.h"
#include "boost/iterator/counting_iterator.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

// rpn
template <typename Dtype>
__global__ void rpn_cmp_conf_bbox_kernel(
        const int threads, const int num_anchors, 
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
        Dtype* conf_data, Dtype* bbox_data) {
    int map_size = map_height * map_width;
    CUDA_KERNEL_LOOP(index, threads) {
        int w = index % map_width;
        int h = (index / map_width) % map_height;
        int a = index / map_size;
        int off = h * map_width + w;

        Dtype  score = prob_data[(num_anchors + a) * map_size + off];
        if (score < thr_obj) {
            conf_data[index] = 0.0;
            continue;
        }

        int ax4 = a * 4;
        Dtype anchor_ctr_x = anc_data[ax4];
        Dtype anchor_ctr_y = anc_data[ax4 + 1];
        Dtype anchor_width = anc_data[ax4 + 2];
        Dtype anchor_height = anc_data[ax4 + 3];

        Dtype input_ctr_x = w * heat_map_a + heat_map_b + anchor_ctr_x;
        Dtype input_ctr_y = h * heat_map_a + heat_map_b + anchor_ctr_y;

        if (allow_border >= Dtype(0.0) 
                || allow_border_ratio >= Dtype(0.0)) {
            Dtype x1 = input_ctr_x - 0.5 * (anchor_width - bsz01); 
            Dtype y1 = input_ctr_y - 0.5 * (anchor_height - bsz01); 
            Dtype x2 = x1 + anchor_width - bsz01; 
            Dtype y2 = y1 + anchor_height - bsz01; 
            if (allow_border >= Dtype(0.0) && (
                        x1 < -allow_border || y1 < -allow_border 
                        || x2 > input_width - 1 + allow_border ||  
                        y2 > input_height - 1 + allow_border)) {
                conf_data[index] = 0.0;
                continue;
            } else if (allow_border_ratio >= Dtype(0.0)) {
                Dtype x11 = max(Dtype(0), x1);
                Dtype y11 = max(Dtype(0), y1);
                Dtype x22 = min(input_width - 1, x2);
                Dtype y22 = min(input_height - 1, y2);
                if ((y22 - y11 + bsz01) * (x22 - x11 + bsz01) 
                        / ((y2 - y1 + bsz01) * (x2 - x1 + bsz01)) 
                        < (1.0 - allow_border_ratio)) {
                    conf_data[index] = 0.0;
                    continue;
                }   
            }   
        }

        Dtype tg0 = tgt_data[ax4 * map_size + off];
        Dtype tg1 = tgt_data[(ax4 + 1) * map_size + off];
        Dtype tg2 = tgt_data[(ax4 + 2) * map_size + off];
        Dtype tg3 = tgt_data[(ax4 + 3) * map_size + off];
        if (do_bbox_norm) {
            tg0 = tg0 * std0 + mean0;
            tg1 = tg1 * std1 + mean1;
            tg2 = tg2 * std2 + mean2;
            tg3 = tg3 * std3 + mean3;
        }
        Dtype tw = anchor_width * exp(tg2);
        Dtype th = anchor_height * exp(tg3);

        Dtype ctx = tg0 * anchor_width + input_ctr_x;
        Dtype cty = tg1 * anchor_height + input_ctr_y;
        Dtype ltx = ctx - 0.5 * (tw - bsz01);
        Dtype lty = cty - 0.5 * (th - bsz01);
        Dtype rbx = ltx + tw - bsz01;
        Dtype rby = lty + th - bsz01;

        if (refine_out_of_map_bbox) {
            ltx = min(max(ltx, Dtype(0.0)), input_width -1); 
            lty = min(max(lty, Dtype(0.0)), input_height -1); 
            rbx = min(max(rbx, Dtype(0.0)), input_width -1); 
            rby = min(max(rby, Dtype(0.0)), input_height -1); 
        }

        if (min_size_mode_and_else_or) {
            if ((rbx - ltx + bsz01) < min_size_w 
                    || (rby - lty + bsz01) < min_size_h) {
                conf_data[index] = 0.0;
                continue;
            }
        } else {
            if ((rbx - ltx + bsz01) < min_size_w 
                    && (rby - lty + bsz01) < min_size_h) {
                conf_data[index] = 0.0;
                continue;
            }
        }

        conf_data[index] = score;
        bbox_data[index * 4] = ltx;
        bbox_data[index * 4 + 1] = lty;
        bbox_data[index * 4 + 2] = rbx;
        bbox_data[index * 4 + 3] = rby;

    }
}

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
        Dtype* conf_data, Dtype* bbox_data) {
    int threads = num_anchors * map_height * map_width;
    rpn_cmp_conf_bbox_kernel<Dtype><<<CAFFE_GET_BLOCKS(threads), 
        CAFFE_CUDA_NUM_THREADS>>>(threads, num_anchors,
                map_height, map_width,
                input_height, input_width,
                heat_map_a, heat_map_b,
                allow_border, allow_border_ratio,
                min_size_w, min_size_h,
                min_size_mode_and_else_or, thr_obj,
                bsz01, do_bbox_norm,
                mean0, mean1, mean2, mean3,
                std0, std1, std2, std3,
                refine_out_of_map_bbox, anc_data, 
                prob_data, tgt_data, 
                conf_data, bbox_data);
    CUDA_POST_KERNEL_CHECK;
}
template void rpn_cmp_conf_bbox_gpu(const int num_anchors, 
        const int map_height, const int map_width,
        const float input_height, const float input_width,
        const float heat_map_a, const float heat_map_b,
        const float allow_border, const float allow_border_ratio,
        const float min_size_w, const float min_size_h,
        const bool min_size_mode_and_else_or, const float thr_obj,  
        const float bsz01, const bool do_bbox_norm,
        const float mean0, const float mean1, 
        const float mean2, const float mean3,
        const float std0, const float std1,
        const float std2, const float std3,
        const bool refine_out_of_map_bbox, const float* anc_data, 
        const float* prob_data, const float* tgt_data, 
        float* conf_data, float* bbox_data);
template void rpn_cmp_conf_bbox_gpu(const int num_anchors, 
        const int map_height, const int map_width,
        const double input_height, const double input_width,
        const double heat_map_a, const double heat_map_b,
        const double allow_border, const double allow_border_ratio,
        const double min_size_w, const double min_size_h,
        const bool min_size_mode_and_else_or, const double thr_obj,  
        const double bsz01, const bool do_bbox_norm,
        const double mean0, const double mean1, 
        const double mean2, const double mean3,
        const double std0, const double std1,
        const double std2, const double std3,
        const bool refine_out_of_map_bbox, const double* anc_data, 
        const double* prob_data, const double* tgt_data, 
        double* conf_data, double* bbox_data);

// rcnn
template <typename Dtype>
__global__ void rcnn_cmp_conf_bbox_kernel(const int num_rois, 
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
        const Dtype* tgt_data, Dtype* conf_data, 
        Dtype* bbox_data) {
    int probs_dim = num_class + 1;
    int cords_dim = (regress_agnostic ? 2 : (num_class + 1)) * 4;
    CUDA_KERNEL_LOOP(index, num_rois) {
        const Dtype* probs = prob_data + index * probs_dim;
        const Dtype* cords = tgt_data + index * cords_dim;
        const Dtype* rois = rois_data + index * 5;

        if ((1.0 - probs[0]) < thr_obj) {
            conf_data[index] = 0.0;
            continue;
        }

        if (int(rois[0]) == -1) {
            conf_data[index] = 0.0;
            continue;
        }

        Dtype score_max = -10e6;
        int cls_max = -1;
        for (int c = 0; c < num_class; c++) {
            Dtype score_c = probs[c + 1] - thr_cls[c];
            if (score_c > score_max) {
                score_max = score_c;
                cls_max = c;
            }
        }
        if (score_max < 0) {
            conf_data[index] = 0.0;
            continue;
        }

        if (allow_border >= 0.0
                || allow_border_ratio >= 0.0) {
            Dtype x1 = rois[1]; 
            Dtype y1 = rois[2]; 
            Dtype x2 = rois[3]; 
            Dtype y2 = rois[4]; 
            if (allow_border >= 0.0 && (
                        x1 < -allow_border || y1 < -allow_border 
                        || x2 > input_width - 1 + allow_border || 
                        y2 > input_height - 1 + allow_border )) {
                conf_data[index] = 0.0;
                continue;
            } else if (allow_border_ratio >= 0.0) {
                Dtype x11 = max(Dtype(0.0), x1);
                Dtype y11 = max(Dtype(0.0), y1);
                Dtype x22 = min(input_width - 1, x2);
                Dtype y22 = min(input_height - 1, y2);
                if ((y22 - y11 + bsz01) * (x22 - x11 + bsz01) 
                        / ((y2 - y1 + bsz01) * (x2 - x1 +bsz01)) 
                        < (1.0 - allow_border_ratio)) {
                    conf_data[index] = 0.0;
                    continue;
                }
            }
        }

        Dtype rois_w = rois[3] - rois[1] + bsz01;
        Dtype rois_h = rois[4] - rois[2] + bsz01;
        Dtype rois_ctr_x = rois[1] + 0.5 * (rois_w - bsz01); 
        Dtype rois_ctr_y = rois[2] + 0.5 * (rois_h - bsz01); 

        int cdst = regress_agnostic ? 4 : ((cls_max + 1) * 4);
        Dtype tg0 = cords[cdst];
        Dtype tg1 = cords[cdst + 1];
        Dtype tg2 = cords[cdst + 2];
        Dtype tg3 = cords[cdst + 3];
        if (do_bbox_norm) {
            tg0 = tg0 * std0 + mean0;
            tg1 = tg1 * std1 + mean1;
            tg2 = tg2 * std2 + mean2;
            tg3 = tg3 * std3 + mean3;
        }
        Dtype tw = rois_w * exp(tg2);
        Dtype th = rois_h * exp(tg3);

        Dtype ctx = tg0 * rois_w + rois_ctr_x;
        Dtype cty = tg1 * rois_h + rois_ctr_y;
        Dtype ltx = ctx - 0.5 * (tw - bsz01);
        Dtype lty = cty - 0.5 * (th - bsz01);
        Dtype rbx = ltx + tw - bsz01;
        Dtype rby = lty + th - bsz01;

        if (refine_out_of_map_bbox) {
            ltx = min(max(ltx, Dtype(0.0)), input_width -1); 
            lty = min(max(lty, Dtype(0.0)), input_height -1); 
            rbx = min(max(rbx, Dtype(0.0)), input_width -1); 
            rby = min(max(rby, Dtype(0.0)), input_height -1); 
        }

        if (min_size_mode_and_else_or) {
            if ((rbx - ltx + bsz01) < min_size_w 
                    || (rby - lty + bsz01) < min_size_h) {
                conf_data[index] = 0.0;
                continue;
            }
        } else {
            if ((rbx - ltx + bsz01) < min_size_w 
                    && (rby - lty + bsz01) < min_size_h) {
                conf_data[index] = 0.0;
                continue;
            }
        }

        conf_data[index] = probs[cls_max + 1];
        bbox_data[index * 4] = ltx;
        bbox_data[index * 4 + 1] = lty;
        bbox_data[index * 4 + 2] = rbx;
        bbox_data[index * 4 + 3] = rby;
    }
}

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
        const Dtype* tgt_data, Dtype* conf_data, 
        Dtype* bbox_data) {
    int threads = num_rois;
    rcnn_cmp_conf_bbox_kernel<Dtype><<<CAFFE_GET_BLOCKS(threads), 
        CAFFE_CUDA_NUM_THREADS>>>(num_rois,
                input_height, input_width,
                allow_border, allow_border_ratio,
                min_size_w, min_size_h,
                min_size_mode_and_else_or, thr_obj,  
                bsz01, do_bbox_norm,
                mean0, mean1, 
                mean2, mean3,
                std0, std1,
                std2, std3,
                refine_out_of_map_bbox, regress_agnostic,
                num_class, thr_cls,
                rois_data, prob_data, 
                tgt_data, conf_data, 
                bbox_data);
    CUDA_POST_KERNEL_CHECK;
}
template void rcnn_cmp_conf_bbox_gpu(const int num_rois, 
        const double input_height, const double input_width,
        const double allow_border, const double allow_border_ratio,
        const double min_size_w, const double min_size_h,
        const bool min_size_mode_and_else_or, const double thr_obj,  
        const double bsz01, const bool do_bbox_norm,
        const double mean0, const double mean1, 
        const double mean2, const double mean3,
        const double std0, const double std1,
        const double std2, const double std3,
        const bool refine_out_of_map_bbox, const bool regress_agnostic,
        const int num_class, const double* thr_cls,
        const double* rois_data, const double* prob_data, 
        const double* tgt_data, double* conf_data, 
        double* bbox_data);
template void rcnn_cmp_conf_bbox_gpu(const int num_rois, 
        const float input_height, const float input_width,
        const float allow_border, const float allow_border_ratio,
        const float min_size_w, const float min_size_h,
        const bool min_size_mode_and_else_or, const float thr_obj,  
        const float bsz01, const bool do_bbox_norm,
        const float mean0, const float mean1, 
        const float mean2, const float mean3,
        const float std0, const float std1,
        const float std2, const float std3,
        const bool refine_out_of_map_bbox, const bool regress_agnostic,
        const int num_class, const float* thr_cls,
        const float* rois_data, const float* prob_data, 
        const float* tgt_data, float* conf_data, 
        float* bbox_data);

// nms, copy and modify some cuda codes form yolo
template <typename Dtype>
__host__ __device__ Dtype bbox_size_gpu(const Dtype *bbox, const Dtype bsz01) {
    if (bbox[2] < bbox[0] || bbox[3] < bbox[1]) {
        return Dtype(0.);
    } else {
        return (bbox[2] - bbox[0] + bsz01) * (bbox[3] - bbox[1] + bsz01);
    }
}

template <typename Dtype>
__host__ __device__ Dtype jaccard_overlap_gpu(const Dtype *bbox1,
        const Dtype *bbox2, const Dtype bsz01) {
    if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] ||
            bbox2[1] > bbox1[3] || bbox2[3] < bbox1[1]) {
        return Dtype(0.);
    } else {
        const Dtype inter_xmin = max(bbox1[0], bbox2[0]);
        const Dtype inter_ymin = max(bbox1[1], bbox2[1]);
        const Dtype inter_xmax = min(bbox1[2], bbox2[2]);
        const Dtype inter_ymax = min(bbox1[3], bbox2[3]);

        const Dtype inter_width = inter_xmax - inter_xmin + bsz01;
        const Dtype inter_height = inter_ymax - inter_ymin + bsz01;
        const Dtype inter_size = inter_width * inter_height;

        const Dtype bbox1_size = bbox_size_gpu(bbox1, bsz01);
        const Dtype bbox2_size = bbox_size_gpu(bbox2, bsz01);

        return inter_size / (bbox1_size + bbox2_size - inter_size);
    }
}

template <typename Dtype>
__global__ void compute_overlapped_by_idx_kernel(
        const int nthreads, const Dtype *bbox_data, const int bbox_step,
        const Dtype overlap_threshold, const int *idx, const int num_idx, 
        const Dtype bsz01, bool *overlapped_data) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
            index < (nthreads); index += blockDim.x * gridDim.x) {
        const int j = index % num_idx;
        const int i = index / num_idx;
        if (i == j) {
            // Ignore same bbox.
            return;
        }
        // Compute overlap between i-th bbox and j-th bbox.
        const int start_loc_i = idx[i] * bbox_step;
        const int start_loc_j = idx[j] * bbox_step;
        const Dtype overlap = jaccard_overlap_gpu(bbox_data + start_loc_i,
                bbox_data + start_loc_j,
                bsz01);
        overlapped_data[index] = overlap > overlap_threshold;
    }
}

template <typename Dtype>
void compute_overlapped_by_idx_gpu(
        const int nthreads, const Dtype *bbox_data, const int bbox_step,
        const Dtype overlap_threshold, const int *idx, const int num_idx,
        const Dtype bsz01, bool *overlapped_data) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    const int thread_size = 512;
    int block_size = (nthreads + thread_size - 1) / thread_size;
    compute_overlapped_by_idx_kernel << < block_size, thread_size >> > (
            nthreads, bbox_data, bbox_step, overlap_threshold, idx, num_idx, 
            bsz01, overlapped_data);
}

template <typename Dtype>
void compute_overlapped_by_idx_gpu(
        const int nthreads, const Dtype *bbox_data, const int bbox_step,
        const Dtype overlap_threshold, const int *idx, const int num_idx,
        const Dtype bsz01, bool *overlapped_data, const cudaStream_t &stream) {
    // NOLINT_NEXT_LINE(whitespace/operators)
    const int thread_size = 512;
    int block_size = (nthreads + thread_size - 1) / thread_size;
    compute_overlapped_by_idx_kernel << < block_size, thread_size, 0, stream >> > (
            nthreads, bbox_data, bbox_step, overlap_threshold, idx, num_idx, 
            bsz01, overlapped_data);
}

// Do nms, modified by mingli.
void apply_nms(const bool *overlapped, const int num, const int top_k, 
        const std::vector<int> &idxes, std::vector<int> *indices, 
        const int nmsed_num = 0, const int nmsed_loc = 0) {
    vector<bool> mask(num, false);
    if (nmsed_num > 0) {
        int k_x_num_add_nmsed_num = nmsed_num;
        for (int k = 0; k < nmsed_num; k++) {
            int k_x_num_add_p = k_x_num_add_nmsed_num;
            for (int p = nmsed_num; p < num; p++) {
                if (overlapped[k_x_num_add_p++]) {
                    mask[p] = true;
                }
            }
            k_x_num_add_nmsed_num += num;
        }
    }
    int count = nmsed_num;
    int k_x_num = (nmsed_num -1) * num;
    for (int k = nmsed_num; k < num; k++) {
        k_x_num += num;
        if (mask[k]) {
            continue;
        } else {
            indices->push_back(idxes[nmsed_loc + k - nmsed_num]);
            if (++count >= top_k) {
                break;
            }
            int k_x_num_add_p = k_x_num + k + 1;
            for (int p = k + 1; p < num; p++) {
                if (overlapped[k_x_num_add_p++]) {
                    mask[p] = true;
                }
            }
        }
    }
}

template <typename Dtype>
void apply_nms_gpu(const Dtype *bbox_data, const Dtype *conf_data,
        const int num_bboxes, const int bbox_step, const Dtype confidence_threshold,
        const int max_candidate_n, const int top_k, const Dtype nms_threshold, 
        const Dtype bsz01, std::vector<int> *indices,
        boost::shared_ptr<SyncedMemory> overlapped, boost::shared_ptr<SyncedMemory> idx_sm,
        const cudaStream_t &stream, std::vector<int> *idx_ptr, 
        const int conf_step, const int conf_idx, const int nms_gpu_max_n_per_time) {
    indices->clear();
    std::vector<int> idx;
    std::vector<Dtype> confidences;
    if (idx_ptr == NULL) {
        if (conf_step == 1) {
            for (int i = 0; i < num_bboxes; ++i) {
                if (conf_data[i] > confidence_threshold) {
                    idx.push_back(i);
                    confidences.push_back(conf_data[i]);
                }
            } 
        } else {
            int i_x_step_add_idx = conf_idx;
            for (int i = 0; i < num_bboxes; ++i) {
                if (conf_data[i_x_step_add_idx] > confidence_threshold) {
                    idx.push_back(i);
                    confidences.push_back(conf_data[i_x_step_add_idx]);
                }
                i_x_step_add_idx += conf_step;
            } 
        }
    } else {
        if (conf_step == 1) {
            for (int k = 0; k < idx_ptr->size(); k++) {
                int i = (*idx_ptr)[k];
                if (conf_data[i] > confidence_threshold) {
                    idx.push_back(i);
                    confidences.push_back(conf_data[i]);
                }
            } 
        } else {
            for (int k = 0; k < idx_ptr->size(); k++) {
                int i = (*idx_ptr)[k];
                int i_x_step_add_idx = i * conf_step + conf_idx;
                if (conf_data[i_x_step_add_idx] > confidence_threshold) {
                    idx.push_back(i);
                    confidences.push_back(conf_data[i_x_step_add_idx]);
                }
            }
        }
    }
    int num_remain = confidences.size();
    if (num_remain == 0) {
        return;
    }
    if (nms_threshold >= Dtype(1.0)) {
        for (int i = 0; i < idx.size(); i++) {
            indices->push_back(idx[i]);
        }
        return;
    }

    thrust::sort_by_key(&confidences[0], &confidences[0] + num_remain, &idx[0],
            thrust::greater<Dtype>());
    if (max_candidate_n > -1 && max_candidate_n < num_remain) {
        num_remain = max_candidate_n;
    }

    int idx_loc = 0;
    int indices_size_pre = 0;
    while (idx_loc < num_remain && indices->size() < top_k) {
        int *idx_data = (int *) idx_sm->mutable_cpu_data();
        std::copy(indices->begin() + indices_size_pre, 
                indices->end(), idx_data + indices_size_pre);
        int idx_num_cur_time = min(int(nms_gpu_max_n_per_time - indices->size()),
                int(num_remain - idx_loc));
        std::copy(idx.begin() + idx_loc, idx.begin() + idx_loc + idx_num_cur_time, 
                idx_data + indices->size());
        int candidate_n_cur_time = indices->size() + idx_num_cur_time;
        int total_bboxes = candidate_n_cur_time * candidate_n_cur_time;
        bool *overlapped_data = (bool *) overlapped->mutable_gpu_data();
        compute_overlapped_by_idx_gpu(total_bboxes, bbox_data, bbox_step, 
                nms_threshold, (const int *) idx_sm->gpu_data(), 
                candidate_n_cur_time, bsz01, overlapped_data, stream);
        const bool *overlapped_results = (const bool *) overlapped->cpu_data();
        indices_size_pre = indices->size();
        apply_nms(overlapped_results, candidate_n_cur_time, top_k, 
                idx, indices, indices->size(), idx_loc);
        idx_loc += idx_num_cur_time;
    }
}
template void apply_nms_gpu(const float *bbox_data, const float *conf_data,
        const int num_bboxes, const int bbox_step, const float confidence_threshold,
        const int max_candidate_n, const int top_k, const float nms_threshold, 
        const float bsz01, std::vector<int> *indices, 
        boost::shared_ptr<SyncedMemory> overlapped, boost::shared_ptr<SyncedMemory> idx_sm,
        const cudaStream_t &stream, std::vector<int> *idx_ptr,
        const int conf_step, const int conf_idx, const int nms_gpu_max_n_per_time);
template void apply_nms_gpu(const double *bbox_data, const double *conf_data,
        const int num_bboxes, const int bbox_step, const double confidence_threshold,
        const int max_candidate_n, const int top_k, const double nms_threshold, 
        const double bsz01, std::vector<int> *indices,
        boost::shared_ptr<SyncedMemory> overlapped, boost::shared_ptr<SyncedMemory> idx_sm,
        const cudaStream_t &stream, std::vector<int> *idx_ptr,
        const int conf_step, const int conf_idx, const int nms_gpu_max_n_per_time);
}

