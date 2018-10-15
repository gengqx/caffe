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

#include "caffe/layers/rcnn_proposal_layer.hpp"
#include "caffe/util/dtout.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{

template <typename Dtype>
void RCNNProposalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top)//check
{
    Dtype input_height = this->im_height_, input_width = this->im_width_;
    Dtype min_size_w_cur = this->min_size_w_;
    Dtype min_size_h_cur = this->min_size_h_;
    vector<Dtype> im_width_scale = vector<Dtype>(1, this->read_width_scale_);
    vector<Dtype> im_height_scale = vector<Dtype>(1, this->read_height_scale_);
    vector<Dtype> cords_offset_x = vector<Dtype>(1, Dtype(0));
    vector<Dtype> cords_offset_y = vector<Dtype>(1, this->read_height_offset_);
    CHECK_EQ(bottom.back()->count(1), 6);
    const Dtype* img_info_data = bottom.back()->cpu_data();
    input_width = img_info_data[0];
    input_height = img_info_data[1];
    CHECK_GT(input_width, 0);
    CHECK_GT(input_height, 0);
    im_width_scale.clear();
    im_height_scale.clear();
    cords_offset_x.clear();
    cords_offset_y.clear();
    for (int n = 0; n < bottom.back()->num(); n++) {
        im_width_scale.push_back(img_info_data[n * 6 + 2]);
        im_height_scale.push_back(img_info_data[n * 6 + 3]);
        CHECK_GT(im_width_scale[n], 0);
        CHECK_GT(im_height_scale[n], 0);
        cords_offset_x.push_back(img_info_data[n * 6 + 4]);
        cords_offset_y.push_back(img_info_data[n * 6 + 5]);
    }

    Dtype bsz01 = this->bbox_size_add_one_ ? Dtype(1.0) : Dtype(0.0);

    Dtype min_size_mode_and_else_or = true;
    if (this->min_size_mode_ ==
                   DetectionOutputSSDParameter_MIN_SIZE_MODE_HEIGHT_OR_WIDTH) {
        min_size_mode_and_else_or = false;
    } else {
        CHECK(this->min_size_mode_ ==
                    DetectionOutputSSDParameter_MIN_SIZE_MODE_HEIGHT_AND_WIDTH);
    }

    const int num_rois = bottom[0]->num();
    const int probs_dim = bottom[0]->channels();
    const int cords_dim = bottom[1]->channels();
    const int pre_rois_dim = bottom[2]->channels();
    CHECK_EQ(num_rois, bottom[1]->num());
    CHECK_EQ(num_rois, bottom[2]->num());
    CHECK_EQ(probs_dim, this->num_class_ + 1);
    if (this->regress_agnostic_) {
        CHECK_EQ(cords_dim, 2 * 4);
    } else {
        CHECK_EQ(cords_dim, (this->num_class_ + 1) * 4);
    }
    CHECK_EQ(pre_rois_dim, 5); // imid, x1, y1, x2, y2

    const Dtype* prob_gpu_data = bottom[0]->gpu_data();
    const Dtype* tgt_gpu_data = bottom[1]->gpu_data();
    const Dtype* rois_gpu_data = bottom[2]->gpu_data();
    dt_conf_.Reshape(num_rois, 1, 1, 1);
    dt_bbox_.Reshape(num_rois, 4, 1, 1);
    Dtype* conf_gpu_data = dt_conf_.mutable_gpu_data();
    Dtype* bbox_gpu_data = dt_bbox_.mutable_gpu_data();

    rcnn_cmp_conf_bbox_gpu(num_rois,
            input_height, input_width,
            this->allow_border_, this->allow_border_ratio_,
            min_size_w_cur, min_size_h_cur,
            min_size_mode_and_else_or, this->threshold_objectness_,
            bsz01, this->do_bbox_norm_,
            this->bbox_means_[0], this->bbox_means_[1],
            this->bbox_means_[2], this->bbox_means_[3],
            this->bbox_stds_[0], this->bbox_stds_[1],
            this->bbox_stds_[2], this->bbox_stds_[3],
            this->refine_out_of_map_bbox_, this->regress_agnostic_,
            this->num_class_, thr_cls_.gpu_data(), 
            rois_gpu_data, prob_gpu_data, tgt_gpu_data,
            conf_gpu_data, bbox_gpu_data);

    const Dtype* prob_data = bottom[0]->cpu_data();
    const Dtype* rois_data = bottom[2]->cpu_data();
    const Dtype* conf_data = dt_conf_.cpu_data();
    const Dtype* bbox_data = dt_bbox_.cpu_data();

    // cmp valid idxes per img
    vector<vector<int> > idx_per_img_vec;
    for (int i = 0; i < num_rois; i++) {
        if (conf_data[i] == Dtype(0.0)) {
            continue;
        }
        int imid = int(rois_data[i * 5]);
        if (imid + 1 > idx_per_img_vec.size()) {
            idx_per_img_vec.resize(imid + 1);
        }
        idx_per_img_vec[imid].push_back(i);
    }

    vector<vector<BBox<Dtype> > > proposal_per_class(this->num_class_);
    vector<vector<vector<Dtype> > > proposal_batch_vec(top.size());
    if (top.size() != 0 || this->nms_among_classes_) {
        for (int imid = 0; imid < idx_per_img_vec.size(); imid++) {
            if (idx_per_img_vec[imid].size() == 0) {
                continue;
            }
            std::vector<int> indices;
            apply_nms_gpu(bbox_gpu_data, conf_data, num_rois, 4,
                    Dtype(0.0), this->nms_max_candidate_n_[0], 
                    this->nms_top_n_[0], this->nms_overlap_ratio_[0], 
                    bsz01, &indices, overlapped_, idx_sm_, stream_, 
                    &idx_per_img_vec[imid], 1, 0,
                    this->nms_gpu_max_n_per_time_);
            if (top.size() == 0) {
                for (int k = 0; k < indices.size(); k++) {
                    BBox<Dtype> bbox;
                    bbox.id = imid;
                    int idk = indices[k];
                    int idkx4 = idk * 4;
                    int imid_cur = im_width_scale.size() > 1 ? imid : 0;
                    CHECK_LT(imid_cur, im_width_scale.size());
                    bbox.x1 = bbox_data[idkx4] / im_width_scale[imid_cur] 
                        + cords_offset_x[imid_cur];
                    bbox.y1 = bbox_data[idkx4 + 1] / im_height_scale[imid_cur] 
                        + cords_offset_y[imid_cur]; 
                    bbox.x2 = bbox_data[idkx4 + 2] / im_width_scale[imid_cur] 
                        + cords_offset_x[imid_cur]; 
                    bbox.y2 = bbox_data[idkx4 + 3] / im_height_scale[imid_cur] 
                        + cords_offset_y[imid_cur]; 
                    const Dtype* probs = prob_data + idk * probs_dim;
                    for (int c = 0; c < this->num_class_; ++c) {
                        if (probs[c + 1] < this->threshold_[c]) {
                            continue;
                        }
                        bbox.score = probs[c + 1];
                        proposal_per_class[c].push_back(bbox);
                    }
                }
            } else if (top.size() == 1) {
                for (int k = 0; k < indices.size(); k++) {
                    vector<Dtype> bbox(6 + probs_dim, 0);
                    bbox[0] = imid;
                    int idk = indices[k];
                    int idkx4 = idk * 4;
                    bbox[1] = conf_data[idk];
                    bbox[2] = bbox_data[idkx4];
                    bbox[3] = bbox_data[idkx4 + 1];
                    bbox[4] = bbox_data[idkx4 + 2];
                    bbox[5] = bbox_data[idkx4 + 3];
                    const Dtype* probs = prob_data + idk * probs_dim;
                    for (int c = 0; c < probs_dim; ++c) {
                        bbox[6 + c] = probs[c];
                    }
                    proposal_batch_vec[0].push_back(bbox);
                }
            } else {
                for (int k = 0; k < indices.size(); k++) {
                    vector<Dtype> bbox(6 + probs_dim, 0);
                    bbox[0] = imid;
                    int idk = indices[k];
                    int idkx4 = idk * 4;
                    bbox[1] = conf_data[idk];
                    bbox[2] = bbox_data[idkx4];
                    bbox[3] = bbox_data[idkx4 + 1];
                    bbox[4] = bbox_data[idkx4 + 2];
                    bbox[5] = bbox_data[idkx4 + 3];
                    const Dtype* probs = prob_data + idk * probs_dim;
                    for (int c = 0; c < probs_dim; ++c) {
                        bbox[6 + c] = probs[c];
                    }
                    Dtype bw = bbox[4] - bbox[2] + bsz01;
                    Dtype bh = bbox[5] - bbox[3] + bsz01;
                    Dtype bwxh = bw * bh;
                    for(int t = 0; t < top.size(); t++) {
                        if(bwxh > this->proposal_min_area_vec_[t] 
                                && bwxh < this->proposal_max_area_vec_[t]) {
                            proposal_batch_vec[t].push_back(bbox);
                        }
                    }
                }
            }
        }
    } else {
        for (int imid = 0; imid < idx_per_img_vec.size(); imid++) {
            if (idx_per_img_vec[imid].size() == 0) {
                continue;
            }
            for (int c = 0; c < this->num_class_; ++c) {
                std::vector<int> indices;
                apply_nms_gpu(bbox_gpu_data, prob_data, num_rois, 4,
                        this->threshold_[c], this->nms_max_candidate_n_[c], 
                        this->nms_top_n_[c], this->nms_overlap_ratio_[c], 
                        bsz01, &indices, overlapped_, idx_sm_, stream_, 
                        &idx_per_img_vec[imid], probs_dim, c + 1,
                        this->nms_gpu_max_n_per_time_);
                for (int k = 0; k < indices.size(); k++) {
                    BBox<Dtype> bbox;
                    bbox.id = imid;
                    int idk = indices[k];
                    int idkx4 = idk * 4;
                    int imid_cur = im_width_scale.size() > 1 ? imid : 0;
                    CHECK_LT(imid_cur, im_width_scale.size());
                    bbox.x1 = bbox_data[idkx4] / im_width_scale[imid_cur] 
                        + cords_offset_x[imid_cur];
                    bbox.y1 = bbox_data[idkx4 + 1] / im_height_scale[imid_cur] 
                        + cords_offset_y[imid_cur]; 
                    bbox.x2 = bbox_data[idkx4 + 2] / im_width_scale[imid_cur] 
                        + cords_offset_x[imid_cur]; 
                    bbox.y2 = bbox_data[idkx4 + 3] / im_height_scale[imid_cur] 
                        + cords_offset_y[imid_cur]; 
                    const Dtype* probs = prob_data + idk * probs_dim;
                    bbox.score = probs[c + 1];
                    proposal_per_class[c].push_back(bbox);
                }
            }
        }
    }

    if (top.size() != 0) {
        for(int t = 0; t < top.size(); t++) {
            if(proposal_batch_vec[t].empty()) {
                // for special case when there is no box
                top[t]->Reshape(1, rois_dim_, 1, 1);
                Dtype* top_boxes_scores = top[t]->mutable_cpu_data();
                caffe_set(top[t]->count(), Dtype(0), top_boxes_scores); 
            } else {
                const int top_num = proposal_batch_vec[t].size();
                top[t]->Reshape(top_num, rois_dim_, 1, 1);
                Dtype* top_boxes_scores = top[t]->mutable_cpu_data();
                for (int k = 0; k < top_num; k++) {
                    top_boxes_scores[k*rois_dim_] = proposal_batch_vec[t][k][0];
                    top_boxes_scores[k*rois_dim_+1] =
                                                    proposal_batch_vec[t][k][2];
                    top_boxes_scores[k*rois_dim_+2] =
                                                    proposal_batch_vec[t][k][3];
                    top_boxes_scores[k*rois_dim_+3] =
                                                    proposal_batch_vec[t][k][4];
                    top_boxes_scores[k*rois_dim_+4] =
                                                    proposal_batch_vec[t][k][5];
                    if (this->rpn_proposal_output_score_) {
                        for (int c = 0; c < probs_dim; c++) {
                            top_boxes_scores[k * rois_dim_ + 5 + c] = 
                                proposal_batch_vec[t][k][6 + c];
                        }
                    }
                }
            }
        }
    } else {
        for (int class_id = 0; class_id < this->num_class_; ++class_id) {
            this->output_bboxes_[class_id] = proposal_per_class[class_id];
        }   
    }
}

template <typename Dtype>
void RCNNProposalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, 
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) {}

INSTANTIATE_LAYER_GPU_FUNCS(RCNNProposalLayer);

}  // namespace caffe
