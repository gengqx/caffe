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

#include "caffe/layers/rpn_proposal_ssd_layer.hpp"
#include "caffe/util/dtout.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{

template <typename Dtype>
void RPNProposalSSDLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
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
    if (this->min_size_mode_ == DetectionOutputSSDParameter_MIN_SIZE_MODE_HEIGHT_OR_WIDTH) {
        min_size_mode_and_else_or = false;
    } else {
        CHECK(this->min_size_mode_ == DetectionOutputSSDParameter_MIN_SIZE_MODE_HEIGHT_AND_WIDTH);
    }

    const int num = bottom[0]->num();
    const int map_height = bottom[0]->height();
    const int map_width  = bottom[0]->width();
    const Dtype heat_map_a = this->heat_map_a_vec_[0];
    const Dtype heat_map_b = this->heat_map_b_vec_[0];
    CHECK_EQ(bottom[0]->channels(), num_anchors_ * 2);
    CHECK_EQ(bottom[1]->num(), num);
    CHECK_EQ(bottom[1]->channels(), num_anchors_ * 4);
    CHECK_EQ(bottom[1]->height(), map_height);
    CHECK_EQ(bottom[1]->width(), map_width);

    const Dtype* prob_gpu_data = bottom[0]->gpu_data();
    const Dtype* tgt_gpu_data = bottom[1]->gpu_data();

    int num_bboxes = num_anchors_ * map_height * map_width;
    dt_conf_ahw_.Reshape(num_bboxes, 1, 1, 1);
    dt_bbox_ahw_.Reshape(num_bboxes, 4, 1, 1);

    vector<BBox<Dtype> > proposal_all;
    vector<vector<vector<Dtype> > > proposal_batch_vec(top.size());
    for (int i = 0; i < num; ++i) {
        //Timer tm;
        //tm.Start();
        rpn_cmp_conf_bbox_gpu(num_anchors_,
                map_height, map_width,
                input_height, input_width,
                heat_map_a, heat_map_b,
                this->allow_border_, this->allow_border_ratio_,
                min_size_w_cur, min_size_h_cur,
                min_size_mode_and_else_or, this->threshold_objectness_,
                bsz01, this->do_bbox_norm_,
                this->bbox_means_[0], this->bbox_means_[1],
                this->bbox_means_[2], this->bbox_means_[3],
                this->bbox_stds_[0], this->bbox_stds_[1],
                this->bbox_stds_[2], this->bbox_stds_[3],
                this->refine_out_of_map_bbox_, anc_.gpu_data(), 
                prob_gpu_data + bottom[0]->offset(i, 0, 0, 0), 
                tgt_gpu_data + bottom[0]->offset(i, 0, 0, 0), 
                dt_conf_ahw_.mutable_gpu_data(),
                dt_bbox_ahw_.mutable_gpu_data());
        //LOG(INFO)<<"nms rpn_cmp_conf_bbox time: "<<tm.MilliSeconds();
        //tm.Start();

        //do nms by gpu
        const Dtype* conf_data = dt_conf_ahw_.cpu_data();
        const Dtype* bbox_gpu_data = dt_bbox_ahw_.gpu_data();
        std::vector<int> indices;
        apply_nms_gpu(bbox_gpu_data, conf_data, num_bboxes, 4,
                Dtype(0.0), this->nms_max_candidate_n_[0], 
                this->nms_top_n_[0], this->nms_overlap_ratio_[0], 
                bsz01, &indices, overlapped_, idx_sm_, stream_,
                NULL, 1, 0, this->nms_gpu_max_n_per_time_);
        //LOG(INFO)<<"nms apply_nms_gpu time: "<<tm.MilliSeconds();

        const Dtype* bbox_data = dt_bbox_ahw_.cpu_data();
        if (top.size() == 0) {
            for (int k = 0; k < indices.size(); k++) {
                BBox<Dtype> bbox;
                bbox.id = i;
                int idk = indices[k];
                int idkx4 = idk * 4;
                bbox.score = conf_data[idk];
                int imid_cur = im_width_scale.size() > 1 ? i : 0;
                CHECK_LT(imid_cur, im_width_scale.size());
                bbox.x1 = bbox_data[idkx4] / im_width_scale[imid_cur] 
                    + cords_offset_x[imid_cur];
                bbox.y1 = bbox_data[idkx4 + 1] / im_height_scale[imid_cur] 
                    + cords_offset_y[imid_cur]; 
                bbox.x2 = bbox_data[idkx4 + 2] / im_width_scale[imid_cur] 
                    + cords_offset_x[imid_cur]; 
                bbox.y2 = bbox_data[idkx4 + 3] / im_height_scale[imid_cur] 
                    + cords_offset_y[imid_cur]; 
                proposal_all.push_back(bbox);
            }
        } else if (top.size() == 1) {
            for (int k = 0; k < indices.size(); k++) {
                vector<Dtype> bbox(6, 0);
                bbox[0] = i;
                int idk = indices[k];
                int idkx4 = idk * 4;
                bbox[1] = conf_data[idk];
                bbox[2] = bbox_data[idkx4];
                bbox[3] = bbox_data[idkx4 + 1];
                bbox[4] = bbox_data[idkx4 + 2];
                bbox[5] = bbox_data[idkx4 + 3];
                proposal_batch_vec[0].push_back(bbox);
            }
        } else {
            for (int k = 0; k < indices.size(); k++) {
                vector<Dtype> bbox(6, 0);
                bbox[0] = i;
                int idk = indices[k];
                int idkx4 = idk * 4;
                bbox[1] = conf_data[idk];
                bbox[2] = bbox_data[idkx4];
                bbox[3] = bbox_data[idkx4 + 1];
                bbox[4] = bbox_data[idkx4 + 2];
                bbox[5] = bbox_data[idkx4 + 3];
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
                top_boxes_scores[k*rois_dim_+1] = proposal_batch_vec[t][k][2];
                top_boxes_scores[k*rois_dim_+2] = proposal_batch_vec[t][k][3];
                top_boxes_scores[k*rois_dim_+3] = proposal_batch_vec[t][k][4];
                top_boxes_scores[k*rois_dim_+4] = proposal_batch_vec[t][k][5];
                if (this->rpn_proposal_output_score_) {
                    top_boxes_scores[k*rois_dim_+5] = proposal_batch_vec[t][k][1];
                }
            }
        }
    }

    if (top.size() == 0) {
        for (int class_id = 0; class_id < this->num_class_; ++class_id) {
            this->output_bboxes_[class_id] = proposal_all;
        }
    }
}

template <typename Dtype>
void RPNProposalSSDLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, 
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

INSTANTIATE_LAYER_GPU_FUNCS(RPNProposalSSDLayer);

}  // namespace caffe
