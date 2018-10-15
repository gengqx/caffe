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

#ifndef CAFFE_ROI_OUTPUT_SSD_LAYER_HPP_
#define CAFFE_ROI_OUTPUT_SSD_LAYER_HPP_

#include <string>
#include <utility>

#include "caffe/proto/caffe.pb.h"

#include "caffe/util/util_others.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/util_img.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/io.hpp"

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"


namespace caffe
{

    template <typename Dtype>
    class ROIOutputSSDLayer : public Layer <Dtype> {
        public:
            explicit ROIOutputSSDLayer(const LayerParameter& param) :
                                                          Layer<Dtype>(param) {
            }
            virtual ~ROIOutputSSDLayer(){
            };
            virtual void LayerSetUp(const std::vector<Blob<Dtype>*>& bottom, 
                    const std::vector<Blob<Dtype>*>& top);//check
            virtual void Reshape(const std::vector<Blob<Dtype>*>& bottom, 
                    const std::vector<Blob<Dtype>*>& top){
            };

            virtual inline const char* type() const { 
                return "ROIOutputSSD"; 
            }
            virtual inline int  MinBottomBlobs() const { 
                return 1; 
            }
            virtual inline int  MaxBottomBlobs() const { 
                return 2; 
            }
            virtual inline int  ExactNumTopBlobs() const { 
                return 2; 
            }

            virtual void Forward_cpu(const std::vector<Blob<Dtype>*>& bottom, 
                    const std::vector<Blob<Dtype>*>& top){
            };
            virtual void Forward_gpu(const std::vector<Blob<Dtype>*>& bottom, 
                    const std::vector<Blob<Dtype>*>& top){
            };

            virtual void Backward_cpu(const std::vector<Blob<Dtype>*>& top, 
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob<Dtype>*>& bottom) {
            }
            virtual void Backward_gpu(const std::vector<Blob<Dtype>*>& top, 
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob<Dtype>*>& bottom) {
            }

            inline std::vector<BBox<Dtype> >& GetFilteredBBox(int class_id) {
                return output_bboxes_[class_id];
            }
            inline std::vector<std::vector< BBox<Dtype> > >& GetFilteredBBox() {
                return output_bboxes_;
            }
            inline int GetNumClass() {
                return num_class_;
            }
            inline std::vector<string> GetClassNames() {
                return class_names_;
            }

        protected:
            std::vector<std::vector<BBox<Dtype> > > all_candidate_bboxes_;
            std::vector<bool> is_candidate_bbox_selected_;
            std::vector< std::vector< BBox<Dtype> > > output_bboxes_;
            //int bbox_data_size, bbox_info_size;
            std::vector<Dtype> threshold_;
            bool nms_need_nms_;
            std::vector<Dtype> nms_overlap_ratio_;
            std::vector<int> nms_top_n_;
            // added by liming
            int nms_gpu_max_n_per_time_;
            std::vector<int> nms_max_candidate_n_;
            std::vector<bool> nms_use_soft_nms_;
            Dtype threshold_objectness_;
            bool nms_among_classes_;
            std::vector<bool> nms_voting_;
            std::vector<Dtype> nms_vote_iou_;
            bool nms_add_score_;
            bool refine_out_of_map_bbox_;
            int channel_per_scale_;
            int num_class_;
            std::vector<string> class_names_;
            std::vector<int> class_indexes_;
            std::vector<Dtype> heat_map_a_vec_;
            std::vector<Dtype> heat_map_b_vec_;
            std::vector<Dtype> anchor_width_vec_;
            std::vector<Dtype> anchor_height_vec_;
            std::vector<Dtype> anchor_x1_vec_;
            std::vector<Dtype> anchor_y1_vec_;
            std::vector<Dtype> anchor_x2_vec_;
            std::vector<Dtype> anchor_y2_vec_;
            std::vector<Dtype> proposal_min_area_vec_;
            std::vector<Dtype> proposal_max_area_vec_;
            bool bg_as_one_of_softmax_;
            bool use_target_type_rcnn_;
            bool do_bbox_norm_;
            std::vector<Dtype> bbox_means_;
            std::vector<Dtype> bbox_stds_;
            Dtype im_width_;
            Dtype im_height_;
            bool rpn_proposal_output_score_;
            bool regress_agnostic_;
            bool show_time_;
            Dtype time_get_bbox_, time_total_, time_nms_, time_bbox_to_blob_;
            Timer timer_get_bbox_;
            Timer timer_total_;
            Timer timer_nms_;
            Timer timer_bbox_to_blob_;
            Dtype allow_border_;
            Dtype allow_border_ratio_;
            bool bbox_size_add_one_;
            Dtype read_width_scale_;
            Dtype read_height_scale_;
            unsigned int read_height_offset_;
            bool zero_anchor_center_;
            Dtype min_size_h_;
            Dtype min_size_w_;
            DetectionOutputSSDParameter_MIN_SIZE_MODE min_size_mode_;
            std::vector<Dtype> reg_means_;
            std::vector<Dtype> reg_stds_;
            //kpts params
            bool has_kpts_;
            bool kpts_reg_as_classify_;
            int kpts_exist_bottom_idx_;
            int kpts_reg_bottom_idx_;
            int kpts_classify_width_;
            int kpts_classify_height_;
            bool kpts_do_norm_;
            int kpts_reg_norm_idx_st_;
            std::vector<int> kpts_st_for_each_class_;
            std::vector<int> kpts_ed_for_each_class_;
            Dtype kpts_classify_pad_ratio_;
            //atrs params
            bool has_atrs_;
            int atrs_reg_bottom_idx_;
            bool atrs_do_norm_;
            int atrs_reg_norm_idx_st_;
            std::vector<ATRSParameter_NormType> atrs_norm_type_;
            //ftrs params
            bool has_ftrs_;
            int ftrs_bottom_idx_;
            //spmp params
            bool has_spmp_;
            int spmp_bottom_idx_;
            int num_spmp_; 
            std::vector<bool> spmp_class_aware_;
            std::vector<int> spmp_label_width_;
            std::vector<int> spmp_label_height_;
            std::vector<Dtype> spmp_pad_ratio_;
            std::vector<int> spmp_dim_st_;
            std::vector<int> spmp_dim_;
            int spmp_dim_sum_;
            //cam3d params
            bool has_cam3d_;
            int cam3d_bottom_idx_;
    };

}  // namespace caffe

#endif  // CAFFE_ROI_OUTPUT_SSD_LAYER_HPP_
