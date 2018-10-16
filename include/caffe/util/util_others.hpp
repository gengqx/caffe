
#ifndef CAFFE_UTIL_UTIL_OTHERS_HPP_
#define CAFFE_UTIL_UTIL_OTHERS_HPP_

#include <string>
#include <map>
#include <utility>
#include <vector>

#include "caffe/common.hpp"

#include "caffe/proto/caffe.pb.h"

using namespace std;

namespace caffe
{

template <typename Dtype>
struct InfoCam3d 
{
    InfoCam3d() {
        x = y = z = h = w = l = o = 0;
    }
    Dtype x;
    Dtype y;
    Dtype z;
    Dtype h;
    Dtype w;
    Dtype l;
    Dtype o;
    vector<vector<Dtype> > pts3d; 
    vector<vector<Dtype> > pts2d; 
};

// 0410Rui //
template <typename Dtype>
struct BBox
{
    BBox()
    {
        id = center_h = center_w = score = x1 = x2 = y1 = y2 = 0;
    }
    Dtype score, x1, y1, x2, y2, center_h, center_w, id;

    Dtype fdlx, fdly, fdrx, fdry, bdrx, bdry, bdlx, bdly;
    Dtype fulx, fuly, furx, fury, burx, bury, bulx, buly;
    Dtype l_3d, w_3d, h_3d;
    Dtype thl, yaw;

    // tracking fea extra //
    int scale_id;
    int heat_map_y;
    int heat_map_x;

    std::vector<Dtype> data;

    vector<Dtype> prbs;
    vector<Dtype> ftrs;
    vector<Dtype> atrs;
    vector<std::pair<Dtype, Dtype> > kpts;
    vector<Dtype> kpts_prbs;
    //spatial maps for each instance
    vector<vector<Dtype> > spmp;
    InfoCam3d<Dtype> cam3d;

    static bool greater(const BBox<Dtype>& a, const BBox<Dtype>& b)
                                                    {return a.score > b.score;}
};

/////////////////////////////////////////////////////////
vector<string> std_split(string str, string reg);

/////////////////////////////////////////////////////////
template <typename Dtype>
Dtype GetArea(const vector<Dtype>& bbox);
template <typename Dtype>
Dtype GetArea(const Dtype x1, const Dtype y1, const Dtype x2, const Dtype y2);

// intersection over union
enum OverlapType{ OVERLAP_UNION,OVERLAP_BOX1, OVERLAP_BOX2 };
template <typename Dtype>
Dtype GetOverlap(const vector<Dtype>& bbox1, const vector<Dtype>& bbox2);
template <typename Dtype>
Dtype GetOverlap(const Dtype x11, const Dtype y11, const Dtype x12,
                 const Dtype y12, const Dtype x21, const Dtype y21,
                 const Dtype x22, const Dtype y22,
                 const OverlapType overlap_type);

/////////////////////////////////////////////////////////
template <typename Dtype>
bool compareCandidate(const pair<Dtype, vector<float> >& c1,
                      const pair<Dtype, vector<float> >& c2);
template <typename Dtype>
bool compareCandidate_v2(const vector<Dtype>  & c1, const  vector<Dtype>  & c2);

/*    Non-maximum suppression. return a mask which elements are selected overlap
*   Overlap threshold for suppression For a selected box Bi, all boxes Bj that are covered by more than overlap are suppressed.
*   Note that 'covered' is is |Bi \cap Bj| / |Bj|, not the PASCAL intersection over union measure.
*     if addscore == true, then the scores of all the overlap bboxes will be added
*/
template <typename Dtype>
const vector<bool> nms(vector<pair<Dtype, vector<float> > >& candidates,
                       const float overlap, const int top_N,
                       const bool addScore = false);

//    Non-maximum suppression. return a mask which elements are selected
//  overlap   Overlap threshold for suppression
//  For a selected box Bi, all boxes Bj that are covered by
//  more than overlap are suppressed. Note that 'covered' is
//  is |Bi \cap Bj| / min(|Bj|,|Bi|), n
//     if addscore == true, then the scores of all the overlap bboxes will be added
template <typename Dtype>
const vector<bool> nms(vector<vector<Dtype> >& candidates, const Dtype overlap,
                        const int top_N, const bool addScore = false);
template <typename Dtype>
const vector<bool> nms(vector< BBox<Dtype> >& candidates, const Dtype overlap,
                        const int top_N, const bool addScore = false);

/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
template <typename Dtype>
void PushBBoxTo(std::ofstream & out_result_file,
                const vector< BBox<Dtype> >& bboxes, bool with_cam3d = false);

/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////
/* This function return a vector showing whether the candidate is correct according to overlap ratio. */
vector<bool> GetPredictedResult(
         const vector< std::pair<int, vector<float> > > &gt_instances,
         const vector< std::pair<float, vector<float> > > &pred_instances,
         float ratio = 0.5);

/////////////////////////////////////////////////////////
float GetPRPoint_FDDB(
    vector< std::pair<float, vector<float> > >& pred_instances_with_gt,
          const int n_positive,vector<float>& precision,vector<float> &recall);
    
//  from mscnn
template <typename Dtype>
Dtype BoxIOU(const Dtype x1, const Dtype y1, const Dtype w1, const Dtype h1,
    const Dtype x2, const Dtype y2, const Dtype w2, const Dtype h2, 
    const string mode, bool bbox_size_add_one = false);

template <typename Dtype>
void coords2targets(const Dtype ltx, const Dtype lty, const Dtype rbx,
    const Dtype rby, 
    const Dtype acx, const Dtype acy, const Dtype acw, const Dtype ach,
    const bool use_target_type_rcnn, const bool do_bbox_norm, 
    const vector<Dtype>& bbox_means, const vector<Dtype>& bbox_stds,
    Dtype& tg0, Dtype& tg1, Dtype& tg2, Dtype& tg3,
    bool bbox_size_add_one = false);

template <typename Dtype>
void targets2coords(const Dtype tg0, const Dtype tg1, const Dtype tg2,
    const Dtype tg3, 
    const Dtype acx, const Dtype acy, const Dtype acw, const Dtype ach,
    const bool use_target_type_rcnn, const bool do_bbox_norm, 
    const vector<Dtype>& bbox_means, const vector<Dtype>& bbox_stds,
    Dtype& ltx, Dtype& lty, Dtype& rbx, Dtype& rby,
    bool bbox_size_add_one = false);

// nms 
template <typename Dtype>
const vector<bool> nms_lm(vector< BBox<Dtype> >& candidates, 
    const Dtype overlap, const int top_N, const bool addScore = false, 
    const int max_candidate_N = -1, bool bbox_size_add_one = true,
    bool voting = true, Dtype vote_iou = 0.5);

// soft nms
template <typename Dtype>
const vector<bool> soft_nms_lm(vector< BBox<Dtype> >& candidates, 
    const Dtype iou_std, const int top_N, 
    const int max_candidate_N = -1, bool bbox_size_add_one = true,
    bool voting = true, Dtype vote_iou = 0.5);

template <typename Dtype>
void coef2dTo3d(Dtype cam_xpz, Dtype cam_xct, Dtype cam_ypz, 
    Dtype cam_yct, Dtype cam_pitch, Dtype px, Dtype py,
    Dtype & k1, Dtype & k2, Dtype & u, Dtype & v);

template <typename Dtype>
void cord2dTo3d(Dtype k1, Dtype k2, Dtype u, 
    Dtype v, Dtype ph, Dtype rh,
    Dtype & x, Dtype & y, Dtype & z);
}  // namespace caffe

#endif  // CAFFE_UTIL_UTIL_OTHERS_HPP_
