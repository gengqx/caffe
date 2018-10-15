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

#include "caffe/util/util_others.hpp"

namespace caffe
{

///////////////////////////////////////////////////////////////
vector<string> std_split(string str, string pattern)
{
	std::string::size_type pos;
	std::vector<std::string> result;
	str+=pattern;
	int size=str.size();

	for(int i=0; i<size; i++)
	{
	pos=str.find(pattern,i);
	if(pos<size)
	{
	  std::string s=str.substr(i,pos-i);
	  result.push_back(s);
	  i=pos+pattern.size()-1;
	}
	}
	return result;

}

///////////////////////////////////////////////////////////////
template <typename Dtype>
Dtype  GetArea(const vector<Dtype>& bbox)
{
	Dtype w = bbox[2] - bbox[0] + 1;
	Dtype h = bbox[3] - bbox[1] + 1;
	if (w <= 0 || h <= 0) return Dtype(0.0);
	return w * h;
}
template float GetArea(const vector<float>& bbox);
template double GetArea(const vector<double>& bbox);
template <typename Dtype>
Dtype GetArea(const Dtype x1, const Dtype y1, const Dtype x2, const Dtype y2)
{
	Dtype w = x2- x1 + 1;
	Dtype h = y2 - y1 + 1;
	if (w <= 0 || h <= 0) return Dtype(0.0);
	return w * h;
}
template float GetArea(const float x1, const float y1, const float x2, const float y2);
template double GetArea(const double x1, const double y1, const double x2, const double y2);

template <typename Dtype>
Dtype GetOverlap(const vector<Dtype>& bbox1, const vector<Dtype>& bbox2)
{
	Dtype x1 = std::max<Dtype>(bbox1[0], bbox2[0]);
	Dtype y1 = std::max<Dtype>(bbox1[1], bbox2[1]);
	Dtype x2 = std::min<Dtype>(bbox1[2], bbox2[2]);
	Dtype y2 = std::min<Dtype>(bbox1[3], bbox2[3]);
	Dtype w = x2 - x1 + 1;
	Dtype h = y2 - y1 + 1;
	if (w <= 0 || h <= 0) return Dtype(0.0);

	Dtype intersection = w * h;
	Dtype area1 = GetArea(bbox1);
	Dtype area2 = GetArea(bbox2);
	Dtype u = area1 + area2 - intersection;

	return intersection / u;
}
template float GetOverlap(const vector<float>& bbox1, const vector<float>& bbox2);
template double GetOverlap(const vector<double>& bbox1, const vector<double>& bbox2);

template <typename Dtype>
Dtype GetOverlap(const Dtype x11, const Dtype y11, const Dtype x12, const Dtype y12, const Dtype x21, const Dtype y21, const Dtype x22, const Dtype y22,const OverlapType overlap_type)
{
	Dtype x1 = std::max<Dtype>(x11, x21);
	Dtype y1 = std::max<Dtype>(y11, y21);
	Dtype x2 = std::min<Dtype>(x12, x22);
	Dtype y2 = std::min<Dtype>(y12, y22);
	Dtype w = x2 - x1 + 1;
	Dtype h = y2 - y1 + 1;
	if (w <= 0 || h <= 0) return Dtype(0.0);

	Dtype intersection = w * h;
	Dtype area1 = GetArea(x11, y11, x12, y12);
	Dtype area2 = GetArea(x21, y21, x22, y22);
	Dtype u = 0;
	switch(overlap_type)
	{
		case caffe::OVERLAP_UNION:
		{
			u = area1 + area2 - intersection;
			break;
		}
		case caffe::OVERLAP_BOX1:
		{
			u = area1 ;
			break;
		}
		case caffe::OVERLAP_BOX2:
		{
			u = area2 ;
			break;
		}
		default:
			LOG(FATAL) << "Unknown type " << overlap_type;
	}

	return intersection / u;
}
template float GetOverlap(const float x11, const float y11, const float x12, const float y12, const float x21, const float y21, const float x22, const float y22,const OverlapType overlap_type);
template double GetOverlap(const double x11, const double y11, const double x12, const double y12, const double x21, const double y21, const double x22, const double y22,const OverlapType overlap_type);

///////////////////////////////////////////////////////////////
template <typename Dtype>
bool compareCandidate(const pair<Dtype, vector<float> >& c1, const pair<Dtype, vector<float> >& c2){return c1.first >= c2.first;}
template bool compareCandidate<float>(const pair<float, vector<float> >& c1, const pair<float, vector<float> >& c2);
template bool compareCandidate<double>(const pair<double, vector<float> >& c1, const pair<double, vector<float> >& c2);

template <typename Dtype>
bool compareCandidate_v2(const vector<Dtype>  & c1, const  vector<Dtype>  & c2) {return c1[0] >= c2[0];}
template bool compareCandidate_v2(const vector<float>  & c1, const  vector<float>  & c2);
template bool compareCandidate_v2(const vector<double>  & c1, const  vector<double>  & c2);

template <typename Dtype>
const vector<bool> nms(vector<pair<Dtype, vector<float> > >& candidates, const float overlap, const int top_N, const bool addScore)
{
  vector<bool> mask(candidates.size(), false);

  if (mask.size() == 0) return mask;

  vector<bool> skip(candidates.size(), false);
  std::stable_sort(candidates.begin(), candidates.end(), compareCandidate<Dtype>);

  vector<float> areas(candidates.size(), 0);
  for (int i = 0; i < candidates.size(); ++i) {
  	areas[i] = (candidates[i].second[2] - candidates[i].second[0] + 1)
				* (candidates[i].second[3] - candidates[i].second[1] + 1);
  }

  for (int count = 0, i = 0; count < top_N && i < mask.size(); ++i) {
    if (skip[i]) continue;

    mask[i] = true;
    ++count;

    // suppress the significantly covered bbox
    for (int j = i + 1; j < mask.size(); ++j) {
      if (skip[j]) continue;

      // get intersections
      float xx1 = std::max<Dtype>(candidates[i].second[0], candidates[j].second[0]);
      float yy1 = std::max<Dtype>(candidates[i].second[1], candidates[j].second[1]);
      float xx2 = std::min<Dtype>(candidates[i].second[2], candidates[j].second[2]);
      float yy2 = std::min<Dtype>(candidates[i].second[3], candidates[j].second[3]);
      float w = xx2 - xx1 + 1;
      float h = yy2 - yy1 + 1;
      if (w > 0 && h > 0) {
        // compute overlap
        float o = w * h / areas[j];
        if (o > overlap) {
          skip[j] = true;

          if (addScore) {
          	candidates[i].first += candidates[j].first;
          }
        }
      }
    }
  }

  return mask;
}
template const vector<bool> nms<float> (vector<pair<float, vector<float> > >& candidates, const float overlap, const int top_N, const bool addScore = false);
template const vector<bool> nms<double> (vector<pair<double, vector<float> > >& candidates, const float overlap, const int top_N, const bool addScore = false);

template <typename Dtype>
const vector<bool> nms(vector < vector<Dtype>   >& candidates, const Dtype overlap, const int top_N, const bool addScore)
{
  vector<bool> mask(candidates.size(), false);
  if (mask.size() == 0) return mask;
  //LOG(INFO)<<"overlap: "<<overlap;
  vector<bool> skip(candidates.size(), false);
  std::stable_sort(candidates.begin(), candidates.end(), compareCandidate_v2<Dtype>);

  vector<Dtype> areas(candidates.size(), 0);
  for (int i = 0; i < candidates.size(); ++i)
  {
  	areas[i] = (candidates[i][3] - candidates[i][1] + 1) * (candidates[i][4] - candidates[i][2] + 1);
  }

  for (int count = 0, i = 0; count < top_N && i < mask.size(); ++i)
  {
    if (skip[i])
    	continue;
    mask[i] = true;
    ++count;
    // suppress the significantly covered bbox
    for (int j = i + 1; j < mask.size(); ++j)
    {
      if (skip[j])
    	  continue;
      // get intersections
      Dtype xx1 = std::max<Dtype>(candidates[i][1], candidates[j][1]);
      Dtype yy1 = std::max<Dtype>(candidates[i][2], candidates[j][2]);
      Dtype xx2 = std::min<Dtype>(candidates[i][3], candidates[j][3]);
      Dtype yy2 = std::min<Dtype>(candidates[i][4], candidates[j][4]);
      Dtype w = xx2 - xx1 + 1;
      Dtype h = yy2 - yy1 + 1;
      //LOG(INFO)<<"xx1:"<<xx1<<"  yy1:"<<yy1<<"  xx2:"<<xx2<<"  yy2:"<<yy2;
      if (w > 0 && h > 0)
      {
        // compute overlap
    	Dtype o = w * h / std::min(areas[j],areas[i]);
       // LOG(INFO)<<o;
        if (o > overlap)
        {
          skip[j] = true;
          if (addScore)
          {
          	candidates[i][0] += candidates[j][0];
          }
        }
      }
    }
  }
  return mask;
}
template const vector<bool> nms (vector < vector<float>   >& candidates, const float overlap, const int top_N, const bool addScore);
template const vector<bool> nms (vector < vector<double>   >& candidates, const double overlap, const int top_N, const bool addScore);

template <typename Dtype>
const vector<bool> nms(vector< BBox<Dtype> >& candidates, const Dtype overlap, const int top_N, const bool addScore)
{
  vector<bool> mask(candidates.size(), false);

  if (mask.size() == 0)
	  return mask;

  vector<bool> skip(candidates.size(), false);
  std::stable_sort(candidates.begin(), candidates.end(), BBox<Dtype>::greater);

  vector<float> areas(candidates.size(), 0);
  for (int i = 0; i < candidates.size(); ++i) {
  	areas[i] = (candidates[i].x2 - candidates[i].x1 + 1)
				* (candidates[i].y2- candidates[i].y1 + 1);
  }

  for (int count = 0, i = 0; count < top_N && i < mask.size(); ++i) {
    if (skip[i]) continue;

    mask[i] = true;
    ++count;

    // suppress the significantly covered bbox
    for (int j = i + 1; j < mask.size(); ++j) {
      if (skip[j]) continue;

      // get intersections
      float xx1 = std::max<Dtype>(candidates[i].x1, candidates[j].x1);
      float yy1 = std::max<Dtype>(candidates[i].y1, candidates[j].y1);
      float xx2 = std::min<Dtype>(candidates[i].x2, candidates[j].x2);
      float yy2 = std::min<Dtype>(candidates[i].y2, candidates[j].y2);
      float w = xx2 - xx1 + 1;
      float h = yy2 - yy1 + 1;
      if (w > 0 && h > 0) {
        // compute overlap
      //float o = w * h / areas[j];//bug
    	float o = w * h / std::min(areas[j], areas[i]);
        if (o > overlap) {
          skip[j] = true;

          if (addScore) {
          	candidates[i].score += candidates[j].score;
          }
        }
      }
    }
  }

  return mask;
}
template const vector<bool> nms  (vector< BBox<float> >&  candidates, const float overlap, const int top_N, const bool addScore );
template const vector<bool> nms  (vector< BBox<double> >&  candidates, const double overlap, const int top_N, const bool addScore );

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
template <typename Dtype>
void PushBBoxTo(std::ofstream & out_result_file, 
        const vector< BBox<Dtype> >& bboxes, bool with_cam3d) {
	for(int candidate_id=0; candidate_id < bboxes.size(); ++candidate_id)
	{
	    out_result_file << bboxes[candidate_id].x1 << " " 
            << bboxes[candidate_id].y1 << " " 
            << bboxes[candidate_id].x2 - bboxes[candidate_id].x1 + 1 
            << " " << bboxes[candidate_id].y2 - bboxes[candidate_id].y1 + 1 << " ";
        if (with_cam3d) {
            out_result_file << bboxes[candidate_id].cam3d.x << " ";
            out_result_file << bboxes[candidate_id].cam3d.y << " ";
            out_result_file << bboxes[candidate_id].cam3d.z << " ";
            out_result_file << bboxes[candidate_id].cam3d.h << " ";
            out_result_file << bboxes[candidate_id].cam3d.w << " ";
            out_result_file << bboxes[candidate_id].cam3d.l << " ";
            out_result_file << bboxes[candidate_id].cam3d.o << " ";
        }
        out_result_file << bboxes[candidate_id].score << std::endl;
	}
}
template void PushBBoxTo(std::ofstream & out_result_file,
        const vector<BBox<float> >& bboxes, bool with_cam3d);
template void PushBBoxTo(std::ofstream & out_result_file,
        const vector<BBox<double> >& bboxes, bool with_cam3d);


///////////////////////////////////////////////////////////////
vector<bool> GetPredictedResult(const vector< std::pair<int, vector<float> > > &gt_instances, const vector< std::pair<float, vector<float> > > &pred_instances, float ratio)
{
	vector<bool> res;
	vector<bool> used_gt_instance ;
	used_gt_instance.resize(gt_instances.size(),false);
	for(int pred_id = 0 ; pred_id < pred_instances.size(); pred_id ++)
	{
		float max_overlap = 0;
		float used_id = -1;
		for(int gt_id = 0; gt_id < gt_instances.size(); ++ gt_id)
		{
			float overlap = GetOverlap(pred_instances[pred_id].second,gt_instances[gt_id].second);
			if( overlap >  max_overlap)
			{
				max_overlap = overlap;
				used_id = gt_id;
			}
		}
		if(used_id != -1)
		{
			res.push_back(max_overlap >= ratio && used_gt_instance[used_id] == false);
			used_gt_instance[used_id] = true;
		}
		else
		{
			res.push_back(false);
		}
	}
	return res;
}

///////////////////////////////////////////////////////////////
float GetPRPoint_FDDB(vector< std::pair<float, vector<float> > >& pred_instances_with_gt, const int n_positive, vector<float>& precision,vector<float> &recall)
{
	std::stable_sort(pred_instances_with_gt.begin(), pred_instances_with_gt.end(), compareCandidate<float>);
	precision.clear();
	recall.clear();
	int corrected_count = 0;
	for(int i=0; i< pred_instances_with_gt.size(); i++)
	{
		corrected_count += int(pred_instances_with_gt[i].second[4]) == 1 ? 1:0;
		precision.push_back(corrected_count/(i+0.0+1));
		recall.push_back(corrected_count/(0.0 + n_positive));
	}
	float ap = precision[0] * recall[0];
	for(int i=1; i< pred_instances_with_gt.size(); i++)
	{
		ap += precision[i] * (recall[i]-recall[i-1]);
	}
	return ap;
}
    
    // added by ming li
    template <typename Dtype>
    Dtype BoxIOU(const Dtype x1, const Dtype y1, const Dtype w1, const Dtype h1,
        const Dtype x2, const Dtype y2, const Dtype w2, const Dtype h2, 
        const string mode, bool bbox_size_add_one) {

        if (w1 <= 0 || h1 <= 0 || w2 <= 0 || h2 <= 0) {
            return Dtype(0);
        }

        Dtype bsz01 = bbox_size_add_one ? Dtype(1.0) : Dtype(0.0);

        Dtype tlx = std::max(x1, x2);
        Dtype tly = std::max(y1, y2);
        Dtype brx = std::min(x1 + w1 - bsz01, x2 + w2 - bsz01);
        Dtype bry = std::min(y1 + h1 - bsz01, y2 + h2 - bsz01);

        Dtype over;
        if ((tlx > brx) || (tly > bry)) {
            over = Dtype(0);
        } else {
            over = (brx - tlx + bsz01) * (bry - tly + bsz01);
        }

        Dtype u;
        if (mode == "IOMU") {
            u = std::min(w1 * h1, w2 * h2);
        } else if (mode == "IOFU") {
            u = w1 * h1;
        } else {
            u = w1 * h1 + w2 * h2 - over;
        }

        return over / u;
    }  
    template
    float BoxIOU(const float x1, const float y1, const float w1, const float h1,
        const float x2, const float y2, const float w2, const float h2, 
        const string mode, bool bbox_size_add_one);
    template
    double BoxIOU(const double x1, const double y1, const double w1, const double h1,
        const double x2, const double y2, const double w2, const double h2, 
        const string mode, bool bbox_size_add_one);

    template <typename Dtype>
    void coords2targets(const Dtype ltx, const Dtype lty, const Dtype rbx, const Dtype rby, 
        const Dtype acx, const Dtype acy, const Dtype acw, const Dtype ach,
        const bool use_target_type_rcnn, const bool do_bbox_norm, 
        const vector<Dtype>& bbox_means, const vector<Dtype>& bbox_stds,
        Dtype& tg0, Dtype& tg1, Dtype& tg2, Dtype& tg3, bool bbox_size_add_one) {

        if (use_target_type_rcnn) {
            Dtype bsz01 = bbox_size_add_one?Dtype(1.0):Dtype(0.0);
            Dtype bxw = Dtype(rbx - ltx + bsz01);
            Dtype bxh = Dtype(rby - lty + bsz01);
            Dtype ctx = ltx + 0.5 * (bxw - bsz01);
            Dtype cty = lty + 0.5 * (bxh - bsz01);

            tg0 = Dtype((ctx - acx) / acw);
            tg1 = Dtype((cty - acy) / ach);
            tg2 = Dtype(log(bxw / acw));
            tg3 = Dtype(log(bxh / ach));
        } else {
            tg0 = Dtype(ltx - acx);
            tg1 = Dtype(lty - acy);
            tg2 = Dtype(rbx - acx);
            tg3 = Dtype(rby - acy);
        }
        if (do_bbox_norm) {
            tg0 -= bbox_means[0]; 
            tg0 /= bbox_stds[0]; 
            tg1 -= bbox_means[1]; 
            tg1 /= bbox_stds[1];
            tg2 -= bbox_means[2]; 
            tg2 /= bbox_stds[2]; 
            tg3 -= bbox_means[3]; 
            tg3 /= bbox_stds[3];
        }
    }
    template 
    void coords2targets(const float ltx, const float lty, const float rbx, const float rby, 
        const float acx, const float acy, const float acw, const float ach,
        const bool use_target_type_rcnn, const bool do_bbox_norm, 
        const vector<float>& bbox_means, const vector<float>& bbox_stds,
        float& tg0, float& tg1, float& tg2, float& tg3, bool bbox_size_add_one);
    template 
    void coords2targets(const double ltx, const double lty, const double rbx, const double rby, 
        const double acx, const double acy, const double acw, const double ach,
        const bool use_target_type_rcnn, const bool do_bbox_norm, 
        const vector<double>& bbox_means, const vector<double>& bbox_stds,
        double& tg0, double& tg1, double& tg2, double& tg3, bool bbox_size_add_one);

    template <typename Dtype>
    void targets2coords(const Dtype tg0, const Dtype tg1, const Dtype tg2, const Dtype tg3, 
        const Dtype acx, const Dtype acy, const Dtype acw, const Dtype ach,
        const bool use_target_type_rcnn, const bool do_bbox_norm, 
        const vector<Dtype>& bbox_means, const vector<Dtype>& bbox_stds,
        Dtype& ltx, Dtype& lty, Dtype& rbx, Dtype& rby, bool bbox_size_add_one) {

        Dtype ntg0 = tg0, ntg1 = tg1, ntg2 = tg2, ntg3 = tg3;
        if (do_bbox_norm) {
            ntg0 *= bbox_stds[0]; 
            ntg0 += bbox_means[0];
            ntg1 *= bbox_stds[1]; 
            ntg1 += bbox_means[1];
            ntg2 *= bbox_stds[2]; 
            ntg2 += bbox_means[2];
            ntg3 *= bbox_stds[3]; 
            ntg3 += bbox_means[3];
        }
        if (use_target_type_rcnn) {
            Dtype bsz01 = bbox_size_add_one ? Dtype(1.0) : Dtype(0.0);
            Dtype ctx = ntg0 * acw + acx;
            Dtype cty = ntg1 * ach + acy;
            Dtype tw = Dtype(acw * exp(ntg2));
            Dtype th = Dtype(ach * exp(ntg3));
            ltx = Dtype(ctx - 0.5 * (tw - bsz01));
            lty = Dtype(cty - 0.5 * (th - bsz01));
            rbx = Dtype(ltx + tw - bsz01);
            rby = Dtype(lty + th - bsz01);
        } else {
            ltx = ntg0 + acx;
            lty = ntg1 + acy;
            rbx = ntg2 + acx;
            rby = ntg3 + acy;
        }
    }
    template 
    void targets2coords(const float tg0, const float tg1, const float tg2, const float tg3, 
        const float acx, const float acy, const float acw, const float ach,
        const bool use_target_type_rcnn, const bool do_bbox_norm, 
        const vector<float>& bbox_means, const vector<float>& bbox_stds,
        float& ltx, float& lty, float& rbx, float& rby, bool bbox_size_add_one);
    template 
    void targets2coords(const double tg0, const double tg1, const double tg2, const double tg3, 
        const double acx, const double acy, const double acw, const double ach,
        const bool use_target_type_rcnn, const bool do_bbox_norm, 
        const vector<double>& bbox_means, const vector<double>& bbox_stds,
        double& ltx, double& lty, double& rbx, double& rby, bool bbox_size_add_one);
    
    template <typename Dtype>
    const vector<bool> nms_lm(vector< BBox<Dtype> >& candidates, 
        const Dtype overlap, const int top_N, const bool addScore, 
        const int max_candidate_N, bool bbox_size_add_one, bool voting,
        Dtype vote_iou) {

        Dtype bsz01 = bbox_size_add_one ? Dtype(1.0) : Dtype(0.0);
        std::stable_sort(candidates.begin(), candidates.end(), BBox<Dtype>::greater);
        vector<bool> mask(candidates.size(), false);
        if (mask.size() == 0) {
            return mask;
        }
        int consider_size = candidates.size();
        if (max_candidate_N > 0) {
            consider_size = min<int>(consider_size, max_candidate_N);
        }
        vector<bool> skip(consider_size, false);
        vector<float> areas(consider_size, 0);
        for (int i = 0; i < consider_size; ++i) {
            areas[i] = (candidates[i].x2 - candidates[i].x1 + bsz01) 
                    * (candidates[i].y2- candidates[i].y1 + bsz01);
        }
        for (int count = 0, i = 0; count < top_N && i < consider_size; ++i) {
            if (skip[i]) {
                continue;
            }
            mask[i] = true;
            ++count;
            Dtype s_vt = candidates[i].score;
            Dtype x1_vt = 0.0; 
            Dtype y1_vt = 0.0;
            Dtype x2_vt = 0.0;
            Dtype y2_vt = 0.0;
            if (voting) {
                CHECK_GE(s_vt, 0);
                x1_vt = candidates[i].x1 * s_vt;
                y1_vt = candidates[i].y1 * s_vt;
                x2_vt = candidates[i].x2 * s_vt;
                y2_vt = candidates[i].y2 * s_vt;
            }
            // suppress the significantly covered bbox
            for (int j = i + 1; j < consider_size; ++j) {
                if (skip[j]) {
                    continue;
                }
                // get intersections
                float xx1 = std::max<Dtype>(candidates[i].x1, candidates[j].x1);
                float yy1 = std::max<Dtype>(candidates[i].y1, candidates[j].y1);
                float xx2 = std::min<Dtype>(candidates[i].x2, candidates[j].x2);
                float yy2 = std::min<Dtype>(candidates[i].y2, candidates[j].y2);
                float w = xx2 - xx1 + bsz01;
                float h = yy2 - yy1 + bsz01;
                if (w > 0 && h > 0) {
                    // compute overlap
                    //float o = w * h / areas[j];
                    float o = w * h;
                    o = o / (areas[i] + areas[j] - o);
                    if (o > overlap) {
                        skip[j] = true;
                        if (addScore) {
                            candidates[i].score += candidates[j].score;
                        }
                    }
                    if (voting && o > vote_iou) {
                        Dtype s_vt_cur = candidates[j].score;
                        CHECK_GE(s_vt_cur, 0);
                        s_vt += s_vt_cur;
                        x1_vt += candidates[j].x1 * s_vt_cur;
                        y1_vt += candidates[j].y1 * s_vt_cur;
                        x2_vt += candidates[j].x2 * s_vt_cur;
                        y2_vt += candidates[j].y2 * s_vt_cur;
                    }
                }
            }
            if (voting && s_vt > 0.0001) {
                candidates[i].x1 = x1_vt / s_vt;
                candidates[i].y1 = y1_vt / s_vt;
                candidates[i].x2 = x2_vt / s_vt;
                candidates[i].y2 = y2_vt / s_vt;
            }
        }
        return mask;
    }
    template const vector<bool> nms_lm(vector< BBox<float> >&  candidates, 
        const float overlap, const int top_N, const bool addScore, 
        const int max_candidate_N, bool bbox_size_add_one, bool voting,
        float vote_iou);
    template const vector<bool> nms_lm(vector< BBox<double> >&  candidates,
        const double overlap, const int top_N, const bool addScore, 
        const int max_candidate_N, bool bbox_size_add_one, bool voting, 
        double vote_iou);

    // soft nms, added by mingli
    template <typename Dtype>
    const vector<bool> soft_nms_lm(vector< BBox<Dtype> >& candidates, 
        const Dtype iou_std, const int top_N, const int max_candidate_N, 
        bool bbox_size_add_one, bool voting, Dtype vote_iou) {

        Dtype bsz01 = bbox_size_add_one?Dtype(1.0):Dtype(0.0);
        std::stable_sort(candidates.begin(), candidates.end(), BBox<Dtype>::greater);
        vector<bool> mask(candidates.size(), false);
        if (mask.size() == 0) {
            return mask;
        }
        int consider_size = candidates.size();
        if (max_candidate_N > 0) {
            consider_size = min<int>(consider_size, max_candidate_N);
        }
        vector<float> areas(consider_size, 0);
        for (int i = 0; i < consider_size; ++i) {
            areas[i] = (candidates[i].x2 - candidates[i].x1 + bsz01) 
                * (candidates[i].y2- candidates[i].y1 + bsz01);
        }
        int top_n_real = min<int>(consider_size, top_N);
        for (int count = 0; count < top_n_real; ++count) {
            int max_box_idx = -1;
            for (int i = 0; i < consider_size; ++i) {
                if (mask[i]) {
                    continue;
                }
                if (max_box_idx == -1 || candidates[i].score > candidates[max_box_idx].score) {
                    max_box_idx = i;
                }
            }
            CHECK(max_box_idx != -1);
            mask[max_box_idx] = true;
            Dtype s_vt = candidates[max_box_idx].score;
            Dtype x1_vt = 0.0; 
            Dtype y1_vt = 0.0;
            Dtype x2_vt = 0.0;
            Dtype y2_vt = 0.0;
            if (voting) {
                CHECK_GE(s_vt, 0);
                x1_vt = candidates[max_box_idx].x1 * s_vt;
                y1_vt = candidates[max_box_idx].y1 * s_vt;
                x2_vt = candidates[max_box_idx].x2 * s_vt;
                y2_vt = candidates[max_box_idx].y2 * s_vt;
            }
            // suppress the significantly covered bbox
            for (int j = 0; j < consider_size; ++j) {
                if (mask[j]) {
                    continue;
                }
                // get intersections
                float xx1 = std::max<Dtype>(candidates[max_box_idx].x1, candidates[j].x1);
                float yy1 = std::max<Dtype>(candidates[max_box_idx].y1, candidates[j].y1);
                float xx2 = std::min<Dtype>(candidates[max_box_idx].x2, candidates[j].x2);
                float yy2 = std::min<Dtype>(candidates[max_box_idx].y2, candidates[j].y2);
                float w = xx2 - xx1 + bsz01;
                float h = yy2 - yy1 + bsz01;
                if (w > 0 && h > 0) {
                    // compute overlap
                    float o = w * h;
                    o = o / (areas[max_box_idx] + areas[j] - o);
                    candidates[j].score *= std::exp(-1.0 * o * o / iou_std);
                    if (voting && o > vote_iou) {
                        Dtype s_vt_cur = candidates[j].score;
                        CHECK_GE(s_vt_cur, 0);
                        s_vt += s_vt_cur;
                        x1_vt += candidates[j].x1 * s_vt_cur;
                        y1_vt += candidates[j].y1 * s_vt_cur;
                        x2_vt += candidates[j].x2 * s_vt_cur;
                        y2_vt += candidates[j].y2 * s_vt_cur;
                    }
                }
            }
            if (voting && s_vt > 0.0001) {
                candidates[max_box_idx].x1 = x1_vt / s_vt;
                candidates[max_box_idx].y1 = y1_vt / s_vt;
                candidates[max_box_idx].x2 = x2_vt / s_vt;
                candidates[max_box_idx].y2 = y2_vt / s_vt;
            }
        }
        std::stable_sort(candidates.begin(), 
            candidates.begin() + consider_size, BBox<Dtype>::greater);
        mask.clear();
        mask.resize(top_n_real, true);
        mask.resize(candidates.size(), false);
        return mask;
    }
    template const vector<bool> soft_nms_lm(vector< BBox<float> >& candidates, 
        const float iou_std, const int top_N, const int max_N, 
        bool bbox_size_add_one, bool voting, float vote_iou);
    template const vector<bool> soft_nms_lm(vector< BBox<double> >& candidates, 
        const double iou_std, const int top_N, const int max_N, 
        bool bbox_size_add_one, bool voting, double vote_iou);

    template <typename Dtype>
    void coef2dTo3d(Dtype cam_xpz, Dtype cam_xct, Dtype cam_ypz, 
            Dtype cam_yct, Dtype cam_pitch, Dtype px, Dtype py,
            Dtype & k1, Dtype & k2, Dtype & u, Dtype & v) {
        k1 = (px - cam_xct) / cam_xpz;
        k2 = (py - cam_yct) / cam_ypz;
        Dtype sin_ = sin(cam_pitch);
        Dtype cos_ = cos(cam_pitch);
        Dtype tmp1 = cam_xpz * k1 * sin_;
        Dtype tmp2 = cam_ypz * (k2 * sin_ + cos_);
        u = sqrt(tmp1 * tmp1 + tmp2 * tmp2);
        v = sin_ * sin_;
    }
    template void coef2dTo3d(double cam_xpz, double cam_xct, double cam_ypz, 
        double cam_yct, double cam_pitch, double px, double py,
        double & k1, double & k2, double & u, double & v); 
    template void coef2dTo3d(float cam_xpz, float cam_xct, float cam_ypz, 
        float cam_yct, float cam_pitch, float px, float py,
        float & k1, float & k2, float & u, float & v); 

    template <typename Dtype>
    void cord2dTo3d(Dtype k1, Dtype k2, Dtype u,
            Dtype v, Dtype ph, Dtype rh,
            Dtype & x, Dtype & y, Dtype & z) {
        Dtype uph = u / ph;
        z = 0.5 * rh * (uph + sqrt(uph * uph + v));
        x = k1 * z;
        y = k2 * z;
    }
    template void cord2dTo3d(double k1, double k2, double u,
        double v, double ph, double rh,
        double & x, double & y, double & z);
    template void cord2dTo3d(float k1, float k2, float u,
        float v, float ph, float rh,
        float & x, float & y, float & z);
    
    // end mingli

}
// namespace caffe
