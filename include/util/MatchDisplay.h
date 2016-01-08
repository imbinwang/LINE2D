#ifndef _MatchDisplay_H_
#define _MatchDisplay_H_

#include "Config.pb.h"
#include "..\Detector.h"
#include <string>
#include <set>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

namespace rl2d
{
	const ObjectsConfig_ObjectConfig& objectConfigById(const ObjectsConfig &objectsConfig, const int classId);

	// draw the best draw_match_num matches
	bool DrawMatches(const std::vector<Match> &matches,  
		const cv::Ptr<Detector> &detector,
		const ObjectsConfig &objectsConfig,
		int draw_match_num, 
		cv::Mat &displayImg);

	void DrawFeatures(const std::vector<Feature> &features,  
		const cv::Point &offset,
		const cv::Scalar &color,
		cv::Mat &displayImg);

	void SelectBestMatch4EachClass(const std::vector<Match> &matches, std::vector<Match> &bestMatches);

	void GroupMatchesByClass(const std::vector<Match> &matches, std::map<int,std::vector<Match*> > &classMatchMap);

	void CvtGroupedMatches2Rects(const std::map<int,std::vector<Match*> > &classMatchMap, 
		const cv::Ptr<Detector> &detector,
		std::map<int,std::vector<cv::Rect> > &classRectMap);

	void GroupRects(std::vector<cv::Rect> &rectList, 
		std::vector<int> &labels, 
		std::set<int> &validLabelValues,  
		int groupThreshold, float eps);

	void MergeMatches(const std::vector<Match> &matches,
		const cv::Ptr<Detector> &detector,
		std::vector<Match> &mergedMatches, 
		std::vector<cv::Rect> &mergedRects,
		int eachGroupMinNum = 2,
		double eps = 0.2);

	void DrawMergedMatches(const std::vector<Match> &matches, 
		const std::vector<cv::Rect> &boundingBoxes,
		const ObjectsConfig &objectsConfig,
		int draw_match_num, cv::Mat &displayImg);
}

#endif // _MatchDisplay_H_