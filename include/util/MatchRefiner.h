#ifndef _MATCH_REFINER_H
#define _MATCH_REFINER_H

#include <string>
#include <vector>
#include <map>
#include <opencv/cv.h>

#include "../Detector.h"
#include "../Match.h"
#include "Config.pb.h"

namespace rl2d
{
	class MatchRefiner
	{
	public:
		static void offlineCalculateHSHist(const ObjectsConfig &objectsConfig,
			int HBIN, int SBIN,
			std::map<int, std::vector<cv::Mat> > &hsHistograms);

		static void refine(std::vector<Match>& candidatesMatches, 
			std::vector<cv::Rect> &correspondingRects,
			const cv::Mat& frameImg,
			const std::map<int, std::vector<cv::Rect> > &templateImgBoundingBoxes,
			std::map<int, std::vector<cv::Mat> > &hsHists, 
			int HBIN, int SBIN,
			float match_threshold);
	};
}


#endif //_MATCH_REFINER_H