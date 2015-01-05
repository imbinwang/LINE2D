/*****************************************************************************************
 Improve PAMI2012 "Gradient Response Maps for Real-time Detection of Textureless Objects "
 Transform quantized gradient orientation, that's to say, do some feature engineering
 @author Bin Wang <binwangsdu@gmail.com>
 @date 2014/11/17
*****************************************************************************************/

#ifndef _rl2d_GradientTransform_H_
#define _rl2d_GradientTransform_H_

#include <vector>
#include <opencv2\core\core.hpp>

namespace rl2d
{
	// original method in the PAMI12 paper
	class GradientTransformL2D
	{
	public:
		GradientTransformL2D(){};
		~GradientTransformL2D(){};

		void spread(const cv::Mat &src, const int T, cv::Mat &dst) const;

		void computeResponseMaps(const cv::Mat &src, std::vector<cv::Mat> &response_maps) const;

		void linearize(const std::vector<cv::Mat> &response_maps, const int T, 
			std::vector<cv::Mat> &linearizeds) const;
	};
}

#endif //_GradientTransform_H_