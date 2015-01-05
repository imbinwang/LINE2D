/*****************************************************************************************
 Improve PAMI2012 "Gradient Response Maps for Real-time Detection of Textureless Objects "
 Image Pyramid Processing
 @author Bin Wang <binwangsdu@gmail.com>
 @date 2014/11/20
*****************************************************************************************/
#ifndef _rl2d_GradientPyramid_H_
#define _rl2d_GradientPyramid_H_

#include "Template.h"
#include "GradientCalculator.h"
#include <vector>
#include <opencv2\core\core.hpp>

namespace rl2d
{
	class GradientPyramid
	{
	public:
		GradientPyramid(const cv::Mat &a_src, const cv::Mat &a_mask);
		GradientPyramid(const cv::Mat &a_src, const cv::Mat &a_mask, 
			float a_weak_threshold, int a_num_features, float a_strong_threshold);
		~GradientPyramid();

		void quantize(cv::Mat& dst) const;
		
		bool extractTemplate(Template& templ);
		
		void pyrDown();
		
		cv::Rect cropTemplates(std::vector<Template> &templates);

		cv::Mat& cloneSrc() const;

	protected:
		void update();

		int pyramid_level;

		GradientCalculator grad_cal;

		cv::Mat src;
		cv::Mat mask;
		cv::Mat direction;
		cv::Mat orientation;
		cv::Mat magnitude;

		float weak_threshold;
		float strong_threshold;
		int num_features;

		/* candidate feature point with a score in cpu */
		struct Candidate
		{
			Feature f;
			float score;

			Candidate(int x, int y, int label, float score)
				: f(x, y, label), score(score)
			{
			}

			// sort candidates with high score to the front
			// overwrite operator for STL container ascent sort
			bool operator<(const Candidate& rhs) const
			{
				return score > rhs.score;
			}
		};

		// choose candidate features so that they are not bunched together.
		void selectScatteredFeatures(const std::vector<Candidate>& candidates,
			std::vector<Feature> &features, float distance);
	};
}
#endif //_GradientPyramid_H_