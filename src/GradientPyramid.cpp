#include "..\include\GradientPyramid.h"
#include <opencv2\opencv.hpp>

//#define _TIMER_
#ifdef _TIMER_
#include "..\include\util\CvUtility.h"
static Timer t;
#endif

namespace rl2d
{
	GradientPyramid::GradientPyramid(const cv::Mat &a_src, const cv::Mat &a_mask)
		:src(a_src), mask(a_mask), weak_threshold(20.0f), num_features(63), 
		strong_threshold(60.0f), pyramid_level(0)
	{
			update();
	}

	GradientPyramid::GradientPyramid(const cv::Mat& a_src, const cv::Mat &a_mask, 
		float a_weak_threshold, int a_num_features, float a_strong_threshold)
		:src(a_src), mask(a_mask), weak_threshold(a_weak_threshold), num_features(a_num_features), 
		strong_threshold(a_strong_threshold), pyramid_level(0)
	{
			update();
	}

	GradientPyramid::~GradientPyramid()
	{
	}

	void GradientPyramid::update()
	{
		grad_cal.compute(src, magnitude, direction);
		grad_cal.quantize(magnitude, direction, orientation);
	}

	void GradientPyramid::quantize(cv::Mat& dst) const
	{
		dst = cv::Mat::zeros(orientation.size(), CV_8U);
		orientation.copyTo(dst, mask);
	}

	void GradientPyramid::pyrDown()
	{
		// Some parameters need to be adjusted
		// is it valid to reduce the number of features?
		// here we choose to keep the features number unchanged
		// initial version the features number is reduced to half
		//  num_features /= 2;

		num_features /= 2;
		++pyramid_level;

		// Downsample the current inputs
		cv::Size size(src.cols/2, src.rows/2);
		cv::Mat next_src;
		cv::pyrDown(src, next_src, size);
		src = next_src;
		if (!mask.empty())
		{
			cv::Mat next_mask;
			cv::resize(mask, next_mask, size, 0.0, 0.0, CV_INTER_NN);
			mask = next_mask;
		}

		update();
	}

	bool GradientPyramid::extractTemplate(Template& templ)
	{
#ifdef _TIMER_
		t.start();
#endif
		// Size determined externally, needs to match templates for other modalities
		templ.width = -1;
		templ.height = -1;
		templ.pyramid_level = pyramid_level;
		templ.features.resize(num_features, Feature());

		// Create sorted list of all pixels with magnitude greater than strong_threshold
		std::vector<Candidate> candidates;
		for (int r = 0; r < magnitude.rows; ++r)
		{
			const uchar* orientation_r = orientation.ptr<uchar>(r);
			const float* magnitude_r = magnitude.ptr<float>(r);

			for (int c = 0; c < magnitude.cols; ++c)
			{
				uchar quantized = orientation_r[c];
				if (quantized > 0)
				{
					float score = magnitude_r[c];
					if (score > strong_threshold)
					{
						int quantized_label = grad_cal.getLabel(quantized);
						candidates.push_back(Candidate(c, r, quantized_label, score));
					}
				}
			}
		}

		// We require a certain number of features
		if (candidates.size() < num_features)
			return false;
		// NOTE: Stable sort to agree with old code, which used std::list::sort()
		std::stable_sort(candidates.begin(), candidates.end());

		// Use heuristic based on surplus of candidates in narrow outline for initial distance threshold
		float distance = static_cast<float>(candidates.size() / num_features + 1);
		selectScatteredFeatures(candidates, templ.features, distance);

#ifdef _TIMER_
		t.stop();
		double ett=t.time();
		printf("cpu extract template: %lf \n", ett);
#endif

		return true;
	}

	void GradientPyramid::selectScatteredFeatures(const std::vector<Candidate>& candidates,
		std::vector<Feature> &features, float distance)
	{
		int addedFeaturesNum = 0;
		int i = 0;
		while (addedFeaturesNum < num_features)
		{
			Candidate c = candidates[i];

			// Add if sufficient distance away from any previously chosen feature
			bool keep = true;
			for (int j = 0; (j < addedFeaturesNum) && keep; ++j)
			{
				Feature f = features[j];
				keep = CV_SQR(c.f.x - f.x) + CV_SQR(c.f.y - f.y) >= CV_SQR(distance);
			}
			if (keep)
				features[addedFeaturesNum++]=c.f;

			if (++i == (int)candidates.size())
			{
				// Start back at beginning, and relax required distance
				i = 0;
				distance -= 1.0f;
			}
		}
	}

	cv::Rect GradientPyramid::cropTemplates(std::vector<Template> &templates)
	{
#ifdef _TIMER_
		t.start();
#endif
		int min_x = std::numeric_limits<int>::max();
		int min_y = std::numeric_limits<int>::max();
		int max_x = std::numeric_limits<int>::min();
		int max_y = std::numeric_limits<int>::min();

		// First pass: find min/max feature x,y over all pyramid levels
		for (int i = 0; i < (int)templates.size(); ++i)
		{
			Template& templ = templates[i];

			for (int j = 0; j < (int)templ.features.size(); ++j)
			{
				int x = templ.features[j].x << templ.pyramid_level;
				int y = templ.features[j].y << templ.pyramid_level;
				min_x = std::min(min_x, x);
				min_y = std::min(min_y, y);
				max_x = std::max(max_x, x);
				max_y = std::max(max_y, y);
			}
		}

		// Second pass: set width/height and shift all feature positions
		for (int i = 0; i < (int)templates.size(); ++i)
		{
			Template& templ = templates[i];
			templ.width = (max_x - min_x) >> templ.pyramid_level;
			templ.height = (max_y - min_y) >> templ.pyramid_level;
			int offset_x = min_x >> templ.pyramid_level;
			int offset_y = min_y >> templ.pyramid_level;

			for (int j = 0; j < (int)templ.features.size(); ++j)
			{
				templ.features[j].x -= offset_x;
				templ.features[j].y -= offset_y;
			}
		}

#ifdef _TIMER_
		t.stop();
		double ctt=t.time();
		printf("cpu crop templates: %lf \n", ctt);
#endif
		return cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
	}


	cv::Mat& GradientPyramid::cloneSrc() const
	{
		cv::Mat ret = src.clone();
		return ret;
	}
}