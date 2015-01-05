/*****************************************************************************************
Improve PAMI2012 "Gradient Response Maps for Real-time Detection of Textureless Objects "
Detector using gradient related features
@author Bin Wang <binwangsdu@gmail.com>
@date 2014/11/25
*****************************************************************************************/
#ifndef _rl2d_Detector_H_
#define _rl2d_Detector_H_

#include <vector>
#include <map>
#include <opencv2\core\core.hpp>

#include "..\include\GradientPyramid.h"
#include "..\include\GradientTransform.h"
#include "..\include\Template.h"
#include "..\include\Match.h"


namespace rl2d
{
	class Detector
	{
	public:
		Detector();
		Detector(const std::vector<int>& a_T_at_level);
		~Detector();

		int getT(int pyramid_level) const;
		int getPyramidLevels() const;

		const std::vector<Template> &getTemplates(int class_id, int template_id) const;

		int getTemplatesNum() const;
		int getTemplatesNum(const int class_id) const;
		int getClassesNum() const;

		std::vector<int> getClassIds() const;

		void match(const cv::Mat &source, float threshold, std::vector<Match>& matches,
			const std::vector<int>& class_ids = std::vector<int>(),
			const cv::Mat &mask = cv::Mat(),
			cv::OutputArrayOfArrays quantized_images = cv::noArray()) const;

		int addTemplate(const cv::Mat &source, const int class_id, cv::Rect *bounding_box = NULL);

	protected:
		std::vector<int> T_at_level; 
		int pyramid_levels;

		GradientTransformL2D grad_trans;

		typedef std::vector<Template> TemplatePyramid;
		typedef std::map<int, std::vector<TemplatePyramid> > TemplatesMap;
		TemplatesMap class_templates;

		typedef std::vector<cv::Mat> LinearMemories;
		// Indexed as [pyramid level][quantized label]
		typedef std::vector<LinearMemories> LinearMemoryPyramid;

		void matchClass(const LinearMemoryPyramid& lm_pyramid,
			const std::vector<cv::Size>& sizes,
			float threshold,
			std::vector<Match>& matches,
			const int class_id,
			const std::vector<TemplatePyramid>& template_pyramids) const;

		const uchar* accessLinearMemory(const LinearMemories& linear_memories,
			const Feature& f, int T, int W) const;
		void similarity(const LinearMemories& linear_memories, const Template& templ,
			cv::Mat& dst, cv::Size size, int T) const;
		void similarityLocal(const LinearMemories& linear_memories, const Template& templ,
			cv::Mat& dst, cv::Size size, int T, cv::Point center) const;
	};

	// detector factory 
	// factory function for detector using LINE algorithm with color gradients
	cv::Ptr<Detector> getDefaultLINE2D();
}

#endif // _rl2d_Detector_H_