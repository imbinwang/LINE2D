/*****************************************************************************************
 Improve PAMI2012 "Gradient Response Maps for Real-time Detection of Textureless Objects "
 Compute gradient and extract gradient related features
 @author Bin Wang <binwangsdu@gmail.com>
 @date 2014/11/10
*****************************************************************************************/

#ifndef _rl2d_GradientCalculator_H_
#define _rl2d_GradientCalculator_H_

#include <vector>

#include <opencv2\core\core.hpp>
#include <opencv2\core\internal.hpp>

namespace rl2d
{
	class GradientCalculator
	{
	public:
		enum QuantizeType{QUANTIZE8, QUANTIZE16};

		GradientCalculator();
		GradientCalculator(QuantizeType type, int threshold);
		~GradientCalculator();

		int compute(const cv::Mat &src, cv::Mat &mag, cv::Mat &ori) const;

		void quantize(const cv::Mat &mag, const cv::Mat &ori, 
			 cv::Mat &quantizedOri) const;

		void visualize(const cv::Mat &quantizedOri, cv::Mat &dst) const;
		int getLabel(int quantized) const;

	private:
		void quantize8(const cv::Mat &mag, const cv::Mat &ori, 
			 cv::Mat &quantizedOri) const;
		void quantize16(const cv::Mat &mag, const cv::Mat &ori, 
			 cv::Mat &quantizedOri) const;
		void visualizeQuantize8(const cv::Mat &quantizedOri, cv::Mat &dst) const;
		void visualizeQuantize16(const cv::Mat &quantizedOri, cv::Mat &dst) const;

	private:
		QuantizeType m_type;
		int m_threshold;
	};
}

#endif // _GradientCalculator_H_