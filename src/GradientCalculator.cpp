#include <vector>
#include <glog\logging.h>
#include <opencv2\opencv.hpp>

#include "..\include\GradientCalculator.h"

//#define _TIMER_
#ifdef _TIMER_
#include "..\include\util\CvUtility.h"
static Timer t;
#endif

#include <fstream>


namespace rl2d
{
	// bgr color lookup tabel: from blue to yellow
	static cv::Vec3b colorLut[16]=
	{
		/*cv::Vec3b(255,0,51),
		cv::Vec3b(226,0,80),
		cv::Vec3b(197,0,109),
		cv::Vec3b(168,0,138),
		cv::Vec3b(109,0,197),
		cv::Vec3b(80,0,226),
		cv::Vec3b(51,0,255),
		cv::Vec3b(45,32,255),
		cv::Vec3b(38,64,255),
		cv::Vec3b(32,96,255),
		cv::Vec3b(26,128,255),
		cv::Vec3b(19,159,255),
		cv::Vec3b(13,191,255),
		cv::Vec3b(6,223,255),
		cv::Vec3b(0,255,255)*/
		cv::Vec3b(0, 0, 255),
		cv::Vec3b(0, 170, 255),
		cv::Vec3b(0, 255, 170),
		cv::Vec3b(0, 255, 0),
		cv::Vec3b(170, 255, 0),
		cv::Vec3b(255, 170, 0),
		cv::Vec3b(255, 0, 0),
		cv::Vec3b(255, 0, 170),
		cv::Vec3b(0, 0, 255),
		cv::Vec3b(0, 170, 255),
		cv::Vec3b(0, 255, 170),
		cv::Vec3b(0, 255, 0),
		cv::Vec3b(170, 255, 0),
		cv::Vec3b(255, 170, 0),
		cv::Vec3b(255, 0, 0),
		cv::Vec3b(255, 0, 170)
	};

	GradientCalculator::GradientCalculator()
		:m_type(QUANTIZE8), m_threshold(20)
	{
	}

	GradientCalculator::GradientCalculator(QuantizeType type, int threshold)
		:m_type(type), m_threshold(threshold)
	{
	}

	GradientCalculator::~GradientCalculator()
	{
	}

	int GradientCalculator::compute(
		const cv::Mat &src, 
		cv::Mat &mag, 
		cv::Mat &ori) const
	{
		// only for 1 or 3 channels image
		cv::Size srcSize = src.size();
		int srcChannel = src.channels();	
		if(srcChannel!=1 && srcChannel!=3)
		{
			LOG(ERROR) << "Souce image is not gray or color image.";
			return 0;
		}

#ifdef _TIMER_
		t.start();
#endif
		// use guassian blur to denoise
		const int KERNEL_SIZE = 5;
		cv::Mat smoothed;
		cv::GaussianBlur(src, smoothed, 
			cv::Size(KERNEL_SIZE, KERNEL_SIZE), 0.0, 0.0, cv::BORDER_REPLICATE);
#ifdef _TIMER_
		t.stop();
		double gbt=t.time();
		printf("cpu gaussian filter: %lf \n", gbt);
#endif

#ifdef _TIMER_
		t.start();
#endif
		// use sobel kernel to compute derivatives in x direction and y direction
		cv::Mat sobel_dx;
		cv::Mat sobel_dy;
		cv::Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
		cv::Sobel(smoothed, sobel_dy, CV_32F, 0, 1, 3, 1.0, 0.0, cv::BORDER_REPLICATE);
		//cv::Scharr(smoothed, sobel_dx, CV_32F, 1, 0, 1.0, 0.0, cv::BORDER_REPLICATE);
		//cv::Scharr(smoothed, sobel_dy, CV_32F, 0, 1, 1.0, 0.0, cv::BORDER_REPLICATE);
#ifdef _TIMER_
		t.stop();
		double st=t.time();
		printf("cpu Sobel: %lf \n", st);
#endif

		mag = cv::Mat::zeros(srcSize, CV_32FC1);
		ori = cv::Mat::zeros(srcSize, CV_32FC1);
		if(srcChannel==1)
		{
#ifdef _TIMER_
			t.start();
#endif
			// combine dx and dy to get gradient magnitude and orientation
			cv::cartToPolar(sobel_dx, sobel_dy, mag, ori, true);
#ifdef _TIMER_
			t.stop();
			double c2pt=t.time();
			printf("cpu cartToPolar: %lf \n", c2pt);
#endif
		}
		if(srcChannel==3)
		{
#ifdef _TIMER_
			t.start();
#endif
			cv::Mat sobel_dx_max = cv::Mat::zeros(srcSize, CV_32FC1);
			cv::Mat sobel_dy_max = cv::Mat::zeros(srcSize, CV_32FC1);

			for(int i=0; i<srcSize.height; ++i)
			{
				const cv::Vec3f *dx_ptr = sobel_dx.ptr<cv::Vec3f>(i);
				const cv::Vec3f *dy_ptr = sobel_dy.ptr<cv::Vec3f>(i);
				float *dx_max_ptr = sobel_dx_max.ptr<float>(i);
				float *dy_max_ptr = sobel_dy_max.ptr<float>(i);
				for(int j=0; j<srcSize.width; ++j)
				{
					// Use the gradient orientation of the channel whose magnitude is largest
					float mag0 = CV_SQR(dx_ptr[j][0]) + CV_SQR(dy_ptr[j][0]);
					float mag1 = CV_SQR(dx_ptr[j][1]) + CV_SQR(dy_ptr[j][1]);
					float mag2 = CV_SQR(dx_ptr[j][2]) + CV_SQR(dy_ptr[j][2]);

					if (mag0 >= mag1 && mag0 >= mag2)
					{
						dx_max_ptr[j] = dx_ptr[j][0];
						dy_max_ptr[j] = dy_ptr[j][0];
					}
					else if (mag1 >= mag0 && mag1 >= mag2)
					{
						dx_max_ptr[j] = dx_ptr[j][1];
						dy_max_ptr[j] = dy_ptr[j][1];
					}
					else
					{
						dx_max_ptr[j] = dx_ptr[j][2];
						dy_max_ptr[j] = dy_ptr[j][2];
					}
				}
			}
#ifdef _TIMER_
			t.stop();
			double maxst=t.time();
			printf("cpu max Sobel: %lf \n", maxst);
#endif

#ifdef _TIMER_
			t.start();
#endif
			// combine dx and dy to get gradient magnitude and orientation
			cv::cartToPolar(sobel_dx_max, sobel_dy_max, mag, ori, true);
#ifdef _TIMER_
			t.stop();
			double c2pt=t.time();
			printf("cpu cartToPolar: %lf \n", c2pt);
#endif
		}

		return 1;
	}

	void GradientCalculator::quantize(const cv::Mat &mag, const cv::Mat &ori, 
		cv::Mat &quantizedOri) const
	{
#ifdef _TIMER_
		t.start();
#endif
		if(m_type == QUANTIZE8)
			quantize8(mag, ori, quantizedOri);
		else if(m_type == QUANTIZE16)
			quantize16(mag, ori, quantizedOri);
		else
		{
			LOG(ERROR) << "Invalid quantization type.";
		}
#ifdef _TIMER_
		t.stop();
		double qt=t.time();
		printf("cpu quantize: %lf \n", qt);
#endif
	}	

		void GradientCalculator::quantize8(const cv::Mat &mag, const cv::Mat &ori, 
		cv::Mat &quantizedOri) const
	{
		// Quantize 360 degree range of orientations into 16 buckets
		// Note that [0, 22.5)get mapped in the end to label 0,
		// for stability of horizontal and vertical features.
		cv::Mat_<uchar> quantized_unfiltered;
		ori.convertTo(quantized_unfiltered, CV_8U, 16.0/360.0);

		// Mask 16 buckets into 8 quantized orientations
		// Note that 0 and 8 represent different direction but have the same orientation
		// two directions have 180 degreee between them, they has same orientation
		for (int r = 0; r < ori.rows; ++r)
		{
			uchar* quant_r = quantized_unfiltered.ptr<uchar>(r);
			for (int c = 0; c < ori.cols; ++c)
			{
				quant_r[c] &= 7;
			}
		}

		// Filter the raw quantized image. Only accept pixels where the magnitude is above some
		// threshold, and there is local agreement on the quantization.
		// NOTE: here we should guarantee that the histogram just corver strong gradient location
		// modified by Bin Wang
		quantizedOri = cv::Mat::zeros(ori.size(), CV_8U);
		for (int r = 1; r < ori.rows - 1; ++r)
		{
			const float* mag_r = mag.ptr<float>(r);
			const float* mag_ru = mag_r - mag.step1();
			const float* mag_rd = mag_r + mag.step1();

			for (int c = 1; c < ori.cols - 1; ++c)
			{
				if (mag_r[c] > m_threshold)
				{
					// Compute histogram of quantized bins in 3x3 patch around pixel
					int histogram[8] = {0, 0, 0, 0, 0, 0, 0, 0};

					uchar* patch3x3_row = &quantized_unfiltered(r-1, c-1);
					if(mag_ru[c-1]>1) ++histogram[patch3x3_row[0]];
					if(mag_ru[c]>1) ++histogram[patch3x3_row[1]];
					if(mag_ru[c+1]>1) ++histogram[patch3x3_row[2]];

					patch3x3_row += quantized_unfiltered.step1();
					if(mag_r[c-1]>1) ++histogram[patch3x3_row[0]];
									 ++histogram[patch3x3_row[1]];
					if(mag_r[c+1]>1) ++histogram[patch3x3_row[2]];

					patch3x3_row += quantized_unfiltered.step1();
					if(mag_rd[c-1]>1) ++histogram[patch3x3_row[0]];
					if(mag_rd[c]>1) ++histogram[patch3x3_row[1]];
					if(mag_rd[c+1]>1) ++histogram[patch3x3_row[2]];

					// Find bin with the most votes from the patch
					int max_votes = 0;
					int index = -1;
					for (int i = 0; i < 8; ++i)
					{
						if (max_votes < histogram[i])
						{
							index = i;
							max_votes = histogram[i];
						}
					}

					// Only accept the quantization if majority of pixels in the patch agree
					static const int NEIGHBOR_THRESHOLD = 3;
					if (max_votes >= NEIGHBOR_THRESHOLD)
						quantizedOri.at<uchar>(r, c) = uchar(1 << index);
				}
			}
		}
	}

	void GradientCalculator::quantize16(const cv::Mat &mag, const cv::Mat &ori, 
		cv::Mat &quantizedOri) const
	{
		// Quantize 360 degree range of orientations into 32 buckets
		// Note that [0, 11.25)get mapped in the end to label 0
		// Note that [11.25, 22.5)get mapped in the end to label 1 ...
		// for stability of horizontal and vertical features.
		cv::Mat_<uchar> quantized_unfiltered;
		ori.convertTo(quantized_unfiltered, CV_8U, 32.0/360.0);

		// Mask 32 buckets into 16 quantized orientations
		// Note that 0 and 16 represent different direction but have the same orientation
		// two directions have 180 degreee between them, they has same orientation
		for (int r = 0; r < ori.rows; ++r)
		{
			uchar* quant_r = quantized_unfiltered.ptr<uchar>(r);
			for (int c = 0; c < ori.cols; ++c)
			{
				quant_r[c] &= 15;
			}
		}

		// Filter the raw quantized image. Only accept pixels where the magnitude is above some
		// threshold, and there is local agreement on the quantization.
		// NOTE: here we should guarantee that the histogram just corver strong gradient location
		// modified by Bin Wang
		quantizedOri = cv::Mat::zeros(ori.size(), CV_16U);
		for (int r = 1; r < ori.rows - 1; ++r)
		{
			const float* mag_r = mag.ptr<float>(r);
			const float* mag_ru = mag_r - mag.step1();
			const float* mag_rd = mag_r + mag.step1();

			for (int c = 1; c < ori.cols - 1; ++c)
			{
				if (mag_r[c] > m_threshold)
				{
					// Compute histogram of quantized bins in 3x3 patch around pixel
					int histogram[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

					uchar* patch3x3_row = &quantized_unfiltered(r-1, c-1);
					if(mag_ru[c-1]>1) ++histogram[patch3x3_row[0]];
					if(mag_ru[c]>1) ++histogram[patch3x3_row[1]];
					if(mag_ru[c+1]>1) ++histogram[patch3x3_row[2]];

					patch3x3_row += quantized_unfiltered.step1();
					if(mag_r[c-1]>1) ++histogram[patch3x3_row[0]];
									 ++histogram[patch3x3_row[1]];
					if(mag_r[c+1]>1) ++histogram[patch3x3_row[2]];

					patch3x3_row += quantized_unfiltered.step1();
					if(mag_rd[c-1]>1) ++histogram[patch3x3_row[0]];
					if(mag_rd[c]>1) ++histogram[patch3x3_row[1]];
					if(mag_rd[c+1]>1) ++histogram[patch3x3_row[2]];

					// Find bin with the most votes from the patch
					int max_votes = 0;
					int index = -1;
					for (int i = 0; i < 16; ++i)
					{
						if (max_votes < histogram[i])
						{
							index = i;
							max_votes = histogram[i];
						}
					}

					// Only accept the quantization if majority of pixels in the patch agree
					static const int NEIGHBOR_THRESHOLD = 3;
					if (max_votes >= NEIGHBOR_THRESHOLD)
						quantizedOri.at<unsigned short>(r, c) = static_cast<unsigned short>(1 << index);
				}
			}
		}
	}

	int GradientCalculator::getLabel(int quantized) const
	{
		switch (quantized)
		{
		case 1:   return 0;
		case 2:   return 1;
		case 4:   return 2;
		case 8:   return 3;
		case 16:  return 4;
		case 32:  return 5;
		case 64:  return 6;
		case 128: return 7;
		case 256: return 8;
		case 512: return 9;
		case 1024: return 10;
		case 2048: return 11;
		case 4096: return 12;
		case 8192: return 13;
		case 16384: return 14;
		case 32768: return 15;
		default:  
			LOG(ERROR) << "Invalid quantized orientation.";
			return -1;
		}
	}

	void GradientCalculator::visualize(const cv::Mat &quantizedOri, cv::Mat &dst) const
	{
		if(m_type == QUANTIZE8)
			visualizeQuantize8(quantizedOri, dst);
		else if(m_type == QUANTIZE16)
			visualizeQuantize16(quantizedOri, dst);
		else
		{
			LOG(ERROR) << "Invalid quantization type for visualization.";
		}
	}

	void GradientCalculator::visualizeQuantize8(const cv::Mat &quantizedOri, cv::Mat &dst) const
	{
		dst = cv::Mat::zeros(quantizedOri.size(), CV_8UC3);
		for (int r = 0; r < dst.rows; ++r)
		{
			const uchar* quant_r = quantizedOri.ptr(r);
			cv::Vec3b* dst_r = dst.ptr<cv::Vec3b>(r);
			for (int c = 0; c < dst.cols; ++c)
			{
				uchar q = quant_r[c];
				if(q) dst_r[c] = colorLut[getLabel(q)];
			}
		}
	}

	void GradientCalculator::visualizeQuantize16(const cv::Mat &quantizedOri, cv::Mat &dst) const
	{
		dst = cv::Mat::zeros(quantizedOri.size(), CV_8UC3);
		for (int r = 0; r < dst.rows; ++r)
		{
			const uchar* quant_r = quantizedOri.ptr(r);
			cv::Vec3b* dst_r = dst.ptr<cv::Vec3b>(r);
			for (int c = 0; c < dst.cols; ++c)
			{
				uchar q = quant_r[c];
				if(q) dst_r[c] = colorLut[getLabel(q)];
			}
		}
	}

}