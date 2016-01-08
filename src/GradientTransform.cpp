#include "..\include\GradientTransform.h"

#include <opencv2\opencv.hpp>
#include <opencv2\core\internal.hpp>

//#define _TIMER_
#ifdef _TIMER_
#include "..\include\util\CvUtility.h"
static Timer t;
#endif

#define ALIGN_16 __declspec(align(16))

namespace rl2d
{
	static void orUnaligned8u(const uchar * src, const int src_stride,
		uchar * dst, const int dst_stride,
		const int width, const int height)
	{
#if CV_SSE2
		volatile bool haveSSE2 = cv::checkHardwareSupport(CV_CPU_SSE2);
#if CV_SSE3
		volatile bool haveSSE3 = cv::checkHardwareSupport(CV_CPU_SSE3);
#endif
#endif

		for (int r = 0; r < height; ++r)
		{
			int c = 0;

#if CV_SSE2
#if CV_SSE3
			// Use LDDQU for fast unaligned load
			if (haveSSE3)
			{
				for ( ; c < width - 15; c += 16)
				{
					__m128i src_val = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(src + c));
					__m128i* dst_ptr = reinterpret_cast<__m128i*>(dst + c);
					__m128i dst_val = _mm_loadu_si128(dst_ptr);
					__m128i or_val = _mm_or_si128(src_val, dst_val);
					_mm_storeu_si128( dst_ptr, or_val);
				}
			}else
#endif
			// Use unaligned loads if possible
			if (haveSSE2)
			{
				for ( ; c < width - 15; c += 16)
				{
					__m128i src_val = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + c));
					__m128i* dst_ptr = reinterpret_cast<__m128i*>(dst + c);
					__m128i dst_val = _mm_loadu_si128(dst_ptr);
					__m128i or_val = _mm_or_si128(src_val, dst_val);
					_mm_storeu_si128( dst_ptr, or_val);
				}
			}
#endif
			for ( ; c < width; ++c)
				dst[c] |= src[c];

			// Advance to next row
			src += src_stride;
			dst += dst_stride;
		}
	}

	void GradientTransformL2D::spread(const cv::Mat &src, const int T, cv::Mat &dst) const
	{
#ifdef _TIMER_
		t.start();
#endif
		// Allocate and zero-initialize spread (OR'ed) image
		dst = cv::Mat::zeros(src.size(), CV_8U);

		// Fill in spread gradient image
		for (int r = 0; r < T; ++r)
		{
			int height = src.rows - r;
			for (int c = 0; c < T; ++c)
			{
				orUnaligned8u(&src.at<unsigned char>(r, c), static_cast<const int>(src.step1()), dst.ptr(),
					static_cast<const int>(dst.step1()), src.cols - c, height);
			}
		}
#ifdef _TIMER_
		t.stop();
		double qt=t.time();
		printf("cpu spread: %lf \n", qt);
#endif
	}

	// the similiarity is calculated as the location difference of 1 value bit, for example,
	// similarity(0000 0001, **** ***1) = 4; similarity(0000 0001, **** 001*) = 4-1;
	// ...;similarity(0000 0001, 1*** ****) = 4-1;
	static ALIGN_16 const uchar SIMILARITY_LUT[256] = 
	{
		// for orientation 0000 0001 looking up least significant 4 bits
		0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4,
		// for orientation 0000 0001 looking up highest significant 4 bits
		0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,  
		0, 3, 4, 4, 3, 3, 4, 4, 2, 3, 4, 4, 3, 3, 4, 4, 
		0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 
		0, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 
		0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 
		0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 
		0, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3, 
		0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 
		0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4, 
		0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 
		0, 3, 4, 4, 3, 3, 4, 4, 2, 3, 4, 4, 3, 3, 4, 4, 
		0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 
		0, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 
		// for orientation 1000 0000 looking up least significant 4 bits
		0, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3, 
		// for orientation 1000 0000 looking up highest significant 4 bits
		0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4  
	};
	//static ALIGN_16 const uchar SIMILARITY_LUT[256] = 
	//{
	//	// for orientation 0000 0001 looking up least significant 4 bits
	//	0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4,
	//	// for orientation 0000 0001 looking up highest significant 4 bits
	//	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  
	//	0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4, 
	//	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	//	0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4, 
	//	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	//	0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 
	//	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	//	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	//	0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 
	//	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	//	0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4, 0, 0, 4, 4, 
	//	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	//	0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4, 
	//	// for orientation 1000 0000 looking up least significant 4 bits
	//	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	//	// for orientation 1000 0000 looking up highest significant 4 bits
	//	0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4  
	//};

	void GradientTransformL2D::computeResponseMaps(const cv::Mat &src,
		std::vector<cv::Mat> &response_maps) const
	{
#ifdef _TIMER_
		t.start();
#endif
		cv::Mat lsb4(src.size(), CV_8U);
		cv::Mat msb4(src.size(), CV_8U);

		int src_width = src.cols;
		int src_height = src.rows;
		int l_w = src_width / 16;
		int l_u = l_w * 16;
		int l_wr = src_width - (l_w * 16); 

		for (int r = 0; r < src_height; ++r)
		{
			const uchar* src_r = src.ptr(r);
			uchar* lsb4_r = lsb4.ptr(r);
			uchar* msb4_r = msb4.ptr(r);

			for (int c = 0; c < src_width; ++c)
			{
				// Least significant 4 bits of spread image pixel
				lsb4_r[c] = src_r[c] & 15;
				// Most significant 4 bits, right-shifted to be in [0, 16)
				msb4_r[c] = (src_r[c] & 240) >> 4;
			}
		}

		// For each of the 8 quantized orientations...
		response_maps.resize(8);

#if CV_SSSE3
		volatile bool haveSSSE3 = cv::checkHardwareSupport(CV_CPU_SSSE3);
		if (haveSSSE3)
		{
			const __m128i* lut = reinterpret_cast<const __m128i*>(SIMILARITY_LUT);
#pragma omp parallel for
			for (int ori = 0; ori < 8; ++ori)
			{
				response_maps[ori].create(src.size(), CV_8U);

				// Precompute the 2D response map S_i (section 2.4)
				for( int r=0; r<src_height; ++r)
				{
					__m128i* map_data = reinterpret_cast<__m128i*>(response_maps[ori].ptr<uchar>(r));
					__m128i* lsb4_data = reinterpret_cast<__m128i*>(lsb4.ptr<uchar>(r));
					__m128i* msb4_data = reinterpret_cast<__m128i*>(msb4.ptr<uchar>(r));

					for( int c=0; c<l_w; ++c)
					{
						__m128i lsb4_val = _mm_lddqu_si128(lsb4_data+c); 
						__m128i msb4_val = _mm_lddqu_si128(msb4_data+c);

						// Using SSE shuffle for table lookup on 4 orientations at a time
						// The most/least significant 4 bits are used as the LUT index
						// _mm_shuffle_epi8(a,mask) reshuffle the a as mask indicates 
						__m128i res1 = _mm_shuffle_epi8(lut[2*ori + 0], lsb4_val);
						__m128i res2 = _mm_shuffle_epi8(lut[2*ori + 1], msb4_val);

						// compare and find max as the response
						__m128i max_val = _mm_max_epu8(res1, res2);
						//_mm_storeu_si128 can store aligned/unaligned data
						_mm_storeu_si128( map_data+c, max_val); 
					}

					// manipulate the resident uchar values
					uchar* map_resident_data = response_maps[ori].ptr<uchar>(r)+l_u;
					uchar* lsb4_resident_data = lsb4.ptr<uchar>(r)+l_u;
					uchar* msb4_resident_data = msb4.ptr<uchar>(r)+l_u;

					for (int c = 0; c < l_wr; ++c)
					{
						uchar lsb4_val = lsb4_resident_data[c]; 
						uchar msb4_val = msb4_resident_data[c];

						uchar res1 = SIMILARITY_LUT[(2*ori+0)*16+lsb4_val];
						uchar res2 = SIMILARITY_LUT[(2*ori+1)*16+msb4_val];
						map_resident_data[c] = std::max(res1,res2);						
					}
				}
			}
		}
		else
#endif
		{
#pragma omp parallel for
			for (int ori = 0; ori < 8; ++ori)
			{
				response_maps[ori].create(src.size(), CV_8U);

				// Precompute the 2D response map S_i (section 2.4)
				for( int r=0; r<src_height; ++r)
				{
					uchar* map_data = response_maps[ori].ptr<uchar>(r);
					uchar* lsb4_data = lsb4.ptr<uchar>(r);
					uchar* msb4_data = msb4.ptr<uchar>(r);

					for( int c=0; c<src_width; ++c)
					{
						uchar lsb4_val = lsb4_data[c]; 
						uchar msb4_val = msb4_data[c];

						uchar res1 = SIMILARITY_LUT[(2*ori+0)*16+lsb4_val];
						uchar res2 = SIMILARITY_LUT[(2*ori+1)*16+msb4_val];

						uchar max_val = res1>res2 ? res1:res2;
						map_data[c] = max_val;
					}
				}
			}
		}

#ifdef _TIMER_
		t.stop();
		double qt=t.time();
		printf("cpu response_map: %lf \n", qt);
#endif
	}

	void GradientTransformL2D::linearize(const std::vector<cv::Mat> &response_maps, const int T, 
		std::vector<cv::Mat> &linearizeds) const
	{
#ifdef _TIMER_
		t.start();
#endif
		// linearized has T^2 rows, where each row is a linear memory
		// NOTE: here we discard the resident values in response maps
		// NOTE: when object is in frame edge, this may lead to missing the correct detection
		linearizeds.resize(response_maps.size());
#pragma omp parallel for
		for(int i=0; i<response_maps.size(); ++i)
		{
			const cv::Mat &response_map = response_maps[i];
			cv::Mat &linearized = linearizeds[i];
			int mem_width = response_map.cols / T;
			int mem_height = response_map.rows / T;
			linearized.create(T*T, mem_width * mem_height, CV_8U);

			// Outer two for loops iterate over top-left T^2 starting pixels
			int index = 0;
			for (int r_start = 0; r_start < T; ++r_start)
			{
				for (int c_start = 0; c_start < T; ++c_start)
				{
					uchar* memory = linearized.ptr<uchar>(index);
					++index;

					// Inner two loops copy every T-th pixel into the linear memory
					for (int r = 0; r < mem_height; ++r)
					{
						const uchar* response_data = response_map.ptr<uchar>(r_start+r*T);
						for (int c = 0; c < mem_width; ++c)
						{
							*memory = response_data[c_start+c*T];
							++memory;
						}
					}
				}
			}
		}
#ifdef _TIMER_
		t.stop();
		double qt=t.time();
		printf("cpu linearize: %lf \n", qt);
#endif
	}

}