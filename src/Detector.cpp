#include "..\include\Detector.h"
#include "..\include\Match.h"
#include <opencv2\opencv.hpp>

#define _TIMER_
#ifdef _TIMER_
#include "..\include\util\CvUtility.h"
static Timer t;
#endif

#include <glog\logging.h>

namespace rl2d
{
	/***CPU code******CPU code******CPU code******CPU code******CPU code******CPU code***/
	/***CPU code******CPU code******CPU code******CPU code******CPU code******CPU code***/
	/***CPU code******CPU code******CPU code******CPU code******CPU code******CPU code***/
	Detector::Detector()
	{
	}

	Detector::Detector(const std::vector<int>& a_T_at_level)
		:T_at_level(a_T_at_level), pyramid_levels(a_T_at_level.size())
	{
	}

	Detector::~Detector()
	{
	}

	int Detector::getT(int pyramid_level) const
	{
		return T_at_level[pyramid_level];
	}

	int Detector::getPyramidLevels() const
	{
		return pyramid_levels;
	}

	const std::vector<Template>& Detector::getTemplates(int class_id, int template_id) const
	{
		TemplatesMap::const_iterator i = class_templates.find(class_id);
		CHECK(i != class_templates.end()) << "No specified class.";
		CHECK(i->second.size() > size_t(template_id)) << "No specified template.";
		return i->second[template_id];
	}

	int Detector::getTemplatesNum() const
	{
		int ret = 0;
		TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
		for ( ; i != iend; ++i)
			ret += static_cast<int>(i->second.size());
		return ret;
	}

	int Detector::getTemplatesNum(const int class_id) const
	{
		TemplatesMap::const_iterator i = class_templates.find(class_id);
		if (i == class_templates.end())
			return 0;
		return static_cast<int>(i->second.size());
	}

	int Detector::getClassesNum() const
	{
		return static_cast<int>(class_templates.size());
	}

	std::vector<int> Detector::getClassIds() const
	{
		std::vector<int> ids;
		TemplatesMap::const_iterator i = class_templates.begin(), iend = class_templates.end();
		for ( ; i != iend; ++i)
		{
			ids.push_back(i->first);
		}

		return ids;
	}

	int Detector::addTemplate(const cv::Mat &source, const int class_id, cv::Rect *bounding_box)
	{
		// current templates for class_id
		std::vector<TemplatePyramid>& template_pyramids = class_templates[class_id];
		int template_id = template_pyramids.size();

		// extract template pyramid from source image
		TemplatePyramid tp;
		tp.resize(pyramid_levels);

		// Extract template pyramid for each pyramid level
		cv::Ptr<GradientPyramid> grad_pyr = new GradientPyramid(source, cv::Mat());
		for (int l = 0; l < pyramid_levels; ++l)
		{
			if (l > 0)
				grad_pyr->pyrDown();

			bool success = grad_pyr->extractTemplate(tp[l]);

			if (!success)
				return -1;
		}

		cv::Rect bb = grad_pyr->cropTemplates(tp);
		if (bounding_box)
			*bounding_box = bb;

		template_pyramids.push_back(tp);
		return template_id;
	}

	void Detector::match(const cv::Mat &source, float threshold, std::vector<Match>& matches,
		const std::vector<int>& class_ids, const cv::Mat &mask,cv::OutputArrayOfArrays quantized_images) const
	{
#ifdef _TIMER_
		t.start();
#endif
		matches.clear();
		if (quantized_images.needed()) //judge if the outputAarry should be allocate memory
			quantized_images.create(1, pyramid_levels, CV_8U);

		// Initialize gradient pyramid
		cv::Ptr<GradientPyramid> grad_pyr = new GradientPyramid(source, mask);

		// pyramid level -> quantization
		LinearMemoryPyramid lm_pyramid(pyramid_levels, LinearMemories(8));

		// For each pyramid level, precompute linear memories
		std::vector<cv::Size> sizes(pyramid_levels);
		for (int l = 0; l < pyramid_levels; ++l)
		{
			int T = T_at_level[l];
			LinearMemories &lm_level = lm_pyramid[l];

			if (l > 0)
				grad_pyr->pyrDown();

			cv::Mat quantized, spread_quantized;
			std::vector<cv::Mat> response_maps;

			grad_pyr->quantize(quantized);
			grad_trans.spread(quantized, T, spread_quantized);
			grad_trans.computeResponseMaps(spread_quantized, response_maps);
			grad_trans.linearize(response_maps, T, lm_level);

			if (quantized_images.needed())
				quantized_images.getMatRef(l) = quantized;

			sizes[l] = quantized.size();
		}

		if (class_ids.empty())
		{
			// Match all classes
			for (TemplatesMap::const_iterator it = class_templates.begin(); it != class_templates.end(); ++it)
				matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second);
		}
		else
		{
			// Match only templates for the requested class IDs
			for (int i = 0; i < (int)class_ids.size(); ++i)
			{
				TemplatesMap::const_iterator it = class_templates.find(class_ids[i]);
				if (it != class_templates.end())
					matchClass(lm_pyramid, sizes, threshold, matches, it->first, it->second);
			}
		}

		// Sort matches by similarity, and prune any duplicates introduced by pyramid refinement
		std::sort(matches.begin(), matches.end());
		std::vector<Match>::iterator new_end = std::unique(matches.begin(), matches.end());
		matches.erase(new_end, matches.end());
#ifdef _TIMER_
		t.stop();
		double cmt=t.time();
		printf("cpu match time: %lf \n", cmt);
#endif
	}

	// Used to filter out weak matches
	struct MatchPredicate
	{
		MatchPredicate(float _threshold) : threshold(_threshold) {}
		bool operator() (const Match& m) { return m.similarity < threshold; }
		float threshold;
	};

	void Detector::matchClass(const LinearMemoryPyramid& lm_pyramid,
		const std::vector<cv::Size>& sizes,
		float threshold,
		std::vector<Match>& matches,
		const int class_id,
		const std::vector<TemplatePyramid>& template_pyramids) const
	{
#pragma omp parallel for
		for (int template_id = 0; template_id < (int)template_pyramids.size(); ++template_id)
		{
			const TemplatePyramid& tp = template_pyramids[template_id];

			// First match over the whole image at the lowest pyramid level
			// i.e. first match on the minimum size, then refined on the larger size
			const LinearMemories &lowest_lm = lm_pyramid.back();

			// Compute similarity map at lowest pyramid level
			cv::Mat similarityLowest;
			int lowest_T = T_at_level.back();
			const Template& templLowest = tp[pyramid_levels - 1];
			int num_features_lowest = (int) templLowest.features.size();

			similarity(lowest_lm, templLowest, similarityLowest, sizes.back(), lowest_T);

			// Convert user-friendly percentage to raw similarity threshold
			// NOTE: This assumes max per-feature response is 4
			// NOTE: the max per-feature response is now defined as MAX_PER_FEATURE_POINT_RESPONSE
			static const int MAX_PER_FEATURE_POINT_RESPONSE = 4;
			int raw_threshold = 
				static_cast<int>(0.5*MAX_PER_FEATURE_POINT_RESPONSE*(1+threshold/100)*num_features_lowest + 0.5f);
			/*int raw_threshold 
			= (threshold / 100.f) * (MAX_PER_FEATURE_POINT_RESPONSE * num_features_lowest) + 0.5f;*/

			// Find initial matches on the smallest size pyramid
			std::vector<Match> candidates;
			for (int r = 0; r < similarityLowest.rows; ++r)
			{	
				uchar* row = similarityLowest.ptr<uchar>(r);
				for (int c = 0; c < similarityLowest.cols; ++c)
				{
					int raw_score = static_cast<int>(row[c]);
					if (raw_score > raw_threshold)
					{
						int offset = lowest_T / 2 + (lowest_T % 2 - 1);
						int x = c * lowest_T + offset;
						int y = r * lowest_T + offset;
						float score 
							= (raw_score * 100.f) / (MAX_PER_FEATURE_POINT_RESPONSE * num_features_lowest);
						candidates.push_back(Match(x, y, score, class_id, template_id));
					}
				}
			}

			// Locally refine each match by marching up the pyramid
			for (int l = pyramid_levels - 2; l >= 0; --l)
			{
				const LinearMemories &lms = lm_pyramid[l];
				int T = T_at_level[l];
				cv::Size size = sizes[l];
				int border = 8 * T; // because loacally refine in 16*16 patches
				int offset = T / 2 + (T % 2 - 1);

				const Template& templ = tp[l];
				int max_x = size.width - templ.width - border;
				int max_y = size.height - templ.height - border;

				int num_features_curr = (int) templ.features.size();
				cv::Mat similaritiyBuff;
				for (int m = 0; m < (int)candidates.size(); ++m)
				{
					Match& match = candidates[m];
					int x = match.x * 2 + 1; 
					int y = match.y * 2 + 1;

					// Require 8 (reduced) row/cols to the up/left
					x = std::max(x, border);
					y = std::max(y, border);
					// Require 8 (reduced) row/cols to the down/left, plus the template size
					x = std::min(x, max_x);
					y = std::min(y, max_y);

					// Compute local similarity maps							
					similarityLocal(lms, templ, similaritiyBuff, size, T, cv::Point(x, y));
					
					// Find best local adjustment in 16x16 patch
					int best_score = 0;
					int best_r = -1, best_c = -1;
					for (int r = 0; r < similaritiyBuff.rows; ++r)
					{
						uchar* row = similaritiyBuff.ptr<uchar>(r);
						for (int c = 0; c < similaritiyBuff.cols; ++c)
						{
							int score = row[c];
							if (score > best_score)
							{
								best_score = score;
								best_r = r;
								best_c = c;
							}
						}
					}

					// Update current match
					match.x = (x / T - 8 + best_c) * T + offset;
					match.y = (y / T - 8 + best_r) * T + offset;
					match.similarity = (best_score * 100.f) / (MAX_PER_FEATURE_POINT_RESPONSE * num_features_curr);
				}

				// Filter out any matches that drop below the similarity threshold
				std::vector<Match>::iterator new_end 
					= std::remove_if(candidates.begin(), candidates.end(), MatchPredicate(threshold));
				candidates.erase(new_end, candidates.end());
			}

#pragma omp critical
			matches.insert(matches.end(), candidates.begin(), candidates.end());
		}
	}

	void Detector::similarity(const LinearMemories& linear_memories, const Template& templ,
		cv::Mat& dst, cv::Size size, int T) const
	{
		// 63 features or less is a special case because the max similarity per-feature is 4.
		// 255/4 = 63, so up to that many we can add up similarities in 8 bits without worrying
		// about overflow. Therefore here we use _mm_add_epi8 as the workhorse, whereas the
		// more general energy_more functions use _mm_add_epi16.
		// for usigned short 65535/4 = 16383 features
		static const int MAXIMUM_MATCHING_POINT_NUMBER = 63;
		CHECK(templ.features.size() <= MAXIMUM_MATCHING_POINT_NUMBER)
			<< "The number of features in one template is over-ranged.";

		// Decimate input image size by factor of T
		int W = size.width / T;
		int H = size.height / T;

		// feature dimensions, decimated by factor T and rounded up
		int wf = (templ.width - 1) / T + 1;
		int hf = (templ.height - 1) / T + 1;

		// span is the range over which we can shift the template around the input image
		// (span_x, span_y) is the most right-bottom position (when the left-top is origin point)
		int span_x = W - wf;
		int span_y = H - hf;

		// compute number of contiguous (in memory) pixels to check when sliding feature over
		// image. This allows template to wrap around left/right border incorrectly, so any
		// wrapped template matches must be filtered out!
		// template_positions is the end position, so add 1
		int template_positions = span_y * W + span_x + 1;

#if CV_SSE2
		volatile bool haveSSE2 = cv::checkHardwareSupport(CV_CPU_SSE2);
#if CV_SSE3
		volatile bool haveSSE3 = cv::checkHardwareSupport(CV_CPU_SSE3);
#endif
#endif
		cv::Mat dst_continuous = cv::Mat::zeros(1, H*W, CV_8U);
		uchar *dst_ptr = dst_continuous.ptr<uchar>();

		// compute the similarity measure for this template by accumulating the contribution of
		// each feature
		for (int i = 0; i < (int)templ.features.size(); ++i)
		{
			// Add the linear memory at the appropriate offset computed from the location of
			// the feature in the template
			const Feature &f = templ.features[i];
			// Discard feature if out of bounds
			if (f.x < 0 || f.x >= size.width || f.y < 0 || f.y >= size.height)
				continue;
			const uchar* lm_ptr = accessLinearMemory(linear_memories, f, T, W);

			int j=0;
#if CV_SSE2	
#if CV_SSE3
			if (haveSSE3)
			{
				// Use LDDQU for fast unaligned load
				for(; j<template_positions-15; j+=16)
				{
					__m128i aligned1 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr + j));
					__m128i* curr_dst_ptr = reinterpret_cast<__m128i*>(dst_ptr + j);
					__m128i aligned2 = _mm_lddqu_si128(curr_dst_ptr);
					__m128i add_val = _mm_add_epi8(aligned2, aligned1);
					_mm_storeu_si128(curr_dst_ptr, add_val);
				}
			}else
#endif
			if (haveSSE2)
			{
				// Use unaligned loads if possible
				for(; j<template_positions-15; j+=16)
				{
					__m128i aligned1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr + j));
					__m128i* curr_dst_ptr = reinterpret_cast<__m128i*>(dst_ptr + j);
					__m128i aligned2 = _mm_loadu_si128(curr_dst_ptr);
					__m128i add_val = _mm_add_epi8(aligned2, aligned1);
					_mm_storeu_si128(curr_dst_ptr, add_val);
				}
			}

#endif
			for(; j<template_positions; ++j)
			{
				dst_ptr[j] = lm_ptr[j]+dst_ptr[j];
			}
		}

		dst = dst_continuous.reshape(0, H);
	}

	void Detector::similarityLocal(const LinearMemories& linear_memories, const Template& templ,
		cv::Mat& dst, cv::Size size, int T, cv::Point center) const
	{
		// Similar to whole-image similarity() above. This version takes a position 'center'
		// and computes the energy in the 16x16 patch centered on it.
		static const int MAXIMUM_MATCHING_POINT_NUMBER = 63;
		CHECK(templ.features.size() <= MAXIMUM_MATCHING_POINT_NUMBER)
			<< "The number of features in one template is over-ranged.";

		// Compute the similarity map in a 16x16 patch around center
		int W = size.width / T;
		dst = cv::Mat::zeros(16, 16, CV_8U);

		// Offset each feature point by the requested center. Further adjust to (-8,-8) from the
		// center to get the top-left corner of the 16x16 patch.
		// NOTE: We make the offsets multiples of T to agree with results of the original code.
		int offset_x = (center.x / T - 8) * T;
		int offset_y = (center.y / T - 8) * T;

#if CV_SSE2
		volatile bool haveSSE2 = cv::checkHardwareSupport(CV_CPU_SSE2);
#if CV_SSE3
		volatile bool haveSSE3 = cv::checkHardwareSupport(CV_CPU_SSE3);
#endif
		__m128i* dst_ptr_sse = dst.ptr<__m128i>();
#endif

		for (int i = 0; i < (int)templ.features.size(); ++i)
		{
			Feature f = templ.features[i];
			f.x += offset_x;
			f.y += offset_y;
			// Discard feature if out of bounds, possibly due to applying the offset
			if (f.x < 0 || f.y < 0 || f.x >= size.width || f.y >= size.height)
				continue;

			const uchar* lm_ptr = accessLinearMemory(linear_memories, f, T, W);

			// Process whole row at a time if vectorization possible
#if CV_SSE2
#if CV_SSE3
			if (haveSSE3)
			{
				// LDDQU may be more efficient than MOVDQU for unaligned load of 16 responses from current row
				for (int row = 0; row < 16; ++row)
				{
					__m128i aligned = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(lm_ptr));
					dst_ptr_sse[row] = _mm_add_epi8(dst_ptr_sse[row], aligned);
					lm_ptr += W; // Step to next row
				}
			}
			else
#endif
			if (haveSSE2)
			{
				// Fall back to MOVDQU
				for (int row = 0; row < 16; ++row)
				{
					__m128i aligned = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lm_ptr));
					dst_ptr_sse[row] = _mm_add_epi8(dst_ptr_sse[row], aligned);
					lm_ptr += W; // Step to next row
				}
			}
			else
#endif			
			{
				uchar* dst_ptr = dst.ptr<uchar>();
				for (int row = 0; row < 16; ++row)
				{
					for (int col = 0; col < 16; ++col)
						dst_ptr[col] = uchar(dst_ptr[col] + lm_ptr[col]);
					dst_ptr += 16;
					lm_ptr += W;
				}
			}
		}
	}

	const uchar* Detector::accessLinearMemory(const LinearMemories& linear_memories,
		const Feature& f, int T, int W) const
	{
		// Retrieve the TxT grid of linear memories associated with the feature label
		const cv::Mat& memory_grid = linear_memories[f.label];

		CHECK(memory_grid.rows == T*T) << "Invalid linear memory for current feature.";
		CHECK(f.x >= 0 && f.y >= 0) << "Feature coordinates over-ranged.";

		// The LM we want is at (x%T, y%T) in the TxT grid (stored as the rows of memory_grid)
		int grid_x = f.x % T;
		int grid_y = f.y % T;
		int grid_index = grid_y * T + grid_x;
		CHECK(grid_index >= 0 && grid_index < memory_grid.rows)
			<< "Invalid row index for linear memory.";
		const uchar* memory = memory_grid.ptr<uchar>(grid_index);

		// Within the LM, the feature is at (x/T, y/T). W is the "width" of the LM, the
		// input image width decimated by T.
		int lm_x = f.x / T;
		int lm_y = f.y / T;
		int lm_index = lm_y * W + lm_x;
		CHECK(lm_index >= 0 && lm_index < memory_grid.cols)
			<< "Invalid column index for linear memory.";

		return memory + lm_index;
	}

	/***detector factory******detector factory******detector factory******detector factory***/
	/***detector factory******detector factory******detector factory******detector factory***/
	/***detector factory******detector factory******detector factory******detector factory***/
	static const int T_DEFAULTS[] = {5,8};
	cv::Ptr<Detector> getDefaultLINE2D()
	{
		return new Detector(std::vector<int>(T_DEFAULTS, T_DEFAULTS + 2));
	}
}