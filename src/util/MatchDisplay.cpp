#include "..\..\include\util\MatchDisplay.h"
#include <glog\logging.h>
#include <map>

namespace rl2d
{
	const ObjectConfig& objectConfigById(const ObjectsConfig &objectsConfig, const int classId)
	{
		const int classNum = objectsConfig.object_config_size();
		for(int i=0; i<classNum; ++i)
		{
			const ObjectConfig &objectConfig = objectsConfig.object_config(i);
			if(objectConfig.id()==classId)
				return objectConfig;
		}
	}

	// draw each feature point with a direction
	static const int length=10;
	static cv::Point lablePointsMap[8][2] = 
	{
		{cv::Point(-5, 0), cv::Point(5, 0)},
		{cv::Point(-4, -2), cv::Point(4, 2)},
		{cv::Point(-3, -3), cv::Point(3, 3)},
		{cv::Point(-2, -4), cv::Point(2, 4)},
		{cv::Point(0, -5), cv::Point(0, 5)},
		{cv::Point(2, -4), cv::Point(-2, 4)},
		{cv::Point(3, -3), cv::Point(-3, 3)},
		{cv::Point(4, -2), cv::Point(-4, 2)},
	};
	void DrawFeatures(const std::vector<Feature> &features,  
		const cv::Point &offset,
		const cv::Scalar &color,
		cv::Mat &displayImg)
	{
		for(int i=0; i<(int)features.size(); ++i)
		{
			const Feature &f = features[i];
			cv::Point position(f.x + offset.x, f.y + offset.y);
			int label = f.label;
			cv::line(displayImg, lablePointsMap[label][0]+position, lablePointsMap[label][1]+position,
				CV_RGB(255,255,255), 1, CV_AA);
		}
	}

	bool DrawMatches(const std::vector<Match> &matches,  
		const cv::Ptr<Detector> &detector,
		const ObjectsConfig &objectsConfig,
		int draw_match_num, 
		cv::Mat &displayImg)
	{
		CHECK(draw_match_num>0) << "Invalid match number for drawing on original image.";

		draw_match_num = draw_match_num>(int)matches.size() ? (int)matches.size():draw_match_num;
		//LOG(INFO)<<"Draw "<<draw_match_num<<" matching results.";

		for(int i=0; i<draw_match_num; ++i)
		{
			const Match &m = matches[i];
			int x = m.x;
			int y = m.y;
			float similarity = m.similarity;
			int class_id = m.class_id;
			int template_id = m.template_id;

			// get the predefined color from configuration file
			const const ObjectConfig& objectConfig = objectConfigById(objectsConfig, class_id);
			const ObjectConfig_RGB &bb_color = objectConfig.bounding_box_color();

			std::vector<Template> templatePyramid = detector->getTemplates(class_id, template_id);

			int templateWidth = 0;
			int templateHeight = 0;
			if(templatePyramid.size()>0)
			{
				std::vector<Feature> &fs = templatePyramid[0].features;
				templateWidth = templatePyramid[0].width;
				templateHeight = templatePyramid[0].height;

				// draw matching points in the largest scale pyramid
				DrawFeatures(fs, cv::Point(x,y), CV_RGB(bb_color.r(),bb_color.g(),bb_color.b()), displayImg);
			}else
				return false;

			// draw the best match with bold bounding box
			if( i==0)
			{
				cv::rectangle(displayImg, cv::Rect(x, y, templateWidth, templateHeight),
					CV_RGB(0,0,0), 3);
				cv::rectangle(displayImg, cv::Rect(x, y, templateWidth, templateHeight), 
					CV_RGB(bb_color.r(),bb_color.g(),bb_color.b()), 2);
			}else
			{ 
				cv::rectangle(displayImg, cv::Rect(x, y, templateWidth, templateHeight), 
					CV_RGB(0,0,0), 2);
				cv::rectangle(displayImg, cv::Rect(x, y, templateWidth, templateHeight), 
					CV_RGB(bb_color.r(),bb_color.g(),bb_color.b()), 1);
			}
		}

		return true;
	}

	void SelectBestMatch4EachClass(const std::vector<Match> &matches, std::vector<Match> &bestMatches)
	{
		bestMatches.clear();
		size_t matchesNum = matches.size();
		std::set<int> visited;

		if(matchesNum>0)
		{
			for(size_t i=0;i<matchesNum;++i)
			{
				const Match &m = matches[i];
				if(visited.insert(m.class_id).second)
					bestMatches.push_back(m);
			}			
		}
	}

	/*
	group the matches by their class id, this is for multi-class instance detection,
	so matches for the same class id will be grouped in one vect.
	*/
	void GroupMatchesByClass(const std::vector<Match> &matches, 
		std::map<int, std::vector<Match*> > &classMatchMap)
	{
		classMatchMap.clear();
		size_t matchesNum = matches.size();
		std::set<int> visited;

		if(matchesNum>0)
		{
			for(size_t i=0;i<matchesNum;++i)
			{
				const Match &m = matches[i];
				const int cls_id = m.class_id;
				if(visited.insert(cls_id).second)
				{
					std::vector<Match*> firstInThisClass;
					firstInThisClass.push_back(const_cast<Match*>(&m));
					classMatchMap.insert( std::make_pair(m.class_id, firstInThisClass) );
				}
				else
				{
					std::map<int, std::vector<Match*> >::iterator cur_group = classMatchMap.find(cls_id);
					cur_group->second.push_back(const_cast<Match*>(&m));					
				}
			}		
		}
	}

	/* 
	find the corresponding bounding boxes for each grouped matches
	*/
	void CvtGroupedMatches2Rects(const std::map<int,std::vector<Match*> > &classMatchMap, 
		const cv::Ptr<Detector> &detector,
		std::map<int, std::vector<cv::Rect> > &classRectMap)
	{
		classRectMap.clear();

		int classMatchMapSize = static_cast<int>(classMatchMap.size());

		for(std::map<int, std::vector<Match*> >::const_iterator mit = classMatchMap.begin(),
			mend = classMatchMap.end(); mit != mend; ++mit)
		{
			size_t matchNum = mit->second.size();
			std::vector<cv::Rect> cur_rects(matchNum);
			for(size_t i = 0; i<matchNum; ++i)
			{
				std::vector<Template> templatePyramid 
					= detector->getTemplates(mit->second[i]->class_id, mit->second[i]->template_id);
				int templateWidth = templatePyramid[0].width;
				int templateHeight = templatePyramid[0].height;
				cur_rects[i] = cv::Rect(mit->second[i]->x, mit->second[i]->y, templateWidth, templateHeight);
			}
			classRectMap.insert(std::make_pair(mit->first, cur_rects));
		}

	}

	/* 
	class for grouping object detection candidates, 
	instance of the class is to be passed to cv::partition,
	measure two bounding boxes similarity
	*/
	struct SimilarRects
	{
		SimilarRects(float a_eps) : eps(a_eps) {}

		// define the similarity of two Rect
		inline bool operator()(const cv::Rect& r1, const cv::Rect& r2) const
		{
			float delta = eps*(std::min(r1.width, r2.width) + std::min(r1.height, r2.height))*0.5;
			return std::abs(r1.x - r2.x) <= delta &&
				std::abs(r1.y - r2.y) <= delta &&
				std::abs(r1.x + r1.width - r2.x - r2.width) <= delta &&
				std::abs(r1.y + r1.height - r2.y - r2.height) <= delta;
		}

		float eps;
	};
	void GroupRects(std::vector<cv::Rect> &rectList, 
		std::vector<int> &labels, 
		std::set<int> &validLabelValues,  
		int groupThreshold, float eps)
	{
		if( groupThreshold <= 0 || rectList.empty() )
			return;

		// cluster the rectangle when four points are close
		labels.clear();
		validLabelValues.clear();
		int nclasses = partition(rectList, labels, SimilarRects(eps));
		std::vector<cv::Rect> rrects(nclasses);
		std::vector<int> rweights(nclasses, 0);
		std::vector<int> rejectLevels(nclasses, 0); // what's this for?
		std::vector<double> rejectWeights(nclasses, DBL_MIN); // and this
		int i, j, nlabels = (int)labels.size();
		for( i = 0; i < nlabels; i++ )
		{
			// sum all the retangles
			int cls = labels[i];
			rrects[cls].x += rectList[i].x;
			rrects[cls].y += rectList[i].y;
			rrects[cls].width += rectList[i].width;
			rrects[cls].height += rectList[i].height;

			// counter the number of retangles in one class
			rweights[cls]++;
		}

		for( i = 0; i < nclasses; i++ )
		{
			cv::Rect r = rrects[i];
			float s = 1.f/rweights[i];
			rrects[i] = cv::Rect(cv::saturate_cast<int>(r.x*s),
				cv::saturate_cast<int>(r.y*s),
				cv::saturate_cast<int>(r.width*s),
				cv::saturate_cast<int>(r.height*s));
		}

		rectList.clear();

		for( i = 0; i < nclasses; i++ )
		{
			cv::Rect r1 = rrects[i];
			int n1 = rweights[i];
			double w1 = rejectWeights[i];
			if( n1 < groupThreshold )
				continue;
			// filter out small rectangles inside large rectangles
			for( j = 0; j < nclasses; j++ )
			{
				int n2 = rweights[j];

				if( j == i || n2 < groupThreshold )
					continue;
				cv::Rect r2 = rrects[j];

				int dx = cv::saturate_cast<int>( r2.width * eps );
				int dy = cv::saturate_cast<int>( r2.height * eps );

				if( i != j &&
					r1.x >= r2.x - dx &&
					r1.y >= r2.y - dy &&
					r1.x + r1.width <= r2.x + r2.width + dx &&
					r1.y + r1.height <= r2.y + r2.height + dy )
					break;
			}

			if( j == nclasses )
			{
				rectList.push_back(r1);
				validLabelValues.insert(i);
			}
		}
	}

	void MergeMatches(const std::vector<Match> &matches, 
		const cv::Ptr<Detector> &detector,
		std::vector<Match> &mergedMatches, 
		std::vector<cv::Rect> &mergedRects,
		int eachGroupMinNum,
		double eps)
	{
		mergedMatches.clear();
		mergedRects.clear();

		// first groud the matches by id
		std::map<int, std::vector<Match*> > classMatchMap;
		GroupMatchesByClass(matches, classMatchMap);

		// then get the bounding boxes of grouped matches
		std::map<int, std::vector<cv::Rect> > classRectMap;
		CvtGroupedMatches2Rects(classMatchMap, detector, classRectMap);

		// then groud the rectangle bounding boxes for each class
		for(std::map<int, std::vector<cv::Rect> >::iterator iter = classRectMap.begin();
			iter != classRectMap.end(); ++iter)
		{
			std::vector<int> lables;
			std::set<int> validLableValues;
			GroupRects(iter->second, lables, validLableValues, eachGroupMinNum, eps);
			mergedRects.insert(mergedRects.end(), iter->second.begin(), iter->second.end());

			// select a best match for each grouped rects
			int lableNum = static_cast<int>(lables.size());
			std::map<int, std::vector<Match*> >::iterator curr_matches_iter = classMatchMap.find(iter->first);
			for(std::set<int>::const_iterator set_iter = validLableValues.begin();
				set_iter !=  validLableValues.end(); ++set_iter)
			{
				int validLableValue = (*set_iter);
				for(int i=0; i<lableNum; ++i)
				{
					// fist the first valid match
					if(lables[i]==validLableValue)
					{
						mergedMatches.push_back( *(curr_matches_iter->second[i]) );
						break;
					}
				}
			}
		}
	}

	void DrawMergedMatches(const std::vector<Match> &matches, 
		const std::vector<cv::Rect> &boundingBoxes,
		const ObjectsConfig &objectsConfig,
		int draw_match_num, cv::Mat &displayImg)
	{
		CV_Assert(draw_match_num>0);
		CV_Assert(matches.size() == boundingBoxes.size());

		draw_match_num = draw_match_num>(int)matches.size() ? (int)matches.size():draw_match_num;


		for(int i=0; i<draw_match_num; ++i)
		{
			const Match &m = matches[i];
			// draw the match with bold bounding box
			const const ObjectConfig& objectConfig = objectConfigById(objectsConfig, m.class_id);
			const ObjectConfig_RGB &bb_color = objectConfig.bounding_box_color();
			cv::rectangle(displayImg, boundingBoxes[i], CV_RGB(0,0,0), 3);
			cv::rectangle(displayImg, boundingBoxes[i], CV_RGB(bb_color.r(),bb_color.g(),bb_color.b()), 2);
		}
	}
}