#include "opencv2/opencv.hpp"
#include "../../include/util/MatchRefiner.h"
#include "../../include/util/Filex.h"

namespace rl2d
{
	void calcHSHist(const cv::Mat &srcImg, cv::Mat& hsHist, int HBIN, int SBIN, cv::Mat &mask)
	{
		if(mask.empty())
		{
			hsHist = cv::Mat(HBIN, SBIN, CV_32FC1,cv::Scalar(0));

			cv::Mat img;
			cv::cvtColor(srcImg, img, CV_BGR2HSV);

			float count = 0;
			for(int m=0; m<img.rows; ++m)
			{
				for(int n=0; n<img.cols; ++n)
				{
					if( (srcImg.at<cv::Vec3b>(m,n)[0]==0 && srcImg.at<cv::Vec3b>(m,n)[1]==0 &&srcImg.at<cv::Vec3b>(m,n)[2]==0) ||
						(srcImg.at<cv::Vec3b>(m,n)[0]==255 && srcImg.at<cv::Vec3b>(m,n)[1]==255 &&srcImg.at<cv::Vec3b>(m,n)[2]==255) )
						continue;

					if(img.at<cv::Vec3b>(m,n)[0] != 0 &&
						img.at<cv::Vec3b>(m,n)[1] != 0)
					{
						int _h = img.at<cv::Vec3b>(m,n)[0]*HBIN/180;
						int _s = img.at<cv::Vec3b>(m,n)[1]*SBIN/256;
						hsHist.at<float>(_h,_s) += 1;
						count++;
					}					
				}
			}

			for(int i=0; i<HBIN; i++)
			{
				for(int j=0; j<SBIN; j++)
				{
					hsHist.at<float>(i,j) /= count;
				}
			}
		}
		else
		{
			hsHist = cv::Mat(HBIN, SBIN, CV_32FC1,cv::Scalar(0));

			cv::Mat img;
			cv::cvtColor(srcImg, img, CV_BGR2HSV);

			int count = 0;
			for(int m=0; m<img.rows; ++m)
			{
				for(int n=0; n<img.cols; ++n)
				{
					if(mask.at<cv::Vec3b>(m,n)[0]!=0 && mask.at<cv::Vec3b>(m,n)[1]!=0 &&
						mask.at<cv::Vec3b>(m,n)[2]!=0)
					{
						if(img.at<cv::Vec3b>(m,n)[0] != 0 &&
							img.at<cv::Vec3b>(m,n)[1] != 0)
						{
							int _h = img.at<cv::Vec3b>(m,n)[0]*HBIN/180;
							int _s = img.at<cv::Vec3b>(m,n)[1]*SBIN/256;
							hsHist.at<float>(_h,_s) += 1;
							count++;
						}				
					}
				}
			}

			for(int i=0; i<HBIN; i++)
			{
				for(int j=0; j<SBIN; j++)
				{
					hsHist.at<float>(i,j) /= count;
				}
			}
		}
	}

	float histCompare(const cv::Mat &hist1, const cv::Mat &hist2, int HBIN, int SBIN)
	{
		float histSimilarity = 0;
		float hist1Mod = 0;
		float hist2Mod = 0;

		for(int u=0; u<HBIN; ++u)
		{
			for(int v=0; v<SBIN; ++v)
			{
				histSimilarity += hist1.at<float>(u,v)*hist2.at<float>(u,v);
				hist1Mod += hist1.at<float>(u,v)*hist1.at<float>(u,v);
				hist2Mod += hist2.at<float>(u,v)*hist2.at<float>(u,v);
			}
		}	

		if(hist1Mod>0 && hist2Mod>0)
			histSimilarity = histSimilarity/(sqrt(hist1Mod)*sqrt(hist2Mod));
		printf("---%f %f %f---\n", hist1Mod, hist2Mod, histSimilarity);
		return histSimilarity;
	}

	void offlineCalcHSHist(const std::string &imgsPath, int HBIN, int SBIN, cv::Mat &hist)
	{
		hist = cv::Mat(HBIN, SBIN, CV_32FC1,cv::Scalar(0));

		// load images	
		std::vector<std::string> fileName;
		if( GetFileNames(imgsPath, fileName) )
		{
			float pxCount = 0;
			for(size_t k=0; k<fileName.size(); ++k)
			{
				std::string file = imgsPath + "/" + fileName[k];
				cv::Mat img=cv::imread( file );
				cv::Mat hsvimg;
				cv::cvtColor(img, hsvimg, CV_BGR2HSV);

				for(int m=0; m<hsvimg.rows; ++m)
				{
					for(int n=0; n<hsvimg.cols; ++n)
					{
						if( (img.at<cv::Vec3b>(m,n)[0]==0 && img.at<cv::Vec3b>(m,n)[1]==0 && img.at<cv::Vec3b>(m,n)[2]==0) ||
							(img.at<cv::Vec3b>(m,n)[0]==255 && img.at<cv::Vec3b>(m,n)[1]==255 && img.at<cv::Vec3b>(m,n)[2]==255) )
							continue;

						int _h = hsvimg.at<cv::Vec3b>(m,n)[0]*HBIN/180;
						int _s = hsvimg.at<cv::Vec3b>(m,n)[1]*SBIN/256;
						hist.at<float>(_h,_s) += 1;
						pxCount++;
					}
				}
			}

			for(int m=0; m<HBIN; m++)
			{
				for(int n=0; n<SBIN; n++)
				{
					hist.at<float>(m,n) /= pxCount;
				}
			}
		}		
	}

	void MatchRefiner::offlineCalculateHSHist(const ObjectsConfig &objectsConfig, 
		int HBIN, int SBIN,
		std::map<int, std::vector<cv::Mat> > &hsHistograms)
	{
		const int objectNum = objectsConfig.object_config_size();
		for(int i=0; i<objectNum; ++i)
		{
			const ObjectConfig &objectConfig = objectsConfig.object_config(i);
			const int classId = objectConfig.id();

			const int colorHistNum = objectConfig.color_hist_dir_size();
			std::vector<cv::Mat> hsHists(colorHistNum);
			for(int j=0; j<colorHistNum; ++j)
			{
				const std::string &colorHistImgsDir = objectConfig.color_hist_dir(j);

				cv::Mat hsHist;
				offlineCalcHSHist(colorHistImgsDir, HBIN, SBIN, hsHist);
				hsHists[j] = hsHist;

				std::cout<<hsHist<<std::endl;
				cv::waitKey();
			}

			hsHistograms.insert(std::pair<int, std::vector<cv::Mat> >(classId, hsHists));
		}
	}

	void MatchRefiner::refine(std::vector<Match>& candidatesMatches,
		std::vector<cv::Rect> &correspondingRects,
		const cv::Mat& frameImg,
		const std::map<int, std::vector<cv::Rect> > &templateImgBoundingBoxes,
		std::map<int, std::vector<cv::Mat> > &hsHists,
		int HBIN, int SBIN,
		float match_threshold)
	{
		std::vector<Match> refinedMatches;
		std::vector<cv::Rect> refinedCorrespondingRects;

		int candidates_num = (int)candidatesMatches.size();

		for(int i=0; i<candidates_num; ++i)
		{
			const Match &m = candidatesMatches[i];
			int x = m.x;
			int y = m.y;
			float similarity = m.similarity;
			int class_id = m.class_id;
			int template_id = m.template_id;

			std::map<int, std::vector<cv::Mat> >::iterator currHistsMap = hsHists.find(class_id);
			std::vector<cv::Mat> &currHists = currHistsMap->second;

			const std::map<int, std::vector<cv::Rect> >::const_iterator currRectMap = templateImgBoundingBoxes.find(class_id);
			const cv::Rect &currTmplRect = currRectMap->second[template_id];

			// candidate area for object
			int r_width = (x+currTmplRect.width)>frameImg.cols-1 ? (frameImg.cols-x):currTmplRect.width;
			int r_height = (y+currTmplRect.height)>frameImg.rows-1 ? (frameImg.rows-y):currTmplRect.height;
			cv::Rect frame_roi = cv::Rect(x, y, r_width, r_height);
			cv::Mat frame_cutout = frameImg(frame_roi);

			//calculate the histogram for cutout object
			cv::Mat cutoutHSHist;
			calcHSHist(frame_cutout, cutoutHSHist, HBIN, SBIN, cv::Mat());

			//match the histogram
			float histSimilarity = 0;

			for(int j=0; j<currHists.size(); ++j)
			{
				float histScore = histCompare(cutoutHSHist, currHists[j], HBIN, SBIN);
				histSimilarity = histSimilarity>histScore ? histSimilarity:histScore;
			}

			
			if(histSimilarity>match_threshold)
			{
				refinedMatches.push_back( m );
				refinedCorrespondingRects.push_back( correspondingRects[i] );
			}
			cv::waitKey();
		}

		std::vector<Match>().swap(candidatesMatches);
		std::vector<cv::Rect>().swap(correspondingRects);

		candidatesMatches = refinedMatches;
		correspondingRects = refinedCorrespondingRects;
	}
}