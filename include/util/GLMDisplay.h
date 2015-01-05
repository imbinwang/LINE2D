#ifndef _GLMDisplay_H_
#define _GLMDisplay_H_

#include <vector>
#include <map>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include "..\Match.h"
#include "..\Detector.h"
#include "CameraCalibration.h"
#include "GLM.h"
#include "Config.pb.h"


namespace rl2d
{
	// rectify initial pose from template images
	void RectifyPose(const cv::Matx33f &camIntinsic, 
		const cv::Matx61f &pose, 
		const cv::Point2f &objCenterInImg, 
		cv::Matx61f &newPose);

	// project 3d coordinate system to 2d image
	void DrawCoordinateSystem(const CameraCalibration &camCalib,
		const std::vector<float> &rVec, 
		const std::vector<float> &tVec, 
		const int axisLength,
		cv::Mat &img);

	// project 3d GlmModel to 2d image
	void ProjectModel2Img(const cv::Ptr<Detector> &detector, 
		const Match &m, 
		const CameraCalibration &camCalib,
		const std::map<int, std::vector<cv::Matx61f> > &poses, 
		const std::map<int, GLMmodel*> &objModels,
		const cv::Scalar &color,
		cv::Mat &dst);

	// find the corresponding pose for a specific match
	void FindPose(const std::map<int, std::vector<cv::Matx61f> > &poses, 
		const Match &match, 
		cv::Matx61f &pose);

	// find the corresponding obj model for a specific match
	void FindModel(const std::map<int, GLMmodel*> &objModels,
		const Match &match, 
		GLMmodel** model);
}

#endif // _GLMDisplay_H_