#include "..\..\include\util\GLMDisplay.h"
#include "..\..\include\util\CameraCalibration.h"

#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <glog\logging.h>

namespace rl2d
{
	// rectify initial pose from template images,
	// the pose attached to each template image are the pose of object which is at the center of camera focus,
	// when the object is detectd at the location where the object is not at the image center, then need rectify
	void RectifyPose(const cv::Matx33f &camIntinsic, 
		const cv::Matx61f &pose, 
		const cv::Point2f &objCenterInImg, 
		cv::Matx61f &newPose)
	{
		float tx = pose.val[3]; float ty = pose.val[4]; float tz = pose.val[5];

		float u = objCenterInImg.x; float v = objCenterInImg.y;  

		float fx = camIntinsic.val[0]; float cx = camIntinsic.val[2]; 
		float fy = camIntinsic.val[4]; float cy = camIntinsic.val[5];

		float newTx = (u-cx)*tz/fx;
		float newTy = (v-cy)*tz/fy;

		newPose.val[0] = pose.val[0];
		newPose.val[1] = pose.val[1]; 
		newPose.val[2] = pose.val[2]; 

		newPose.val[3] = newTx;
		newPose.val[4] = newTy; 
		newPose.val[5] = tz; 
	}

	// project 3d obj model to 2d image
	void DrawCoordinateSystem(const CameraCalibration &camCalib,
		const std::vector<float> &rVec, 
		const std::vector<float> &tVec, 
		const int axisLength,
		cv::Mat &img)
	{
		// camera parameters and distortion coefficients
		const cv::Matx33f camIntrinsic = camCalib.getIntrinsic();
		const cv::Matx<float,5,1> camDistortion = camCalib.getDistorsion();

		// coordiantes points in 3d
		std::vector<cv::Point3f> coor_points_3d;
		coor_points_3d.push_back(cv::Point3f(0.f,0.f,0.f));
		coor_points_3d.push_back(cv::Point3f(axisLength,0.f,0.f));
		coor_points_3d.push_back(cv::Point3f(0.f,axisLength,0.f));
		coor_points_3d.push_back(cv::Point3f(0.f,0.f,axisLength));

		// project above points to 2d
		std::vector<cv::Point2f> coor_points_2d;
		cv::projectPoints(coor_points_3d, rVec, tVec, camIntrinsic, camDistortion, coor_points_2d);

		cv::line(img, cv::Point((int)coor_points_2d[0].x, (int)coor_points_2d[0].y),
			cv::Point((int)coor_points_2d[1].x, (int)coor_points_2d[1].y), CV_RGB(255,0,0),2);
		cv::line(img, cv::Point((int)coor_points_2d[1].x, (int)coor_points_2d[1].y),
			cv::Point((int)coor_points_2d[1].x, (int)coor_points_2d[1].y), CV_RGB(255,0,0),5);

		cv::line(img, cv::Point((int)coor_points_2d[0].x, (int)coor_points_2d[0].y),
			cv::Point((int)coor_points_2d[2].x, (int)coor_points_2d[2].y), CV_RGB(0,255,0),2);
		cv::line(img, cv::Point((int)coor_points_2d[2].x, (int)coor_points_2d[2].y),
			cv::Point((int)coor_points_2d[2].x, (int)coor_points_2d[2].y), CV_RGB(0,255,0),5);

		cv::line(img, cv::Point((int)coor_points_2d[0].x, (int)coor_points_2d[0].y),
			cv::Point((int)coor_points_2d[3].x, (int)coor_points_2d[3].y), CV_RGB(0,0,255),2);
		cv::line(img, cv::Point((int)coor_points_2d[3].x, (int)coor_points_2d[3].y),
			cv::Point((int)coor_points_2d[3].x, (int)coor_points_2d[3].y), CV_RGB(0,0,255),5);
	}

	void ProjectModel2Img(const cv::Ptr<Detector> &detector, 
		const Match &m, 
		const CameraCalibration &camCalib,
		const std::map<int, std::vector<cv::Matx61f> > &poses, 
		const std::map<int, GLMmodel*> &objModels, 
		const cv::Scalar &color,
		cv::Mat &dst)
	{
		//get template size
		const std::vector<Template>& templates = detector->getTemplates(m.class_id, m.template_id);
		int templWidth = templates[0].width;
		int templHeight = templates[0].height;

		//object center in image coordinate system
		float objCenterX = m.x + templWidth/2.0;
		float objCenterY = m.y + templHeight/2.0;

		// camera parameters and distortion coefficients
		const cv::Matx33f camIntrinsic = camCalib.getIntrinsic();
		const cv::Matx<float,5,1> camDistortion = camCalib.getDistorsion();

		//get pose and rectify it
		cv::Matx61f pose = cv::Matx61f::eye();
		FindPose(poses, m,pose);
		cv::Matx61f rectifiedPose = cv::Matx61f::eye();

		//std::vector<float> rVec(pose.val, pose.val+3);
		//std::vector<float> tVec(pose.val+3, pose.val+6);
		RectifyPose(camIntrinsic, pose, cv::Point2f(objCenterX, objCenterY), rectifiedPose);
		std::vector<float> rVec(rectifiedPose.val, rectifiedPose.val+3);
		std::vector<float> tVec(rectifiedPose.val+3, rectifiedPose.val+6);

		//printf("%f %f %f %f %f %f\n", rVec[0],rVec[1],rVec[2],tVec[0],tVec[1],tVec[2]);

		// get obj vertices data
		GLMmodel *objModel = NULL;
		FindModel(objModels, m, &objModel);
		unsigned int verticesNum = objModel->numvertices;
		GLfloat *vertices = objModel->vertices;
		std::vector<cv::Point3f> points3d(verticesNum);
		for(size_t i=1; i<=verticesNum; ++i)
		{
			points3d[i-1] = cv::Point3f(vertices[3 * i + 0],
				vertices[3 * i + 1],
				vertices[3 * i + 2]);
		}

		//get obj edges data
		unsigned int edgesNum = objModel->numLines;
		GLMLine *edges = objModel->lines;

		// project above points to 2d
		std::vector<cv::Point2f> points2d;
		points2d.clear();
		cv::projectPoints(points3d, rVec, tVec, camIntrinsic, camDistortion, points2d);

		//draw the 2d points
		for(size_t j=0; j<verticesNum; ++j)
		{
			cv::line(dst, cv::Point((int)points2d[j].x, (int)points2d[j].y),
				cv::Point((int)points2d[j].x, (int)points2d[j].y), color, 2);
		}

		// and draw the edges
		for(size_t j=0; j<edgesNum; ++j)
		{
			unsigned int p0= edges[j].vindices[0]-1;
			unsigned int p1 = edges[j].vindices[1]-1;
			cv::line(dst, cv::Point((int)points2d[p0].x, (int)points2d[p0].y),
				cv::Point((int)points2d[p1].x, (int)points2d[p1].y), color, 1);
		}

		// draw the coordinates system
		int axisLength = static_cast<int>(1.5*glmMaxRadius(objModel));
		DrawCoordinateSystem(camCalib, rVec, tVec, axisLength, dst);
	}

	// find pose for a specific match
	void FindPose(const std::map<int, std::vector<cv::Matx61f> > &poses, 
		const Match &match, 
		cv::Matx61f &pose)
	{
		//get pose
		std::map<int, std::vector<cv::Matx61f> >::const_iterator pose_iter = poses.find(match.class_id);
		CHECK( pose_iter!=poses.end()) << "No corresponding pose data.";

		pose= pose_iter->second[match.template_id];
	}

	// find obj model for a specific match
	void FindModel(const std::map<int, GLMmodel*> &objModels,
		const Match &match, 
		GLMmodel** model)
	{
		std::map<int, GLMmodel*>::const_iterator obj_iter = objModels.find(match.class_id);
		CHECK( obj_iter!=objModels.end() ) << "No corresponding obj model.";

		*model = obj_iter->second;
	}
}