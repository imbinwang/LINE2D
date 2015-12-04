#include "..\..\include\util\OfflineTemplatesLoader.h"
#include "..\..\include\util\Filex.h"
#include "..\..\include\util\GLRender.h"

#include <fstream>
#include <opencv2\highgui\highgui.hpp>

#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <glog\logging.h>

namespace rl2d
{
	OfflineTemplatesLoader::OfflineTemplatesLoader()
	{
	}

	OfflineTemplatesLoader::~OfflineTemplatesLoader()
	{
	}

#if 0
	// load template images
	bool OfflineTemplatesLoader::loadTemplateImages(
		const ObjectsConfig &objectsConfig,
		cv::Ptr<Detector> &detector, 
		std::map<int, std::vector<cv::Mat> > &images, 
		std::map<int, std::vector<cv::Rect> > &boundingBoxes)
	{
		const int objectNum = objectsConfig.object_config_size();
		CHECK(objectNum>0) << "No template images.";

#pragma omp parallel for
		for(int i=0; i<objectNum; ++i)
		{
			const ObjectConfig &objectConfig = objectsConfig.object_config(i);
			const int classId = objectConfig.id();
			const std::string &className = objectConfig.name();
			const std::string &imagesDir = objectConfig.img_dir();

			//read images
			std::vector<std::string> fileName;
			CHECK(GetFileNames(imagesDir, fileName))
				<< "Cannot get files name in specific directory";

			std::vector<cv::Mat> imgs;
			std::vector<cv::Rect> bbs;
			for(int k=0; k<(int)fileName.size(); ++k)
			{
				std::string file = imagesDir + "/" + fileName[k];
				cv::Mat img=cv::imread( file );					

				//extract a new template for current image
				cv::Rect tmplBoundingBox;
				int template_id = detector->addTemplate(img, classId, &tmplBoundingBox);

				//save all images, it is not a good idea when template images are large
				//but to do matches refinement, there's not good way
				imgs.push_back(img);
				bbs.push_back(tmplBoundingBox);
			}
#pragma omp critical
			{
				images.insert(std::pair<int, std::vector<cv::Mat> >(classId, imgs));
				boundingBoxes.insert(std::pair<int, std::vector<cv::Rect> >(classId, bbs));
			}
		}

		return true;
	}
#endif

	// load template images, poses of images, obj model file
	bool OfflineTemplatesLoader::loadTemplateImagesAndPoses(
		const ObjectsConfig &objectsConfig,
		cv::Ptr<Detector> &detector, 
		std::map<int, std::vector<cv::Mat> > &images, 
		std::map<int, std::vector<cv::Rect> > &boundingBoxes,
		std::map<int, std::vector<cv::Matx61f> > &poses, 
		std::map<int, GLMmodel*> &objModels)
	{
		const int objectNum = objectsConfig.object_config_size();
		CHECK(objectNum>0) << "No template images.";

		//#pragma omp parallel for
		for(int i=0; i<objectNum; ++i)
		{
			const ObjectConfig &objectConfig = objectsConfig.object_config(i);
			const int classId = objectConfig.id();
			const std::string &className = objectConfig.name();
			//const std::string &imagesDir = objectConfig.img_dir();
			//const std::string &posePath = objectConfig.pose_path();
			const std::string &modelPath = objectConfig.model_path();

			//CHECK(!posePath.empty())<< "Please configure the file path of pose file.";
			CHECK(!modelPath.empty())<< "Please configure the file path of obj file.";

#if 0
			// read images
			std::vector<std::string> fileName;
			CHECK(GetFileNames(imagesDir, fileName))
				<< "Cannot get files name in specific directory";

			std::vector<cv::Mat> imgs;
			std::vector<cv::Rect> bbs;
			for(int k=0; k<(int)fileName.size(); ++k)
			{
				std::string file = imagesDir + "/" + fileName[k];
				cv::Mat img=cv::imread( file );					

				//extract a new template for current image
				cv::Rect tmplBoundingBox;
				int template_id = detector->addTemplate(img, classId, &tmplBoundingBox);

				//save all images, it is not a good idea when template images are large
				//but to do matches refinement, there's not good way
				imgs.push_back(img);
				bbs.push_back(tmplBoundingBox);
			}

			// read pose file
			std::ifstream poseInFile(posePath.c_str());
			CHECK(poseInFile.is_open()) << "Cannot open pose file.";

			std::vector<cv::Matx61f> posesParam;
			for(int j=0; j<(int)fileName.size(); ++j)
			{
				cv::Matx61f poseTmp;

				// rotation vector and translation vector
				poseInFile >> poseTmp.val[0] >> poseTmp.val[1] >> poseTmp.val[2]
				>> poseTmp.val[3] >> poseTmp.val[4] >> poseTmp.val[5];

				posesParam.push_back(poseTmp);
			}
#endif

			// read template images and corresponding poses
			std::vector<cv::Mat> imgs;
			std::vector<cv::Rect> bbs;
			std::vector<cv::Matx61f> posesParam;
			for(int j=0; j<objectConfig.imagedir_posefile_pair_size(); ++j)
			{
				const ObjectConfig_ImageDirAndPoseFilePair& imagedir_posefile = objectConfig.imagedir_posefile_pair(j);
				const std::string &imagesDir = imagedir_posefile.img_dir();
				const std::string &posePath = imagedir_posefile.pose_path();

				// read images
				std::vector<std::string> fileName;
				CV_Assert(GetFileNames(imagesDir, fileName));

				for(int k=0; k<(int)fileName.size(); ++k)
				{
					std::string file = imagesDir + "/" + fileName[k];
					cv::Mat img=cv::imread( file );	

					//extract a new template for current image
					cv::Rect tmplBoundingBox;
					int template_id = detector->addTemplate(img, classId, &tmplBoundingBox);

					//save all images, it is not a good idea when template images are large
					//but to do matches refinement, there's not good way
					imgs.push_back(img);
					bbs.push_back(tmplBoundingBox);
				}

				// read poses
				std::ifstream poseInFile(posePath.c_str());
				CV_Assert(poseInFile.is_open());

				for(int j=0; j<(int)fileName.size(); ++j)
				{
					cv::Matx61f poseTmp;

					// rotation vector and translation vector
					poseInFile >> poseTmp.val[0] >> poseTmp.val[1] >> poseTmp.val[2]
					>> poseTmp.val[3] >> poseTmp.val[4] >> poseTmp.val[5];

					posesParam.push_back(poseTmp);
				}
			}

			// read obj file
			GLMmodel *modelTmp = glmReadOBJ(const_cast<char*>(modelPath.c_str()));

			//#pragma omp critical
			{
				images.insert(std::pair<int, std::vector<cv::Mat> >(classId, imgs));
				boundingBoxes.insert(std::pair<int, std::vector<cv::Rect> >(classId, bbs));
				poses.insert( std::pair<int, std::vector<cv::Matx61f> >(classId, posesParam) );
				objModels.insert(std::pair<int, GLMmodel*>(classId, modelTmp));
			}
		}

		return true;
	}

	// call the opengl to render template images
	void OfflineTemplatesLoader::renderTemplateImages(int argc, char **argv,
		GLMmodel *model,
		CameraCalibration &camCalib,
		float minRadius, float maxRadius, float radiusStep,
		float minLatitude, float maxLatitude, float latitudeStep,
		float minLongitude, float maxLongitude, float longitudeStep,
		float minInplane, float maxInplane, float inplaneStep,
		std::vector<cv::Mat> &images,
		std::vector<cv::Matx61f> &poses)
	{
		const float nearPlane = 0.01;
		const float farPlane = 1000;
		GLRender::init(argc, argv, model, camCalib, nearPlane, farPlane,
			minRadius, maxRadius, radiusStep, 
			minLatitude, maxLatitude, latitudeStep,
			minLongitude, maxLongitude, longitudeStep,
			minInplane, maxInplane, inplaneStep);

		images = GLRender::m_viewImgs;
		poses = GLRender::m_viewPoses;
	}

}

