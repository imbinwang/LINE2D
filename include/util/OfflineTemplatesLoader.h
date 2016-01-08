#ifndef _OfflineTemplatesLoader_H_
#define _OfflineTemplatesLoader_H_

#include "Config.pb.h"
#include "..\Detector.h"
#include "CameraCalibration.h"
#include "GLM.h"

#include <map>
#include <vector>
#include <opencv2\core\core.hpp>

namespace rl2d
{
	class OfflineTemplatesLoader
	{
	public:
		OfflineTemplatesLoader();
		~OfflineTemplatesLoader();

		// load template images
		bool loadTemplateImages(const ObjectsConfig &objectsConfig,
			cv::Ptr<Detector> &detector,
			std::map<int, std::vector<cv::Mat> > &images, 
			std::map<int, std::vector<cv::Rect> > &boundingBoxes);

		// load template images, poses of images, obj model file
		bool loadTemplateImagesAndPoses(const ObjectsConfig &objectsConfig,
			cv::Ptr<Detector> &detector, 
			std::map<int, std::vector<cv::Mat> > &images, 
			std::map<int, std::vector<cv::Rect> > &boundingBoxes,
			std::map<int, std::vector<cv::Matx61f> > &poses, 
			std::map<int, GLMmodel*> &objModels);

		// render template images by gl
		void renderTemplateImages(int argc, char **argv,
			GLMmodel *model,
			CameraCalibration &camCalib,
			float minRadius, float maxRadius, float radiusStep,
			float minLatitude, float maxLatitude, float latitudeStep,
			float minLongitude, float maxLongitude, float longitudeStep,
			float minInplane, float maxInplane, float inplaneStep,
			const std::string &out_dir);

	};

}

#endif // _OfflineTemplatesLoader_H_