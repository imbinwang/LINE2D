#include <opencv2\highgui\highgui.hpp>
#include <glog\logging.h>

#include "..\include\util\OfflineTemplatesLoader.h"
#include "..\include\util\Config.pb.h"
#include "..\include\util\ProtobufIO.h"
#include "..\include\util\MatchDisplay.h"
#include "..\include\util\GLMDisplay.h"
#include "..\include\Detector.h"
#include "..\include\util\GLRender.h"

using namespace rl2d;

int main(int argc, char *argv[])
{
	//camera parameters for the Logi web camare
	float distortionCoeff[5] = {0.10895365623692939, -2.4227559050149443, 
		0., 0.,13.608371846992902};
	CameraCalibration camCalib(804.53,804.53,319.5,239.5,distortionCoeff);

	std::string configFile = "D:\\project\\VSProject\\LINE2D4IKEA\\Bin_LINE2DIKEA\\data\\config.prototxt";
	ObjectsConfig objectsConfig;
	ReadProtoFromTextFileOrDie(configFile.c_str(), &objectsConfig);

	float threshold = 90;
	cv::Ptr<Detector> detector = getDefaultLINE2D();

	OfflineTemplatesLoader offlineLoader;
	std::map<int, std::vector<cv::Mat> > images;
	std::map<int, std::vector<cv::Rect> > boundingBoxes;
	std::map<int, std::vector<cv::Matx61f> > templatePoses;
	std::map<int, GLMmodel*> objModels; 
	//offlineLoader.loadTemplateImages(objectsConfig, detector, images, boundingBoxes);
	offlineLoader.loadTemplateImagesAndPoses(objectsConfig, detector, 
		images, boundingBoxes, templatePoses, objModels);

	// open camera
	cv::VideoCapture capture(0);
	//cv::VideoCapture capture("D:/project/VSProject/LINE_MOD/LINE_MOD/data/duplo_motion_01.avi");
	CHECK(capture.isOpened()) << "Could not open camera.";

	cv::namedWindow("detected", 1);
	std::vector<Match> matches;
	std::vector<Match> mergedMatches;
	std::vector<cv::Rect> mergedRects;
	cv::Mat color;
	for(;;)
	{
		capture >> color;

		detector->match(color, threshold, matches);
		MergeMatches(matches, detector, mergedMatches, mergedRects);

		cv::Mat displayImg = color.clone();
		//DrawMatches(matches, detector, objectsConfig, 100, displayImg);
		
		if(matches.size()>0)
			ProjectModel2Img(detector, matches[0], camCalib, templatePoses, objModels, CV_RGB(255,255,255), displayImg);
		DrawMergedMatches(mergedMatches, mergedRects, objectsConfig, 10, displayImg);

		printf("matches size %d\n", (int)matches.size());

		cv::imshow("detected", displayImg);
		cv::waitKey(1);
	}

	return 1;
}