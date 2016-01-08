#include <opencv2\highgui\highgui.hpp>
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <glog\logging.h>

#include "..\include\util\OfflineTemplatesLoader.h"
#include "..\include\util\Config.pb.h"
#include "..\include\util\ProtobufIO.h"
#include "..\include\util\MatchDisplay.h"
#include "..\include\util\GLMDisplay.h"
#include "..\include\Detector.h"
#include "..\include\util\GLRender.h"
#include "..\include\util\MatchRefiner.h"
#include "..\include\util\Filex.h"

using namespace rl2d;

#if 1
int main(int argc, char *argv[])
{
	// insert the configuration file path
	std::string configFile;
	if(argc==1) // for lazy monkey, you can insert path by hard coding
		configFile = "D:\\project\\VSProject\\LINE2D4IKEA\\Bin_LINE2DIKEA\\data\\HBLTS8\\config.prototxt";
	else // the second param in cmd line
		configFile = std::string(argv[1]);
	RuntimeConfig runtimeConfig;
	ReadProtoFromTextFileOrDie(configFile.c_str(), &runtimeConfig);

	// camera intrinsic and distortion parameters (camera should be calibrated in advance)
	const CameraIntrinsic &cam_intrinsic = runtimeConfig.cam_intrinsic();
	float distortionCoeff[5] = {cam_intrinsic.distortions(0),
	                            cam_intrinsic.distortions(1),
		                        cam_intrinsic.distortions(2),
		                        cam_intrinsic.distortions(3),
	                            cam_intrinsic.distortions(4)};
	CameraCalibration camCalib(cam_intrinsic.fx(),
		                       cam_intrinsic.fx(),
		                       cam_intrinsic.cx(),
		                       cam_intrinsic.cy(),
		                       distortionCoeff);

	// detector configuration
	const DetectorConfig &detectorConfig = runtimeConfig.detector_config();
	float threshold = detectorConfig.match_threshold();
	cv::Ptr<Detector> detector = getDefaultLINE2D(detectorConfig);

	// objects configuration
	const ObjectsConfig &objectsConfig = runtimeConfig.objects_config();
	OfflineTemplatesLoader offlineLoader;
	std::map<int, std::vector<cv::Mat> > images;
	std::map<int, std::vector<cv::Rect> > boundingBoxes;
	std::map<int, std::vector<cv::Matx61f> > templatePoses;
	std::map<int, GLMmodel*> objModels; 

	// load template image and corresponding poses
	offlineLoader.loadTemplateImagesAndPoses(objectsConfig, detector, 
		images, boundingBoxes, templatePoses, objModels);

	// calculate hsv color histograms
	const HSColorHistogram &hsch = runtimeConfig.hs_color_hist();
	bool HS_COLOR_REFINE = hsch.hs_color_refinement();
	int HBIN = hsch.h_bin();
	int SBIN = hsch.s_bin();
	float HS_REFINE_THRESHOLD = hsch.threshold();
	std::map<int, std::vector<cv::Mat> > templateColorHists;
    if(HS_COLOR_REFINE)
		MatchRefiner::offlineCalculateHSHist(objectsConfig, HBIN, SBIN, templateColorHists);

	// manipulate data
	const DataSource &dataSource = runtimeConfig.data_source();
	const DataSource_DataType dataType = dataSource.type();
	if(dataType==DataSource_DataType_ONLINE_VIDEO)
	{
		// open camera
		cv::VideoCapture capture(0);
		//cv::VideoCapture capture("D:/project/VSProject/LINE_MOD/LINE_MOD/data/duplo_motion_01.avi");
		CHECK(capture.isOpened()) << "Could not open camera.";

		cv::namedWindow("Detected", 1);
		std::vector<Match> matches;
		std::vector<Match> mergedMatches;
		std::vector<cv::Rect> mergedRects;
		cv::Mat color;
		for(;;)
		{
			capture >> color;

			detector->match(color, threshold, matches);
			MergeMatches(matches, detector, mergedMatches, mergedRects);
			if(HS_COLOR_REFINE)
				MatchRefiner::offlineCalculateHSHist(objectsConfig, HBIN, SBIN, templateColorHists);

			cv::Mat displayImg = color.clone();
			DrawMergedMatches(mergedMatches, mergedRects, objectsConfig, 5, displayImg);

			printf("matches size %d\n", (int)mergedMatches.size());

			cv::imshow("Detected", displayImg);
			cv::waitKey(1);
		}
	}
	if(dataType==DataSource_DataType_OFFLINE_VIDEO)
	{
		// read video
		std::string video_path = dataSource.offline_video_path();
		CHECK(!video_path.empty()) << "Invalid video path.";
		cv::VideoCapture capture(video_path);
		CHECK(capture.isOpened()) << "Could not open video file.";

		cv::namedWindow("Detected", 1);
		std::vector<Match> matches;
		std::vector<Match> mergedMatches;
		std::vector<cv::Rect> mergedRects;
		cv::Mat color;
		for(;;)
		{
			capture >> color;

			detector->match(color, threshold, matches);
			MergeMatches(matches, detector, mergedMatches, mergedRects);
			if(HS_COLOR_REFINE)
				MatchRefiner::offlineCalculateHSHist(objectsConfig, HBIN, SBIN, templateColorHists);

			cv::Mat displayImg = color.clone();
			DrawMergedMatches(mergedMatches, mergedRects, objectsConfig, 5, displayImg);

			printf("matches size %d\n", (int)mergedMatches.size());

			cv::imshow("Detected", displayImg);
			cv::waitKey(1);
		}
	}
	if(dataType==DataSource_DataType_IMAGES)
	{
		std::string imagesDir = dataSource.imgs_dir();
		CHECK(!imagesDir.empty()) << "Invalid images directory.";
		// read images name
		std::vector<std::string> fileNames;
		bool getImgsFinished = GetFileNames(imagesDir, fileNames);
		CHECK(getImgsFinished) << "Cannot get images.";

		for(int k=0; k<(int)fileNames.size(); ++k)
		{
			printf("Image: %s\n",fileNames[k].c_str());
			std::string img_path = imagesDir + "/" + fileNames[k];
			size_t dot_idx = fileNames[k].find_first_of('.');
			std::string name_ = fileNames[k].substr(0,dot_idx);
			std::string out_img_path = imagesDir + "/" + name_ + "_detection.png";
			cv::Mat color = cv::imread(img_path);

			std::vector<Match> matches;
			std::vector<Match> mergedMatches;
			std::vector<cv::Rect> mergedRects;

			detector->match(color, threshold, matches);
			MergeMatches(matches, detector, mergedMatches, mergedRects);
			if(HS_COLOR_REFINE)
				MatchRefiner::refine(mergedMatches, mergedRects, color, boundingBoxes, templateColorHists, HBIN, SBIN, HS_REFINE_THRESHOLD);


			cv::Mat displayImg = color.clone();
			DrawMergedMatches(mergedMatches, mergedRects, objectsConfig, 5, displayImg);
			printf("matches size %d\n", (int)mergedMatches.size());

			cv::imwrite(out_img_path, displayImg);

			cv::namedWindow("detected", 1);
			cv::imshow("detected", displayImg);
			cv::waitKey(1);
		}	
	}

	return 1;
}
#endif


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//+++++++++++ I am lazy. Maybe another file for rendering is better. ++++++++++++++
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#if 0
#include <direct.h>
int main(int argc, char *argv[])
{
	// insert the configuration file path
	std::string configFile;
	if(argc==1) // for lazy monkey, you can insert path by hard coding
		configFile = "D:\\project\\VSProject\\LINE2D4IKEA\\Bin_LINE2DIKEA\\data\\HBLTS8\\config.prototxt";
	else // the second param in cmd line
		configFile = std::string(argv[1]);
	RuntimeConfig runtimeConfig;
	ReadProtoFromTextFileOrDie(configFile.c_str(), &runtimeConfig);

	// camera intrinsic and distortion parameters (camera should be calibrated in advance)
	const CameraIntrinsic &cam_intrinsic = runtimeConfig.cam_intrinsic();
	float distortionCoeff[5] = {cam_intrinsic.distortions(0),
		cam_intrinsic.distortions(1),
		cam_intrinsic.distortions(2),
		cam_intrinsic.distortions(3),
		cam_intrinsic.distortions(4)};
	CameraCalibration camCalib(cam_intrinsic.fx(),
		cam_intrinsic.fx(),
		cam_intrinsic.cx(),
		cam_intrinsic.cy(),
		distortionCoeff);

	// objects configuration
	const ObjectsConfig &objectsConfig = runtimeConfig.objects_config();
	const int objectsNum = objectsConfig.object_config_size();
	OfflineTemplatesLoader otl;
	std::vector<cv::Mat> images;
	std::vector<cv::Matx61f> poses;
	for(int i=0; i<objectsNum; ++i)
	{
		const ObjectsConfig_ObjectConfig &objectConfig = objectsConfig.object_config(i);
		const int classId = objectConfig.id();
		const std::string &className = objectConfig.name();
		const std::string &modelPath = objectConfig.model_path();
		CHECK(!modelPath.empty())<< "Please configure the file path of obj file.";
		printf("Start to render %s\n", className.c_str());

		// read obj file
		GLMmodel *objModel = glmReadOBJ(const_cast<char*>(modelPath.c_str()));

		// read viewpoint sampling information
		images.clear();
		vector<cv::Mat>(images).swap(images);
		poses.clear();
		vector<cv::Matx61f>(poses).swap(poses);
		const ObjectsConfig_ObjectConfig_OnlineRenderingParams &orp = objectConfig.online_rendering_params();
		otl.renderTemplateImages(argc, argv,
			objModel,
			camCalib,
			orp.min_radius(), orp.max_radius(), orp.radius_step(),
			orp.min_latitude(), orp.max_latitude(), orp.latitude_step(),
			orp.min_longitude(), orp.max_longitude(), orp.longitude_step(),
			orp.min_inplane(), orp.max_inplane(), orp.inplane_step(),
			orp.out_dir());

		
	}

	return 0;
}
#endif