#ifndef _Render_H_
#define _Render_H_

#include <GL\freeglut.h>
#include <opencv2\opencv.hpp>
#include <vector>
#include "CameraCalibration.h"
#include "GLM.h"

#ifndef PI
#define PI 3.14159265f
#endif

/*
**Render using opengl
**Static class
*/
class GLRender
{
	// methods
public:
	static void computeProjectionMatrix();
	static void computeModelViewMatrix();

	static void loadProjectionMatrix(bool reset=true);
	static void loadModelViewMatrix(bool reset=true);

	static void init(int argc, char **argv,
		GLMmodel *model,
		CameraCalibration& calibration, 
		float nearPlane, float farPlane,
		float minRadius, float maxRadius, float radiusStep,
		float minLatitude, float maxLatitude, float latitudeStep,
		float minLongitude, float maxLongitude, float longitudeStep,
		float minInplane, float maxInplane, float inplaneStep);
	static void display();
	static void reshape(int width, int height);
	static void moveCamera(int value);
	static cv::Mat getRenderedColorImage();

	// variables
public:
	/* projection(intrinsic) related */
	static CameraCalibration m_calibration; // calibrated camera intrinsic parameters
	static float m_nearPlane; // near clipping plane
	static float m_farPlane; // far clipping plane
	static int m_screenWidth; // screen width
	static int m_screenHeight; // screen height	
	static GLfloat m_projectionMatrix[16]; // Buffered projection matrix.

	/* model-view (extrinsic) related */
	static float m_radius, m_latitude, m_longitude, m_inplane;
	static float m_minRadius, m_maxRadius, m_radiusStep; // ball radius
	static float m_minLatitude, m_maxLatitude, m_latitudeStep; // camera lagitude (in degree)
	static float m_minLongitude, m_maxLongitude, m_longitudeStep; // camera longitude (in degree)
	static float m_minInplane, m_maxInplane, m_inplaneStep; // camera inplane (in degree)
	static float m_rotation[3]; // coordinate system axises represented in Rodrigus format
	static float m_translation[3]; // coordinate system origin position
	static GLfloat m_modelViewMatrix[16]; // Buffered model view matrix

	/* the object to be rendered */
	static GLMmodel *m_model;

	/* rendered buffer data for each frame related */
	static uchar* m_pixelBuffer; // buffered rendered pixels
	static float* m_depthBuffer; // buffered rendered depth data

	/* storage for each frame and camera pose during camera viewpoint sampling */
	static std::vector<cv::Mat> m_viewImgs;
	static std::vector<cv::Matx61f> m_viewPoses;
};

#endif // _Render_H_