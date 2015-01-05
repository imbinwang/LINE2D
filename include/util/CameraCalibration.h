#ifndef _CameraCalibration_H_
#define _CameraCalibration_H_

#include <opencv2/opencv.hpp>

/**
* A camera calibration class that stores intrinsic matrix and distortion coefficients.
* Camera calibration results are from mannual chessboard calibration.
*/
class CameraCalibration
{
public:
	CameraCalibration();
	CameraCalibration(float fx, float fy, float cx, float cy);
	CameraCalibration(float fx, float fy, float cx, float cy, float distorsionCoeff[5]);


	const cv::Matx33f& getIntrinsic() const;
	const cv::Matx<float,5,1>&  getDistorsion() const;


	float& fx();
	float& fy();


	float& cx();
	float& cy();


	float fx() const;
	float fy() const;


	float cx() const;
	float cy() const;

	/*The field of view is the fov in Y(vertical) direction. 
	It sets the Y(vertical) aperture of the Camera.*/
	float fieldOfView() const;

	/*Returns the Camera aspect ratio defined by cx() / cy()*/
	float aspectRatio() const;

private:
	cv::Matx33f     m_intrinsic;
	cv::Matx<float,5,1>     m_distortion;
};

#endif //_CameraCalibration_H_

