
#include "..\..\include\util\CameraCalibration.h"


CameraCalibration::CameraCalibration()
{
}


CameraCalibration::CameraCalibration(float _fx, float _fy, float _cx, float _cy)
{
	m_intrinsic = cv::Matx33f::eye();

	fx() = _fx;
	fy() = _fy;
	cx() = _cx;
	cy() = _cy;

	m_distortion = cv::Matx<float,5,1>::zeros();
}


CameraCalibration::CameraCalibration(float _fx, float _fy, float _cx, float _cy, float distorsionCoeff[5])
{
	m_intrinsic = cv::Matx33f::eye();

	fx() = _fx;
	fy() = _fy;
	cx() = _cx;
	cy() = _cy;

	m_distortion = cv::Matx<float,5,1>::zeros();
	for (int i=0; i<5; i++)
		m_distortion(i) = distorsionCoeff[i];
}


const cv::Matx33f& CameraCalibration::getIntrinsic() const
{
	return m_intrinsic;
}


const cv::Matx<float,5,1>&  CameraCalibration::getDistorsion() const
{
	return m_distortion;
}


float& CameraCalibration::fx()
{
	return m_intrinsic(0,0);
}


float& CameraCalibration::fy()
{
	return m_intrinsic(1,1);
}


float& CameraCalibration::cx()
{
	return m_intrinsic(0,2);
}


float& CameraCalibration::cy()
{
	return m_intrinsic(1,2);
}


float CameraCalibration::fx() const
{
	return m_intrinsic(0,0);
}


float CameraCalibration::fy() const
{
	return m_intrinsic(1,1);
}


float CameraCalibration::cx() const
{
	return m_intrinsic(0,2);
}


float CameraCalibration::cy() const
{
	return m_intrinsic(1,2);
}

float CameraCalibration::fieldOfView() const
{
	float fov = 2 * atan( cy()/fy() );
	return fov;
}

float CameraCalibration::aspectRatio() const
{
	float ar = cx()/cy();
	return ar;
}
