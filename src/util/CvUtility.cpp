#include "..\..\include\util\CvUtility.h"

Timer::Timer():start_(0), time_(0) 
{
}

void Timer::start()
{
	start_ = cv::getTickCount();
}

void Timer::stop()
{
	CV_Assert(start_ != 0);
	int64 end = cv::getTickCount();
	time_ += end - start_;
	start_ = 0;
}

double Timer::time()
{
	double ret = time_ / cv::getTickFrequency();
	time_ = 0;
	return ret;
}
