#ifndef _CvUtility_H_
#define _CvUtility_H_

#include <opencv/cv.h>

class Timer
{
public:
	Timer();

	void start();

	void stop();

	double time();

private:
	int64 start_, time_;
};


#endif // _CvUtility_H_