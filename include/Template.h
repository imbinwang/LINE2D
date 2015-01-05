/*****************************************************************************************
 Improve PAMI2012 "Gradient Response Maps for Real-time Detection of Textureless Objects "
 Template consists of multiple Features which are selected as matching pixels in cpu
 @author Bin Wang <binwangsdu@gmail.com>
 @date 2014/11/20
*****************************************************************************************/

#ifndef _rl2d_Template_H_
#define _rl2d_Template_H_

#include "Feature.h"
#include <vector>

namespace rl2d
{
	struct Template
	{
		int width;
		int height;
		int pyramid_level;
		std::vector<Feature> features;
	};
}

#endif //_Template_H_