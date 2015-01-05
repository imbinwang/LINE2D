/*****************************************************************************************
Improve PAMI2012 "Gradient Response Maps for Real-time Detection of Textureless Objects "
Feature Point for matching in cpu
@author Bin Wang <binwangsdu@gmail.com>
@date 2014/11/20
*****************************************************************************************/

#ifndef _rl2d_Feature_H_
#define _rl2d_Feature_H_

namespace rl2d
{
	// Feature struct can be used in cpu and gpu
	struct Feature
	{
		int x; 
		int y; 
		int label; 

		Feature():x(-1), y(-1), label(-1){}

		Feature(int a_x, int a_y, int a_label):x(a_x), y(a_y), label(a_label){}

		Feature(const Feature &f):x(-1), y(-1), label(-1)
		{
			x = f.x;
			y = f.y;
			label = f.label;
		}
	};

}
#endif // _Feature_H_