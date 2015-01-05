/*****************************************************************************************
Improve PAMI2012 "Gradient Response Maps for Real-time Detection of Textureless Objects "
Match: a good detection result defined
@author Bin Wang <binwangsdu@gmail.com>
@date 2014/11/26
*****************************************************************************************/
#ifndef _rl2d_Match_H_
#define _rl2d_Match_H_

namespace rl2d
{
	struct Match
	{
		int x;
		int y;
		float similarity;
		int class_id;
		int template_id;

		Match()
		{
		}

		Match(int a_x, int a_y, float a_similarity, 
			const int a_class_id, int a_template_id)
			: x(a_x), y(a_y), similarity(a_similarity), class_id(a_class_id), template_id(a_template_id)
		{
		}

		/// Sort matches with high similarity to the front
		bool operator<(const Match& rhs) const
		{
			// Secondarily sort on template_id for the sake of duplicate removal
			if (similarity != rhs.similarity)
				return similarity > rhs.similarity;
			else
				return template_id < rhs.template_id;
		}

		bool operator==(const Match& rhs) const
		{
			return x == rhs.x && y == rhs.y && 
				similarity == rhs.similarity && class_id == rhs.class_id;
		}
	};
}

#endif // _rl2d_Match_H_