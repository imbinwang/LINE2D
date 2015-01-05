#ifndef _Filex_H_
#define _Filex_H_

#include <windows.h>
#include <vector>
#include <map>
#include <string>
#include <opencv2\core\core.hpp>
#include "..\Detector.h"

using std::string;
using std::vector;

// read the file namses under the directoryPath
bool GetFileNames(const string &directoryPath, vector<string> &fileNames);

// read the subdirectories under the directoryPath
bool GetSubDirNames(const string &directoryPath, vector<string> &dirNames);


#endif // _Filex_H_