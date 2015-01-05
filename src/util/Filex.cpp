#include "..\..\include\util\Filex.h"

bool GetFileNames(const string &directoryPath, vector<string> &fileNames)
{
	WIN32_FIND_DATA data;  
	HANDLE hFind;   

	string allNameAndType = directoryPath + "/*" ;

	hFind = FindFirstFile(allNameAndType.c_str(), &data);
	while( hFind!=INVALID_HANDLE_VALUE )   
	{   
		if(data.cFileName[0] != '.' && data.dwFileAttributes != FILE_ATTRIBUTE_DIRECTORY )
		{
			fileNames.push_back( string(data.cFileName) ); 
		}      
		if(!FindNextFile(hFind,&data))   
		{     
			hFind=INVALID_HANDLE_VALUE;   
		}   
	}

	FindClose(hFind);
	return 1;   
}


bool GetSubDirNames(const string &directoryPath, vector<string> &dirNames)
{
	WIN32_FIND_DATA data;  
	HANDLE hFind;   

	string allNameAndType = directoryPath + "/*" ;

	hFind = FindFirstFile(allNameAndType.c_str(), &data);
	while( hFind!=INVALID_HANDLE_VALUE )   
	{   
		if(data.cFileName[0] != '.' && data.dwFileAttributes == FILE_ATTRIBUTE_DIRECTORY )
		{
			dirNames.push_back( string(data.cFileName) ); 
		}      
		if(!FindNextFile(hFind,&data))   
		{     
			hFind=INVALID_HANDLE_VALUE;   
		}   
	}

	FindClose(hFind);
	return 1;   
}