#include "..\..\include\util\Filex.h"

LPCWSTR stringToLPCWSTR(std::string orig)
{
	size_t origsize = orig.length() + 1;
	const size_t newsize = 100;
	size_t convertedChars = 0;
	wchar_t *wcstring = (wchar_t *)malloc(sizeof(wchar_t)*(orig.length() - 1));
	mbstowcs_s(&convertedChars, wcstring, origsize, orig.c_str(), _TRUNCATE);

	return wcstring;
}
string WCharToMByte(LPCWSTR lpcwszStr)
{
	string str;
	DWORD dwMinSize = 0;
	LPSTR lpszStr = NULL;
	dwMinSize = WideCharToMultiByte(CP_OEMCP, NULL, lpcwszStr, -1, NULL, 0, NULL, FALSE);
	if (0 == dwMinSize)
	{
		return FALSE;
	}
	lpszStr = new char[dwMinSize];
	WideCharToMultiByte(CP_OEMCP, NULL, lpcwszStr, -1, lpszStr, dwMinSize, NULL, FALSE);
	str = lpszStr;
	delete[] lpszStr;
	return str;
}

bool GetFileNames(const string &directoryPath, vector<string> &fileNames)
{
	WIN32_FIND_DATA data;  
	HANDLE hFind;   

	string allNameAndType = directoryPath + "/*" ;

	hFind = FindFirstFile(stringToLPCWSTR(allNameAndType), &data);
	while( hFind!=INVALID_HANDLE_VALUE )   
	{   
		if(data.cFileName[0] != '.' && data.dwFileAttributes != FILE_ATTRIBUTE_DIRECTORY )
		{
			fileNames.push_back(WCharToMByte(data.cFileName) );
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

	hFind = FindFirstFile(stringToLPCWSTR(allNameAndType), &data);
	while( hFind!=INVALID_HANDLE_VALUE )   
	{   
		if(data.cFileName[0] != '.' && data.dwFileAttributes == FILE_ATTRIBUTE_DIRECTORY )
		{
			dirNames.push_back(WCharToMByte(data.cFileName) );
		}      
		if(!FindNextFile(hFind,&data))   
		{     
			hFind=INVALID_HANDLE_VALUE;   
		}   
	}

	FindClose(hFind);
	return 1;   
}