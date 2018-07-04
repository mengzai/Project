#pragma once


//-------------------------------------------------------------------------------------------------
// load training/testing data: X[num*dim], y[num]
class Data
{
public:
	Data();
	~Data();

	// cr: true - classification; false - regression
	
	// data: label attribute attribute ... attribute
	bool LoadDataTxt(bool cr, const char *data);
	
	// head: num dim
	// data: label attribute attribute ... attribute
	bool LoadDataBin(bool cr, const char *data);

	// sparse data in libsvm format
	// data: label index:attribute index:attribute ... index:attribute
	bool LoadDataSpa(bool cr, const char *data);

	int   n;  // number of data samples
	int   d;  // feature dimension
	float*X;  // feature vectors
	int  *yc; // classification, binary: -1,+1 or 0,1, multiclass: 0,1,...,c-1
	float*yr; // regression

private:
	void Release();
};
