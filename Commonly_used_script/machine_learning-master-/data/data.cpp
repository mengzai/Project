#include <stdio.h>
#include <string.h>
#include "data.h"
#pragma warning(disable:4996)


//-------------------------------------------------------------------------------------------------
Data::Data()
{
	n = 0;
	d = 0;
	X = 0;
	yc = 0;
	yr = 0;
}


//-------------------------------------------------------------------------------------------------
Data::~Data()
{
	Release();
}


//-------------------------------------------------------------------------------------------------
void Data::Release()
{
	n = 0;
	d = 0;
	if(X!=0){ delete[] X; X = 0; }
	if(yc!=0){ delete[] yc; yc = 0; }
	if(yr!=0){ delete[] yr; yr = 0; }
}


//-------------------------------------------------------------------------------------------------
// data: label attribute attribute ... attribute
bool Data::LoadDataTxt(bool cr, const char *data)
{
	Release();

	FILE *pfile = fopen(data, "rt");
	if(pfile==0){ printf("Can not open input file %s\n", data); return false; }
	bool ret = true;

	// find how many samples
	const int len = 4096;
	char *buf = new char[len];
	while(fgets(buf, len, pfile)!=0)
	{
		int yci = 0, di = 0;
		float xi = 0, yri = 0;
		
		// check class label yi
		if( cr && (sscanf(buf, "%d %[^\n]", &yci, buf)!=2 || yci<-1)
		|| !cr &&  sscanf(buf, "%f %[^\n]", &yri, buf)!=2)
		{
			ret = false;
			printf("row %d is invalid\n", n); // error yi
			goto label;
		}

		// check feature vector
		while(true)
		{
			int ret = sscanf(buf, "%f %[^\n]", &xi, buf);
			if(ret>=1) di++;
			if(ret!=2) break;
		}
		if(di>0 && d==0) d = di; // update feature dimension
		if(di==0 || di!=d)
		{
			ret = false;
			printf("row %d is invalid\n", n); // error xi
			goto label;
		}
		
		n++;
	}
	
	// load all samples
	if(n>1)
	{
		X = new float[n*d];
		if(cr) yc = new int[n];
		else yr = new float[n];

		fseek(pfile, 0, SEEK_SET);
		for(int i=0; i<n; i++)
		{
			if(cr) fscanf(pfile, "%d", yc+i);
			else fscanf(pfile, "%f", yr+i);
			float *xi = X+i*d;
			for(int j=0; j<d; j++) fscanf(pfile, "%f", xi+j);
		}
	}
	else ret = false;

label:
	if(!ret) n = d = 0;
	delete[] buf;
	
	fclose(pfile);
	if(ret) printf("load samples: num = %d, dim = %d\n", n, d);
	return ret;
}


//-------------------------------------------------------------------------------------------------
// head: num dim
// data: label attribute attribute ... attribute
bool Data::LoadDataBin(bool cr, const char *data)
{
	Release();

	FILE *pfile = fopen(data, "rb");
	if(pfile==0){ printf("Can not open input file %s\n", data); return false; }
	bool ret = true;

	// number of samples
	if(fread(&n, sizeof(int), 1, pfile)!=1 || n<2){ ret = false; goto label; }

	// feature dimension
	if(fread(&d, sizeof(int), 1, pfile)!=1 || d<1){ ret = false; goto label; }

	// load all samples
	X = new float[n*d];
	if(cr) yc = new int[n];
	else yr = new float[n];
	for(int i=0; i<n; i++)
	{
		// load class label
		if( cr && (fread(yc+i, sizeof(int),   1, pfile)!=1 || yc[i]<-1)
		|| !cr &&  fread(yr+i, sizeof(float), 1, pfile)!=1)
		{ ret = false; goto label; }

		// load feature vector
		if(fread(X+i*d, sizeof(float), d, pfile)!=d) { ret = false; goto label; }
	}

label:
	fclose(pfile);
	
	if(!ret){ printf("format error in %s\n", data); Release(); }
	else printf("load samples: num = %d, dim = %d\n", n, d);
	return ret;
}


//-------------------------------------------------------------------------------------------------
// sparse data in libsvm format
// data: label index:attribute index:attribute ... index:attribute
bool Data::LoadDataSpa(bool cr, const char *data)
{
	Release();

	FILE *pfile = fopen(data, "rt");
	if(pfile==0){ printf("Can not open input file %s\n", data); return false; }

	bool ret = true;
	const int len = 4096;
	char *buf = new char[len];

	// find how many samples and dimensions
	n = d = 0;
	while(true)
	{
		if(fgets(buf, len, pfile)==0) break;

		int   i = 0;
		float f = 0;
		int read = cr ? sscanf(buf, "%d %[^\n]", &i, buf) : sscanf(buf, "%f %[^\n]", &f, buf);
		if(read!=2 || cr && i<-1){ ret = false; goto label; } // error format

		while(true)
		{
			read = sscanf(buf, "%d:%f %[^\n]", &i, &f, buf);
			if(read<2){ ret = false; goto label; } // error format
			else
			{
				if(d<i) d = i;
				if(read==2) break; // last elements
			}
		}

		n++;
	}

	// load training samples
	if(n>1 && d>0)
	{
		X = new float[n*d];
		if(cr) yc = new int[n];
		else yr = new float[n];

		memset(X, 0, n*d*sizeof(float));

		int *ci = yc;
		float *xi = X, *ri = yr;
		fseek(pfile, 0, SEEK_SET);
		while(true)
		{
			if(fgets(buf, len, pfile)==0) break;

			int read = cr ? sscanf(buf, "%d %[^\n]", ci, buf) : sscanf(buf, "%f %[^\n]", ri, buf);
			if(read!=2 || cr && *ci<-1){ ret = false; goto label; } // error format

			while(true)
			{
				int   i;
				float f;
				read = sscanf(buf, "%d:%f %[^\n]", &i, &f, buf);
				if(read<2){ ret = false; goto label; } // error format
				else
				{
					xi[i-1] = f;
					if(read==2) break; // last elements
				}
			}

			if(cr) ci++;
			else ri++;
			xi += d;
		}
	}
	else ret = false;

label:
	delete[] buf;
	fclose(pfile);

	if(!ret) Release();
	else printf("load samples: num = %d, dim = %d\n", n, d);
	return ret;
}
