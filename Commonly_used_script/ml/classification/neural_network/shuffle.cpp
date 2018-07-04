#include "shuffle.h"
#include <stdlib.h>
#include <time.h>


//-------------------------------------------------------------------------------------------------
Shuffle::Shuffle(int num)
{
	this->num = num;
	shu = new int[num];
}


//-------------------------------------------------------------------------------------------------
Shuffle::~Shuffle()
{
	delete[] shu;
}


//-------------------------------------------------------------------------------------------------
void Shuffle::SetRandomSeed()
{
	srand((unsigned int)time(0));
}


//-------------------------------------------------------------------------------------------------
int* Shuffle::RandomShuffle()
{
	// initialize permutation
	for(int i=0; i<num; i++) shu[i] = i;

	// random shuffle
	for(int i=num-1; i>0; i--)
	{
		// select the i-th element randomly
		int index = rand() % (i+1);

		// swap the two elements
		int temp = shu[i];
		shu[i] = shu[index];
		shu[index] = temp;
	}

	return shu;
}
