#pragma once


//-------------------------------------------------------------------------------------------------
class Shuffle
{
public:
	Shuffle(int num);
	~Shuffle();
	
	void SetRandomSeed();
	int* RandomShuffle();

private:
	int  num;
	int *shu;
};
