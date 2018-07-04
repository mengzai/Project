#pragma once
#include <string.h>


//-------------------------------------------------------------------------------------------------
template<class type> class HeapSort
{
public:
	// a - array, b - index
	void SortAsce(int n, type *a, int *b);
	void SortDesc(int n, type *a, int *b);

	// a - array, ak - top-k, bk - top-k index
	void SortMaxK(int n, type *a, int k, type *ak, int *bk);
	void SortMinK(int n, type *a, int k, type *ak, int *bk);

private:
	void AdjustMinHeap(int n, type *a, int *b, int i);
	void BuildMinHeap(int n, type *a, int *b);

	void AdjustMaxHeap(int n, type *a, int *b, int i);
	void BuildMaxHeap(int n, type *a, int *b);
};


//-------------------------------------------------------------------------------------------------
template<class type>
void HeapSort<type>::AdjustMinHeap(int n, type *a, int *b, int i)
{
	int j = 2*i+1; //���ӽ��
	while(j<n)
	{
		//ȡ�����ӽ���е�С��
		if(j+1<n && a[j]>a[j+1]) j++; //���ӽ��

		//���ӽ��ȸ����С���򽻻�
		if(a[j]<a[i])
		{
			type t = a[i];
			a[i] = a[j];
			a[j] = t;

			int bi = b[i];
			b[i] = b[j];
			b[j] = bi;

			i = j;     //���������
			j = 2*i+1; //���ӽ��
		}
		else break;
	}
}


//-------------------------------------------------------------------------------------------------
template<class type>
void HeapSort<type>::BuildMinHeap(int n, type *a, int *b)
{
	//�����һ������㿪ʼ����ǰ�����ѣ����n-1�ĸ������(n-2)/2
	for(int i=(n-2)/2; i>=0; i--) AdjustMinHeap(n, a, b, i);
}


//-------------------------------------------------------------------------------------------------
template<class type>
void HeapSort<type>::AdjustMaxHeap(int n, type *a, int *b, int i)
{
	int j = 2*i+1; //���ӽ��
	while(j<n)
	{
		//ȡ�����ӽ���еĴ���
		if(j+1<n && a[j]<a[j+1]) j++; //���ӽ��

		//���ӽ��ȸ������򽻻�
		if(a[j]>a[i])
		{
			type t = a[i];
			a[i] = a[j];
			a[j] = t;

			int bi = b[i];
			b[i] = b[j];
			b[j] = bi;

			i = j;     //���������
			j = 2*i+1; //���ӽ��
		}
		else break;
	}
}


//-------------------------------------------------------------------------------------------------
template<class type>
void HeapSort<type>::BuildMaxHeap(int n, type *a, int *b)
{
	//�����һ������㿪ʼ����ǰ�����ѣ����n-1�ĸ������(n-2)/2
	for(int i=(n-2)/2; i>=0; i--) AdjustMaxHeap(n, a, b, i);
}


//-------------------------------------------------------------------------------------------------
template<class type>
void HeapSort<type>::SortDesc(int n, type *a, int *b)
{
	for(int i=0; i<n; i++) b[i] = i;

	BuildMinHeap(n, a, b);

	for(int i=0, i1=n-1; i<i1; i++)
	{
		//������������
		type t = a[0];
		int ni = i1-i;
		a[0] = a[ni];
		a[ni] = t;
		
		int b0 = b[0];
		b[0] = b[ni];
		b[ni] = b0;

		//����ʣ���
		AdjustMinHeap(ni, a, b, 0);
	}
}


//-------------------------------------------------------------------------------------------------
template<class type>
void HeapSort<type>::SortAsce(int n, type *a, int *b)
{
	for(int i=0; i<n; i++) b[i] = i;

	BuildMaxHeap(n, a, b);

	for(int i=n-1; i>0; i--)
	{
		//������������
		type t = a[0];
		a[0] = a[i];
		a[i] = t;

		int b0 = b[0];
		b[0] = b[i];
		b[i] = b0;

		//����ʣ���
		AdjustMaxHeap(i, a, b, 0);
	}
}


//-------------------------------------------------------------------------------------------------
template<class type>
void HeapSort<type>::SortMaxK(int n, type *a, int k, type *ak, int *bk)
{
	if(k>n) return;

	//�Ȱ�ǰk��Ԫ�ط������
	memcpy(ak, a, k*sizeof(type));
	for(int i=0; i<k; i++) bk[i] = i;

	//����Ϊ��С�ѣ�����Ƚ�
	BuildMinHeap(k, ak, bk);

	//����һ�Σ��ҵ�����ǰk��Ԫ��
	for(int i=k; i<n; i++)
	{
		if(ak[0]<a[i])
		{
			//�û��Ѷ�
			ak[0] = a[i];
			bk[0] = i;

			//���µ���һ�ζ�
			AdjustMinHeap(k, ak, bk, 0);
		}
	}

	//����ǰk��Ԫ��������
	for(int i=0, i1=k-1; i<i1; i++)
	{
		//������������
		type t = ak[0];
		int ki = i1-i;
		ak[0] = ak[ki];
		ak[ki] = t;

		int bk0 = bk[0];
		bk[0] = bk[ki];
		bk[ki] = bk0;

		//����ʣ���
		AdjustMinHeap(ki, ak, bk, 0);
	}
}


//-------------------------------------------------------------------------------------------------
template<class type>
void HeapSort<type>::SortMinK(int n, type *a, int k, type *ak, int *bk)
{
	if(k>n) return;

	//�Ȱ�ǰk��Ԫ�ط������
	memcpy(ak, a, k*sizeof(type));
	for(int i=0; i<k; i++) bk[i] = i;

	//����Ϊ���ѣ�����Ƚ�
	BuildMaxHeap(k, ak, bk);

	//����һ�Σ��ҵ���С��ǰk��Ԫ��
	for(int i=k; i<n; i++)
	{
		if(ak[0]>a[i])
		{
			//�û��Ѷ�
			ak[0] = a[i];
			bk[0] = i;

			//���µ���һ�ζ�
			AdjustMaxHeap(k, ak, bk, 0);
		}
	}

	//��С��ǰk��Ԫ��������
	for(int i=0, i1=k-1; i<i1; i++)
	{
		//������������
		type t = ak[0];
		int ki = i1-i;
		ak[0] = ak[ki];
		ak[ki] = t;

		int bk0 = bk[0];
		bk[0] = bk[ki];
		bk[ki] = bk0;

		//����ʣ���
		AdjustMaxHeap(ki, ak, bk, 0);
	}
}
