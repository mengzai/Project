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
	int j = 2*i+1; //左子结点
	while(j<n)
	{
		//取左右子结点中的小者
		if(j+1<n && a[j]>a[j+1]) j++; //右子结点

		//若子结点比父结点小，则交换
		if(a[j]<a[i])
		{
			type t = a[i];
			a[i] = a[j];
			a[j] = t;

			int bi = b[i];
			b[i] = b[j];
			b[j] = bi;

			i = j;     //检查其子树
			j = 2*i+1; //左子结点
		}
		else break;
	}
}


//-------------------------------------------------------------------------------------------------
template<class type>
void HeapSort<type>::BuildMinHeap(int n, type *a, int *b)
{
	//从最后一个父结点开始，向前调整堆，结点n-1的父结点是(n-2)/2
	for(int i=(n-2)/2; i>=0; i--) AdjustMinHeap(n, a, b, i);
}


//-------------------------------------------------------------------------------------------------
template<class type>
void HeapSort<type>::AdjustMaxHeap(int n, type *a, int *b, int i)
{
	int j = 2*i+1; //左子结点
	while(j<n)
	{
		//取左右子结点中的大者
		if(j+1<n && a[j]<a[j+1]) j++; //右子结点

		//若子结点比父结点大，则交换
		if(a[j]>a[i])
		{
			type t = a[i];
			a[i] = a[j];
			a[j] = t;

			int bi = b[i];
			b[i] = b[j];
			b[j] = bi;

			i = j;     //检查其子树
			j = 2*i+1; //左子结点
		}
		else break;
	}
}


//-------------------------------------------------------------------------------------------------
template<class type>
void HeapSort<type>::BuildMaxHeap(int n, type *a, int *b)
{
	//从最后一个父结点开始，向前调整堆，结点n-1的父结点是(n-2)/2
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
		//依次输出根结点
		type t = a[0];
		int ni = i1-i;
		a[0] = a[ni];
		a[ni] = t;
		
		int b0 = b[0];
		b[0] = b[ni];
		b[ni] = b0;

		//调整剩余堆
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
		//依次输出根结点
		type t = a[0];
		a[0] = a[i];
		a[i] = t;

		int b0 = b[0];
		b[0] = b[i];
		b[i] = b0;

		//调整剩余堆
		AdjustMaxHeap(i, a, b, 0);
	}
}


//-------------------------------------------------------------------------------------------------
template<class type>
void HeapSort<type>::SortMaxK(int n, type *a, int k, type *ak, int *bk)
{
	if(k>n) return;

	//先把前k个元素放入堆中
	memcpy(ak, a, k*sizeof(type));
	for(int i=0; i<k; i++) bk[i] = i;

	//调整为最小堆，方便比较
	BuildMinHeap(k, ak, bk);

	//遍历一次，找到最大的前k个元素
	for(int i=k; i<n; i++)
	{
		if(ak[0]<a[i])
		{
			//置换堆顶
			ak[0] = a[i];
			bk[0] = i;

			//重新调整一次堆
			AdjustMinHeap(k, ak, bk, 0);
		}
	}

	//最大的前k个元素做排序
	for(int i=0, i1=k-1; i<i1; i++)
	{
		//依次输出根结点
		type t = ak[0];
		int ki = i1-i;
		ak[0] = ak[ki];
		ak[ki] = t;

		int bk0 = bk[0];
		bk[0] = bk[ki];
		bk[ki] = bk0;

		//调整剩余堆
		AdjustMinHeap(ki, ak, bk, 0);
	}
}


//-------------------------------------------------------------------------------------------------
template<class type>
void HeapSort<type>::SortMinK(int n, type *a, int k, type *ak, int *bk)
{
	if(k>n) return;

	//先把前k个元素放入堆中
	memcpy(ak, a, k*sizeof(type));
	for(int i=0; i<k; i++) bk[i] = i;

	//调整为最大堆，方便比较
	BuildMaxHeap(k, ak, bk);

	//遍历一次，找到最小的前k个元素
	for(int i=k; i<n; i++)
	{
		if(ak[0]>a[i])
		{
			//置换堆顶
			ak[0] = a[i];
			bk[0] = i;

			//重新调整一次堆
			AdjustMaxHeap(k, ak, bk, 0);
		}
	}

	//最小的前k个元素做排序
	for(int i=0, i1=k-1; i<i1; i++)
	{
		//依次输出根结点
		type t = ak[0];
		int ki = i1-i;
		ak[0] = ak[ki];
		ak[ki] = t;

		int bk0 = bk[0];
		bk[0] = bk[ki];
		bk[ki] = bk0;

		//调整剩余堆
		AdjustMaxHeap(ki, ak, bk, 0);
	}
}
