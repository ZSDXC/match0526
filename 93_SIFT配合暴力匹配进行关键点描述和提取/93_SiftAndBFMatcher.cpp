// Ransac Match.cpp: �������̨Ӧ�ó������ڵ㡣
//
//#include "stdafx.h"
#include <opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <time.h>
using namespace cv;
using namespace std;

int cornerScore(const uchar* ptr, const int pixel[], int threshold)
{
	const int K = 8, N = K * 3 + 1;
	//vΪ��ǰ����ֵ  
	int k, v = ptr[0];
	short d[N];
	//���㵱ǰ����ֵ����Բ������ֵ֮��Ĳ�ֵ  
	for (k = 0; k < N; k++)
		d[k] = (short)(v - ptr[pixel[k]]);

#if CV_SSE2  
	__m128i q0 = _mm_set1_epi16(-1000), q1 = _mm_set1_epi16(1000);
	for (k = 0; k < 16; k += 8)
	{
		__m128i v0 = _mm_loadu_si128((__m128i*)(d + k + 1));
		__m128i v1 = _mm_loadu_si128((__m128i*)(d + k + 2));
		__m128i a = _mm_min_epi16(v0, v1);
		__m128i b = _mm_max_epi16(v0, v1);
		v0 = _mm_loadu_si128((__m128i*)(d + k + 3));
		a = _mm_min_epi16(a, v0);
		b = _mm_max_epi16(b, v0);
		v0 = _mm_loadu_si128((__m128i*)(d + k + 4));
		a = _mm_min_epi16(a, v0);
		b = _mm_max_epi16(b, v0);
		v0 = _mm_loadu_si128((__m128i*)(d + k + 5));
		a = _mm_min_epi16(a, v0);
		b = _mm_max_epi16(b, v0);
		v0 = _mm_loadu_si128((__m128i*)(d + k + 6));
		a = _mm_min_epi16(a, v0);
		b = _mm_max_epi16(b, v0);
		v0 = _mm_loadu_si128((__m128i*)(d + k + 7));
		a = _mm_min_epi16(a, v0);
		b = _mm_max_epi16(b, v0);
		v0 = _mm_loadu_si128((__m128i*)(d + k + 8));
		a = _mm_min_epi16(a, v0);
		b = _mm_max_epi16(b, v0);
		v0 = _mm_loadu_si128((__m128i*)(d + k));
		q0 = _mm_max_epi16(q0, _mm_min_epi16(a, v0));
		q1 = _mm_min_epi16(q1, _mm_max_epi16(b, v0));
		v0 = _mm_loadu_si128((__m128i*)(d + k + 9));
		q0 = _mm_max_epi16(q0, _mm_min_epi16(a, v0));
		q1 = _mm_min_epi16(q1, _mm_max_epi16(b, v0));
	}
	q0 = _mm_max_epi16(q0, _mm_sub_epi16(_mm_setzero_si128(), q1));
	q0 = _mm_max_epi16(q0, _mm_unpackhi_epi64(q0, q0));
	q0 = _mm_max_epi16(q0, _mm_srli_si128(q0, 4));
	q0 = _mm_max_epi16(q0, _mm_srli_si128(q0, 2));
	threshold = (short)_mm_cvtsi128_si32(q0) - 1;
#else  
	//a0Ϊ��ֵ  
	int a0 = threshold;
	//����ǵ�����1ʱ��������ֵ  
	for (k = 0; k < 16; k += 2)
	{
		//aΪd[k+1]��d[k+2]��d[k+3]�е���Сֵ  
		int a = std::min((int)d[k + 1], (int)d[k + 2]);
		a = std::min(a, (int)d[k + 3]);
		//���aС����ֵ���������һ��ѭ��  
		if (a <= a0)
			continue;
		//������ֵ  
		//aΪ��d[k+1]��d[k+8]�е���Сֵ  
		a = std::min(a, (int)d[k + 4]);
		a = std::min(a, (int)d[k + 5]);
		a = std::min(a, (int)d[k + 6]);
		a = std::min(a, (int)d[k + 7]);
		a = std::min(a, (int)d[k + 8]);
		//��d[k]��d[k+9]�е���Сֵ��a0�Ƚϣ��ĸ����ĸ���Ϊ�µ���ֵ  
		a0 = std::max(a0, std::min(a, (int)d[k]));
		a0 = std::max(a0, std::min(a, (int)d[k + 9]));
	}
	//����ǵ�����2ʱ��������ֵ  
	int b0 = -a0;
	for (k = 0; k < 16; k += 2)
	{
		int b = std::max((int)d[k + 1], (int)d[k + 2]);
		b = std::max(b, (int)d[k + 3]);
		b = std::max(b, (int)d[k + 4]);
		b = std::max(b, (int)d[k + 5]);
		if (b >= b0)
			continue;
		b = std::max(b, (int)d[k + 6]);
		b = std::max(b, (int)d[k + 7]);
		b = std::max(b, (int)d[k + 8]);

		b0 = std::min(b0, std::max(b, (int)d[k]));
		b0 = std::min(b0, std::max(b, (int)d[k + 9]));
	}

	threshold = -b0 - 1;
#endif  

#if VERIFY_CORNERS  
	testCorner(ptr, pixel, K, N, threshold);
#endif  
	//���º����ֵ��Ϊ���  
	return threshold;
}


void makeOffsets(int pixel[25], int rowStride, int patternSize)
{
	//�ֱ����������飬���ڱ�ʾpatternSizeΪ16��12��8ʱ��Բ�����ض���Բ�ĵ��������λ��
	static const int offsets16[][2] =
	{
		{ 0,  3 },{ 1,  3 },{ 2,  2 },{ 3,  1 },{ 3, 0 },{ 3, -1 },{ 2, -2 },{ 1, -3 },
	{ 0, -3 },{ -1, -3 },{ -2, -2 },{ -3, -1 },{ -3, 0 },{ -3,  1 },{ -2,  2 },{ -1,  3 }
	};

	static const int offsets12[][2] =
	{
		{ 0,  2 },{ 1,  2 },{ 2,  1 },{ 2, 0 },{ 2, -1 },{ 1, -2 },
	{ 0, -2 },{ -1, -2 },{ -2, -1 },{ -2, 0 },{ -2,  1 },{ -1,  2 }
	};

	static const int offsets8[][2] =
	{
		{ 0,  1 },{ 1,  1 },{ 1, 0 },{ 1, -1 },
	{ 0, -1 },{ -1, -1 },{ -1, 0 },{ -1,  1 }
	};
	//����patternSizeֵ���õ�����Ӧ�����涨����ĸ�����
	const int(*offsets)[2] = patternSize == 16 ? offsets16 :
		patternSize == 12 ? offsets12 :
		patternSize == 8 ? offsets8 : 0;

	CV_Assert(pixel && offsets);

	int k = 0;
	//��������ͼ��ÿ�е����ظ������õ�Բ�����صľ�������λ��
	for (; k < patternSize; k++)
		pixel[k] = offsets[k][0] + offsets[k][1] * rowStride;
	//����Ҫ�������������أ����Ҫѭ���Ķ��г�һЩֵ
	for (; k < 25; k++)
		pixel[k] = pixel[k - patternSize];
}


void FAST_tDu(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold, int threshold2, bool nonmax_suppression, int patternSize)
{
	//int patternSize = 16;
	Mat img = _img.getMat();    //��ȡ������ͼ�����
								//KΪԲ���������صĸ���
								//N����ѭ��Բ�ܵ����ص㣬��ΪҪ��β���ӣ�����NҪ��ʵ��Բ������������K+1��
	const int K = patternSize / 2, N = patternSize + K + 1;

	int i, j, k, pixel[25];
	//�ҵ�Բ�����ص������Բ�ĵ�ƫ����
	makeOffsets(pixel, (int)img.step, patternSize);
	//��������������
	keypoints.clear();
	//��֤��ֵ������255����С��0
	threshold = std::min(std::max(threshold, 0), 255);
	threshold2 = std::min(std::max(threshold2, 0), 255);

	// threshold_tabΪ��ֵ�б��ڽ�����ֵ�Ƚϵ�ʱ��ֻ���ñ���
	uchar threshold_tab[512];
	/*Ϊ��ֵ�б�ֵ���ñ��Ϊ���Σ���һ�δ�threshold_tab[0]��threshold_tab[255 - threshold]��ֵΪ1�����ڸ������ֵ��ʾ����ǵ��ж�����2���ڶ��δ�threshold_tab[255 �C threshold]��threshold_tab[255 + threshold]��ֵΪ0�����ڸ������ֵ��ʾ���ǽǵ㣻�����δ�threshold_tab[255 + threshold]��threshold_tab[511]��ֵΪ2�����ڸ������ֵ��ʾ����ǵ��ж�����1*/
	for (i = -255; i <= 255; i++)
	{
		threshold_tab[i + 255] = (uchar)(i < -threshold2 ? 0 : i < -threshold ? 1 : i < threshold ? 0 : i < threshold2 ? 2 : 0);

	}

	//threshold_tab[i + 255] = (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);
	//����һ���ڴ�ռ�
	AutoBuffer<uchar> _buf((img.cols + 16) * 3 * (sizeof(int) + sizeof(uchar)) + 128);
	uchar* buf[3];
	/*buf[0��buf[1]��buf[2]�ֱ��ʾͼ���ǰһ�С���ǰ�кͺ�һ�С���Ϊ�ڷǼ���ֵ���ƵĲ���2�У���Ҫ��3��3�Ľǵ������ڽ��бȽϣ������Ҫ���е�ͼ�����ݡ���Ϊֻ�еõ��˵�ǰ�е����ݣ����Զ�����һ����˵���Ŵչ����������е����ݣ��������ķǼ���ֵ���ƵĽ������һ�����ݵĴ�����*/
	buf[0] = _buf; buf[1] = buf[0] + img.cols; buf[2] = buf[1] + img.cols;
	//cpbuf�洢�ǵ������λ�ã�Ҳ����Ҫ�������е�����
	int* cpbuf[3];
	cpbuf[0] = (int*)alignPtr(buf[2] + img.cols, sizeof(int)) + 1;
	cpbuf[1] = cpbuf[0] + img.cols + 1;
	cpbuf[2] = cpbuf[1] + img.cols + 1;
	memset(buf[0], 0, img.cols * 3);    //buf�����ڴ�����
										//��������ͼ�����أ�Ѱ�ҽǵ�
										//����Բ�İ뾶Ϊ3�����أ����ͼ������ܱ߽綼����3�����صĿ��
	for (i = 3; i < img.rows - 2; i++)
	{
		//�õ�ͼ���е��׵�ַָ��
		const uchar* ptr = img.ptr<uchar>(i) + 3;
		//�õ�buf��ĳ�����飬���ڴ洢��ǰ�еĵ÷ֺ�����ֵV
		uchar* curr = buf[(i - 3) % 3];
		//�õ�cpbuf��ĳ�����飬���ڴ洢��ǰ�еĽǵ�����λ��
		int* cornerpos = cpbuf[(i - 3) % 3];
		memset(curr, 0, img.cols);    //����
		int ncorners = 0;    //��⵽�Ľǵ�����

		if (i < img.rows - 3)
		{
			//ÿһ�ж�����3�����صĿ��
			j = 3;

			for (; j < img.cols - 3; j++, ptr++)
			{
				//��ǰ���صĻҶ�ֵ
				int v = ptr[0];
				//�ɵ�ǰ���صĻҶ�ֵ��ȷ��������ֵ�б��е�λ��
				const uchar* tab = &threshold_tab[0] - v + 255;
				//pixel[0]��ʾԲ���ϱ��Ϊ0�����������Բ�������ƫ����
				//ptr[pixel[0]��ʾԲ���ϱ��Ϊ0������ֵ
				//tab[ptr[pixel[0]]]��ʾ����ڵ�ǰ���أ���Բ�ģ�Բ���ϱ��Ϊ0������ֵ����ֵ�б�threshold_tab������ѯ�õ���ֵ�����Ϊ1��˵��I0 < Ip - t�����Ϊ2��˵��I0 > Ip + t�����Ϊ0��˵�� Ip �C t < I0 < Ip + t�����ͨ��tab���Ϳ��Եõ���ǰ�����Ƿ�����ǵ�������
				//���Ϊ0��8����ֱ����Բ���ϵ��������ص㣩���б��е�ֵ����õ�d��d=0˵�����Ϊ0��8��ֵ����0��d=1˵�����Ϊ0��8��ֵ������һ��Ϊ1������һ������Ϊ2��d=2˵�����Ϊ0��8��ֵ������һ��Ϊ2������һ������Ϊ1��d=3˵�����Ϊ0��8��ֵ��һ��Ϊ1����һ��Ϊ2��ֻ�����������������
				int d = tab[ptr[pixel[0]]] | tab[ptr[pixel[8]]];
				//d=0˵��Բ���ϲ�����������12����������ǵ���������˵�ǰֵһ�����ǽǵ㣬�����˳��˴�ѭ����������һ��ѭ��
				if (d == 0)
					continue;
				//������������ֱ�����������ص���ж�
				d &= tab[ptr[pixel[2]]] | tab[ptr[pixel[10]]];
				d &= tab[ptr[pixel[4]]] | tab[ptr[pixel[12]]];
				d &= tab[ptr[pixel[6]]] | tab[ptr[pixel[14]]];
				//d=0˵������d��������һ��dΪ0�����Կ϶����ǽǵ㣻��һ�������һ��dΪ2������һ��dΪ1�������ҲΪ0����˵��һ��������ǵ�����1������һ������ǵ�����2�����Կ϶�Ҳ����������12����������ͬһ���ǵ������ģ����Ҳһ�����ǽǵ㡣
				if (d == 0)
					continue;
				//�����ж�Բ����ʣ������ص�
				d &= tab[ptr[pixel[1]]] | tab[ptr[pixel[9]]];
				d &= tab[ptr[pixel[3]]] | tab[ptr[pixel[11]]];
				d &= tab[ptr[pixel[5]]] | tab[ptr[pixel[13]]];
				d &= tab[ptr[pixel[7]]] | tab[ptr[pixel[15]]];
				//�������if��������˵���п�������ǵ�����2
				if (d & 1)
				{
					//vtΪ�����Ľǵ���������Ip �C t��countΪ�������صļ���ֵ
					int vt = v - threshold, count = 0;
					//��������Բ��
					for (k = 0; k < N; k++)
					{
						int x = ptr[pixel[k]];    //��ȡ��Բ���ϵ�����ֵ
						if (x < vt)    //�����������2
						{
							//�������������ж��Ƿ����K��KΪԲ�����ص�һ�룩
							if (++count > K)
							{
								//�����if��䣬˵���Ѿ��õ�һ���ǵ�
								//����õ��λ�ã����ѵ�ǰ�еĽǵ�����1
								cornerpos[ncorners++] = j;
								//���зǼ���ֵ���Ƶĵ�һ��������÷ֺ���
								if (nonmax_suppression)
									curr[j] = (uchar)cornerScore(ptr, pixel, threshold);
								break;    //�˳�ѭ��
							}
						}
						else
							count = 0;    //�������صļ���ֵ����
					}
				}
				//�������if��������˵���п�������ǵ�����1
				if (d & 2)
				{
					//vtΪ�����Ľǵ���������Ip + t��countΪ�������صļ���ֵ
					int vt = v + threshold, count = 0;
					//��������Բ��
					for (k = 0; k < N; k++)
					{
						int x = ptr[pixel[k]];    //��ȡ��Բ���ϵ�����ֵ
						if (x > vt)    //�����������1
						{
							//�������������ж��Ƿ����K��KΪԲ�����ص�һ�룩
							if (++count > K)
							{
								//�����if��䣬˵���Ѿ��õ�һ���ǵ�
								//����õ��λ�ã����ѵ�ǰ�еĽǵ�����1
								cornerpos[ncorners++] = j;
								//���зǼ���ֵ���Ƶĵ�һ��������÷ֺ���
								if (nonmax_suppression)
									curr[j] = (uchar)cornerScore(ptr, pixel, threshold);
								break;    //�˳�ѭ��
							}
						}
						else
							count = 0;    //�������صļ���ֵ����
					}
				}
			}
		}
		//���浱ǰ������⵽�Ľǵ���
		cornerpos[-1] = ncorners;
		//i=3˵��ֻ����������һ�е����ݣ������ܽ��зǼ���ֵ���Ƶĵڶ��������Բ������������Ĳ�����ֱ�ӽ�����һ��ѭ��
		if (i == 3)
			continue;
		//���´����ǽ��зǼ���ֵ���Ƶĵڶ���������3��3�Ľǵ������ڶԵ÷ֺ�����ֵ���зǼ���ֵ���ơ���Ϊ�����������ļ��㣬�Ѿ��õ��˵�ǰ�е����ݣ����Կ��Խ�����һ�еķǼ���ֵ���ơ��������Ĵ�����е�����һ�еķǼ���ֵ���ơ�
		//��ȡ����һ�к������е�ͼ������
		const uchar* prev = buf[(i - 4 + 3) % 3];
		const uchar* pprev = buf[(i - 5 + 3) % 3];
		//��ȡ����һ������⵽�Ľǵ�λ��
		cornerpos = cpbuf[(i - 4 + 3) % 3];
		//��ȡ����һ�еĽǵ���
		ncorners = cornerpos[-1];
		//����һ���ڱ���������⵽�Ľǵ�
		for (k = 0; k < ncorners; k++)
		{
			j = cornerpos[k];    //�õ��ǵ��λ��
			int score = prev[j];    //�õ��ýǵ�ĵ÷ֺ���ֵ
									//��3��3�Ľǵ������ڣ����㵱ǰ�ǵ��Ƿ�Ϊ���ֵ���������ѹ������ֵ������
			if (!nonmax_suppression ||
				(score > prev[j + 1] && score > prev[j - 1] &&
					score > pprev[j - 1] && score > pprev[j] && score > pprev[j + 1] &&
					score > curr[j - 1] && score > curr[j] && score > curr[j + 1]))
			{
				keypoints.push_back(KeyPoint((float)j, (float)(i - 1), 7.f, -1, (float)score));
			}
		}

	}
}


int main()
{
	clock_t start = clock();
	Mat Object1, MatchObject1, Object, MatchObject;
	Object1 = imread("E:\\sift\\15.jpg");
	MatchObject1 = imread("E:\\sift\\16.jpg");
	resize(Object1, Object1, Size(300, 300), 0, 0, 1);
	resize(MatchObject1, MatchObject1, Size(300, 300), 0, 0, 1);
	cvtColor(Object1, Object, CV_BGR2GRAY);
	cvtColor(MatchObject1, MatchObject, CV_BGR2GRAY);
	vector<KeyPoint> keypointObject, keypointMatch;
	ORB orb;
	//orb.detect(Object, keypointObject);
	//orb.detect(MatchObject, keypointMatch);
	FAST_tDu(Object, keypointObject, 55, 95, 1, 16);
	FAST_tDu(MatchObject, keypointMatch, 55, 95, 1, 16);
	Mat descriptObjiect, descriptMatch;
	orb.compute(Object, keypointObject, descriptObjiect);
	orb.compute(MatchObject, keypointMatch, descriptMatch);

	BFMatcher matcher(NORM_HAMMING, true);
	vector<DMatch> match;
	matcher.match(descriptObjiect, descriptMatch, match);
	Mat OutMatch;
	drawMatches(Object1, keypointObject, MatchObject1, keypointMatch, match, OutMatch, Scalar(0, 0, 255), Scalar_<double>::all(-1));

	imshow("�˳�ǰ", OutMatch);
	printf("Pr_keypointObject=%d\n", keypointObject.size());
	printf("Pr_keypointMatch=%d\n", keypointMatch.size());
	//


	vector<int> queryIdxs(match.size()), trainIdxs(match.size());
	for (size_t i = 0; i < match.size(); i++)
	{
		queryIdxs[i] = match[i].queryIdx;
		trainIdxs[i] = match[i].trainIdx;
	}

	Mat H12;   //�任����

	vector<Point2f> points1; KeyPoint::convert(keypointObject, points1, queryIdxs);
	vector<Point2f> points2; KeyPoint::convert(keypointMatch, points2, trainIdxs);
	int ransacReprojThreshold = 5;  //�ܾ���ֵ


	H12 = findHomography(Mat(points1), Mat(points2), CV_RANSAC, ransacReprojThreshold);
	vector<char> matchesMask(match.size(), 0);
	Mat points1t;
	perspectiveTransform(Mat(points1), points1t, H12);
	for (size_t i1 = 0; i1 < points1.size(); i1++)  //���桮�ڵ㡯
	{
		if (norm(points2[i1] - points1t.at<Point2f>((int)i1, 0)) <= ransacReprojThreshold) //���ڵ������
		{
			matchesMask[i1] = 1;
		}
	}
	Mat match_img2;   //�˳�����㡯��
	drawMatches(Object1, keypointObject, MatchObject1, keypointMatch, match, match_img2, Scalar(0, 0, 255), Scalar_<double>::all(-1), matchesMask);
	clock_t end = clock();
	cout << "ʱ�䣺" << (double)(end - start) << "ms" << endl;
	int count = match.size();
	cout << "����" << count << endl;
	imshow("�˳���ƥ���", match_img2);

	waitKey(0);
	return 0;
}


