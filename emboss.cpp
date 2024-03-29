// g++ emboss.cpp `pkg-config --cflags --libs opencv` -std=c++11 -o emboss && ./emboss preemboss_res.png
// ./emboss preemboss_res.png

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

typedef pair<int, int> ii;

Mat img;
int filter_size;
Mat filter, bgn, nd;
vector<ii> list;
Mat res;

int useful(int i, int j){
	if(i == 0 || i == img.rows - 1 || j == 0 || j == img.cols - 1) return true;

	return img.at<unsigned char>(i - 1 , j) == 0 || img.at<unsigned char>(i, j + 1) == 0 ||
			img.at<unsigned char>(i + 1, j) == 0 || img.at<unsigned char>(i, j - 1) == 0;
}

int valid_pos(int i, int j){
	return i >= 0 && i < img.rows && j >= 0 && j < img.cols;
}

int limitj(int j){
	return min(img.cols - 1, max(0, j));
}

void make_list(){
	int a, b;

	for(int i = 0; i < filter.rows; i++){
		a = -1;
		for(int j = 0; j < filter.cols && a == -1; j++){
			if(filter.at<bool>(i, j) == 1){
				a = j - filter_size / 2;
			}
		}

		b = -1;
		for(int j = filter.cols - 1; j >= 0 && b == -1; j--){
			if(filter.at<bool>(i, j) == 1){
				b = j - filter_size / 2;
			}
		}

		if(a != -1){
			list.push_back(ii(a, b));
		}
	}
}

void fillbe(){
	for(int i = 0; i < img.rows; i++){
		for(int j = 0; j < img.cols; j++){
			if(img.at<unsigned char>(i, j) != 0 && useful(i, j)){
				for(int k = 0; k < list.size(); k++){
					int l = i - filter_size / 2 + k;
					int ce = limitj(j + list[k].first);
					int cd = limitj(j + list[k].second);
					if(valid_pos(l, ce)){
						bgn.at<unsigned char>(l, ce) += 1;
					}
					if(valid_pos(l, cd)){
						nd.at<unsigned char>(l, cd) += 1;
					}
				}
			}
		}
	}
}

void make_res(){
	for(int i = 0; i < res.rows; i++){
		int balance = 0;
		for(int j = 0; j < res.cols; j++){
			balance += bgn.at<unsigned char>(i, j);
			res.at<unsigned char>(i, j) = 255 * (balance > 0);
			balance -= nd.at<unsigned char>(i, j);
		}
	}
}

float pow2(float a){
	return a * a;
}

void create_filter(int diameter){
	filter = Mat::zeros(diameter, diameter, CV_8U);
	float c = (diameter - 1) / 2.0;

	for(int i = 0; i < diameter; i++){
		for(int j = 0; j < diameter; j++){
			float d = sqrt(pow2(i - c) + pow2(j - c));
			if(d <= c + 0.1){
				filter.at<unsigned char>(i, j) = 1;
			}
		}
	}
}

void expand_white(int fs){
	filter_size = fs;
	create_filter(filter_size);

	list.clear();
	make_list();

	bgn = Mat::zeros(img.rows, img.cols, CV_8U);
	nd = Mat::zeros(img.rows, img.cols, CV_8U);

	fillbe();

	res = Mat(img.rows, img.cols, img.type());
	make_res();

	add(img, res, img);
}

void invert(Mat vec){
	bitwise_not(vec, vec);
}

int cm2pixels600dpi(float cm){
	return cm * 0.393701 * 600;
}

int circulo(float raio){
	return cm2pixels600dpi(raio * 2);
}

int main(int argc, char** argv){
	char stackAxis = 'v';
	
	if(argc == 3){
		stackAxis = argv[2][0];
	}
	
	float roundness = 0.04;

	img = imread(argv[1], IMREAD_GRAYSCALE);
	threshold(img, img, 0, 255, THRESH_BINARY + THRESH_OTSU);

	invert(img);
	expand_white(circulo(0.05 + roundness));
	invert(img);
	expand_white(circulo(roundness));
	expand_white(circulo(0.10 + roundness));
	invert(img);
	expand_white(circulo(roundness));
	invert(img);

	Size sz1 = img.size();
	Size sz2 = img.size();
	
	int heightIm3 = sz1.height;
	int widthIm3 = sz1.width;
	if(stackAxis == 'v'){
		heightIm3 = sz1.height + sz2.height + cm2pixels600dpi(0.5);
	} else {
		widthIm3 = sz1.width + sz2.width + cm2pixels600dpi(0.5);
	}
	Mat im3(heightIm3, widthIm3, img.type());
	
	Mat im1(im3, Rect(0, 0, sz1.width, sz1.height));
	img.copyTo(im1);

	img = imread(argv[1], IMREAD_GRAYSCALE);
	threshold(img, img, 0, 255, THRESH_BINARY + THRESH_OTSU);

	expand_white(circulo(0.05 + roundness));
	invert(img);
	expand_white(circulo(roundness));
	expand_white(circulo(0.10 + roundness));
	invert(img);
	expand_white(circulo(roundness));
	invert(img);

	flip(img, img, 1);

	int offsetYIm2 = 0;
	int offsetXIm2 = 0;
	if(stackAxis == 'v'){
		offsetYIm2 = sz1.height + cm2pixels600dpi(0.5);
	} else {
		offsetXIm2 = sz1.width + cm2pixels600dpi(0.5);
	}
	Mat im2(im3, Rect(offsetXIm2, offsetYIm2, sz2.width, sz2.height));
	img.copyTo(im2);

	imwrite("emboss_res.png", im3);

	return 0;
}
