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

void expand_white(int fs){
	filter_size = fs;
	filter = getStructuringElement(MORPH_ELLIPSE, Size(filter_size, filter_size));

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

int main(int argc, char** argv){
	img = imread(argv[1], IMREAD_GRAYSCALE);
	threshold(img, img, 0, 255, THRESH_BINARY + THRESH_OTSU);
	
	expand_white(37);
	invert(img);
	expand_white(13);
	invert(img);
	
	Size sz1 = img.size();
    Size sz2 = img.size();
    Mat im3(sz1.height + sz2.height + cm2pixels600dpi(0.5), sz1.width, img.type());
    Mat left(im3, Rect(0, 0, sz1.width, sz1.height));
    img.copyTo(left);
	
	img = imread(argv[1], IMREAD_GRAYSCALE);
	threshold(img, img, 0, 255, THRESH_BINARY + THRESH_OTSU);
	
	invert(img);
	expand_white(37);
	invert(img);
	expand_white(13);
	invert(img);
	
	flip(img, img, 1);

	Mat right(im3, Rect(0, sz1.height + cm2pixels600dpi(0.5), sz2.width, sz2.height));
    img.copyTo(right);
	
	imwrite("emboss_res.png", im3);

	return 0;
}
