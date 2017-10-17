#include <stdio.h>
#include "opencv2/opencv.hpp"

using namespace cv;

int main(int, char**)
{
	VideoCapture capture(0); //open default camera)
	if(!capture.isOpened())
		return -1;

	//https://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/find_contours/find_contours.html?highlight=contour
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	Mat proc;
	namedWindow("capture", 1);
	namedWindow("processed", 1);
	moveWindow("capture", 0, 0);
	moveWindow("processed", 600, 5);
	for(;;)
	{
		capture >> proc;

		Mat cap = proc.clone();

		GaussianBlur(proc, proc, Size(9,9), 1.5, 1.5);
		//cvtColor(proc, proc, CV_BGR2GRAY);

		//ISOLATE BLUE
		inRange(proc, Scalar(80,0,0), Scalar(255,100,100), proc);

		// NOISE REDUCTION

		int REDUCE_SIZE = 5;
		//http://opencv-srf.blogspot.ca/2010/09/object-detection-using-color-seperation.html
		//morphological opening (remove small objects from the foreground)
		erode(proc, proc, getStructuringElement(MORPH_ELLIPSE, Size(REDUCE_SIZE, REDUCE_SIZE)) );
		dilate( proc, proc, getStructuringElement(MORPH_ELLIPSE, Size(REDUCE_SIZE, REDUCE_SIZE)) ); 
	  
		//morphological closing (fill small holes in the foreground)
		dilate( proc, proc, getStructuringElement(MORPH_ELLIPSE, Size(REDUCE_SIZE, REDUCE_SIZE)) ); 
		erode(proc, proc, getStructuringElement(MORPH_ELLIPSE, Size(REDUCE_SIZE, REDUCE_SIZE)) );

		Canny(proc, proc, 100, 300, 3);
		
		// Find contours
		findContours( proc, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
		// Draw contours
		Mat drawing = Mat::zeros( proc.size(), CV_8UC3 );
		for( int i = 0; i< contours.size(); i++ )
		{
			double area = contourArea(contours[i], false);
			if(area > 250 && contours[i].size() > 70)
				drawContours( drawing, contours, i, Scalar(255,255,0), 1, CV_AA, hierarchy, 0, Point() );
		}

		imshow("capture", cap);
		imshow("processed", drawing);
		if(waitKey(30) >= 30) break; // wait for key to be pressed
	}

	// camera will be automatically deinitialized by VideoCapture
	return 0;

}
