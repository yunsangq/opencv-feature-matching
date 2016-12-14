
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat cameraMatrix;
Mat distCoeffs;

vector<vector<Point3f>> objectPoints;
vector<vector<Point2f>> imagePoints;
Size imageSize;
int flag = 0;
double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0,
k1 = 0.0, k2 = 0.0, p1 = 0.0, p2 = 0.0;

Scalar pointcolor(0, 0, 255);
Scalar textcolor(255, 255, 255);
int thickness = 2;

void init() {
	for (int i = 0; i < 20; i++) {
		Size boardSize(7, 4);
		string str = "image" + to_string(i) + ".png";
		Mat img = imread("./calib/" + str);
		imageSize = img.size();

		vector<Point3f> objectCorners;
		vector<Point2f> imageCorners;

		Mat img_gray;
		for (int i = 0; i < boardSize.height; i++) {
			for (int j = 0; j < boardSize.width; j++) {
				objectCorners.push_back(Point3f(i, j, 0.0f));
			}
		}

		cvtColor(img, img_gray, COLOR_BGR2GRAY);

		bool found = findChessboardCorners(img, boardSize, imageCorners);
		if (found)
			cornerSubPix(img_gray, imageCorners, Size(11, 11), Size(-1, -1),
				TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		else
			std::cout << found << endl;
		drawChessboardCorners(img, boardSize, Mat(imageCorners), found);

		if (imageCorners.size() == boardSize.area()) {
			imagePoints.push_back(imageCorners);
			objectPoints.push_back(objectCorners);
		}
	}
	vector<Mat> rvecs, tvecs;
	calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
		distCoeffs, rvecs, tvecs, flag | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5);
	cout << cameraMatrix << endl;
	cout << distCoeffs << endl;
	fx = cameraMatrix.at<double>(0, 0);
	cx = cameraMatrix.at<double>(0, 2);
	fy = cameraMatrix.at<double>(1, 1);
	cy = cameraMatrix.at<double>(1, 2);
	k1 = distCoeffs.at<double>(0, 0);
	k2 = distCoeffs.at<double>(0, 1);
	p1 = distCoeffs.at<double>(0, 2);
	p2 = distCoeffs.at<double>(0, 3);
}

Point2f world_to_cam_to_pixel(double w[], Mat R, Mat tvec) {
	Mat Pw(3, 1, CV_64FC1, w);
	Mat Pc = R*Pw + tvec;
	double* pc = (double*)Pc.data;
	double u = pc[0] / pc[2];
	double v = pc[1] / pc[2];
	double x = u*fx + cx;
	double y = v*fy + cy;
	return Point2f(x, y);
}

int main() {
	VideoCapture vc(0);
	if (!vc.isOpened()) return 0;

	std::cout << "Camera Calibration...." << endl;
	init();
	std::cout << "Camera Calibration Complete!!" << endl;

	Mat model_image = imread("model.bmp");
	cvtColor(model_image, model_image, COLOR_BGR2GRAY);
	while (waitKey(33) != 27) {
		Mat input, input_gray;
		vc >> input;
		cvtColor(input, input_gray, COLOR_BGR2GRAY);

		try {
			//detector
			Ptr<Feature2D> fd = BRISK::create();

			//key point
			vector<KeyPoint> keypts1, keypts2;
			fd->detect(model_image, keypts1);
			fd->detect(input_gray, keypts2);

			//descriptor
			Mat descriptor1, descriptor2;
			fd->compute(model_image, keypts1, descriptor1);
			fd->compute(input_gray, keypts2, descriptor2);

			//matching method
			Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce");

			//matching
			vector<DMatch> matches;
			descriptorMatcher->match(descriptor1, descriptor2, matches);

			Mat index;
			int nbMatch = int(matches.size());
			Mat tab(nbMatch, 1, CV_32F);
			for (int i = 0; i < nbMatch; i++) {
				tab.at<float>(i, 0) = matches[i].distance;
			}
			sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
			vector<DMatch> bestMatches;
			for (int i = 0; i < 30; i++) {
				bestMatches.push_back(matches[index.at<int>(i, 0)]);
			}

			vector<Point2f> pts1, pts2;
			for (int i = 0; i < int(bestMatches.size()); i++) {
				int idx1 = bestMatches[i].queryIdx;
				int idx2 = bestMatches[i].trainIdx;
				pts1.push_back(keypts1[idx1].pt);
				pts2.push_back(keypts2[idx2].pt);
			}

			Mat H = findHomography(pts1, pts2, RANSAC);

			int model_h = model_image.rows;
			int model_w = model_image.cols;

			vector<Point2f> corner_pts1;
			corner_pts1.push_back(Point(0, 0));
			corner_pts1.push_back(Point(0, model_h - 1));
			corner_pts1.push_back(Point(model_w - 1, model_h - 1));
			corner_pts1.push_back(Point(model_w - 1, 0));

			vector<Point2f> corner_pts2;
			perspectiveTransform(corner_pts1, corner_pts2, H);

			if (int(corner_pts2.size()) >= 4) {
				vector<Point3f> objectCorners;
				Mat R, rvec, tvec;
				objectCorners.push_back(Point3f(0.0f, 0.0f, 0.0f));
				objectCorners.push_back(Point3f(7.9f, 0.0f, 0.0f));
				objectCorners.push_back(Point3f(7.9f, 8.9f, 0.0f));
				objectCorners.push_back(Point3f(0.0f, 8.9f, 0.0f));
				solvePnP(objectCorners, corner_pts2, cameraMatrix, distCoeffs, rvec, tvec);
				Rodrigues(rvec, R);

				double w0[] = { 0,0,0 };
				double w1[] = { 7.9,0.0,0.0 };
				double w2[] = { 7.9,8.9,0.0 };
				double w3[] = { 0.0,8.9,0.0 };

				double w4[] = { 0,0,8.0 };
				double w5[] = { 7.9,0.0,8.0 };
				double w6[] = { 7.9,8.9,8.0 };
				double w7[] = { 0.0,8.9,8.0 };

				Point2f p0 = world_to_cam_to_pixel(w0, R, tvec);				
				Point2f p1 = world_to_cam_to_pixel(w1, R, tvec);							
				Point2f p2 = world_to_cam_to_pixel(w2, R, tvec);				
				Point2f p3 = world_to_cam_to_pixel(w3, R, tvec);
				
				Point2f p4 = world_to_cam_to_pixel(w4, R, tvec);
				Point2f p5 = world_to_cam_to_pixel(w5, R, tvec);
				Point2f p6 = world_to_cam_to_pixel(w6, R, tvec);
				Point2f p7 = world_to_cam_to_pixel(w7, R, tvec);

				line(input, p0, p1, pointcolor, thickness);
				line(input, p1, p2, pointcolor, thickness);
				line(input, p2, p3, pointcolor, thickness);
				line(input, p3, p0, pointcolor, thickness);

				line(input, p4, p5, pointcolor, thickness);
				line(input, p5, p6, pointcolor, thickness);
				line(input, p6, p7, pointcolor, thickness);
				line(input, p7, p4, pointcolor, thickness);

				line(input, p0, p4, pointcolor, thickness);
				line(input, p1, p5, pointcolor, thickness);
				line(input, p2, p6, pointcolor, thickness);
				line(input, p3, p7, pointcolor, thickness);

				/*
				circle(input, p0, 5, pointcolor, thickness);
				circle(input, p1, 5, pointcolor, thickness);
				circle(input, p2, 5, pointcolor, thickness);
				circle(input, p3, 5, pointcolor, thickness);
				*/
			}


			//display
			Mat result;
			drawMatches(model_image, keypts1, input, keypts2, bestMatches, result);

			imshow("result", result);
		}
		catch (Exception& e){
			cout << e.msg << endl;			
		}
	}

	return 0;
}