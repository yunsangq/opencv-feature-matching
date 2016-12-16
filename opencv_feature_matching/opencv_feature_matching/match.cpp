#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\features2d.hpp>
#include <iostream>
#include <thread>
#include <time.h>

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

void makecube(Mat& input, vector<Point2f> corner_pts2) {
	vector<Point3f> objectCorners;
	Mat R, rvec, tvec;
	objectCorners.push_back(Point3f(0.0f, 7.9f, 0.0f));
	objectCorners.push_back(Point3f(0.0f, 0.0f, 0.0f));
	objectCorners.push_back(Point3f(8.9f, 0.0f, 0.0f));
	objectCorners.push_back(Point3f(8.9f, 7.9f, 0.0f));
	solvePnP(objectCorners, corner_pts2, cameraMatrix, distCoeffs, rvec, tvec);
	Rodrigues(rvec, R);

	double w0[] = { 0,7.9,0 };
	double w1[] = { 0,0,0 };
	double w2[] = { 8.9,0,0 };
	double w3[] = { 8.9,7.9,0 };

	double w4[] = { 0,7.9,8.0 };
	double w5[] = { 0,0,8.0 };
	double w6[] = { 8.9,0,8.0 };
	double w7[] = { 8.9,7.9,8.0 };

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
}

void method1(VideoCapture vc, Mat model_image, Mat descimg1, vector<KeyPoint> keyimg1) {
	long double frame = 0.0;
	time_t start;
	time(&start);
	while (waitKey(1) != 27) {
		Mat input, input_gray;
		vc >> input;
		frame++;
		cvtColor(input, input_gray, COLOR_BGR2GRAY);

		try {

			Ptr<Feature2D> b = BRISK::create();

			Ptr<DescriptorMatcher> descriptorMatcher;
			vector<DMatch> matches;
			vector<KeyPoint> keyimg2;
			Mat descimg2;
			b->detectAndCompute(input_gray, Mat(), keyimg2, descimg2);

			if (keyimg2.size() != 0) {
				descriptorMatcher = DescriptorMatcher::create("BruteForce");
				descriptorMatcher->match(descimg1, descimg2, matches, Mat());

				if (matches.size() > 4) {

					vector<Point2f> pts1, pts2;
					for (int i = 0; i < (int)matches.size(); i++) {
						int idx1 = matches[i].queryIdx;
						int idx2 = matches[i].trainIdx;
						pts1.push_back(keyimg1[idx1].pt);
						pts2.push_back(keyimg2[idx2].pt);
					}

					Mat H = findHomography(pts1, pts2, RANSAC);

					if (matches.size() > 10) {

						int model_h = model_image.rows;
						int model_w = model_image.cols;

						vector<Point2f> corner_pts1;
						corner_pts1.push_back(Point(0, 0));
						corner_pts1.push_back(Point(0, model_h - 1));
						corner_pts1.push_back(Point(model_w - 1, model_h - 1));
						corner_pts1.push_back(Point(model_w - 1, 0));

						vector<Point2f> corner_pts2;
						perspectiveTransform(corner_pts1, corner_pts2, H);

						makecube(input, corner_pts2);
					}
				}
				time_t end;
				time(&end);
				double diff = difftime(end, start);
				double fps = frame / diff;
				string h_fps = format("%.2f", fps);
				cv::putText(input, "Method1 FPS:" + h_fps, Point2f(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

				Mat result;
				cv::drawMatches(model_image, keyimg1, input, keyimg2, matches, result);

				cv::imshow("all + ransac", result);
			}
		}
		catch (Exception& e)
		{
			cout << e.msg << endl;			
		}		
	}
}

void method2(VideoCapture vc, Mat model_image, Mat descimg1, vector<KeyPoint> keyimg1) {
	long double frame = 0.0;
	time_t start;
	time(&start);
	while (waitKey(1) != 27) {
		Mat input, input_gray;
		vc >> input;
		frame++;
		cvtColor(input, input_gray, COLOR_BGR2GRAY);

		try {

			Ptr<Feature2D> b = BRISK::create();

			Ptr<DescriptorMatcher> descriptorMatcher;
			vector<DMatch> matches;
			vector<DMatch> bestMatches;
			vector<KeyPoint> keyimg2;
			Mat descimg2;
			b->detectAndCompute(input_gray, Mat(), keyimg2, descimg2);

			if (keyimg2.size() != 0) {
				descriptorMatcher = DescriptorMatcher::create("BruteForce");
				descriptorMatcher->match(descimg1, descimg2, matches, Mat());

				if (matches.size() > 4) {

					Mat index;
					int nbMatch = int(matches.size());
					Mat tab(nbMatch, 1, CV_32F);
					float* tab_data = (float*)tab.data;
					for (int i = 0; i < nbMatch; i++) {
						tab_data[i] = matches[i].distance;
					}
					sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
					
					int* index_data = (int*)index.data;
					for (int i = 0; i < 30; i++) {
						bestMatches.push_back(matches[index_data[i]]);
					}

					vector<Point2f> pts1, pts2;
					for (int i = 0; i < (int)bestMatches.size(); i++) {
						int idx1 = bestMatches[i].queryIdx;
						int idx2 = bestMatches[i].trainIdx;
						pts1.push_back(keyimg1[idx1].pt);
						pts2.push_back(keyimg2[idx2].pt);
					}

					Mat H = findHomography(pts1, pts2, RANSAC);

					if (bestMatches.size() > 10) {

						int model_h = model_image.rows;
						int model_w = model_image.cols;

						vector<Point2f> corner_pts1;
						corner_pts1.push_back(Point(0, 0));
						corner_pts1.push_back(Point(0, model_h - 1));
						corner_pts1.push_back(Point(model_w - 1, model_h - 1));
						corner_pts1.push_back(Point(model_w - 1, 0));

						vector<Point2f> corner_pts2;
						perspectiveTransform(corner_pts1, corner_pts2, H);

						makecube(input, corner_pts2);
					}
				}
				time_t end;
				time(&end);
				double diff = difftime(end, start);
				double fps = frame / diff;
				string h_fps = format("%.2f", fps);
				cv::putText(input, "Method2 FPS:" + h_fps, Point2f(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

				Mat result;
				cv::drawMatches(model_image, keyimg1, input, keyimg2, bestMatches, result);

				cv::imshow("30-best + ransac", result);
			}
		}
		catch (Exception& e)
		{
			cout << e.msg << endl;
		}
	}
}

void method3(VideoCapture vc, Mat model_image, Mat descimg1, vector<KeyPoint> keyimg1) {
	long double frame = 0.0;
	time_t start;
	time(&start);
	while (waitKey(1) != 27) {
		Mat input, input_gray;
		vc >> input;
		frame++;
		cvtColor(input, input_gray, COLOR_BGR2GRAY);

		try {

			Ptr<Feature2D> b = BRISK::create();

			Ptr<DescriptorMatcher> descriptorMatcher;
			vector<DMatch> matches;
			vector<DMatch> bestMatches;
			vector<KeyPoint> keyimg2;
			Mat descimg2;
			b->detectAndCompute(input_gray, Mat(), keyimg2, descimg2);

			if (keyimg2.size() != 0) {
				descriptorMatcher = DescriptorMatcher::create("BruteForce");
				descriptorMatcher->match(descimg1, descimg2, matches, Mat());

				if (matches.size() > 4) {

					Mat index;
					int nbMatch = int(matches.size());
					Mat tab(nbMatch, 1, CV_32F);
					float* tab_data = (float*)tab.data;
					for (int i = 0; i < nbMatch; i++) {
						tab_data[i] = matches[i].distance;
					}
					sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
					
					int* index_data = (int*)index.data;
					for (int i = 0; i < 30; i++) {
						bestMatches.push_back(matches[index_data[i]]);
					}

					vector<Point2f> pts1, pts2;
					for (int i = 0; i < (int)bestMatches.size(); i++) {
						int idx1 = bestMatches[i].queryIdx;
						int idx2 = bestMatches[i].trainIdx;
						pts1.push_back(keyimg1[idx1].pt);
						pts2.push_back(keyimg2[idx2].pt);
					}

					float reprojectionThreshold = 1.0;
					vector<unsigned char> inliersMask(bestMatches.size());

					Mat H = findHomography(pts1, pts2, CV_FM_RANSAC, reprojectionThreshold, inliersMask);
					vector<cv::DMatch> inliers;
					for (size_t i = 0; i < inliersMask.size(); i++) {
						if (inliersMask[i])
							inliers.push_back(bestMatches[i]);
					}
					bestMatches.swap(inliers);

					if (bestMatches.size() > 10) {
						int model_h = model_image.rows;
						int model_w = model_image.cols;

						vector<Point2f> corner_pts1;
						corner_pts1.push_back(Point(0, 0));
						corner_pts1.push_back(Point(0, model_h - 1));
						corner_pts1.push_back(Point(model_w - 1, model_h - 1));
						corner_pts1.push_back(Point(model_w - 1, 0));

						vector<Point2f> corner_pts2;
						perspectiveTransform(corner_pts1, corner_pts2, H);

						makecube(input, corner_pts2);
					}
				}
				time_t end;
				time(&end);
				double diff = difftime(end, start);
				double fps = frame / diff;
				string h_fps = format("%.2f", fps);
				cv::putText(input, "Method3 FPS:" + h_fps, Point2f(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

				Mat result;
				cv::drawMatches(model_image, keyimg1, input, keyimg2, bestMatches, result);

				cv::imshow("30-best + inlier + ransac", result);
			}
		}
		catch (Exception& e)
		{
			cout << e.msg << endl;
		}
	}
}

void method4(VideoCapture vc, Mat model_image, Mat descimg1, vector<KeyPoint> keyimg1) {
	long double frame = 0.0;
	time_t start;
	time(&start);
	while (waitKey(1) != 27) {
		Mat input, input_gray;
		vc >> input;
		frame++;
		cvtColor(input, input_gray, COLOR_BGR2GRAY);

		try {

			Ptr<Feature2D> b = BRISK::create();

			Ptr<DescriptorMatcher> descriptorMatcher;
			vector<KeyPoint> keyimg2;
			vector<vector<DMatch>> m_knnMatches;
			vector<DMatch>m_Matches;
			Mat descimg2;
			b->detectAndCompute(input_gray, Mat(), keyimg2, descimg2);

			if (keyimg2.size() != 0) {
				descriptorMatcher = DescriptorMatcher::create("BruteForce");

				const float minRatio = 0.8f;
				descriptorMatcher->knnMatch(descimg1, descimg2, m_knnMatches, 2);
				for (size_t i = 0; i<m_knnMatches.size(); i++)
				{
					const cv::DMatch& bestMatch = m_knnMatches[i][0];
					const cv::DMatch& betterMatch = m_knnMatches[i][1];
					float distanceRatio = bestMatch.distance / betterMatch.distance;
					if (distanceRatio <minRatio)
					{
						m_Matches.push_back(bestMatch);
					}
				}
				
				if (m_Matches.size() > 4) {
					vector<Point2f> pts1, pts2;
					for (int i = 0; i < (int)m_Matches.size(); i++) {
						int idx1 = m_Matches[i].queryIdx;
						int idx2 = m_Matches[i].trainIdx;
						pts1.push_back(keyimg1[idx1].pt);
						pts2.push_back(keyimg2[idx2].pt);
					}

					float reprojectionThreshold = 1.0;
					vector<unsigned char> inliersMask(m_Matches.size());

					Mat H = findHomography(pts1, pts2, CV_FM_RANSAC, reprojectionThreshold, inliersMask);
					vector<cv::DMatch> inliers;
					for (size_t i = 0; i < inliersMask.size(); i++) {
						if (inliersMask[i])
							inliers.push_back(m_Matches[i]);
					}
					m_Matches.swap(inliers);

					if (m_Matches.size() > 10) {
						int model_h = model_image.rows;
						int model_w = model_image.cols;

						vector<Point2f> corner_pts1;
						corner_pts1.push_back(Point(0, 0));
						corner_pts1.push_back(Point(0, model_h - 1));
						corner_pts1.push_back(Point(model_w - 1, model_h - 1));
						corner_pts1.push_back(Point(model_w - 1, 0));

						vector<Point2f> corner_pts2;
						perspectiveTransform(corner_pts1, corner_pts2, H);

						makecube(input, corner_pts2);
					}
				}

				time_t end;
				time(&end);
				double diff = difftime(end, start);
				double fps = frame / diff;
				string h_fps = format("%.2f", fps);
				cv::putText(input, "Method4 FPS:" + h_fps, Point2f(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

				Mat result;
				cv::drawMatches(model_image, keyimg1, input, keyimg2, m_Matches, result);

				cv::imshow("ratio test + inlier + ransac", result);
			}
		}
		catch (Exception& e)
		{
			cout << e.msg << endl;
		}
	}
}

void method5(VideoCapture vc, Mat model_image, Mat descimg1, vector<KeyPoint> keyimg1) {
	long double frame = 0.0;
	time_t start;
	time(&start);
	while (waitKey(1) != 27) {
		Mat input, input_gray;
		vc >> input;
		frame++;
		cvtColor(input, input_gray, COLOR_BGR2GRAY);

		try {

			Ptr<Feature2D> b = BRISK::create();

			Ptr<DescriptorMatcher> descriptorMatcher;
			vector<KeyPoint> keyimg2;
			vector<vector<DMatch>> m_knnMatches;
			vector<DMatch>m_Matches;
			Mat descimg2;
			b->detectAndCompute(input_gray, Mat(), keyimg2, descimg2);

			if (keyimg2.size() != 0) {				
				Ptr<DescriptorMatcher> BFMatcher(new BFMatcher(NORM_HAMMING, true));
				BFMatcher->match(descimg1, descimg2, m_Matches);

				if (m_Matches.size() > 4) {
					vector<Point2f> pts1, pts2;
					for (int i = 0; i < (int)m_Matches.size(); i++) {
						int idx1 = m_Matches[i].queryIdx;
						int idx2 = m_Matches[i].trainIdx;
						pts1.push_back(keyimg1[idx1].pt);
						pts2.push_back(keyimg2[idx2].pt);
					}

					float reprojectionThreshold = 1.0;
					vector<unsigned char> inliersMask(m_Matches.size());

					Mat H = findHomography(pts1, pts2, CV_FM_RANSAC, reprojectionThreshold, inliersMask);
					vector<cv::DMatch> inliers;
					for (size_t i = 0; i < inliersMask.size(); i++) {
						if (inliersMask[i])
							inliers.push_back(m_Matches[i]);
					}
					m_Matches.swap(inliers);

					if (m_Matches.size() > 10) {
						int model_h = model_image.rows;
						int model_w = model_image.cols;

						vector<Point2f> corner_pts1;
						corner_pts1.push_back(Point(0, 0));
						corner_pts1.push_back(Point(0, model_h - 1));
						corner_pts1.push_back(Point(model_w - 1, model_h - 1));
						corner_pts1.push_back(Point(model_w - 1, 0));

						vector<Point2f> corner_pts2;
						perspectiveTransform(corner_pts1, corner_pts2, H);

						makecube(input, corner_pts2);
					}
				}

				time_t end;
				time(&end);
				double diff = difftime(end, start);
				double fps = frame / diff;
				string h_fps = format("%.2f", fps);
				cv::putText(input, "Method4 FPS:" + h_fps, Point2f(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

				Mat result;
				cv::drawMatches(model_image, keyimg1, input, keyimg2, m_Matches, result);

				cv::imshow("backward-check + inlier + ransac", result);
			}
		}
		catch (Exception& e)
		{
			cout << e.msg << endl;
		}
	}
}

void method6(VideoCapture vc, Mat model_image, Mat descimg1, vector<KeyPoint> keyimg1) {
	long double frame = 0.0;
	time_t start;
	time(&start);
	while (waitKey(1) != 27) {
		Mat input, input_gray;
		vc >> input;
		frame++;
		cvtColor(input, input_gray, COLOR_BGR2GRAY);

		try {

			Ptr<Feature2D> b = ORB::create();

			Ptr<DescriptorMatcher> descriptorMatcher;
			vector<DMatch> matches;
			vector<DMatch> bestMatches;
			vector<KeyPoint> keyimg2;
			Mat descimg2;
			b->detectAndCompute(input_gray, Mat(), keyimg2, descimg2);

			if (keyimg2.size() != 0) {
				descriptorMatcher = DescriptorMatcher::create("BruteForce-Hamming");
				descriptorMatcher->match(descimg1, descimg2, matches, Mat());

				if (matches.size() > 4) {

					Mat index;
					int nbMatch = int(matches.size());
					Mat tab(nbMatch, 1, CV_32F);
					float* tab_data = (float*)tab.data;
					for (int i = 0; i < nbMatch; i++) {
						tab_data[i] = matches[i].distance;
					}
					sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);

					int* index_data = (int*)index.data;
					for (int i = 0; i < 30; i++) {
						bestMatches.push_back(matches[index_data[i]]);
					}

					vector<Point2f> pts1, pts2;
					for (int i = 0; i < (int)bestMatches.size(); i++) {
						int idx1 = bestMatches[i].queryIdx;
						int idx2 = bestMatches[i].trainIdx;
						pts1.push_back(keyimg1[idx1].pt);
						pts2.push_back(keyimg2[idx2].pt);
					}

					float reprojectionThreshold = 1.0;
					vector<unsigned char> inliersMask(bestMatches.size());

					Mat H = findHomography(pts1, pts2, CV_FM_RANSAC, reprojectionThreshold, inliersMask);
					vector<cv::DMatch> inliers;
					for (size_t i = 0; i < inliersMask.size(); i++) {
						if (inliersMask[i])
							inliers.push_back(bestMatches[i]);
					}
					bestMatches.swap(inliers);

					if (bestMatches.size() > 10) {
						int model_h = model_image.rows;
						int model_w = model_image.cols;

						vector<Point2f> corner_pts1;
						corner_pts1.push_back(Point(0, 0));
						corner_pts1.push_back(Point(0, model_h - 1));
						corner_pts1.push_back(Point(model_w - 1, model_h - 1));
						corner_pts1.push_back(Point(model_w - 1, 0));

						vector<Point2f> corner_pts2;
						perspectiveTransform(corner_pts1, corner_pts2, H);

						makecube(input, corner_pts2);
					}
				}
				time_t end;
				time(&end);
				double diff = difftime(end, start);
				double fps = frame / diff;
				string h_fps = format("%.2f", fps);
				cv::putText(input, "Method6 FPS:" + h_fps, Point2f(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

				Mat result;
				cv::drawMatches(model_image, keyimg1, input, keyimg2, bestMatches, result);

				cv::imshow("ORB 30-best + inlier + ransac", result);
			}
		}
		catch (Exception& e)
		{
			cout << e.msg << endl;
		}
	}
}

void method7(VideoCapture vc, Mat model_image, Mat descimg1, vector<KeyPoint> keyimg1) {
	long double frame = 0.0;
	time_t start;
	time(&start);
	while (waitKey(1) != 27) {
		Mat input, input_gray;
		vc >> input;
		frame++;
		cvtColor(input, input_gray, COLOR_BGR2GRAY);

		try {

			Ptr<Feature2D> b = ORB::create();

			Ptr<DescriptorMatcher> descriptorMatcher;
			vector<KeyPoint> keyimg2;
			vector<vector<DMatch>> m_knnMatches;
			vector<DMatch>m_Matches;
			Mat descimg2;
			b->detectAndCompute(input_gray, Mat(), keyimg2, descimg2);

			if (keyimg2.size() != 0) {
				descriptorMatcher = DescriptorMatcher::create("BruteForce-Hamming");

				const float minRatio = 0.8f;
				descriptorMatcher->knnMatch(descimg1, descimg2, m_knnMatches, 2);
				for (size_t i = 0; i<m_knnMatches.size(); i++)
				{
					const cv::DMatch& bestMatch = m_knnMatches[i][0];
					const cv::DMatch& betterMatch = m_knnMatches[i][1];
					float distanceRatio = bestMatch.distance / betterMatch.distance;
					if (distanceRatio <minRatio)
					{
						m_Matches.push_back(bestMatch);
					}
				}

				if (m_Matches.size() > 4) {
					vector<Point2f> pts1, pts2;
					for (int i = 0; i < (int)m_Matches.size(); i++) {
						int idx1 = m_Matches[i].queryIdx;
						int idx2 = m_Matches[i].trainIdx;
						pts1.push_back(keyimg1[idx1].pt);
						pts2.push_back(keyimg2[idx2].pt);
					}

					float reprojectionThreshold = 1.0;
					vector<unsigned char> inliersMask(m_Matches.size());

					Mat H = findHomography(pts1, pts2, CV_FM_RANSAC, reprojectionThreshold, inliersMask);
					vector<cv::DMatch> inliers;
					for (size_t i = 0; i < inliersMask.size(); i++) {
						if (inliersMask[i])
							inliers.push_back(m_Matches[i]);
					}
					m_Matches.swap(inliers);

					if (m_Matches.size() > 10) {
						int model_h = model_image.rows;
						int model_w = model_image.cols;

						vector<Point2f> corner_pts1;
						corner_pts1.push_back(Point(0, 0));
						corner_pts1.push_back(Point(0, model_h - 1));
						corner_pts1.push_back(Point(model_w - 1, model_h - 1));
						corner_pts1.push_back(Point(model_w - 1, 0));

						vector<Point2f> corner_pts2;
						perspectiveTransform(corner_pts1, corner_pts2, H);

						makecube(input, corner_pts2);
					}
				}

				time_t end;
				time(&end);
				double diff = difftime(end, start);
				double fps = frame / diff;
				string h_fps = format("%.2f", fps);
				cv::putText(input, "Method7 FPS:" + h_fps, Point2f(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

				Mat result;
				cv::drawMatches(model_image, keyimg1, input, keyimg2, m_Matches, result);

				cv::imshow("ORB ratio test + inlier + ransac", result);
			}
		}
		catch (Exception& e)
		{
			cout << e.msg << endl;
		}
	}
}

void method8(VideoCapture vc, Mat model_image, Mat descimg1, vector<KeyPoint> keyimg1) {
	long double frame = 0.0;
	time_t start;
	time(&start);
	while (waitKey(1) != 27) {
		Mat input, input_gray;
		vc >> input;
		frame++;
		cvtColor(input, input_gray, COLOR_BGR2GRAY);

		try {

			Ptr<Feature2D> b = ORB::create();

			Ptr<DescriptorMatcher> descriptorMatcher;
			vector<KeyPoint> keyimg2;
			vector<vector<DMatch>> m_knnMatches;
			vector<DMatch>m_Matches;
			Mat descimg2;
			b->detectAndCompute(input_gray, Mat(), keyimg2, descimg2);

			if (keyimg2.size() != 0) {
				Ptr<DescriptorMatcher> BFMatcher(new BFMatcher(NORM_HAMMING, true));
				BFMatcher->match(descimg1, descimg2, m_Matches);

				if (m_Matches.size() > 4) {
					vector<Point2f> pts1, pts2;
					for (int i = 0; i < (int)m_Matches.size(); i++) {
						int idx1 = m_Matches[i].queryIdx;
						int idx2 = m_Matches[i].trainIdx;
						pts1.push_back(keyimg1[idx1].pt);
						pts2.push_back(keyimg2[idx2].pt);
					}

					float reprojectionThreshold = 1.0;
					vector<unsigned char> inliersMask(m_Matches.size());

					Mat H = findHomography(pts1, pts2, CV_FM_RANSAC, reprojectionThreshold, inliersMask);
					vector<cv::DMatch> inliers;
					for (size_t i = 0; i < inliersMask.size(); i++) {
						if (inliersMask[i])
							inliers.push_back(m_Matches[i]);
					}
					m_Matches.swap(inliers);

					if (m_Matches.size() > 10) {
						int model_h = model_image.rows;
						int model_w = model_image.cols;

						vector<Point2f> corner_pts1;
						corner_pts1.push_back(Point(0, 0));
						corner_pts1.push_back(Point(0, model_h - 1));
						corner_pts1.push_back(Point(model_w - 1, model_h - 1));
						corner_pts1.push_back(Point(model_w - 1, 0));

						vector<Point2f> corner_pts2;
						perspectiveTransform(corner_pts1, corner_pts2, H);

						makecube(input, corner_pts2);
					}
				}

				time_t end;
				time(&end);
				double diff = difftime(end, start);
				double fps = frame / diff;
				string h_fps = format("%.2f", fps);
				cv::putText(input, "Method8 FPS:" + h_fps, Point2f(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

				Mat result;
				cv::drawMatches(model_image, keyimg1, input, keyimg2, m_Matches, result);

				cv::imshow("ORB backward-check + inlier + ransac", result);
			}
		}
		catch (Exception& e)
		{
			cout << e.msg << endl;
		}
	}
}

int main() {
	VideoCapture vc(0);
	if (!vc.isOpened()) return 0;
	std::cout << "Camera Calibration...." << endl;
	init();
	std::cout << "Camera Calibration Complete!!" << endl;

	Mat model_image = imread("model.bmp");
	cvtColor(model_image, model_image, COLOR_BGR2GRAY);
	
	Ptr<Feature2D> b = BRISK::create();
	vector<KeyPoint> keyimg1;
	Mat descimg1;
	b->detectAndCompute(model_image, Mat(), keyimg1, descimg1);

	Ptr<Feature2D> orb_b = ORB::create();
	vector<KeyPoint> orb_keyimg1;
	Mat orb_descimg1;
	orb_b->detectAndCompute(model_image, Mat(), orb_keyimg1, orb_descimg1);

	//BRISK
	// all + ransac
	thread t1(&method1, vc, model_image, descimg1, keyimg1);
	// 30-best + ransac
	thread t2(&method2, vc, model_image, descimg1, keyimg1);
	// 30-best + inlier + ransac
	thread t3(&method3, vc, model_image, descimg1, keyimg1);
	// ratio test + inlier + ransac
	thread t4(&method4, vc, model_image, descimg1, keyimg1);
	// backward-check + inlier + ransac
	thread t5(&method5, vc, model_image, descimg1, keyimg1);

	//ORB
	// 30-best + inlier + ransac
	thread t6(&method6, vc, model_image, orb_descimg1, orb_keyimg1);
	// ratio test + inlier + ransac
	thread t7(&method7, vc, model_image, orb_descimg1, orb_keyimg1);
	// backward-check + inlier + ransac
	thread t8(&method8, vc, model_image, orb_descimg1, orb_keyimg1);


	t1.join();
	t2.join();
	t3.join();
	t4.join();
	t5.join();
	t6.join();
	t7.join();
	t8.join();
	return 0;
}