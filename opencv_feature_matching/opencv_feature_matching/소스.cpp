#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include <vector>  
#include <iostream>  

using namespace cv;
using namespace std;
Mat src, frameImg;
int width;
int height;
vector<Point> srcCorner(4);
vector<Point> dstCorner(4);

static bool createDetectorDescriptorMatcher(const string& detectorType, const string& descriptorType, const string& matcherType,
	Ptr<FeatureDetector>& featureDetector,
	Ptr<DescriptorExtractor>& descriptorExtractor,
	Ptr<DescriptorMatcher>& descriptorMatcher)
{
	cout << "<Creating feature detector, descriptor extractor and descriptor matcher ..." << endl;
	if (detectorType == "SIFT" || detectorType == "SURF")
		initModule_nonfree();
	featureDetector = FeatureDetector::create(detectorType);
	descriptorExtractor = DescriptorExtractor::create(descriptorType);
	descriptorMatcher = DescriptorMatcher::create(matcherType);
	cout << ">" << endl;
	bool isCreated = !(featureDetector.empty() || descriptorExtractor.empty() || descriptorMatcher.empty());
	if (!isCreated)
		cout << "Can not create feature detector or descriptor extractor or descriptor matcher of given types." << endl << ">" << endl;
	return isCreated;
}


bool refineMatchesWithHomography(const std::vector<cv::KeyPoint>& queryKeypoints,
	const std::vector<cv::KeyPoint>& trainKeypoints,
	float reprojectionThreshold,
	std::vector<cv::DMatch>& matches,
	cv::Mat& homography)
{
	const int minNumberMatchesAllowed = 4;
	if (matches.size() <minNumberMatchesAllowed)
		return false;
	// Prepare data for cv::findHomography    
	std::vector<cv::Point2f> queryPoints(matches.size());
	std::vector<cv::Point2f> trainPoints(matches.size());
	for (size_t i = 0; i <matches.size(); i++)
	{
		queryPoints[i] = queryKeypoints[matches[i].queryIdx].pt;
		trainPoints[i] = trainKeypoints[matches[i].trainIdx].pt;
	}
	// Find homography matrix and get inliers mask    
	std::vector<unsigned char> inliersMask(matches.size());
	homography = cv::findHomography(queryPoints,
		trainPoints,
		CV_FM_RANSAC,
		reprojectionThreshold,
		inliersMask);
	std::vector<cv::DMatch> inliers;
	for (size_t i = 0; i<inliersMask.size(); i++)
	{
		if (inliersMask[i])
			inliers.push_back(matches[i]);
	}
	matches.swap(inliers);
	Mat homoShow;
	drawMatches(src, queryKeypoints, frameImg, trainKeypoints, matches, homoShow, Scalar::all(-1), CV_RGB(255, 255, 255), Mat(), 2);
	imshow("homoShow", homoShow);
	return matches.size() > minNumberMatchesAllowed;

}


bool matchingDescriptor(const vector<KeyPoint>& queryKeyPoints, const vector<KeyPoint>& trainKeyPoints,
	const Mat& queryDescriptors, const Mat& trainDescriptors,
	Ptr<DescriptorMatcher>& descriptorMatcher,
	bool enableRatioTest = true)
{
	vector<vector<DMatch>> m_knnMatches;
	vector<DMatch>m_Matches;

	if (enableRatioTest)
	{
		cout << "KNN Matching" << endl;
		const float minRatio = 1.f / 1.5f;
		descriptorMatcher->knnMatch(queryDescriptors, trainDescriptors, m_knnMatches, 2);
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

	}
	else
	{
		cout << "Cross-Check" << endl;
		Ptr<cv::DescriptorMatcher> BFMatcher(new cv::BFMatcher(cv::NORM_HAMMING, true));
		BFMatcher->match(queryDescriptors, trainDescriptors, m_Matches);
	}
	Mat homo;
	float homographyReprojectionThreshold = 1.0;
	bool homographyFound = refineMatchesWithHomography(
		queryKeyPoints, trainKeyPoints, homographyReprojectionThreshold, m_Matches, homo);

	if (!homographyFound)
		return false;
	else
	{
		if (m_Matches.size()>10)
		{
			std::vector<Point2f> obj_corners(4);
			obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(src.cols, 0);
			obj_corners[2] = cvPoint(src.cols, src.rows); obj_corners[3] = cvPoint(0, src.rows);
			std::vector<Point2f> scene_corners(4);
			perspectiveTransform(obj_corners, scene_corners, homo);
			line(frameImg, scene_corners[0], scene_corners[1], CV_RGB(255, 0, 0), 2);
			line(frameImg, scene_corners[1], scene_corners[2], CV_RGB(255, 0, 0), 2);
			line(frameImg, scene_corners[2], scene_corners[3], CV_RGB(255, 0, 0), 2);
			line(frameImg, scene_corners[3], scene_corners[0], CV_RGB(255, 0, 0), 2);
			return true;
		}
		return true;
	}


}
int main()
{
	string filename = "box.png";
	src = imread(filename, 0);
	width = src.cols;
	height = src.rows;
	string detectorType = "SIFT";
	string descriptorType = "SIFT";
	string matcherType = "FlannBased";

	Ptr<FeatureDetector> featureDetector;
	Ptr<DescriptorExtractor> descriptorExtractor;
	Ptr<DescriptorMatcher> descriptorMatcher;
	if (!createDetectorDescriptorMatcher(detectorType, descriptorType, matcherType, featureDetector, descriptorExtractor, descriptorMatcher))
	{
		cout << "Creat Detector Descriptor Matcher False!" << endl;
		return -1;
	}
	//Intial: read the pattern img keyPoint  
	vector<KeyPoint> queryKeypoints;
	Mat queryDescriptor;
	featureDetector->detect(src, queryKeypoints);
	descriptorExtractor->compute(src, queryKeypoints, queryDescriptor);

	VideoCapture cap(0); // open the default camera  
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	if (!cap.isOpened())  // check if we succeeded  
	{
		cout << "Can't Open Camera!" << endl;
		return -1;
	}
	srcCorner[0] = Point(0, 0);
	srcCorner[1] = Point(width, 0);
	srcCorner[2] = Point(width, height);
	srcCorner[3] = Point(0, height);

	vector<KeyPoint> trainKeypoints;
	Mat trainDescriptor;

	Mat frame, grayFrame;
	char key = 0;

	//	frame = imread("box_in_scene.png");  
	while (key != 27)
	{
		cap >> frame;
		if (!frame.empty())
		{
			frame.copyTo(frameImg);
			printf("%d,%d\n", frame.depth(), frame.channels());
			grayFrame.zeros(frame.rows, frame.cols, CV_8UC1);
			cvtColor(frame, grayFrame, CV_BGR2GRAY);
			trainKeypoints.clear();
			trainDescriptor.setTo(0);
			featureDetector->detect(grayFrame, trainKeypoints);

			if (trainKeypoints.size() != 0)
			{
				descriptorExtractor->compute(grayFrame, trainKeypoints, trainDescriptor);

				bool isFound = matchingDescriptor(queryKeypoints, trainKeypoints, queryDescriptor, trainDescriptor, descriptorMatcher);
				imshow("foundImg", frameImg);

			}
		}
		key = waitKey(1);
	}
	cap.release();
	return 0;
}