#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

// include openCV and standard headers
// Computer Vision 2023 (P. Zanuttigh, code M. Carraro) - LAB 2

// copy and paste inside your code

// hists = vector of 3 cv::mat of size nbins=256 with the 3 histograms
// e.g.: hists[0] = cv:mat of size 256 with the red histogram
//       hists[1] = cv:mat of size 256 with the green histogram
//       hists[2] = cv:mat of size 256 with the blue histogram
void showHistogram(std::vector<cv::Mat>& hists)
{
  // Min/Max computation
  double h_max[3] = {0,0,0};
  double min;
  cv::minMaxLoc(hists[0], &min, &h_max[0]);
  cv::minMaxLoc(hists[1], &min, &h_max[1]);
  cv::minMaxLoc(hists[2], &min, &h_max[2]);

  std::string wname[3] = { "Blue", "Green", "Red" };
  cv::Scalar colors[3] = { cv::Scalar(255,0,0), cv::Scalar(0,255,0),
                           cv::Scalar(0,0,255) };

  std::vector<cv::Mat> canvas(hists.size());

  // Display each histogram in a canvas
  for (int i = 0, end = hists.size(); i < end; i++)
  {
    canvas[i] = cv::Mat::ones(125, hists[0].rows, CV_8UC3);

    for (int j = 0, rows = canvas[i].rows; j < hists[0].rows-1; j++)
    {
      cv::line(
            canvas[i],
            cv::Point(j, rows),
            cv::Point(j, rows - (hists[i].at<float>(j) * rows/h_max[i])),
            hists.size() == 1 ? cv::Scalar(200,200,200) : colors[i],
            1, 8, 0
            );
    }

    cv::imshow(hists.size() == 1 ? "Value" : wname[i], canvas[i]);
  }
}

int main(int argc, char** argv)
{
    //------------------------------- PART 1) EQUALIZATÝON AND HISTOGRAMS
    // reading original image
    cv::Mat image = cv::imread("C:/Users/10/OneDrive/Masaüstü/ComputerVision/lab_2_code_and_data/data/countryside.jpg");

    if (image.empty()) {
        std::cerr << "Error: Could not open image file." << std::endl;
        return -1;
    }

    imshow("Original Image", image);
    cv::waitKey();

    // Split the image into its 3 channels
    std::vector<cv::Mat> bgr;
    cv::split(image, bgr);

    // Compute the histograms
    int nbins = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;

    //-------------- 1)ORIGINAL IMAGE AND HISTOGRAM-----------------
    std::vector<cv::Mat> hist(3);
    for (int i = 0; i < 3; i++) {
        cv::calcHist(&bgr[i], 1, 0, cv::Mat(), hist[i], 1, &nbins, &histRange, uniform, accumulate);
    }

    // Display the histograms
    showHistogram(hist);
    cv::waitKey();
    //--------------2)RGB EQUALIZATION-----------------

    //Equalize channels
    
    cv::equalizeHist(bgr[0], bgr[0]);
    cv::equalizeHist(bgr[1], bgr[1]);
    cv::equalizeHist(bgr[2], bgr[2]);
    //Merge channels back into image
    cv::Mat equalized_image;
    cv::merge(bgr, equalized_image);
    // Show RGB equalized image
    imshow("RGB Equalized Image", equalized_image);
    cv::waitKey();
    cv::split(equalized_image, bgr);

    // Equalized histograms
    std::vector<cv::Mat> hist_eq(3);
    for (int i = 0; i < 3; i++) {
        cv::calcHist(&bgr[i], 1, 0, cv::Mat(), hist_eq[i], 1, &nbins, &histRange, uniform, accumulate);
    }

    showHistogram(hist_eq);
    cv::waitKey();

    //--------------3)Luminance EQUALIZATION-----------------
    // Convert to Lab color space 
    cv::Mat lab_image;
    cv::cvtColor(image, lab_image, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_planes;

    // Extract the L channel
    cv::split(lab_image, lab_planes);
    cv::equalizeHist(lab_planes[0], lab_planes[0]);

    // Merge the the color planes back into an Lab image
    cv::merge(lab_planes, lab_image);

    // Convert back to RGB color space and show equalized image and histograms
    cv::Mat equalized_lab_image;
    cv::cvtColor(lab_image, equalized_lab_image, cv::COLOR_Lab2BGR);
    cv::imshow("Luminance Equalized Image", equalized_lab_image);
    cv::waitKey();
    
    std::vector<cv::Mat> hist_eq_2(3);
    std::vector<cv::Mat> bgr_L;
    cv::split(equalized_lab_image, bgr_L);
    for (int i = 0; i < 3; i++) {
        cv::calcHist(&bgr_L[i], 1, 0, cv::Mat(), hist_eq_2[i], 1, &nbins, &histRange, uniform, accumulate);
    }

    showHistogram(hist_eq_2);
    cv::waitKey();

    //-------------------------------------- PART 2 ---- FILTERING--------------

    //# Read the equalized image
    // Denoise the equalized image using Median filter
    cv::Mat denoised_median;
    cv::medianBlur(equalized_lab_image, denoised_median, 11); // kernel size was changed 5x5 or 11x11

    // Denoise the equalized image using Gaussian filter
    cv::Mat denoised_gaussian;
    cv::GaussianBlur(equalized_lab_image, denoised_gaussian, cv::Size(7, 7),0); // kernel size = 7x7 with sigmaX =sigmaY = 0
                                                                                // kernel size = 11x11 with sixmaX=sigmaY=10
    // Denoise the equalized image using Bilateral filter
    cv::Mat denoised_bilateral;
    cv::bilateralFilter(equalized_lab_image, denoised_bilateral, 15, 75, 75); // different values were used
    /*
    
    -If range parameter d increases, the bilateral filter becomes closer to Gaussian blur
    -Increasing the spatial parameter of a filter smooths larger features

    */

    cv::imshow("Denoised with Median filter", denoised_median);
    cv::waitKey(0);
    cv::imshow("Denoised with Gaussian filter", denoised_gaussian);
    cv::waitKey(0);
    cv::imshow("Denoised with Bilateral filter", denoised_bilateral);
    cv::waitKey(0);


    return 0;
} 

