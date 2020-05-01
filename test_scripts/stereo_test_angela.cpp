#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include<string>

using namespace cv;
using namespace std;

int main()
{
    string img1_file_name = "left.jpg";
    Mat img1, img2;
    img1 = imread(img1_file_name, CV_LOAD_IMAGE_COLOR);   // Read the file

    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", img1 );                   // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}