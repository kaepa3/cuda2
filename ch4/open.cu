#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
#pragma comment(lib, "opencv_world470.lib")
using namespace cv;
using namespace std;

int main() {
  printf("hogehoge");
  Mat host = imread("lena.bmp", cv::IMREAD_GRAYSCALE);
  UMat dev, dev1;
  host.copyTo(dev);
  threshold(dev, dev1, 128, 255, THRESH_BINARY);

  imshow("Test", dev1);
  waitKey();

  GaussianBlur(dev, dev1, Size(25, 25), 40);
  imshow("test", dev1);
  waitKey();

  transpose(dev, dev1);
  imshow("test", dev1);
  waitKey();

  return 0;
}
