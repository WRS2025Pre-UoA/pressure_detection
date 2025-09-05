#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <limits>

// -----------------------------------------
// 最小・最大目盛りの角度を検出する（外周マスク＋ヒストグラム法）
// -----------------------------------------
static std::pair<double,double> detect_min_max_angles(const cv::Mat& gray, cv::Point& center, int& radius) {
    cv::Mat gb;
    cv::GaussianBlur(gray, gb, cv::Size(5,5), 0);

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gb, circles, cv::HOUGH_GRADIENT, 1.2, 100, 100, 30, 0, 0);
    if (circles.empty()) {
        center = cv::Point(gray.cols/2, gray.rows/2);
        radius = std::min(gray.cols, gray.rows)/2;
    } else {
        center = cv::Point(cvRound(circles[0][0]), cvRound(circles[0][1]));
        radius = cvRound(circles[0][2]);
    }

    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);
    cv::Mat mask = cv::Mat::zeros(edges.size(), CV_8UC1);
    cv::circle(mask, center, int(radius*0.91), 255, -1);
    cv::circle(mask, center, int(radius*0.75), 0, -1);
    cv::Mat ticks;
    cv::bitwise_and(edges, mask, ticks);

    std::vector<double> angles;
    for (int y=0; y<ticks.rows; y++) {
        for (int x=0; x<ticks.cols; x++) {
            if (ticks.at<uchar>(y,x) > 0) {
                double dx = x - center.x;
                double dy = center.y - y;
                double ang = atan2(dy, dx) * 180.0 / CV_PI;
                if (ang < 0) ang += 360.0;
                if (ang >= 180) angles.push_back(ang);
            }
        }
    }

    const int bins = 180;
    std::vector<int> hist(bins,0);
    for (double a: angles) {
        int idx = int(a) - 180;
        if (0 <= idx && idx < bins) hist[idx]++;
    }

    // 平滑化
    std::vector<double> smooth(hist.size(),0.0);
    for (int i=0; i<bins; i++) {
        for (int k=-3; k<=3; k++) {
            int j=(i+k+bins)%bins;
            smooth[i]+=hist[j];
        }
        smooth[i]/=7.0;
    }

    // 谷を探索
    double thr=1.0;
    std::vector<bool> below(bins,false);
    for (int i=0; i<bins; i++) below[i]=smooth[i]<=thr;

    int best_len=0, best_s=-1, best_e=-1;
    for (int i=0;i<2*bins;i++){
        if(below[i%bins]){
            int j=i;
            while(j<2*bins && below[j%bins]) j++;
            int len=j-i;
            if(len>best_len){best_len=len; best_s=i; best_e=j-1;}
            i=j;
        }
    }

    double min_angle=0,max_angle=0;
    if(best_s>=0){
        int a1=(best_s%bins)+180;
        int a2=(best_e%bins)+180;
        min_angle=std::min(a1,a2);
        max_angle=std::max(a1,a2);
    }

    return {min_angle,max_angle};
}

// -----------------------------------------
// 針を検出し圧力値を返す
// -----------------------------------------
static double pressure_from_angles(double min_angle, double max_angle,
                                   const cv::Point2d& center, const cv::Vec4i& line,
                                   double min_val=0.0, double max_val=1.6) {
    cv::Point2d p1(line[0], line[1]);
    cv::Point2d p2(line[2], line[3]);
    double d1 = cv::norm(p1 - center);
    double d2 = cv::norm(p2 - center);
    cv::Point2d tip = (d1 > d2) ? p1 : p2;

    cv::Point2d v = tip - center;
    double n = hypot(v.x,v.y);
    if(n<1e-9) return std::numeric_limits<double>::quiet_NaN();
    v.x/=n; v.y/=n;

    const cv::Point2d ref(0.0,-1.0);
    double dot=v.x*ref.x+v.y*ref.y;
    dot=std::max(-1.0,std::min(1.0,dot));
    double ang=acos(dot)*180.0/CV_PI;
    if(v.x<0) ang=360.0-ang;

    double cw_range=fmod(min_angle-max_angle+360.0,360.0);
    double cw_from_min=fmod(min_angle-ang+360.0,360.0);
    double frac=std::clamp(cw_from_min/cw_range,0.0,1.0);
    return min_val+frac*(max_val-min_val);
}

int main(int argc,char*argv[]){
    std::string path="../cropped/pic_0.png"; // 入力画像
    cv::Mat img=cv::imread(path);
    if(img.empty()){std::cerr<<"image load failed"<<std::endl; return -1;}
    cv::Mat gray; cv::cvtColor(img,gray,cv::COLOR_BGR2GRAY);

    cv::Point center; int radius;
    auto[min_angle,max_angle]=detect_min_max_angles(gray,center,radius);
    std::cout<<"min_angle="<<min_angle<<", max_angle="<<max_angle<<std::endl;

    // 針の直線をHoughで取得
    cv::Mat edges; cv::Canny(gray,edges,50,150);
    cv::Mat mask=cv::Mat::zeros(edges.size(),CV_8UC1);
    cv::circle(mask,center,int(radius*0.90),255,-1);
    cv::circle(mask,center,int(radius*0.35),0,-1);
    cv::Mat roi; cv::bitwise_and(edges,mask,roi);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(roi,lines,1,CV_PI/180,30,int(radius*0.25),15);
    if(lines.empty()){std::cerr<<"needle not found"<<std::endl; return -1;}

    cv::Vec4i best=lines[0];
    double maxd=0;
    for(auto&L:lines){
        cv::Point p1(L[0],L[1]),p2(L[2],L[3]);
        double d1=cv::norm(p1-center),d2=cv::norm(p2-center);
        double d=std::max(d1,d2);
        if(d>maxd){maxd=d; best=L;}
    }

    double pressure=pressure_from_angles(min_angle,max_angle,center,best);
    std::cout<<"Pressure="<<pressure<<" MPa"<<std::endl;

    // 可視化
    cv::circle(img,center,radius,cv::Scalar(0,0,255),2);
    cv::line(img,cv::Point(best[0],best[1]),cv::Point(best[2],best[3]),cv::Scalar(0,255,0),2);
    cv::imshow("Result",img);
    cv::waitKey(0);
    return 0;
}
