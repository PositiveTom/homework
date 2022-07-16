#include <iostream>
#include <random>
#include "backend/problem.h"

#include<fstream>

using namespace myslam::backend;
using namespace std;

//  a1,代表指数函数，不定义a1.代表二次曲线函数
#define a1


// 曲线模型的顶点，模板参数：优化变量维度和数据类型
class CurveFittingVertex: public Vertex
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CurveFittingVertex(): Vertex(3) {}  // abc: 三个参数， Vertex 是 3 维的
    virtual std::string TypeInfo() const { return "abc"; }
};

// 误差模型 模板参数：观测值维度，类型，连接顶点类型
class CurveFittingEdge: public Edge
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge( double x, double y ): Edge(1,1, std::vector<std::string>{"abc"}) {
        x_ = x;
        y_ = y;
    }
    // 计算曲线模型误差
    virtual void ComputeResidual() override
    {
        Vec3 abc = verticies_[0]->Parameters();  // 估计的参数
        #ifdef a1
        residual_(0) = std::exp( abc(0)*x_*x_ + abc(1)*x_ + abc(2) ) - y_;  // 构建残差
        #else
        residual_(0) = abc(0) *x_*x_ + abc(1)*x_ + abc(2) - y_;
        #endif
    }

    // 计算残差对变量的雅克比
    virtual void ComputeJacobians() override
    {
        Vec3 abc = verticies_[0]->Parameters();
        double exp_y = std::exp( abc(0)*x_*x_ + abc(1)*x_ + abc(2) );

        Eigen::Matrix<double, 1, 3> jaco_abc;  // 误差为1维，状态量 3 个，所以是 1x3 的雅克比矩阵
        
        #ifdef a1
        jaco_abc << x_ * x_ * exp_y, x_ * exp_y , 1 * exp_y;
        #else
        //  任务1.2
        jaco_abc <<  x_ * x_ , x_, 1;

        #endif
        jacobians_[0] = jaco_abc;
    }
    /// 返回边的类型信息
    virtual std::string TypeInfo() const override { return "CurveFittingEdge"; }
public:
    double x_,y_;  // x 值， y 值为 _measurement
};

int main()
{
    double a=1, b=2.0, c=1.0;         // 真实参数值
    int N = 100;                          // 数据点
    double w_sigma= 1.;                 // 噪声Sigma值

    std::default_random_engine generator;
    std::normal_distribution<double> noise(0.,w_sigma);

    // 构建 problem
    Problem problem(Problem::ProblemType::GENERIC_PROBLEM);
    shared_ptr< CurveFittingVertex > vertex(new CurveFittingVertex());

    // 设定待估计参数 a, b, c初始值
    vertex->SetParameters(Eigen::Vector3d (0.,0.,0.));
    // 将待估计的参数加入最小二乘问题
    problem.AddVertex(vertex);

    // 构造 N 次观测

    #ifndef a1
    fstream f("/home/zjj/MyCode/VIO-Homework/CurveFitting_LM/data/data3.txt", ios::out);
    if(!f.is_open())
    {
        cout << "error" << endl;
        return -1;
    }
    #endif

    for (int i = 0; i < N; ++i) {

        double x = i/100.;
        double n = noise(generator);
        // 观测 y
        #ifdef a1
        double y = std::exp( a*x*x + b*x + c ) + n;
        #else
        double y = a*x*x + b*x + c + n;
        #endif
//        double y = std::exp( a*x*x + b*x + c );
#ifndef a1
        f << y-n << " "<< y << " "<< x << endl;
#endif
        // 每个观测对应的残差函数
        shared_ptr< CurveFittingEdge > edge(new CurveFittingEdge(x,y));
        std::vector<std::shared_ptr<Vertex>> edge_vertex;
        edge_vertex.push_back(vertex);
        edge->SetVertex(edge_vertex);

        // 把这个残差添加到最小二乘问题
        problem.AddEdge(edge);
    }
#ifndef a1
    f.close();
#endif
    std::cout<<"\nTest CurveFitting start..."<<std::endl;
    /// 使用 LM 求解
    problem.Solve(30);

    std::cout << "-------After optimization, we got these parameters :" << std::endl;
    std::cout << vertex->Parameters().transpose() << std::endl;
    std::cout << "-------ground truth: " << std::endl;
    std::cout << a << " " << b << " " << c << std::endl;

#ifndef a1
    fstream result("/home/zjj/MyCode/VIO-Homework/CurveFitting_LM/data/result.txt",ios::out);
    VecX k = vertex->Parameters().transpose();
    result << k[0] << " " << k[1] << " " << k[2];
    result.close();
#endif
    // std
    return 0;
}


