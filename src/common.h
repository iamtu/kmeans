#ifndef SEMTP_COMMON_H_
#define SEMTP_COMMON_H_

#include <vector>
#include <unordered_map>
#include <mutex>

#include <time.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <random>
namespace kmeans {

    struct Document {
        std::string doc_id;
        double *vec;
        int k;

        Document(int num, std::string _doc_id) {
            vec = new double[num];
            doc_id = _doc_id;
        }

        ~Document() {
            delete[] vec;
        }
    };

    struct Center {
        double *vec;

        Center(int num) {
            vec = new double[num];
        }

        ~Center() {
            delete vec;
        }
    };

    struct Model {
        std::vector<Center *> centers;
        int num_cluster;
        int dim;

        void init() {
            for (int i = 0; i < num_cluster; i++) {
                Center *c = new Center(dim);
                centers.push_back(c);
            }
        }

        Model(int _num_c, int _dim) {
            num_cluster = _num_c;
            dim = _dim;
        }

        ~Model() {
            for (Center *c: centers) {
                delete c;
            }
        }
    };

    struct DataSet {
        std::vector<Document *> doc;

        //计算y到含i个中心点集合C的最小距离
        double computeD(std::vector<double *> C, double *y, int i, int dim) {
            double close = -1;
            for (int j = 0; j < i; j++) {
                if (C[j] != nullptr) {
                    double cos_v = 0;
                    for (int m = 0; m < dim; m++) {
                        cos_v += C[j][m] * y[m];
                    }
                    if (close < cos_v) close = cos_v;
                }
            }
            if (close == -1) {
                printf("computeD wrong\n");
            }
            return close;
        }


        void init_k(int num_cluster, int dim) {
//  原代码为随机初始化，现在改为MCMC抽样，为了不改变其他函数吗，并迭代一轮，给所有点打上标签k
//      srand(time(NULL));
//      for (Document* d : doc) {
//        int rr = rand()%num_cluster;
//        d->k = rr;
//          }
            std::vector<double *> C(num_cluster, nullptr);
            int doc_size = doc.size();
            //随机抽样出第一个中心点
            srand((unsigned) time(NULL));
            int index = rand() % doc_size;
            C[0] = doc[index]->vec;

            for (int i = 1; i < num_cluster; i++) {
                index = rand() % doc_size;
                double *x = doc[index]->vec;
                double dx = computeD(C, x, i,dim);
                //取 m = 30
                for (int j = 0; j < 30; j++) {
                    index = rand() % doc_size;
                    double *y = doc[index]->vec;
                    double dy = computeD(C, y, i, dim);
                    // 产生0~1的随机浮点数
                    index = rand() % 1000;
                    double p = index / 999;
                    // dx dy 在使用cos衡量情况下，其值越大，距离越小，故这里的分子分母和论文相反
                    if ((dx / dy) > p) {
                        x = y;
                        dx = dy;
                    }
                }
                C[i] = x;
            }

            for (auto d: doc) {
                double close_v = -2;
                int close_id = -1;
                for (int i = 0; i < num_cluster; i++) {
                    double cos_v = 0;
                    for (int j = 0; j < dim; j++) {
                        cos_v += d->vec[j] * C[i][j];
                    }
                    if (cos_v > close_v) {
                        close_v = cos_v;
                        close_id = i;
                    }
                }
                if (close_id == -1) {
                    printf("close_id wrong\n");
                }
                d->k = close_id;
            }

        }
    };

    // Random hack
#define CLOCK (std::chrono::system_clock::now().time_since_epoch().count())

    static std::mt19937 _rng(CLOCK);
    static std::uniform_real_distribution<double> _unif01;
#define STATCLOCK (0x3751)

    static std::mt19937 _statrng(STATCLOCK);

}


#endif // SEMTP_COMMON_H_
