
#include <fstream>
#include <iostream>

#include "common.h"
#include "kmeans_io.h"
#include "kmeans.h"

#include "toml.h"

using std::string;
using std::vector;

using kmeans::DataSet;
using kmeans::Model;

int main(int argc, char* argv[]) {

  std::cout << "x" << std::endl;

  std::ifstream ifs("kmeans.toml");
  toml::ParseResult pr = toml::parse(ifs);

  if (!pr.valid()) {
    std::cout << pr.errorReason << std::endl;
    return 1;
  }

  string location_data = pr.value.get<string>("kmeans.location_data");
  string location_cluster = pr.value.get<string>("kmeans.location_ct");

  int num_cluster = pr.value.get<int>("kmeans.num_cluster");
  int threads = pr.value.get<int>("kmeans.threads");
  int iter_num = pr.value.get<int>("kmeans.iter_num");
  int dim = pr.value.get<int>("kmeans.dim");

  DataSet dataset = kmeans::get_dataset(location_data, dim);
  printf("DataSet %lu\n",dataset.doc.size());

  dataset.init_k(num_cluster, dim);

  Model* model = new Model(num_cluster, dim);
  model->init();
  printf("Model init\n");

  printf("run ...\n");
  for (int i = 0; i < iter_num; ++i) {
    printf("iter %d\n",i);
    int st = time(NULL);
    kmeans::run_iteration(model, dataset, threads);
    int et = time(NULL);
    printf("iter cost %d\n", et-st);

    if(i!=0 && i % 10 == 0){
      kmeans::save(dataset,location_cluster);
    }
  }

  kmeans::save(dataset,location_cluster);

  delete model;
  for (auto d : dataset.doc) {
    delete d;
  }
  return 0;

}
