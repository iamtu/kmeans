#ifndef KMEANS_H_
#define KMEANS_H_

#include "common.h"

#include <string>
#include <unordered_map>
using kmeans::Document;
using kmeans::DataSet;

namespace kmeans {

  void run_iteration(Model* model, DataSet& dataset, int threads);

  void compute_centers(int k, int dim, Center* c, std::vector<Document*>& docs);
  void compute_cluster(Document* d, Model* model);
}

#endif // KMEANS_H_
