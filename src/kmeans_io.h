#ifndef KMEANS_IO_H_
#define KMEANS_IO_H_

#include "common.h"

#include <string>
using std::string;

namespace kmeans {

  DataSet get_dataset(string location, int dim);
  void save(DataSet& dataset, string location);

}

#endif // KMEANS_IO_H_
