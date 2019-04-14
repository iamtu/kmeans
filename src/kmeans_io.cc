#include "kmeans_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <iostream>
#include <vector>

namespace kmeans {

  using std::vector;

  vector<string> string_split(const string& s, const string &c) {
    vector<string> v;
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while(string::npos != pos2){
      v.push_back(s.substr(pos1, pos2-pos1));
      pos1 = pos2 + c.size();
      pos2 = s.find(c, pos1);
    }
    if(pos1 != s.length())
      v.push_back(s.substr(pos1));
    return v;
  }

  string string_join( vector<string>& elements, string delimiter ) {
    std::stringstream ss;
    size_t elems = elements.size(),
    last = elems - 1;

    for( size_t i = 0; i < elems; ++i )
    {
        ss << elements[i];

        if( i != last )
          ss << delimiter;
    }

    return ss.str();
  }

  DataSet get_dataset(string location, int dim) {
    FILE* file = fopen(location.c_str(),"r");

    char* line = NULL;
    size_t len = 0;
    ssize_t read;

    DataSet dataset;

    while((read=getline(&line,&len,file)) != -1) {
      vector<string> s = string_split(string(line), "\t");
      vector<string> v = string_split(s[1]," ");
      string doc_id = s[0];
      int n = v.size();
      if(n!=dim){
        printf("read data len is not equal input dim!\n");
      }
      Document* doc = new Document(n, doc_id);

      double len_v = 0;
      for (int i = 0; i < n; ++i) {
        double wv = std::stod(v[i]);
        doc->vec[i] = wv;
        len_v += wv*wv;
      }
      len_v = std::sqrt(len_v);
      for(int i=0; i <n; i++){
        doc->vec[i] /= len_v;
      }
      dataset.doc.push_back(doc);
    }
    fclose(file);
    return dataset;
  }
  
  void save(DataSet& dataset, string location) {
    FILE* file = fopen(location.c_str(),"w");

    for(auto doc : dataset.doc){
      string doc_id = doc->doc_id;
      int k = doc->k;
      string line = doc_id+"\t"+std::to_string(k)+"\n";
      fputs(line.c_str(),file);
    }

    fclose(file);
  }

}
