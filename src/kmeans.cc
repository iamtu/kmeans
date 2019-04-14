#include "parallel.h"
#include "kmeans.h"

namespace kmeans {
	struct KDocPair {
    	int k;
    	std::vector<Document *> docs;
    	KDocPair(int _k, std::vector<Document *> _docs) {
    		k = _k;
    		docs = _docs;
    	}
    };
    void run_iteration(Model *model, DataSet &dataset, int threads) {
        printf("===REMOVEME - Begin run_iteration\n");
    	Parallel::Parallel pool(threads);

    	std::unordered_map<int, std::vector<Document *>> c_index;
    	for (Document *d : dataset.doc) {
    		int k = d->k;
    		c_index[k].push_back(d);
    	}

	    std::vector<KDocPair> vs;
    	for (auto it : c_index) {
    		int k = it.first;
    		std::vector<Document *> &docs = it.second;
    		KDocPair idx_pair = KDocPair(k, docs);
    		vs.push_back(idx_pair);
    	}

    	int dim = model->dim;
    	pool.foreach (vs.begin(), vs.end(), [&](KDocPair &p) {
    		int k = p.k;
    		auto &docs = p.docs;
    		Center *c = model->centers[k];
    		compute_centers(k, dim, c, docs);
    	});

    	pool.foreach (dataset.doc.begin(), dataset.doc.end(), [&](Document *x) {
    		compute_cluster(x, model);
    	});

	    printf("REMOVEME - end run_iteration\n");
    }

    void compute_cluster(Document *d, Model *model) {
    	int num_cluster = model->num_cluster;
    	int dim = model->dim;

    	double close_v = -2;
    	int close_id = -1;
    	for (int i = 0; i < num_cluster; i++) {
    		double cos_v = 0;
    		Center *c = model->centers[i];
    		for (int j = 0; j < dim; j++)
    			cos_v += d->vec[j] * c->vec[j];
    		if (cos_v > close_v) {
    			close_v = cos_v;
    			close_id = i;
    		}
    	}
    	if (close_id == -1)
    		printf("close_id wrong\n");
    	d->k = close_id;
    }

    void compute_centers(int k, int dim, Center *c, std::vector<Document *> &docs) {
    	int docn = docs.size();
    	double *c_vec = c->vec;
    	for (int i = 0; i < dim; i++)
    		c_vec[i] = 0;
    	for (Document *doc : docs) {
    		double *doc_vec = doc->vec;
    		for (int i = 0; i < dim; i++) {
    			c_vec[i] += doc_vec[i];
    		}
    	}

    	double len_c = 0;
    	for (int i = 0; i < dim; i++) {
    		c_vec[i] /= docn;
    		len_c += c_vec[i] * c_vec[i];
    	}
    	len_c = std::sqrt(len_c);
    	for (int i = 0; i < dim; i++) {
    		c_vec[i] /= len_c;
    	}
    }
} // namespace kmeans
