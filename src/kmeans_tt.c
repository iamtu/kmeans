#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <pthread.h>
#include <time.h>

#define MAX_SIZE 1024
const long long vocab_hash_size = 2000000; // Maximum 50 * 0.7 = 35M words in the vocabulary
int *vocab_hash;
const long long max_w = 50;              // max length of vocabulary entries

FILE *f;
FILE *fo_close;
char train_file[MAX_SIZE], output_file[MAX_SIZE], class_file[MAX_SIZE], close_file[MAX_SIZE];
long long words = 0, size = 0;
float *M;
char *vocab;
int classes;
int layer1_size;

int clcn, iter = 20;
int *centcn; // = (int *)malloc(classes * sizeof(int));
int *cl; // = (int *)calloc(vocab_size, sizeof(int));
double *cent; // = (real *)calloc(classes * layer1_size, sizeof(real));
int num_threads = 12;
long long vocab_size;
pthread_mutex_t mutex;

// Returns hash value of a word
int GetWordHash(char *word) {
	unsigned long long a, hash = 0;
	for (a = 0; a < strlen(word); a++)
		hash = hash * 257 + word[a];
	hash = hash % vocab_hash_size;
	return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
	unsigned int hash = GetWordHash(word);
    //printf("%s %d\n", word, hash);
	while (1) {
		if (vocab_hash[hash] == -1)
			return -1;
		if (!strcmp(word, &vocab[vocab_hash[hash]*max_w]))
			return vocab_hash[hash];
		hash = (hash + 1) % vocab_hash_size;
	}
	return -1;
}

void read_vec(){
  unsigned int hash;
  float len;
  long long a,b;
  f = fopen(train_file, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  if(words==0){
    fscanf(f, "%lld", &words);
    fscanf(f, "%lld", &size);
  }
  layer1_size = (int) size;
  printf("%lld %lld \n", words, size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  vocab_hash = (int *) calloc(vocab_hash_size, sizeof(int));
  for (a = 0; a < vocab_hash_size; a++)
    vocab_hash[a] = -1;
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
  for (b = 0; b < words; b++) {
    a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
    vocab[b * max_w + a] = 0;

	//hash = GetWordHash(&vocab[b*max_w]);
	//while (vocab_hash[hash] != -1){
    //    printf("hash confict word %s hash %d\n",&vocab[b*max_w], hash);
	//	hash = (hash + 1) % vocab_hash_size;
    //}
	//vocab_hash[hash] = b;
    //printf("%s %d\n",&vocab[b*max_w], hash);

    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
  }
  fclose(f);
}

void init_cluster(){
    long long a;
    printf("init cluster!\n");
    f = fopen(class_file, "rb");
    int w_idx, k;
    char word[max_w];
    while(!feof(f)){
        a = 0;
        while (1) {
          word[a] = fgetc(f);
          if (feof(f) || (word[a] == ' ')) break;
          if ((a < max_w) && (word[a] != '\n')) a++;
        }
        word[a] = 0;
        fscanf(f,"%d\n",&k);
        w_idx = SearchVocab(&word);
        if(w_idx != -1){
            cl[w_idx] = k;
            //printf("%s %d %d\n",word, w_idx, k);
        }
    }
    fclose(f);
}

void *KmeansThread(void *id){
    int closeid;
    double xx, closev;
    long long bb, cc, dd;
    int begin = vocab_size/(long long)num_threads*(long long)id;
    int msize = vocab_size/(long long)num_threads;
    int end = begin + msize;
    if((int)id ==(num_threads-1)){
        end = words;
    }
    //printf("id %d begin %d end %d\n",id, begin, end);
    for (cc = begin; cc < end; cc++) {
        closev = -10;
        closeid = -1;
        //printf("id %d c %d\n", id, cc);
        for (dd = 0; dd < clcn; dd++) {
          xx = 0;
          for (bb = 0; bb < layer1_size; bb++){
              xx += cent[layer1_size * dd + bb] * M[cc * layer1_size + bb];
              //printf("id %d c %d cent %d dim %d xx %f \n", id, cc, dd,bb, xx);
          }
          //printf("dd is %d and xx is %f\n",dd,xx);
          if (xx > closev) {
            closev = xx;
            closeid = dd;
          }
        }
        //if(closeid==-1){
        //    printf("xx %f\n", xx);
        //}
        cl[cc] = closeid;
    }
}

void *FindBestdThread(void *id){
    int begin = vocab_size/(long long)num_threads*(long long)id;
    int msize = vocab_size/(long long)num_threads;
    int end = begin + msize;
    if((int)id ==(num_threads-1)){
        end = words;
    }
    //printf("thread id is %d, position is %d, end is %d\n", id, begin, end);
    int N = 500;
    double dist, bestd[N],len;
    int bestw[N];
    long long ii, jj, b, c, d;
    for (c = begin; c < end; c++) {
        for (ii = 0; ii < N; ii++) {
            bestd[ii] = -1;
            bestw[ii] = -1;
        }
        for (d = 0; d < clcn; d++) {
          dist = 0;
          len = 0;
          for (b = 0; b < layer1_size; b++){
            dist += cent[layer1_size * d + b] * M[c * layer1_size + b];
            len += M[b + c * layer1_size] * M[b + c * layer1_size];
          }
          len = sqrt(len);
          dist /= len;
          for (ii = 0; ii < N; ii++){
              if (dist > bestd[ii]){
                for(jj = N - 1; jj > ii; jj--){
                    bestd[jj] = bestd[jj-1];
                    bestw[jj] = bestw[jj-1];
                }
                bestd[ii] = dist;
                bestw[ii] = d;
                break;
              }
          }
        }
        pthread_mutex_lock(&mutex);
        //fprintf(fo_close, "id %d c %d\n", id, c);
        fprintf(fo_close, "%s", &vocab[c*max_w]);
        for(ii=0; ii<N; ii++){
            fprintf(fo_close, " %d:%f",bestw[ii], bestd[ii]);
        }
        fprintf(fo_close, "\n");
        pthread_mutex_unlock(&mutex); 
    }
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++)
		if (!strcmp(str, argv[a])) {
			if (a == argc - 1) {
				printf("Argument missing for %s\n", str);
				exit(1);
			}
			return a;
		}
	return -1;
}
int main(int argc, char **argv)
{
  int i, read_class=0, output_close=0;
  if ((i = ArgPos((char *) "-train", argc, argv)) > 0){
    strcpy(train_file, argv[i + 1]);
  }
  if ((i = ArgPos((char *) "-output", argc, argv)) > 0){
    strcpy(output_file, argv[i + 1]);
  }
  if ((i = ArgPos((char *) "-read-classes", argc, argv)) > 0){
    strcpy(class_file, argv[i + 1]);
    read_class = 1;
  }
  if ((i = ArgPos((char *) "-output-close", argc, argv)) > 0){
    strcpy(close_file, argv[i + 1]);
    output_close = 1;
  }
  if ((i = ArgPos((char *) "-classes", argc, argv)) > 0){
    classes = atoi(argv[i + 1]);
  }
  if ((i = ArgPos((char *) "-threads", argc, argv)) > 0){
    num_threads = atoi(argv[i + 1]);
  }
  if ((i = ArgPos((char *) "-words", argc, argv)) > 0){
    words = atoi(argv[i + 1]);
  }
  if ((i = ArgPos((char *) "-size", argc, argv)) > 0){
    size = atoi(argv[i + 1]);
  }
  if ((i = ArgPos((char *) "-iter", argc, argv)) > 0){
    iter = atoi(argv[i + 1]);
  }

  long long a, b, c, d, th;
  double closev, x;
  read_vec();
  printf("read vec done!\n");

  FILE *fo = fopen(output_file, "wb");
  vocab_size = words;
  if(1){
    // Run K-means on the word vectors
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    clcn = classes;
    centcn = (int *)malloc(classes * sizeof(int));
    cl = (int *)calloc(vocab_size, sizeof(int));
    cent = (double *)calloc(classes * layer1_size, sizeof(double));
    //随机分配所在cluster
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    // 从文件读入上次分类结果
    if(read_class==1) init_cluster();
    for (a = 0; a < iter; a++) {
      // 中心点 向量 初始化为0
      printf("iter %d\n", a);
      time_t start = 0,end = 0;
      time(&start);
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += M[c * layer1_size + d];
        centcn[cl[c]]++;
      }
      //printf("norm centers\n");
      // norm 向量
      for (b = 0; b < clcn; b++) {
        closev = 0;
        //printf("before norm cent %d ",b);
        for (c = 0; c < layer1_size; c++) {
          //printf("%f ", cent[layer1_size * b + c]);
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        //printf("\n");
        closev = sqrt(closev);
        //printf("cent %d ",b);
        for (c = 0; c < layer1_size; c++){
            cent[layer1_size * b + c] /= closev;
            //printf("%f ", cent[layer1_size * b + c]);
        }
        //printf("\n");
      }

      printf("thread starts!\n");
      // 计算所属聚类
      for (th = 0; th < num_threads; th++) pthread_create(&pt[th], NULL, KmeansThread, (void *)th);
      for (th = 0; th < num_threads; th++) pthread_join(pt[th], NULL);
      time(&end);
      printf("cost %d sconds!\n", end-start);
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", &vocab[a*max_w], cl[a]);

    if(output_close == 1){
      fo_close = fopen(close_file, "w");
      for (th = 0; th < num_threads; th++) pthread_create(&pt[th], NULL, FindBestdThread, (void *)th);
      for (th = 0; th < num_threads; th++) pthread_join(pt[th], NULL);
      fclose(fo_close);
    }
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
}
