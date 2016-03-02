#include <stdlib.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "svm.h"

struct svm_node **dense_to_libsvm (double *x, npy_intp *dims);
void set_parameter(struct svm_parameter *param, int svm_type, int kernel_type, int degree,
		double gamma, double coef0, double nu, double cache_size, double C,
		double eps, double p, int shrinking, int probability, int nr_weight,
		char *weight_label, char *weight, int max_iter, int random_seed);
void set_problem(struct svm_problem *problem, char *X, char *Y, char *sample_weight, npy_intp *dims, int kernel_type);
npy_intp get_l(struct svm_model *model);
npy_intp get_nr(struct svm_model *model);
void copy_sv_coef(char *data, struct svm_model *model);
void copy_intercept(char *data, struct svm_model *model, npy_intp *dims);
void copy_support (char *data, struct svm_model *model);
void copy_probA(char *data, struct svm_model *model, npy_intp * dims);
void copy_probB(char *data, struct svm_model *model, npy_intp * dims);
int copy_predict(char *predict, struct svm_model *model, npy_intp *predict_dims,
                 char *dec_values);
int copy_predict_values(char *predict, struct svm_model *model,
                        npy_intp *predict_dims, char *dec_values, int nr_class);
int free_model(struct svm_model *model);
int free_param(struct svm_parameter *param);
void set_verbosity(int verbosity_flag);
