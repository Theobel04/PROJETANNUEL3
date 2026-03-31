// lib/linear_model.h
#ifndef LINEAR_MODEL_H
#define LINEAR_MODEL_H

#define N_CLASSES  3
#define N_FEATURES 1024  // 32x32 grayscale

typedef struct {
    double W[N_CLASSES][N_FEATURES];
    double b[N_CLASSES];
} LinearModel;

LinearModel* lm_create();
void         lm_destroy(LinearModel* model);
void         lm_train(LinearModel* model, double* X, int* y,
                      int n_samples, double lr, int epochs);
int          lm_predict(LinearModel* model, double* x);
double       lm_evaluate(LinearModel* model, double* X, int* y, int n_samples);
void         lm_save(LinearModel* model, const char* path);
LinearModel* lm_load(const char* path);

#endif