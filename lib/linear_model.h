// lib/linear_model.h
#ifndef LINEAR_MODEL_H
#define LINEAR_MODEL_H

#define N_CLASSES  3
#define N_FEATURES 1024  // 32x32 grayscale

// MODELE LINEAIRE

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

// MODELE LINEAIRE GENERIQUE (n_features variable)
// Pour les cas de tests transformés

typedef struct {
    int      n_classes;
    int      n_features;
    double*  W;  // matrice [n_classes x n_features]
    double*  b;  // biais   [n_classes]
} LinearModelGen;

LinearModelGen* lmg_create(int n_classes, int n_features);
void            lmg_destroy(LinearModelGen* model);
void            lmg_train(LinearModelGen* model, double* X, int* y,
                           int n_samples, double lr, int epochs);
int             lmg_predict(LinearModelGen* model, double* x);
double          lmg_evaluate(LinearModelGen* model, double* X, int* y,
                              int n_samples);

// TRANSFORMATIONS NON LINEAIRES

// XOR : (x1, x2) -> (x1, x2, x1*x2)
void transform_xor(double* X, int n_samples, double* X_out);

// Cercles : (x1, x2) -> (x1, x2, x1²+x2²)
void transform_circles(double* X, int n_samples, double* X_out);

// Polynomial : ajoute x² pour les n_feat premières features
// X_in  : n_samples * n_feat
// X_out : n_samples * (n_feat + n_feat) = n_samples * 2*n_feat
void transform_polynomial(double* X_in, int n_samples, int n_feat,
                           double* X_out);

#endif