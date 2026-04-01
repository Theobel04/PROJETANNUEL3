// lib/linear_model.c
#include "linear_model.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

LinearModel* lm_create() {
    LinearModel* m = calloc(1, sizeof(LinearModel));
    return m;  // Poids initialisés à 0
}

void lm_destroy(LinearModel* m) {
    free(m);
}

int lm_predict(LinearModel* m, double* x) {
    int best = 0;
    double best_score = -1e18;
    for (int c = 0; c < N_CLASSES; c++) {
        double score = m->b[c];
        for (int f = 0; f < N_FEATURES; f++)
            score += m->W[c][f] * x[f];
        if (score > best_score) {
            best_score = score;
            best = c;
        }
    }
    return best;
}

void lm_train(LinearModel* m, double* X, int* y,
              int n_samples, double lr, int epochs) {
    for (int e = 0; e < epochs; e++) {
        int errors = 0;
        for (int i = 0; i < n_samples; i++) {
            double* x  = X + (long)i * N_FEATURES;
            int pred   = lm_predict(m, x);
            int truth  = y[i];
            if (pred != truth) {
                errors++;
                for (int f = 0; f < N_FEATURES; f++) {
                    m->W[truth][f] += lr * x[f];
                    m->W[pred][f]  -= lr * x[f];
                }
                m->b[truth] += lr;
                m->b[pred]  -= lr;
            }
        }
        if (e % 10 == 0)
            printf("Epoch %3d — erreurs : %d / %d\n", e, errors, n_samples);
        if (errors == 0) {
            printf("Convergence à l'epoch %d\n", e);
            break;
        }
    }
}

double lm_evaluate(LinearModel* m, double* X, int* y, int n_samples) {
    int correct = 0;
    for (int i = 0; i < n_samples; i++) {
        if (lm_predict(m, X + (long)i * N_FEATURES) == y[i])
            correct++;
    }
    return (double)correct / n_samples * 100.0;
}

void lm_save(LinearModel* m, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Erreur ouverture %s\n", path); return; }
    fwrite(m, sizeof(LinearModel), 1, f);
    fclose(f);
    printf("Modèle sauvegardé : %s\n", path);
}

LinearModel* lm_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Erreur ouverture %s\n", path); return NULL; }
    LinearModel* m = malloc(sizeof(LinearModel));
    if (fread(m, sizeof(LinearModel), 1, f) != 1) {
        fprintf(stderr, "Erreur lecture modèle\n");
        free(m);
        fclose(f);
        return NULL;
    }
    fclose(f);
    printf("Modèle chargé : %s\n", path);
    return m;
}

// MODELE LINEAIRE GENERIQUE

LinearModelGen* lmg_create(int n_classes, int n_features) {
    LinearModelGen* m = malloc(sizeof(LinearModelGen));
    m->n_classes  = n_classes;
    m->n_features = n_features;
    m->W = calloc(n_classes * n_features, sizeof(double));
    m->b = calloc(n_classes, sizeof(double));
    return m;
}

void lmg_destroy(LinearModelGen* m) {
    free(m->W);
    free(m->b);
    free(m);
}

int lmg_predict(LinearModelGen* m, double* x) {
    int best = 0;
    double best_score = -1e18;
    for (int c = 0; c < m->n_classes; c++) {
        double score = m->b[c];
        for (int f = 0; f < m->n_features; f++)
            score += m->W[c * m->n_features + f] * x[f];
        if (score > best_score) {
            best_score = score;
            best = c;
        }
    }
    return best;
}

void lmg_train(LinearModelGen* m, double* X, int* y,
               int n_samples, double lr, int epochs) {
    for (int e = 0; e < epochs; e++) {
        int errors = 0;
        for (int i = 0; i < n_samples; i++) {
            double* x = X + (long)i * m->n_features;
            int pred  = lmg_predict(m, x);
            int truth = y[i];
            if (pred != truth) {
                errors++;
                for (int f = 0; f < m->n_features; f++) {
                    m->W[truth * m->n_features + f] += lr * x[f];
                    m->W[pred  * m->n_features + f] -= lr * x[f];
                }
                m->b[truth] += lr;
                m->b[pred]  -= lr;
            }
        }
        if (e % 10 == 0)
            printf("Epoch %3d — erreurs : %d / %d\n", e, errors, n_samples);
        if (errors == 0) {
            printf("Convergence a l'epoch %d\n", e);
            break;
        }
    }
}

double lmg_evaluate(LinearModelGen* m, double* X, int* y, int n_samples) {
    int correct = 0;
    for (int i = 0; i < n_samples; i++) {
        if (lmg_predict(m, X + (long)i * m->n_features) == y[i])
            correct++;
    }
    return (double)correct / n_samples * 100.0;
}

// TRANSFORMATIONS NON LINEAIRES

void transform_xor(double* X, int n_samples, double* X_out) {
    // (x1, x2) -> (x1, x2, x1*x2)
    for (int i = 0; i < n_samples; i++) {
        double x1 = X[i * 2 + 0];
        double x2 = X[i * 2 + 1];
        X_out[i * 3 + 0] = x1;
        X_out[i * 3 + 1] = x2;
        X_out[i * 3 + 2] = x1 * x2;  // XOR
    }
}

void transform_circles(double* X, int n_samples, double* X_out) {
    // (x1, x2) -> (x1, x2, x1²+x2²)
    for (int i = 0; i < n_samples; i++) {
        double x1 = X[i * 2 + 0];
        double x2 = X[i * 2 + 1];
        X_out[i * 3 + 0] = x1;
        X_out[i * 3 + 1] = x2;
        X_out[i * 3 + 2] = x1*x1 + x2*x2;  // Distance au centre
    }
}

void transform_polynomial(double* X_in, int n_samples, int n_feat,
                           double* X_out) {
    // Ajoute les carrés de chaque feature
    // X_out : n_samples * (2 * n_feat)
    for (int i = 0; i < n_samples; i++) {
        // Features originales
        for (int f = 0; f < n_feat; f++)
            X_out[i * 2*n_feat + f] = X_in[i * n_feat + f];
        // Carrés
        for (int f = 0; f < n_feat; f++)
            X_out[i * 2*n_feat + n_feat + f] = X_in[i * n_feat + f]
                                              * X_in[i * n_feat + f];
    }
}