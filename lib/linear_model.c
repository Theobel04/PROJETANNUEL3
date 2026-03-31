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