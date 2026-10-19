#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <unistd.h>  // pipe(), fork(), close(), read(), write(), _exit()
#include <sys/types.h> // pid_t
#include <sys/wait.h>

/* ========================================================================= */
/* NETWORK ARCHITECTURE                                                    */
/* ========================================================================= */

#define INPUT_DIM	4
#define C1_COLUMN_SIZE 3
#define C2_COLUMN_SIZE 3	/* egal au nombre de sorties */

#define NB_SAMPLES 15

/* ========================================================================= */
/* PRÉ-TRAITEMENT ET FONCTIONS D'ACTIVATION                                 */
/* ========================================================================= */

/* Pré-traitement : Normalisation Min-Max [-1, 1] des données d'entrée */
void normalize(int n_samples, int input_size, double data[n_samples][input_size], double data_norm[n_samples][input_size]);


/* Fonctions d'activation */
void act_tanh(double x[], int len);
void softmax(const double in[], double out[], int len);


/* ========================================================================= */
/* CALCULS NEURONAUX                                                         */
/* ========================================================================= */

/* Produit scalaire entre l'entrée augmentée (avec bias) et la j-ième colonne de poids */
double dot_column(int n_in, int n_out,
    double in_aug[], double coefs[][n_out],
    int neuron_idx);

/* Propagation avant pour une couche complète (ponderated sum with bias) */
void process_column(int n_in, int n_out,
    double in[], double out[],
    double coefs[][n_out]);


int main(int argc, char **argv)
{
	
/* vecteurs de test linearises */
double jeux_test_lineaire[NB_SAMPLES][4] = {
{-0.613472, 0.192301, -4.148675, -0.758351},
{-0.778532, -0.024905, -4.148675, -0.758351},
{-0.943593, 0.061977, -4.324615, -0.758351},
{-1.026123, 0.018536, -3.972734, -0.758351},
{-0.696002, 0.235742, -4.148675, -0.682465},
{0.954602, 0.061977, 1.657359, 0.152280},
{0.459421, 0.061977, 1.305478, 0.228166},
{0.872072, 0.018536, 2.009239, 0.228166},
{-0.283351, -0.328992, 0.425776, 0.076394},
{0.541951, -0.111787, 1.481418, 0.228166},
{0.376891, 0.105419, 3.944584, 0.987025},
{-0.035760, -0.155228, 2.361120, 0.531710},
{1.037132, -0.024905, 3.768643, 0.683482},
{0.376891, -0.068346, 3.240822, 0.455824},
{0.541951, -0.024905, 3.592703, 0.759367}};

/* Sories attendues pour les NB_SAMPLES vecteurs de test */
int sortie_attendue[NB_SAMPLES] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
		
/* Jeu de coefficients pour un réseau 4 entrées, 3 neurones couche1 */
/* 3 neurones couches 2 et 3 sorties */		
/* Les elements [INPUT_DIM][] de coef_MAT1 contiennent les biais*/
/* Les elements [C1_COLUMN_SIZE][] de coef_MAT2 contiennent les biais*/
/* 13 sur 15 résultats (87%) correct avec cet ensemble de coefficients */
double coef_MAT1[INPUT_DIM + 1][C1_COLUMN_SIZE] = 
			{{ +1.70, +2.58, -1.87}, 
			{+1.56, +4.04, -5.81}, 
			{-2.68, -3.84, +4.21}, 
			{-0.02, -6.49, +6.97}, 
			{-1.00, +5.00, -5.00}}; // ---> bias
			
double coef_MAT2[C1_COLUMN_SIZE + 1][C2_COLUMN_SIZE] = 
			{{ +9.00, -10.75, -1.38},
			{-1.12, +4.68, -5.82}, 
			{-3.43, -5.32, +5.60}, 
			{-3.30, +1.42, -1.22 }}; // ---> bias


/*      MAIN PROGRAM        */


// creation of the pipes

int pipe_in[INPUT_DIM][2];
int pipe_out[C2_COLUMN_SIZE][2];

for (int i = 0; i < INPUT_DIM; i++) {
    if (pipe(pipe_in[i]) < 0) {
        perror("Fail when creating pipe_in");
        exit(EXIT_FAILURE);
    }
}

for (int i = 0; i < C2_COLUMN_SIZE; i++) {
    if (pipe(pipe_out[i]) < 0) {
        perror("Fail when creating pipe_out");
        exit(EXIT_FAILURE);
    }
}

// FORKING
pid_t pid = fork();

if (pid != 0) {
    /* MONITOR */

    /*
    // normalize input
    double normalized_input[N_samples][INPUT_DIM];
    normalize(N_samples, INPUT_DIM, jeux_test_lineaire, normalized_input);
    */

    // close unused ends
    for (int i = 0; i < INPUT_DIM; i++) close(pipe_in[i][0]);  // dont read the inputs
    for (int i = 0; i < C2_COLUMN_SIZE; i++) close(pipe_out[i][1]); // dont write the outputs

    // write inputs
    for (int i = 0; i < NB_SAMPLES; i++) {
        for (int j = 0; j < INPUT_DIM; j++) {
            write(pipe_in[j][1], &jeux_test_lineaire[i][j], sizeof(double));
        }
    }

    // read Application outputs
    double out[NB_SAMPLES][C2_COLUMN_SIZE];
    for (int i = 0; i < NB_SAMPLES; i++) {
        for (int j = 0; j < C2_COLUMN_SIZE; j++) {
            read(pipe_out[j][0], &out[i][j], sizeof(double));
        }
    }
    
    // print outputs
    for (int i = 0; i < NB_SAMPLES; i++) {
        printf("saída:\n[");
        for (int j = 0; j < C2_COLUMN_SIZE; j++) {
            printf("%.4F\t", out[i][j]);
        }
        printf("]\n");
    }

    wait(NULL);
}

else {
    /* APPLICATION */
    
    // close unused ends
    for (int i = 0; i < INPUT_DIM; i++) close(pipe_in[i][1]);  // dont read the inputs
    for (int i = 0; i < C2_COLUMN_SIZE; i++) close(pipe_out[i][0]); // dont write the outputs


    for (int i = 0; i < NB_SAMPLES; i++) {

        // read Monitor inputs
        double in[INPUT_DIM];
        for (int j = 0; j < INPUT_DIM; j++){
            read(pipe_in[j][0], &in[j], sizeof(double));
        }

        // process the columns
        double out_C1[C1_COLUMN_SIZE];
        process_column(INPUT_DIM, C1_COLUMN_SIZE, in, out_C1, coef_MAT1);
        act_tanh(out_C1, C1_COLUMN_SIZE);

        double out_C2[C2_COLUMN_SIZE];
        process_column(C1_COLUMN_SIZE, C2_COLUMN_SIZE, out_C1, out_C2, coef_MAT2);

        double out[C2_COLUMN_SIZE];
        softmax(out_C2, out, C2_COLUMN_SIZE);

        // send outputs to Monitor
        for (int j = 0; j < C2_COLUMN_SIZE; j++){
            write(pipe_out[j][1], &out[j], sizeof(double));
        }
    }

    // close descriptors before terminate
    for (int i = 0; i < INPUT_DIM; i++) close(pipe_in[i][0]);
    for (int i = 0; i < C2_COLUMN_SIZE; i++) close(pipe_out[i][1]);
    _exit(0);
}

	return 0;
}


/*/////////////////////////////////////////////////////////////////////////*/
	   
/*  nb est la taille des vecteurs d'entree et de sortie - in et out */
void softmax(const double in[], double out[], int len) {
    /* Trouver la valeur max pour la stabilité numérique */
    double max_softmax = in[0];
    for (int l = 1; l < len; l++) {
        if (in[l] > max_softmax)
				max_softmax = in[l];
    }

    /* Calculer exp(x - max) */
    double somme = 0.0;
    for (int l = 0; l < len; l++) {
        out[l] = exp(in[l] - max_softmax);
        somme += out[l];
    }

    /* Normaliser */
    for (int l = 0; l < len; l++) {
        out[l] /= somme;
    }
}


void act_tanh(double x[], int len) {
    for (int i = 0; i < len; i++){
        x[i] = tanh(x[i]);
    }
}


void process_column(int n_in, int n_out, double in[], double out[], double coefs[][n_out]) {
    // 1. Creation of augmented input array
    double in_aug[n_in + 1];
    for (int i = 0; i < n_in; i++) {
        in_aug[i] = in[i];
    }
    in_aug[n_in] = 1.0;  // for add the bias
    
    // 2. calculate ponderated sum with the bias
    for (int j = 0; j < n_out; j++) {
        out[j] = dot_column(n_in + 1, n_out, in_aug, coefs, j);
    }
}


double dot_column(int n_in, int n_out, double in_aug[], double coefs[][n_out], int neuron_idx) {
    double p = 0.0;
    for (int i = 0; i < n_in; i++) {
        p += in_aug[i] * coefs[i][neuron_idx];
    }
    return p;
}


void normalize(int n_samples, int input_size, double data[n_samples][input_size], double data_norm[n_samples][input_size]) {
    

    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < input_size; j++) {
        }
    }

    
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < input_size; j++) {
        }
    }

    // for each input:
    for (int j = 0; j < input_size; j++) {

        // find data maximum value
        double max = data[0][j];
        double min = data[0][j];
        for (int i = 0; i < n_samples; i++) {
            if (max < data[i][j]) max = data[i][j];
            if (min > data[i][j]) min = data[i][j];
        }

        // change the scale range
        const double range = max - min;
        for (int i = 0; i < n_samples; i++) {
            data_norm[i][j] = (range == 0) ? 0.0 : 2 * (data[i][j] - min) / range - 1;
        }
    }
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < input_size; j++) {
        }
    }
}