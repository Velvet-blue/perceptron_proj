#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <unistd.h>  // pipe(), fork(), close(), read(), write(), _exit()
#include <sys/types.h> // pid_t
#include <sys/wait.h>


/* Structure du réseau */
#define NB_neurones_C1 3
#define NB_neurones_C2 3	/* egal au nombre de sorties */

/* 4 données d'entree et une valeur de sortie */
#define N_entree	4

/* Test avec 15 vecteurs d'entree */
#define NB_data 15


/* Fonctions d'activation */
void act_tanh(double x[], int nb);
void softmax(const double in[], double out[], int nb);

/* Fonctions des neurones */
double dot_column(int n_in, int n_out, double in_aug[], double coefs[][n_out], int col_j);
void process_column(int n_in, int n_out, double in[], double out[], double coefs[][n_out]);


int main(int argc, char **argv)
{
	
/* vecteurs de test linearises */
double jeux_test_lineaire[NB_data][4] = {
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

/* Sories attendues pour les NB_data vecteurs de test */
int sortie_attendue[NB_data] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
		
/* Jeu de coefficients pour un réseau 4 entrées, 3 neurones couche1 */
/* 3 neurones couches 2 et 3 sorties */		
/* Les elements [N_entree][] de coef_MAT1 contiennent les biais*/
/* Les elements [NB_neurones_C1][] de coef_MAT2 contiennent les biais*/
/* 13 sur 15 résultats (87%) correct avec cet ensemble de coefficients */
double coef_MAT1[N_entree + 1][NB_neurones_C1] = 
			{{ +1.70, +2.58, -1.87}, 
			{+1.56, +4.04, -5.81}, 
			{-2.68, -3.84, +4.21}, 
			{-0.02, -6.49, +6.97}, 
			{-1.00, +5.00, -5.00}}; // ---> bias
			
double coef_MAT2[NB_neurones_C1 + 1][NB_neurones_C2] = 
			{{ +9.00, -10.75, -1.38},
			{-1.12, +4.68, -5.82}, 
			{-3.43, -5.32, +5.60}, 
			{-3.30, +1.42, -1.22 }}; // ---> bias


/*      MAIN PROGRAM        */


// creation of the pipes

int pipe_in[N_entree][2];
int pipe_out[NB_neurones_C2][2];

for (int i = 0; i < N_entree; i++) {
    if (pipe(pipe_in[i]) < 0) {
        perror("Fail when creating pipe_in");
        exit(EXIT_FAILURE);
    }
}

for (int i = 0; i < NB_neurones_C2; i++) {
    if (pipe(pipe_out[i]) < 0) {
        perror("Fail when creating pipe_out");
        exit(EXIT_FAILURE);
    }
}


const int N_samples = sizeof(jeux_test_lineaire)/sizeof(jeux_test_lineaire[0]);


// FORKING
pid_t pid = fork();

if (pid != 0) {
    /* MONITOR */

    // close unused ends
    for (int i = 0; i < N_entree; i++) close(pipe_in[i][0]);  // dont read the inputs
    for (int i = 0; i < NB_neurones_C2; i++) close(pipe_out[i][1]); // dont write the outputs

    // write inputs
    for (int i = 0; i < N_samples; i++) {
        for (int j = 0; j < N_entree; j++) {
            write(pipe_in[j][1], &jeux_test_lineaire[i][j], sizeof(double));
        }
    }

    // read Application outputs
    double out[N_samples][NB_neurones_C2];
    for (int i = 0; i < N_samples; i++) {
        for (int j = 0; j < NB_neurones_C2; j++) {
            read(pipe_out[j][0], &out[i][j], sizeof(double));
    }
    
    // print outputs
    for (int i = 0; i < N_samples; i++) {
    printf("saída:\n[");
        for (int j = 0; j < NB_neurones_C2; j++) {
            printf("%.4F\t", out[i][j]);
    }
        printf("]\n");
    }

    wait(NULL);
}

else {
    /* APPLICATION */
    
    // close unused ends
    for (int i = 0; i < N_entree; i++) close(pipe_in[i][1]);  // dont read the inputs
    for (int i = 0; i < NB_neurones_C2; i++) close(pipe_out[i][0]); // dont write the outputs


    for (int i = 0; i < N_samples; i++) {

    // read Monitor inputs
    double in[N_entree];
        for (int j = 0; j < N_entree; j++){
            read(pipe_in[j][0], &in[j], sizeof(double));
    }

    // process the columns
    double out_C1[NB_neurones_C1];
        process_column(N_entree, NB_neurones_C1, in, out_C1, coef_MAT1);
    act_tanh(out_C1, NB_neurones_C1);

    double out_C2[NB_neurones_C2];
    process_column(NB_neurones_C1, NB_neurones_C2, out_C1, out_C2, coef_MAT2);

    double out[NB_neurones_C2];
    softmax(out_C2, out, NB_neurones_C2);

    // send outputs to Monitor
        for (int j = 0; j < NB_neurones_C2; j++){
            write(pipe_out[j][1], &out[j], sizeof(double));
        }
    }

    // close descriptors before terminate
    for (int i = 0; i < N_entree; i++) close(pipe_in[i][0]);
    for (int i = 0; i < NB_neurones_C2; i++) close(pipe_out[i][1]);
    _exit(0);
}

	return 0;
}


/*/////////////////////////////////////////////////////////////////////////*/
	   
/*  nb est la taille des vecteurs d'entree et de sortie - in et out */
void softmax(const double in[], double out[], int nb) {
    /* Trouver la valeur max pour la stabilité numérique */
    double max_softmax = in[0];
    for (int l = 1; l < nb; l++) {
        if (in[l] > max_softmax)
				max_softmax = in[l];
    }

    /* Calculer exp(x - max) */
    double somme = 0.0;
    for (int l = 0; l < nb; l++) {
        out[l] = exp(in[l] - max_softmax);
        somme += out[l];
    }

    /* Normaliser */
    for (int l = 0; l < nb; l++) {
        out[l] /= somme;
    }
}


/* Fonction d'activation tanh */
void act_tanh(double x[], int nb) {
    for (int i = 0; i < nb; i++){
        x[i] = tanh(x[i]);
        printf("act = %.15F\n", x[i]);
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
        // printf("C1 > %.15F\n", out[j]);
    }
}


double dot_column(int n_in, int n_out, double in_aug[], double coefs[][n_out], int col_j) {
    double p = 0.0;
    for (int i = 0; i < n_in; i++) {
        p += in_aug[i] * coefs[i][col_j];
    }
    return p;
}