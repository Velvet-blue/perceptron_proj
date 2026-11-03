#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <unistd.h>  // pipe(), fork(), close(), read(), write(), _exit()
#include <sys/types.h> // pid_t
#include <sys/wait.h>

#include <sys/mman.h>   // shm_open(), mmap(), munmap(), shm_unlink(), PROT_*, MAP_*
#include <sys/stat.h>   // Para constantes de modo (0666)
#include <fcntl.h>      // O_CREAT, O_RDWR, O_RDONLY, O_EXCL
#include <semaphore.h>  // sem_t, sem_open(), sem_wait(), sem_post(), sem_close(), sem_unlink(), SEM_FAILED
#include <signal.h>     // sigaction, sig_atomic_t, SIGUSR1, kill()
#include <time.h>       // nanosleep(), struct timespec


#define SHM_NAME "/shm_perceptron"
#define SEM_NAME "/sem_perceptron"

/* ========================================================================= */
/*              NETWORK ARCHITECTURE                                         */
/* ========================================================================= */

#define INPUT_DIM	4
#define C1_COLUMN_SIZE 3
#define C2_COLUMN_SIZE 3	/* equals the number of outputs*/

#define NB_SAMPLES 15

/* Shared Memory */
typedef struct {
    double coef_MAT1[INPUT_DIM + 1][C1_COLUMN_SIZE];
    double coef_MAT2[C1_COLUMN_SIZE + 1][C2_COLUMN_SIZE];
} WeightsBuffer;

/* ========================================================================= */
/*              PRE-TREATMENT AND ACTIVATION FUNCTIONS                       */
/* ========================================================================= */

/* Pre-treatment : Normalization Min-Max [-1, 1] of the input data */
void normalize(int n_samples, int input_size, double data[n_samples][input_size], double data_norm[n_samples][input_size]);


/* Activation Functions */
void act_tanh(double x[], int len);
void softmax(const double in[], double out[], int len);


/* ========================================================================= */
/*              NEURAL COMPUTATIONS                                          */
/* ========================================================================= */

/* Scalar Product between the augmented input (with bias) and the j-th column of weights */
double dot_column(int n_in, int n_out,
    double in_aug[], double coefs[][n_out],
    int neuron_idx);

/* Forward propagation for a complete layer (weighted sum with bias) */
void process_column(int n_in, int n_out,
    double in[], double out[],
    double coefs[][n_out]);

/* ========================================================================= */
/*              PROGRAM FUNCTIONS                                            */
/* ========================================================================= */

void run_monitor(pid_t pid_reader, int pipe_synth[2], WeightsBuffer *shm_weights, sem_t *sem);
void run_reader(int pipe_in[4][2], int pipe_ref[2]);
void run_application(int pipe_in[INPUT_DIM][2], int pipe_out[C2_COLUMN_SIZE][2]);
void run_analyser(int pipe_out[3][2], int pipe_ref[2], int pipe_synth[2]);

