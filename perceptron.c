#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <unistd.h>  // pipe(), fork(), close(), read(), write(), _exit()
#include <sys/types.h> // pid_t
#include <sys/wait.h>

volatile sig_atomic_t start_reading = 0;
void handle_sigusr1(int sig) {
    (void)sig; // Évite les warnings "unused parameter"
    start_reading = 1;
}

/* ========================================================================= */
/*              NETWORK ARCHITECTURE                                         */
/* ========================================================================= */

#define INPUT_DIM	4
#define C1_COLUMN_SIZE 3
#define C2_COLUMN_SIZE 3	/* equals the number of outputs*/

#define NB_SAMPLES 15

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

void run_monitor(pid_t pid_reader, int pipe_synth[2]);
void run_reader(int pipe_in[4][2], int pipe_ref[2]);
void run_application(int pipe_in[4][2], int pipe_out[3][2]);
void run_analyser(int pipe_out[3][2], int pipe_ref[2], int pipe_synth[2]);


/* ========================================================================= */
/*              MAIN                                                         */
/* ========================================================================= */

/* sample linear test vector */
double linear_sample_vector[NB_SAMPLES][4] = {
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

/* Expected outputs for the N_SAMPLES test vectors */
int ref_values[NB_SAMPLES] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
		
/* Set of coefficients for a network with 4 inputs and 3 neurons in the first layer */
/* 3 neurons in the second layer and 3 outputs */
/* The [INPUT_DIM][] elements of coef_MAT1 contain the biases */
/* The [C1_COLUMN_SIZE][] elements of coef_MAT2 contain the biases */
/* 13 out of 15 results (87%) are correct with this set of coefficients */
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



int main(int argc, char **argv)
{
	


/*      MAIN PROGRAM        */

// creation of the pipes

int pipe_in[INPUT_DIM][2];          // reader  <-> Applicat
int pipe_out[C2_COLUMN_SIZE][2];    // Applicat <-> Analyser
int pipe_ref[2];                    // reader  <-> analyser
int pipe_synth[2];                  // monitor <-> analyser

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

if (pipe(pipe_ref) < 0) {
    perror("Fail when creatin pipe_ref");
    exit(EXIT_FAILURE);
}

if (pipe(pipe_synth) < 0) {
    perror("Fail when creatin pipe_synth");
    exit(EXIT_FAILURE);
}


// Creation of child processes

pid_t pid, pid_reader, pid_app, pid_analyser;

pid_reader = fork();
if (pid_reader == 0) {
    printf("[reader]\t fork\n");
    run_reader(pipe_in, pipe_ref);
    printf("[reader]\t end\n");
    _exit(EXIT_SUCCESS);
}

pid_app = fork();
if (pid_app == 0) {
    printf("[applycat]\t fork\n");
    run_application(pipe_in, pipe_out);
    printf("[applycat]\t end\n");
    _exit(EXIT_SUCCESS);
}

pid_analyser = fork();
if (pid_analyser == 0) {
    printf("[analys]\t fork\n");
    run_analyser(pipe_out, pipe_ref, pipe_synth);
    printf("[analys]\t end\n");
    _exit(EXIT_SUCCESS);
}

// 3. Executes the main routine
run_monitor(pid_reader, pipe_synth);
printf("[monitor]\t end\n");

// 4. Wait the 3 processes
for (int i = 0; i < 3; i++) {
    wait(NULL);
}



return 0;
}


/*/////////////////////////////////////////////////////////////////////////*/
	   
void softmax(const double in[], double out[], int len) {
    /* Find the maximum value for numerical stability */
    double max_softmax = in[0];
    for (int l = 1; l < len; l++) {
        if (in[l] > max_softmax)
				max_softmax = in[l];
    }

    /* Calculates exp(x - max) */
    double somme = 0.0;
    for (int l = 0; l < len; l++) {
        out[l] = exp(in[l] - max_softmax);
        somme += out[l];
    }

    /* Normalize */
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


void run_reader(int pipe_in[4][2], int pipe_ref[2]) {
    /* reader */

    /*
    // normalize input
    double normalized_input[NB_SAMPLES][INPUT_DIM];
    normalize(NB_SAMPLES, INPUT_DIM, linear_sample_vector, normalized_input);
    */


    // close pipes : the process only writes
    for (int i = 0; i < INPUT_DIM; i++) close(pipe_in[i][0]); // Application
    close(pipe_ref[0]); // analyser

    // initialize signal
    signal(SIGUSR1, handle_sigusr1);

    // wait the signal for it starts
    while (!start_reading) {
        pause();
    }

    // write inputs for Application
    for (int i = 0; i < NB_SAMPLES; i++) {
        for (int j = 0; j < INPUT_DIM; j++) {
            write(pipe_in[j][1], &linear_sample_vector[i][j], sizeof(double));
        }
    }

    // write reference values for analyser
    for (int i = 0; i < NB_SAMPLES; i++) {
        write(pipe_ref[1], &ref_values[i], sizeof(int));
    }
}


void run_application(int pipe_in[4][2], int pipe_out[3][2]) {
    /* APPLICATION */
    
    // close unused ends
    for (int i = 0; i < INPUT_DIM; i++) close(pipe_in[i][1]);  // only read the inputs
    for (int i = 0; i < C2_COLUMN_SIZE; i++) close(pipe_out[i][0]); // only write the outputs

    // iterate through the samples
    for (int i = 0; i < NB_SAMPLES; i++) {

        // read the inputs
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

        // send outputs
        for (int j = 0; j < C2_COLUMN_SIZE; j++){
            write(pipe_out[j][1], &out[j], sizeof(double));
        }
    }

    // close descriptors before terminate
    for (int i = 0; i < INPUT_DIM; i++) close(pipe_in[i][0]);
    for (int i = 0; i < C2_COLUMN_SIZE; i++) close(pipe_out[i][1]);
}


void run_analyser(int pipe_out[3][2], int pipe_ref[2], int pipe_synth[2]) {

    // close unused ends
    for (int i = 0; i < C2_COLUMN_SIZE; i++) close(pipe_out[i][1]); // only read the outputs
    close(pipe_ref[1]);   // only read the ref values
    close(pipe_synth[0]); // only write the results

    int ref_value = 0;
    int n_right_results = 0;
    double out_sample[C2_COLUMN_SIZE];
    for (int i = 0; i < NB_SAMPLES; i++){

        // read one ref value
        read(pipe_ref[0], &ref_value, sizeof(int));

        // read one output sample vector
        for (int j = 0; j < C2_COLUMN_SIZE; j++) {
            read(pipe_out[j][0], &out_sample[j], sizeof(double));
        }

        // perform argmax classification
        double out_max_value = out_sample[0];
        int out_max_idx = 0;
        for (int j = 1; j < C2_COLUMN_SIZE; j++) {
            if (out_max_value < out_sample[j]) {
                out_max_value = out_sample[j];
                out_max_idx = j;
            }
        }

        // increment the right predictions counter
        if (out_max_idx == ref_value) {
            n_right_results++;
        }

    }

    // send accuracy percentage scaled by 100 as a rounded int (ex: 8667 means 86.67%)
    // note: +0.5 for each sample for round, not trunc (NB_SAMPLES/2)
    int percent_right_results = (10000 * n_right_results + NB_SAMPLES/2) / NB_SAMPLES;
    write(pipe_synth[1], &percent_right_results, sizeof(int));

    // close descriptors before terminate
    for (int i = 0; i < C2_COLUMN_SIZE; i++) close(pipe_out[i][0]);
    close(pipe_ref[0]);
    close(pipe_synth[1]);
}



void run_monitor(pid_t pid_reader, int pipe_synth[2]) {

    // close unused ends
    close(pipe_synth[1]);

    // Attendre un court instant ou laisser les fils s'initialiser
    usleep(1000); // 1 ms
    // Envoyer le signal SIGUSR1 au Lecteur pour déclencher l'exécution
    if (kill(pid_reader, SIGUSR1) < 0) {
        perror("Erreur lors de l'envoi de SIGUSR1 au Lecteur");
    }

    // read the righ result percentage
    int percent_right_results;
    read(pipe_synth[0], &percent_right_results, sizeof(int));

    // print the result
    printf("percentage of right results = %.2f %%\n", (float) percent_right_results/100 );

    // close descriptors before terminate
    close(pipe_synth[0]);

}