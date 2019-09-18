#include <stdio.h>

/* We'll be using MPI routines, definitions, etc. */
#include <mpi.h>

/* Build a derived datatype for distributing the input data */
void Build_mpi_type(double* a_p, double* b_p, int* n_p,
                    MPI_Datatype* input_mpi_t_p);

/* Get the input values */
void Get_input(int my_rank, int comm_sz, double* a_p, double* b_p,
               int* n_p);

/* Calculate local integral  */
double Trap(double left_endpt, double right_endpt, int trap_count,
            double base_len);

/* Function we're integrating */
double f(double x);

int main(void) {
    int my_rank, comm_sz, n, local_n;
    double a, b, h, local_a, local_b;
    double local_int, total_int;
    double start, finish, loc_elapsed, tempo_minimo, tempo_medio, tempoMaximo;

    /* Let the system do what it needs to start up MPI */
    MPI_Init(NULL, NULL);

    /* Get my process rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* Find out how many processes are being used */
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    Get_input(my_rank, comm_sz, &a, &b, &n);
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    h = (b-a)/n;          /* h is the same for all processes */
    local_n = n/comm_sz;  /* So is the number of trapezoids  */
    /* Length of each process' interval of
     * integration = local_n*h.  So my interval
     * starts at: */
    local_a = a + my_rank*local_n*h;
    local_b = local_a + local_n*h;
    local_int = Trap(local_a, local_b, local_n, h);
    finish = MPI_Wtime();
    loc_elapsed = finish-start;
    MPI_Reduce(&loc_elapsed, &tempoMaximo, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&loc_elapsed, &tempo_minimo, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&loc_elapsed, &tempo_medio, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    /* Add up the integrals calculated by each process */
    MPI_Reduce(&local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    /* Print the result */
    if (my_rank == 0) {
        printf("With n = %d trapezoids, our estimate\n", n);
        printf("of the integral from %f to %f = %.15e\n",
               a, b, total_int);
        printf("Tempo MÃ­nimo gasto = %e\n", tempo_minimo);
        printf("Tempo MÃ¡ximo gasto = %e\n", tempoMaximo);
        printf("Tempo MÃ©dio gasto = %e\n", tempo_medio/comm_sz);
        printf("Tempo Mediana gasto = %e\n", (tempoMaximo + tempo_minimo)/2.0);
    }

    /* Shut down MPI */
    MPI_Finalize();

    return 0;
} 
void Build_mpi_type(
        double*        a_p            /* in  */,
        double*        b_p            /* in  */,
        int*           n_p            /* in  */,
        MPI_Datatype*  input_mpi_t_p  /* out */) {

    int array_of_blocklengths[3] = {1, 1, 1};
    MPI_Datatype array_of_types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_INT};
    MPI_Aint a_addr, b_addr, n_addr;
    MPI_Aint array_of_displacements[3] = {0};

    MPI_Get_address(a_p, &a_addr);
    MPI_Get_address(b_p, &b_addr);
    MPI_Get_address(n_p, &n_addr);
    array_of_displacements[1] = b_addr-a_addr;
    array_of_displacements[2] = n_addr-a_addr;
    MPI_Type_create_struct(3, array_of_blocklengths,
                           array_of_displacements, array_of_types,
                           input_mpi_t_p);
    MPI_Type_commit(input_mpi_t_p);
}  
void Get_input(
        int      my_rank  /* in  */,
        int      comm_sz  /* in  */,
        double*  a_p      /* out */,
        double*  b_p      /* out */,
        int*     n_p      /* out */) {
    MPI_Datatype input_mpi_t;

    Build_mpi_type(a_p, b_p, n_p, &input_mpi_t);

    if (my_rank == 0) {
        printf("Enter a, b, and n\n");
        scanf("%lf %lf %d", a_p, b_p, n_p);
    }
    MPI_Bcast(a_p, 1, input_mpi_t, 0, MPI_COMM_WORLD);

    MPI_Type_free(&input_mpi_t);
} 
double Trap(
        double left_endpt  /* in */,
        double right_endpt /* in */,
        int    trap_count  /* in */,
        double base_len    /* in */) {
    double estimate, x;
    int i;

    estimate = (f(left_endpt) + f(right_endpt))/2.0;
    for (i = 1; i <= trap_count-1; i++) {
        x = left_endpt + i*base_len;
        estimate += f(x);
    }
    estimate = estimate*base_len;

    return estimate;
} 
double f(double x /* in */) {
    return x*x;
} /* f */