#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(void) {
    int my_rank, comm_sz;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(mpi_comm, &comm_sz);
    MPI_Comm_rank(mpi_comm, &my_rank);

    if (my_rank == 0) {
        clock_t tempoInicial, tempoFinal;
        int dadoEnviar = 5;
        int dadoReceber;
        tempoInicial = clock();
        MPI_Send(&dadoEnviar, 1, MPI_INT, 1, 0, mpi_comm);
        MPI_Recv(&dadoReceber, 1, MPI_INT, 1, 0, mpi_comm, MPI_STATUS_IGNORE);
        tempoFinal = clock() - tempoInicial;
        printf("Durou %f milissegundos\n", ((float)tempoFinal)*1000/CLOCKS_PER_SEC);
    } else {
        int dadoReceber;
        int dadoEnviar = 3;
        MPI_Recv(&dadoReceber, 1, MPI_INT, 0, 0, mpi_comm, MPI_STATUS_IGNORE);
        MPI_Send(&dadoEnviar, 1, MPI_INT, 0, 0, mpi_comm);
    }

    MPI_Finalize();
    return 0;
}