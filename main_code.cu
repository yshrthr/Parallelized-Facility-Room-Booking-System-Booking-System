	#include <iostream>
	#include <stdio.h>
	#include <cuda.h>

	#define max_N 100000
	#define max_P 30
	#define BLOCKSIZE 1024
  	#define VAL 100000
  	#define LARGE_VAL 1000000

	using namespace std;


//storing VAL in lock array
	__global__ void get_lock(int* lock, int start,int end)
	{
    int tid = threadIdx.x; 
    int bid = blockIdx.x * blockDim.x + start;    
    		
    if((tid + bid)  < end)
		lock[tid+bid] = VAL;
	}

  
//Kernel to fill slots array 
__global__ void fill_slots(int second, int first, int end, int *d_slots, int* d_capacity)
{
    int tid = threadIdx.x; 
    int bid = blockIdx.x * blockDim.x + second; 
    int id = tid+bid; 

    if(id<end)
      d_slots[id] = d_capacity[first+(id-second)/24]; 
}


//Main kernel to find total successful requests and successfull requests in each centre
__global__ void count_success_req(int *d_succ_reqs, int *d_succ, int begin, int R,int first, int second, int *d_offset,int c,int *d_room_offset,bool check, int *d_lock, int volatile *d_slots, int *d_req_cen, int *d_req_fac, int *d_req_start, int *d_req_slots) 
{
    int id = blockIdx.x *blockDim.x + threadIdx.x;
    
    bool flag = false, execute = true;
    int stop;
    __shared__ bool done;
    id += begin;
    done = true;

    int slot_offset = d_offset[d_req_cen[id]]+d_req_fac[id]*(c+1)*24;
    int room_offset = d_room_offset[d_req_cen[id]]+d_req_fac[id];

    __syncthreads();

    int x = d_req_start[id]-first;
    int y = d_req_start[id]+d_req_slots[id] - second;
    

    while(check) 
    {  
        if (execute != false) 
          atomicMin(&d_lock[room_offset], id);

        else check = true;
        
        done = true;

        __syncthreads();

        if (execute != false) { 
            if (d_lock[room_offset] == id) {
                int begin = slot_offset + x;
                int end = slot_offset + y;
                int temp = slot_offset/24;

                for (int i = begin; i <= end; i++){
                    atomicSub((unsigned int*)&d_slots[i], 1);
                    if (d_slots[i] < c) {
                        flag = true;
                        stop = i;
                        break;
                    }
                }
                

                if (flag == true) {
                      int end2 = stop;

                      while(begin<=end2) {
                        atomicAdd((unsigned int*)&d_slots[begin], 1);
                        begin++;
                      }
                    temp++;
                    flag = false;
                }
                
                else {
                    atomicInc((unsigned int*) &d_succ[0], 100001);
                    atomicInc((unsigned int*) &d_succ_reqs[d_req_cen[id]], 100001);
                }
                
                d_lock[room_offset] = LARGE_VAL;
                execute = false;
            } 

            else if (execute != false) 
             done = false;

            else check = true; 

        }
        __syncthreads();

        if(done)
        {
          check = false;
        } 
    } 
}

//Boiler plate code begins

int main(int argc,char **argv)
{
	// variable declarations...
    int N,*centre,*facility,*capacity,*fac_ids, *succ_reqs, *tot_reqs;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &N ); // N is number of centres
	
    // Allocate memory on cpu
    centre=(int*)malloc(N * sizeof (int));  // Computer  centre numbers
    facility=(int*)malloc(N * sizeof (int));  // Number of facilities in each computer centre
    fac_ids=(int*)malloc(max_P * N  * sizeof (int));  // Facility room numbers of each computer centre
    capacity=(int*)malloc(max_P * N * sizeof (int));  // stores capacities of each facility for every computer centre 


    int success=0;  // total successful requests
    int fail = 0;   // total failed requests
    tot_reqs = (int *)malloc(N*sizeof(int));   // total requests for each centre
    succ_reqs = (int *)malloc(N*sizeof(int)); // total successful requests for each centre

    // Input the computer centres data
    int k1=0 , k2 = 0;
    for(int i=0;i<N;i++)
    {
      fscanf( inputfilepointer, "%d", &centre[i] );
      fscanf( inputfilepointer, "%d", &facility[i] );
      
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &fac_ids[k1] );
        k1++;
      }
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &capacity[k2]);
        k2++;     
      }
    }

    // variable declarations
    int *req_id, *req_cen, *req_fac, *req_start, *req_slots;   // Number of slots requested for every request
    
    // Allocate memory on CPU 
	  int R;
	  fscanf( inputfilepointer, "%d", &R); // Total requests
    req_id = (int *) malloc ((R)*sizeof (int));  // Request ids
    req_cen = (int *) malloc ((R)*sizeof (int));  // Requested computer centre
    req_fac = (int *) malloc ((R)*sizeof (int));  // Requested facility
    req_start = (int *) malloc((R)*sizeof (int));  // Start slot of every request
    req_slots = (int *) malloc((R)*sizeof (int));   // Number of slots requested for every request
    
    // Input the user request data
    for(int j = 0; j < R; j++)
    {
       fscanf( inputfilepointer, "%d", &req_id[j]);
       fscanf( inputfilepointer, "%d", &req_cen[j]);
       fscanf( inputfilepointer, "%d", &req_fac[j]);
       fscanf( inputfilepointer, "%d", &req_start[j]);
       fscanf( inputfilepointer, "%d", &req_slots[j]);
       tot_reqs[req_cen[j]]+=1;  
    }
		
//Boiler plate code ends-------------------------------------------
//Kernel calling code begins

    int v1=0, v2=0;

    //Arrays on host 
  	int *offset = (int*)malloc(N * sizeof (int));
	int *room_offset=(int*)malloc(N * sizeof (int));

    //Declaring device arrays
    int *d_offset, *d_total_succ, *d_succ_reqs, *d_room_offset,*d_lock,*d_req_start, *d_req_cen, *d_req_fac, *d_req_slots;

		// Allocating memory on gpu
    cudaMalloc(&d_lock, k1 * sizeof(int));
    cudaMalloc(&d_total_succ, sizeof(int));
	cudaMalloc(&d_req_cen,R * sizeof(int));
	cudaMalloc(&d_req_fac,R * sizeof(int));
	cudaMalloc(&d_req_start,R * sizeof(int));
	cudaMalloc(&d_req_slots,R * sizeof(int));
	cudaMalloc(&d_succ_reqs,N * sizeof(int));
    cudaMalloc(&d_room_offset, N * sizeof(int));
	cudaMalloc(&d_offset, N*sizeof(int));

    //Device array to keep track of capicity
    int *d_capacity;
    cudaMalloc(&d_capacity, max_P*N * sizeof(int));
	cudaMemcpy(d_capacity, capacity, k1 * sizeof(int), cudaMemcpyHostToDevice);

    //Device array to keep track of empty slots
    int *d_slots;
    cudaMalloc(&d_slots,max_P*N *24* sizeof(int));


    for (int i = 0; i < N; i++) {

      int num_facility = facility[i];
      int end = v2 + 24 * num_facility;

      room_offset[i] = v1;
      offset[i] = v2;

      int nb = ceil(float(end - v2+1)/BLOCKSIZE);

      fill_slots<<<nb,BLOCKSIZE>>>(v2, v1, end, d_slots, d_capacity);
      cudaDeviceSynchronize();

      v1 += num_facility;
      v2 = end;
    }


	cudaMemcpy(d_offset, offset, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_room_offset, room_offset, k1*24 * sizeof(int), cudaMemcpyHostToDevice);

    //function for getting lock
	int num_blocks = ceil((float(k1)/BLOCKSIZE));
    int end = k1;
	get_lock<<<num_blocks,BLOCKSIZE>>>(d_lock,0, end);	

      //This host array stores total successful requests
    int *h_total_succ;
    h_total_succ = (int*)malloc(sizeof (int));

			//Host to device copy
	cudaMemcpy(d_req_fac, req_fac, R * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_cen, req_cen, R*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_req_slots,req_slots, R * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemset(d_total_succ, 0, sizeof(int));
	cudaMemcpy(d_req_start,req_start, R * sizeof(int),cudaMemcpyHostToDevice);		
	cudaMemcpy(d_succ_reqs, succ_reqs, N* sizeof(int), cudaMemcpyHostToDevice);
			
    bool check = true; 
    int nblocks = R/BLOCKSIZE, remaining = R % BLOCKSIZE;
    int total_blocks = 0, blocks = 1;

	//Calling kernel for processing requests in parallel in a batch of 1024
	while(total_blocks<nblocks) {	
    int begin = total_blocks*BLOCKSIZE;
	count_success_req<<<blocks,BLOCKSIZE>>>(d_succ_reqs,d_total_succ, begin, R, 1, 2, d_offset,0, d_room_offset, check,d_lock, d_slots, d_req_cen, d_req_fac, d_req_start, d_req_slots );
	cudaDeviceSynchronize();
	total_blocks++;
	}

    //Calling kernel if any remaining requests
    if(remaining !=0){
    	int begin = BLOCKSIZE*nblocks;
		count_success_req<<<blocks,remaining>>>(d_succ_reqs,d_total_succ, begin, R,1,2, d_offset,0, d_room_offset, check,d_lock, d_slots, d_req_cen, d_req_fac, d_req_start, d_req_slots);
    }
      
    //get back the result
	cudaMemcpy(succ_reqs, d_succ_reqs, N* sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_total_succ, d_total_succ, sizeof(int), cudaMemcpyDeviceToHost);
      
    success = h_total_succ[0];
    fail = R-success;
   //printf("%d %d\n", success, fail);
  //kernel calling and computation ends-------------------
    // Output
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    fprintf( outputfilepointer, "%d %d\n", success, fail);
    for(int j = 0; j < N; j++)
    {
        fprintf( outputfilepointer, "%d %d\n", succ_reqs[j], tot_reqs[j]-succ_reqs[j]);
    }
    fclose( inputfilepointer );
    fclose( outputfilepointer );
    cudaDeviceSynchronize();
	return 0;
}
