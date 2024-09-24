//
// Created by richard on 9/24/24.
//

/* Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 3
 * of the programming guide with some additions like error checking.
 *
 */

// device code
extern "C" __global__ void VecAdd_kernel(const float* A, const float* B, float* C, int numElems) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElems) {
        C[i] = A[i] + B[i];
    }
}
