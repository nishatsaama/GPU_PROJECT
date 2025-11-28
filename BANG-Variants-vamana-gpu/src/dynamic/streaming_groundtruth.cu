#include "streaming_groundtruth.h"
#include "../vamana.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

/**
 * GPU Kernel: Compute L2 Distance Matrix
 *
 * Computes distances from batchSize queries to numPoints base points
 * Output: distMatrix[query_idx * numPoints + point_idx] = L2_distance
 *
 * Uses shared memory for query caching to reduce global memory traffic
 */
__global__ void computeDistanceMatrixKernel(
    uint8_t* d_graph, float* d_queries, float* d_distMatrix,
    unsigned* d_deleted, unsigned batchSize, unsigned numPoints,
    unsigned dim, unsigned graphEntrySize)
{
    // Each block handles one query
    unsigned queryIdx = blockIdx.x;
    if (queryIdx >= batchSize) return;

    // Thread handles multiple points
    unsigned pointIdx = blockIdx.y * blockDim.x + threadIdx.x;

    // Load query into shared memory for fast access
    extern __shared__ float s_query[];
    for (unsigned d = threadIdx.x; d < dim; d += blockDim.x) {
        s_query[d] = d_queries[queryIdx * dim + d];
    }
    __syncthreads();

    if (pointIdx >= numPoints) return;

    // Check if point is deleted
    if (d_deleted != nullptr && d_deleted[pointIdx] != 0) {
        d_distMatrix[queryIdx * numPoints + pointIdx] = FLT_MAX;
        return;
    }

    // Extract point vector from graph
    float* pointVec = (float*)(d_graph + pointIdx * graphEntrySize);

    // Compute L2 distance
    float dist = 0.0f;
    #pragma unroll 8
    for (unsigned d = 0; d < dim; d++) {
        float diff = s_query[d] - pointVec[d];
        dist += diff * diff;
    }

    d_distMatrix[queryIdx * numPoints + pointIdx] = dist;
}

/**
 * GPU Kernel: Warp-Level Top-K Selection (Optimized for k <= 32)
 *
 * Uses warp shuffle primitives for efficient parallel reduction
 * Each warp processes one query
 */
__global__ void warpTopKKernel(
    float* d_distMatrix, unsigned* d_topkIds, float* d_topkDists,
    unsigned batchSize, unsigned numPoints, unsigned k)
{
    unsigned queryIdx = blockIdx.x;
    if (queryIdx >= batchSize) return;

    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    unsigned laneId = warp.thread_rank();

    // Each query's distance row
    float* queryDists = d_distMatrix + queryIdx * numPoints;

    // Local top-k storage (each thread maintains k candidates)
    float localDists[32];
    unsigned localIds[32];

    for (unsigned i = 0; i < k; i++) {
        localDists[i] = FLT_MAX;
        localIds[i] = 0;
    }

    // Process points in chunks across warp
    for (unsigned i = laneId; i < numPoints; i += 32) {
        float dist = queryDists[i];
        unsigned id = i;

        // Insert into local top-k if better
        if (dist < localDists[k - 1]) {
            // Find insertion position
            int pos = k - 1;
            while (pos > 0 && dist < localDists[pos - 1]) {
                pos--;
            }

            // Shift and insert
            for (int j = k - 1; j > pos; j--) {
                localDists[j] = localDists[j - 1];
                localIds[j] = localIds[j - 1];
            }
            localDists[pos] = dist;
            localIds[pos] = id;
        }
    }

    // Warp-level merge: each thread has k candidates, merge across warp
    // Use shuffle to gather all candidates
    __shared__ float s_mergeDists[32 * 32];  // Max k=32, 32 threads
    __shared__ unsigned s_mergeIds[32 * 32];

    for (unsigned i = 0; i < k; i++) {
        s_mergeDists[laneId * k + i] = localDists[i];
        s_mergeIds[laneId * k + i] = localIds[i];
    }
    __syncwarp();

    // Thread 0 performs final merge
    if (laneId == 0) {
        float finalDists[32];
        unsigned finalIds[32];

        for (unsigned i = 0; i < k; i++) {
            finalDists[i] = FLT_MAX;
            finalIds[i] = 0;
        }

        // Merge all 32 threads' top-k lists
        for (unsigned t = 0; t < 32; t++) {
            for (unsigned i = 0; i < k; i++) {
                float dist = s_mergeDists[t * k + i];
                unsigned id = s_mergeIds[t * k + i];

                if (dist < finalDists[k - 1]) {
                    // Find insertion position
                    int pos = k - 1;
                    while (pos > 0 && dist < finalDists[pos - 1]) {
                        pos--;
                    }

                    // Shift and insert
                    for (int j = k - 1; j > pos; j--) {
                        finalDists[j] = finalDists[j - 1];
                        finalIds[j] = finalIds[j - 1];
                    }
                    finalDists[pos] = dist;
                    finalIds[pos] = id;
                }
            }
        }

        // Write results
        for (unsigned i = 0; i < k; i++) {
            d_topkDists[queryIdx * k + i] = finalDists[i];
            d_topkIds[queryIdx * k + i] = finalIds[i];
        }
    }
}

/**
 * GPU Kernel: Simple Top-K Selection (for larger k)
 *
 * Each block handles one query
 * Uses block-level parallel selection
 */
__global__ void selectTopKSimpleKernel(
    float* d_distMatrix, unsigned* d_topkIds, float* d_topkDists,
    unsigned batchSize, unsigned numPoints, unsigned k)
{
    unsigned queryIdx = blockIdx.x;
    if (queryIdx >= batchSize) return;

    // Shared memory for block-level top-k
    extern __shared__ char s_mem[];
    float* s_topkDists = (float*)s_mem;
    unsigned* s_topkIds = (unsigned*)(s_mem + k * sizeof(float));

    // Initialize shared top-k with infinity
    for (unsigned i = threadIdx.x; i < k; i += blockDim.x) {
        s_topkDists[i] = FLT_MAX;
        s_topkIds[i] = 0;
    }
    __syncthreads();

    // Each thread processes subset of points
    float* queryDists = d_distMatrix + queryIdx * numPoints;

    for (unsigned i = threadIdx.x; i < numPoints; i += blockDim.x) {
        float dist = queryDists[i];

        // Check if this distance should be in top-k
        if (dist < s_topkDists[k - 1]) {
            // Atomically find insertion position and insert
            // Use critical section for simplicity (could optimize with atomic min)
            atomicMin((unsigned*)&s_topkDists[k - 1], __float_as_uint(dist));
            __syncthreads();

            // If our distance was inserted, update ID
            for (unsigned j = 0; j < k; j++) {
                if (s_topkDists[j] == dist) {
                    s_topkIds[j] = i;
                    break;
                }
            }
        }
    }
    __syncthreads();

    // Sort the top-k (simple bubble sort in shared memory)
    for (unsigned i = threadIdx.x; i < k - 1; i += blockDim.x) {
        for (unsigned j = 0; j < k - i - 1; j++) {
            if (s_topkDists[j] > s_topkDists[j + 1]) {
                // Swap
                float tmpD = s_topkDists[j];
                unsigned tmpId = s_topkIds[j];
                s_topkDists[j] = s_topkDists[j + 1];
                s_topkIds[j] = s_topkIds[j + 1];
                s_topkDists[j + 1] = tmpD;
                s_topkIds[j + 1] = tmpId;
            }
        }
        __syncthreads();
    }

    // Write results to global memory
    for (unsigned i = threadIdx.x; i < k; i += blockDim.x) {
        d_topkDists[queryIdx * k + i] = s_topkDists[i];
        d_topkIds[queryIdx * k + i] = s_topkIds[i];
    }
}

/**
 * Optimized: Radix-Select based Top-K (for very large numPoints)
 *
 * Uses parallel radix select algorithm for better performance
 * with large point sets
 */
__global__ void bitonicTopKKernel(
    float* d_distMatrix, unsigned* d_topkIds, float* d_topkDists,
    unsigned batchSize, unsigned numPoints, unsigned k)
{
    // Each block processes one query
    unsigned queryIdx = blockIdx.x;
    if (queryIdx >= batchSize) return;

    float* queryDists = d_distMatrix + queryIdx * numPoints;

    // Allocate shared memory for storing (distance, id) pairs
    extern __shared__ char s_data[];
    float* s_dists = (float*)s_data;
    unsigned* s_ids = (unsigned*)(s_data + blockDim.x * sizeof(float));

    // Each thread loads multiple elements
    unsigned elemsPerThread = (numPoints + blockDim.x - 1) / blockDim.x;

    for (unsigned iter = 0; iter < elemsPerThread; iter++) {
        unsigned idx = threadIdx.x + iter * blockDim.x;
        if (idx < numPoints) {
            s_dists[threadIdx.x] = queryDists[idx];
            s_ids[threadIdx.x] = idx;
        } else {
            s_dists[threadIdx.x] = FLT_MAX;
            s_ids[threadIdx.x] = UINT_MAX;
        }
        __syncthreads();

        // Bitonic sort (simplified - assumes k is power of 2)
        for (unsigned size = 2; size <= blockDim.x; size *= 2) {
            unsigned dir = (threadIdx.x / size) % 2;
            for (unsigned stride = size / 2; stride > 0; stride /= 2) {
                unsigned pos = threadIdx.x;
                unsigned partner = pos ^ stride;

                float dist1 = s_dists[pos];
                float dist2 = s_dists[partner];

                bool swap = (dist1 > dist2) == dir;
                if (swap && partner > pos) {
                    s_dists[pos] = dist2;
                    s_dists[partner] = dist1;
                    unsigned tmpId = s_ids[pos];
                    s_ids[pos] = s_ids[partner];
                    s_ids[partner] = tmpId;
                }
                __syncthreads();
            }
        }
    }

    // Write top-k results
    if (threadIdx.x < k) {
        d_topkDists[queryIdx * k + threadIdx.x] = s_dists[threadIdx.x];
        d_topkIds[queryIdx * k + threadIdx.x] = s_ids[threadIdx.x];
    }
}

// ============== StreamingGroundtruth Implementation ==============

StreamingGroundtruth::StreamingGroundtruth(uint8_t* d_graph, unsigned numPoints,
                                         unsigned dim, unsigned maxBatchSize,
                                         unsigned maxK, unsigned graphEntrySize)
    : d_graph(d_graph), numPoints(numPoints), dim(dim),
      graphEntrySize(graphEntrySize), d_deleted(nullptr)
{
    // If graphEntrySize not provided, calculate it
    if (graphEntrySize == 0) {
        this->graphEntrySize = D * sizeof(float) + (R + 1) * sizeof(unsigned);
    }

    printf("[StreamingGroundtruth] Initializing...\n");
    printf("  Base points: %u, Dim: %u, Max batch: %u, Max k: %u\n",
           numPoints, dim, maxBatchSize, maxK);

    bool success = initializeBuffers(maxBatchSize, maxK);
    if (!success) {
        fprintf(stderr, "[Error] Failed to initialize StreamingGroundtruth buffers\n");
    }

    printf("[StreamingGroundtruth] Initialization complete\n");
}

StreamingGroundtruth::~StreamingGroundtruth() {
    freeBuffers();
}

bool StreamingGroundtruth::initializeBuffers(unsigned maxBatchSize, unsigned maxK) {
    std::lock_guard<std::mutex> lock(bufferMutex);

    if (buffers.initialized) {
        printf("[Warning] Buffers already initialized\n");
        return true;
    }

    buffers.maxBatchSize = maxBatchSize;
    buffers.numBasePoints = numPoints;
    buffers.dim = dim;
    buffers.maxK = maxK;

    // Create CUDA stream for async operations
    cudaStreamCreate(&buffers.stream);

    // Allocate device memory
    cudaMalloc(&buffers.d_queryBatch, maxBatchSize * dim * sizeof(float));
    cudaMalloc(&buffers.d_distanceMatrix, maxBatchSize * numPoints * sizeof(float));
    cudaMalloc(&buffers.d_topkIds, maxBatchSize * maxK * sizeof(unsigned));
    cudaMalloc(&buffers.d_topkDists, maxBatchSize * maxK * sizeof(float));

    // Allocate pinned host memory for fast transfers
    cudaMallocHost(&buffers.h_topkIds, maxBatchSize * maxK * sizeof(unsigned));
    cudaMallocHost(&buffers.h_topkDists, maxBatchSize * maxK * sizeof(float));

    // Check for allocation errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[Error] CUDA allocation failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    buffers.initialized = true;

    printf("  ✓ Allocated GPU buffers:\n");
    printf("    - Query batch: %.2f MB\n",
           (maxBatchSize * dim * sizeof(float)) / 1024.0 / 1024.0);
    printf("    - Distance matrix: %.2f MB\n",
           (maxBatchSize * numPoints * sizeof(float)) / 1024.0 / 1024.0);
    printf("    - Results: %.2f MB\n",
           (2 * maxBatchSize * maxK * sizeof(float)) / 1024.0 / 1024.0);

    return true;
}

void StreamingGroundtruth::freeBuffers() {
    std::lock_guard<std::mutex> lock(bufferMutex);

    if (!buffers.initialized) return;

    cudaFree(buffers.d_queryBatch);
    cudaFree(buffers.d_distanceMatrix);
    cudaFree(buffers.d_topkIds);
    cudaFree(buffers.d_topkDists);

    cudaFreeHost(buffers.h_topkIds);
    cudaFreeHost(buffers.h_topkDists);

    if (buffers.d_temp_storage) {
        cudaFree(buffers.d_temp_storage);
    }

    cudaStreamDestroy(buffers.stream);

    buffers.initialized = false;
}

void StreamingGroundtruth::computeDistanceMatrixGPU(
    float* d_queries, unsigned batchSize, float* d_distMatrix,
    unsigned* d_deleted, cudaStream_t stream)
{
    // Grid: (batchSize, numBlocks_for_points)
    // Each block handles one query, threads process points
    unsigned threadsPerBlock = GT_THREADS_PER_BLOCK;
    unsigned pointBlocksPerQuery = (numPoints + threadsPerBlock - 1) / threadsPerBlock;

    dim3 grid(batchSize, pointBlocksPerQuery);
    dim3 block(threadsPerBlock);

    unsigned sharedMemSize = dim * sizeof(float);  // Cache query in shared memory

    computeDistanceMatrixKernel<<<grid, block, sharedMemSize, stream>>>(
        d_graph, d_queries, d_distMatrix, d_deleted,
        batchSize, numPoints, dim, graphEntrySize
    );
}

void StreamingGroundtruth::selectTopKGPU(
    float* d_distMatrix, unsigned batchSize, unsigned* d_topkIds,
    float* d_topkDists, unsigned k, cudaStream_t stream)
{
    if (k <= 32) {
        // Use optimized warp-level kernel
        unsigned numBlocks = batchSize;
        unsigned threadsPerBlock = 32;  // One warp per query

        warpTopKKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
            d_distMatrix, d_topkIds, d_topkDists, batchSize, numPoints, k
        );
    } else {
        // Use simple selection kernel for larger k
        unsigned numBlocks = batchSize;
        unsigned threadsPerBlock = 256;
        unsigned sharedMemSize = k * (sizeof(float) + sizeof(unsigned));

        selectTopKSimpleKernel<<<numBlocks, threadsPerBlock, sharedMemSize, stream>>>(
            d_distMatrix, d_topkIds, d_topkDists, batchSize, numPoints, k
        );
    }
}

bool StreamingGroundtruth::computeSingleQuery(
    float* d_queryVec, unsigned* h_topkIds, float* h_topkDists, unsigned k,
    unsigned* d_deleted, cudaStream_t stream)
{
    if (!buffers.initialized) {
        fprintf(stderr, "[Error] Buffers not initialized\n");
        return false;
    }

    if (k > buffers.maxK) {
        fprintf(stderr, "[Error] k=%u exceeds maxK=%u\n", k, buffers.maxK);
        return false;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Use provided stream or default stream
    cudaStream_t execStream = (stream != 0) ? stream : buffers.stream;

    // Use delete bitvector if provided or use stored one
    unsigned* deleteBV = (d_deleted != nullptr) ? d_deleted : this->d_deleted;

    // Compute distance matrix (1 query vs all points)
    computeDistanceMatrixGPU(d_queryVec, 1, buffers.d_distanceMatrix,
                            deleteBV, execStream);

    // Select top-k
    selectTopKGPU(buffers.d_distanceMatrix, 1, buffers.d_topkIds,
                 buffers.d_topkDists, k, execStream);

    // Copy results to host
    cudaMemcpyAsync(h_topkIds, buffers.d_topkIds, k * sizeof(unsigned),
                   cudaMemcpyDeviceToHost, execStream);
    cudaMemcpyAsync(h_topkDists, buffers.d_topkDists, k * sizeof(float),
                   cudaMemcpyDeviceToHost, execStream);

    // Synchronize if using default stream
    if (stream == 0) {
        cudaStreamSynchronize(execStream);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double timeMs = std::chrono::duration<double, std::milli>(end - start).count();

    // Update statistics
    queriesProcessed++;
    batchesProcessed++;
    double oldTime = totalComputeTimeMs.load();
    while (!totalComputeTimeMs.compare_exchange_weak(oldTime, oldTime + timeMs));

    return true;
}

bool StreamingGroundtruth::computeBatchQueries(
    float* d_queryBatch, unsigned batchSize, unsigned* h_topkIds,
    float* h_topkDists, unsigned k, unsigned* d_deleted, cudaStream_t stream)
{
    if (!buffers.initialized) {
        fprintf(stderr, "[Error] Buffers not initialized\n");
        return false;
    }

    if (batchSize > buffers.maxBatchSize) {
        fprintf(stderr, "[Error] batchSize=%u exceeds maxBatchSize=%u\n",
                batchSize, buffers.maxBatchSize);
        return false;
    }

    if (k > buffers.maxK) {
        fprintf(stderr, "[Error] k=%u exceeds maxK=%u\n", k, buffers.maxK);
        return false;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Use provided stream or default stream
    cudaStream_t execStream = (stream != 0) ? stream : buffers.stream;

    // Use delete bitvector if provided or use stored one
    unsigned* deleteBV = (d_deleted != nullptr) ? d_deleted : this->d_deleted;

    // Compute distance matrix (batchSize queries vs all points)
    computeDistanceMatrixGPU(d_queryBatch, batchSize, buffers.d_distanceMatrix,
                            deleteBV, execStream);

    // Select top-k for each query
    selectTopKGPU(buffers.d_distanceMatrix, batchSize, buffers.d_topkIds,
                 buffers.d_topkDists, k, execStream);

    // Copy results to host
    size_t resultSize = batchSize * k * sizeof(unsigned);
    cudaMemcpyAsync(h_topkIds, buffers.d_topkIds, resultSize,
                   cudaMemcpyDeviceToHost, execStream);
    cudaMemcpyAsync(h_topkDists, buffers.d_topkDists, batchSize * k * sizeof(float),
                   cudaMemcpyDeviceToHost, execStream);

    // Synchronize if using default stream
    if (stream == 0) {
        cudaStreamSynchronize(execStream);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double timeMs = std::chrono::duration<double, std::milli>(end - start).count();

    // Update statistics
    queriesProcessed += batchSize;
    batchesProcessed++;
    double oldTime = totalComputeTimeMs.load();
    while (!totalComputeTimeMs.compare_exchange_weak(oldTime, oldTime + timeMs));

    return true;
}

void StreamingGroundtruth::updateBasePoints(uint8_t* d_newGraph, unsigned newNumPoints) {
    std::lock_guard<std::mutex> lock(bufferMutex);

    d_graph = d_newGraph;
    numPoints = newNumPoints;

    // If number of points increased significantly, may need to reallocate distance matrix
    if (newNumPoints > buffers.numBasePoints * 1.5) {
        printf("[Info] Base points increased significantly (%u -> %u), "
               "consider reallocating buffers\n",
               buffers.numBasePoints, newNumPoints);
        // Could implement automatic reallocation here
    }

    buffers.numBasePoints = newNumPoints;
}

void StreamingGroundtruth::setDeleteBitvector(unsigned* d_deleted) {
    this->d_deleted = d_deleted;
}

void StreamingGroundtruth::printStatistics() const {
    printf("\n=== Streaming Groundtruth Statistics ===\n");
    printf("Queries processed: %llu\n", queriesProcessed.load());
    printf("Batches processed: %llu\n", batchesProcessed.load());
    printf("Total compute time: %.2f ms\n", totalComputeTimeMs.load());

    unsigned long long queries = queriesProcessed.load();
    if (queries > 0) {
        printf("Avg time per query: %.3f ms\n", totalComputeTimeMs.load() / queries);
        printf("Throughput: %.1f queries/sec\n",
               queries / (totalComputeTimeMs.load() / 1000.0));
    }

    printf("Buffer utilization:\n");
    printf("  Max batch size: %u\n", buffers.maxBatchSize);
    printf("  Max k: %u\n", buffers.maxK);
    printf("  Base points: %u\n", buffers.numBasePoints);
    printf("=====================================\n\n");
}
