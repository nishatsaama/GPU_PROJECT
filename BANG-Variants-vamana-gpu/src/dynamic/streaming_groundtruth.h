#ifndef STREAMING_GROUNDTRUTH_H
#define STREAMING_GROUNDTRUTH_H

#include <cuda_runtime.h>
#include <vector>
#include <atomic>
#include <mutex>

/**
 * GPU-Accelerated Streaming Groundtruth Computation
 *
 * Features:
 * - GPU-based distance computation using cuBLAS
 * - GPU-based top-k selection using bitonic sort + reduction
 * - Batch processing for multiple queries simultaneously
 * - Handles dynamic insertions/deletions via delete bitvector
 * - Streaming mode: compute groundtruth on-demand for new queries
 * - Memory efficient: reuses pre-allocated buffers
 */

// Configuration
#define GT_BATCH_SIZE 512        // Process up to 512 queries in parallel
#define MAX_GT_K 100            // Maximum k for groundtruth
#define GT_THREADS_PER_BLOCK 256

// Result structure for groundtruth computation
struct GroundtruthResult {
    unsigned* h_ids;      // Host: nearest neighbor IDs (batch_size * k)
    float* h_distances;   // Host: distances (batch_size * k)
    unsigned batchSize;   // Number of queries in batch
    unsigned k;           // Number of nearest neighbors

    GroundtruthResult() : h_ids(nullptr), h_distances(nullptr),
                         batchSize(0), k(0) {}
};

// Pre-allocated GPU buffers for streaming groundtruth computation
struct StreamingGTBuffers {
    // Device buffers for batch processing
    float* d_queryBatch;           // Query vectors (batch_size * dim)
    float* d_basePoints;           // Base points (num_points * dim)
    float* d_distanceMatrix;       // Distance matrix (batch_size * num_points)
    unsigned* d_topkIds;           // Top-k IDs per query (batch_size * k)
    float* d_topkDists;            // Top-k distances per query (batch_size * k)
    unsigned* d_deleted;           // Delete bitvector (shared with executor)

    // Temporary buffers for radix select
    void* d_temp_storage;          // CUB temporary storage
    size_t temp_storage_bytes;

    // Host pinned memory for results
    unsigned* h_topkIds;
    float* h_topkDists;

    // Dimensions
    unsigned maxBatchSize;
    unsigned numBasePoints;
    unsigned dim;
    unsigned maxK;

    // CUDA stream for async operations
    cudaStream_t stream;

    // Flags
    bool initialized;
    std::atomic<bool> inUse{false};

    StreamingGTBuffers() : d_queryBatch(nullptr), d_basePoints(nullptr),
                          d_distanceMatrix(nullptr), d_topkIds(nullptr),
                          d_topkDists(nullptr), d_deleted(nullptr),
                          d_temp_storage(nullptr), temp_storage_bytes(0),
                          h_topkIds(nullptr), h_topkDists(nullptr),
                          maxBatchSize(0), numBasePoints(0), dim(0), maxK(0),
                          initialized(false) {}
};

/**
 * StreamingGroundtruth: GPU-accelerated groundtruth computation
 * for dynamic ANN index with streaming queries
 */
class StreamingGroundtruth {
public:
    /**
     * Constructor
     * @param d_graph: Device pointer to graph structure
     * @param numPoints: Number of base points in index
     * @param dim: Dimensionality of vectors
     * @param maxBatchSize: Maximum queries to process in one batch
     * @param maxK: Maximum k value for groundtruth
     * @param graphEntrySize: Size of each graph entry in bytes
     */
    StreamingGroundtruth(uint8_t* d_graph, unsigned numPoints, unsigned dim,
                        unsigned maxBatchSize = GT_BATCH_SIZE,
                        unsigned maxK = MAX_GT_K,
                        unsigned graphEntrySize = 0);

    ~StreamingGroundtruth();

    /**
     * Compute groundtruth for a single query (on-demand)
     * @param d_queryVec: Device pointer to query vector
     * @param h_topkIds: Host output buffer for top-k IDs (size >= k)
     * @param h_topkDists: Host output buffer for top-k distances (size >= k)
     * @param k: Number of nearest neighbors to find
     * @param d_deleted: Device pointer to delete bitvector (optional)
     * @param stream: CUDA stream for async execution (optional)
     * @return true if successful
     */
    bool computeSingleQuery(float* d_queryVec, unsigned* h_topkIds,
                           float* h_topkDists, unsigned k,
                           unsigned* d_deleted = nullptr,
                           cudaStream_t stream = 0);

    /**
     * Compute groundtruth for a batch of queries
     * @param d_queryBatch: Device pointer to query vectors (batchSize * dim)
     * @param batchSize: Number of queries in batch
     * @param h_topkIds: Host output buffer for top-k IDs (batchSize * k)
     * @param h_topkDists: Host output buffer for top-k distances (batchSize * k)
     * @param k: Number of nearest neighbors to find per query
     * @param d_deleted: Device pointer to delete bitvector (optional)
     * @param stream: CUDA stream for async execution (optional)
     * @return true if successful
     */
    bool computeBatchQueries(float* d_queryBatch, unsigned batchSize,
                            unsigned* h_topkIds, float* h_topkDists, unsigned k,
                            unsigned* d_deleted = nullptr,
                            cudaStream_t stream = 0);

    /**
     * Update base points after insertion
     * @param d_newGraph: Updated graph pointer
     * @param newNumPoints: New number of points
     * Note: This is lightweight - just updates pointer and count
     */
    void updateBasePoints(uint8_t* d_newGraph, unsigned newNumPoints);

    /**
     * Set delete bitvector pointer (shared with executor)
     * @param d_deleted: Device pointer to delete bitvector
     */
    void setDeleteBitvector(unsigned* d_deleted);

    /**
     * Get buffer usage statistics
     */
    void printStatistics() const;

private:
    // GPU graph data
    uint8_t* d_graph;
    unsigned numPoints;
    unsigned dim;
    unsigned graphEntrySize;

    // Delete bitvector (shared, not owned)
    unsigned* d_deleted;

    // Pre-allocated buffers for streaming computation
    StreamingGTBuffers buffers;
    std::mutex bufferMutex;

    // Statistics
    std::atomic<unsigned long long> queriesProcessed{0};
    std::atomic<unsigned long long> batchesProcessed{0};
    std::atomic<double> totalComputeTimeMs{0.0};

    // Internal helper functions
    bool initializeBuffers(unsigned maxBatchSize, unsigned maxK);
    void freeBuffers();

    /**
     * GPU kernel wrappers
     */
    void computeDistanceMatrixGPU(float* d_queries, unsigned batchSize,
                                 float* d_distMatrix, unsigned* d_deleted,
                                 cudaStream_t stream);

    void selectTopKGPU(float* d_distMatrix, unsigned batchSize,
                      unsigned* d_topkIds, float* d_topkDists, unsigned k,
                      cudaStream_t stream);
};

/**
 * GPU Kernel Declarations
 */

// Compute L2 distance matrix: queries (batchSize x dim) vs base (numPoints x dim)
// Output: distMatrix (batchSize x numPoints)
__global__ void computeDistanceMatrixKernel(
    uint8_t* d_graph, float* d_queries, float* d_distMatrix,
    unsigned* d_deleted, unsigned batchSize, unsigned numPoints,
    unsigned dim, unsigned graphEntrySize);

// Extract top-k smallest distances per query using bitonic sort
__global__ void bitonicTopKKernel(
    float* d_distMatrix, unsigned* d_topkIds, float* d_topkDists,
    unsigned batchSize, unsigned numPoints, unsigned k);

// Fallback: Simple selection for small k
__global__ void selectTopKSimpleKernel(
    float* d_distMatrix, unsigned* d_topkIds, float* d_topkDists,
    unsigned batchSize, unsigned numPoints, unsigned k);

// Optimized: Use warp-level primitives for k <= 32
__global__ void warpTopKKernel(
    float* d_distMatrix, unsigned* d_topkIds, float* d_topkDists,
    unsigned batchSize, unsigned numPoints, unsigned k);

/**
 * Utility Functions
 */

// Calculate recall between groundtruth and results
inline double calculateRecallGT(unsigned* groundtruth, unsigned* results,
                               unsigned gtK, unsigned k, unsigned recallK) {
    unsigned matches = 0;
    for (unsigned i = 0; i < recallK && i < k; i++) {
        for (unsigned j = 0; j < recallK && j < gtK; j++) {
            if (results[i] == groundtruth[j]) {
                matches++;
                break;
            }
        }
    }
    return (100.0 * matches) / recallK;
}

#endif // STREAMING_GROUNDTRUTH_H
