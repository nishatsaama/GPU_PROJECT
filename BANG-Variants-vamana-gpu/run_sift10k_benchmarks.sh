#!/bin/bash

# SIFT10K Concurrent FreshDiskANN Benchmarks
# Prerequisites: Run ./run_10k.sh first to set up everything
# Usage: bash run_sift10k_benchmarks.sh

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════╗"
echo "║   SIFT10K Concurrent FreshDiskANN Benchmarks       ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Configuration - Use relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Files
GRAPH_FILE="$SCRIPT_DIR/build/vamana_alpha1.2.out"
QUERY_FILE="$SCRIPT_DIR/data/siftsmall_query.bin"
GT_FILE="$SCRIPT_DIR/data/siftsmall_groundtruth.bin"

# Search parameters
SEARCH_L=20
K=10
ALPHA=1.2
CONSOLIDATE_THRESH=5.0
DYNAMIC_GT="--dynamic-gt"  # Enable dynamic groundtruth for accurate recall

cd "$SCRIPT_DIR"

echo "════════════════════════════════════════════════════"
echo "Checking Prerequisites"
echo "════════════════════════════════════════════════════"

# Check if graph file exists
if [ ! -f "$GRAPH_FILE" ]; then
    echo "Error: Graph file not found: $GRAPH_FILE"
    echo "Please run ./run_10k.sh first to set up SIFT10K!"
    exit 1
fi
echo "✓ Graph file found"

# Check if query file exists
if [ ! -f "$QUERY_FILE" ]; then
    echo "Error: Query file not found: $QUERY_FILE"
    echo "Please run ./run_10k.sh first to set up SIFT10K!"
    exit 1
fi
echo "✓ Query file found"

# Check if groundtruth file exists
if [ ! -f "$GT_FILE" ]; then
    echo "Error: Groundtruth file not found: $GT_FILE"
    echo "Please run ./run_10k.sh first to set up SIFT10K!"
    exit 1
fi
echo "✓ Groundtruth file found"

# Check if concurrent executor exists
if [ ! -f "./bin/concurrent_fresh_diskann" ]; then
    echo "Error: concurrent_fresh_diskann not found"
    echo "Please run ./run_10k.sh first to compile the executor!"
    exit 1
fi
echo "✓ Concurrent executor found"

# Check workloads - use test directory workloads
WORKLOAD_DIR="$SCRIPT_DIR/test"
WORKLOAD_MISSING=0

if [ ! -f "$WORKLOAD_DIR/workload_query_only_400.jsonl" ]; then
    echo "Warning: test/workload_query_only_400.jsonl not found"
    WORKLOAD_MISSING=1
fi
if [ ! -f "$WORKLOAD_DIR/workload_query_heavy_400.jsonl" ]; then
    echo "Warning: test/workload_query_heavy_400.jsonl not found"
    WORKLOAD_MISSING=1
fi
if [ ! -f "$WORKLOAD_DIR/workload_balanced_400.jsonl" ]; then
    echo "Warning: test/workload_balanced_400.jsonl not found"
    WORKLOAD_MISSING=1
fi
if [ ! -f "$WORKLOAD_DIR/workload_insert_heavy_400.jsonl" ]; then
    echo "Warning: test/workload_insert_heavy_400.jsonl not found"
    WORKLOAD_MISSING=1
fi

if [ $WORKLOAD_MISSING -eq 1 ]; then
    echo ""
    echo "Error: Some workload files are missing in test directory!"
    echo "Please ensure workload files exist in $WORKLOAD_DIR"
    exit 1
fi
echo "✓ All workload files found in test/"
echo ""

echo "════════════════════════════════════════════════════"
echo "Running Benchmarks with L=$SEARCH_L, k=$K"
echo "════════════════════════════════════════════════════"
echo ""

# Results file
RESULTS_FILE="benchmark_results_$(date +%Y%m%d_%H%M%S).txt"

echo "SIFT10K Concurrent FreshDiskANN Benchmark Results" > $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "Configuration: L=$SEARCH_L, k=$K, α=$ALPHA" >> $RESULTS_FILE
echo "Dynamic Groundtruth: ENABLED (accurate streaming recall)" >> $RESULTS_FILE
echo "=================================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# 1. Query-Only
echo "1. Query-Only (400 queries)"
echo "-----------------------------------"
./bin/concurrent_fresh_diskann \
    $GRAPH_FILE \
    $WORKLOAD_DIR/workload_query_only_400.jsonl \
    $QUERY_FILE \
    $GT_FILE \
    --searchL $SEARCH_L \
    --k $K \
    --alpha $ALPHA \
    --thresh $CONSOLIDATE_THRESH \
    $DYNAMIC_GT 2>&1 | tee -a $RESULTS_FILE | grep -E "Insert avg|Delete avg|Query avg|Query QPS|recall@|Throughput|Queries processed"
echo ""

# 2. Query-Heavy
echo "2. Query-Heavy (12I + 8D + 380Q)"
echo "-----------------------------------"
./bin/concurrent_fresh_diskann \
    $GRAPH_FILE \
    $WORKLOAD_DIR/workload_query_heavy_400.jsonl \
    $QUERY_FILE \
    $GT_FILE \
    --searchL $SEARCH_L \
    --k $K \
    --alpha $ALPHA \
    --thresh $CONSOLIDATE_THRESH \
    $DYNAMIC_GT 2>&1 | tee -a $RESULTS_FILE | grep -E "Insert avg|Delete avg|Query avg|Query QPS|recall@|Throughput|Queries processed"
echo ""

# 3. Balanced
echo "3. Balanced (100I + 60D + 240Q)"
echo "-----------------------------------"
./bin/concurrent_fresh_diskann \
    $GRAPH_FILE \
    $WORKLOAD_DIR/workload_balanced_400.jsonl \
    $QUERY_FILE \
    $GT_FILE \
    --searchL $SEARCH_L \
    --k $K \
    --alpha $ALPHA \
    --thresh $CONSOLIDATE_THRESH \
    $DYNAMIC_GT 2>&1 | tee -a $RESULTS_FILE | grep -E "Insert avg|Delete avg|Query avg|Query QPS|recall@|Throughput|Queries processed"
echo ""

# 4. Insert-Heavy
echo "4. Insert-Heavy (200I + 40D + 160Q)"
echo "-----------------------------------"
./bin/concurrent_fresh_diskann \
    $GRAPH_FILE \
    $WORKLOAD_DIR/workload_insert_heavy_400.jsonl \
    $QUERY_FILE \
    $GT_FILE \
    --searchL $SEARCH_L \
    --k $K \
    --alpha $ALPHA \
    --thresh $CONSOLIDATE_THRESH \
    $DYNAMIC_GT 2>&1 | tee -a $RESULTS_FILE | grep -E "Insert avg|Delete avg|Query avg|Query QPS|recall@|Throughput|Queries processed"
echo ""

echo "════════════════════════════════════════════════════"
echo "✓ All benchmarks complete!"
echo "Results saved to: $RESULTS_FILE"
echo "════════════════════════════════════════════════════"

# Display GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo ""
echo "Summary:"
echo "  • Dataset: SIFT10K (10,000 points, 128 dimensions)"
echo "  • Graph: $GRAPH_FILE"
echo "  • Workloads (400 ops each):"
echo "    - Query-Only: 400 queries"
echo "    - Query-Heavy: 12I + 8D + 380Q"
echo "    - Balanced: 100I + 60D + 240Q"
echo "    - Insert-Heavy: 200I + 40D + 160Q"
echo "  • Results: $SCRIPT_DIR/$RESULTS_FILE"
echo ""
