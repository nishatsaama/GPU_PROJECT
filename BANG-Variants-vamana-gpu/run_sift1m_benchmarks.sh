#!/bin/bash

# SIFT1M Concurrent FreshDiskANN Benchmarks
# Prerequisites: Run ./run_1m.sh first to set up everything
# Usage: bash run_sift1m_benchmarks.sh

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════╗"
echo "║   SIFT1M Concurrent FreshDiskANN Benchmarks        ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Configuration - Use relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Files
GRAPH_FILE="$SCRIPT_DIR/build/vamana_sift1m_alpha1.2.out"
QUERY_BIN="$SCRIPT_DIR/data/sift1m_query.bin"
GT_BIN="$SCRIPT_DIR/data/sift1m_groundtruth.bin"

# Search parameters
SEARCH_L=20
K=10
WORKERS=8
BATCH=50

cd "$SCRIPT_DIR"

echo "════════════════════════════════════════════════════"
echo "Checking Prerequisites"
echo "════════════════════════════════════════════════════"

# Check if graph file exists
if [ ! -f "$GRAPH_FILE" ]; then
    echo "Error: Graph file not found: $GRAPH_FILE"
    echo "Please run ./run_1m.sh first to set up SIFT1M!"
    exit 1
fi
echo "✓ Graph file found"

# Check if query binary exists
if [ ! -f "$QUERY_BIN" ]; then
    echo "Error: Query binary not found: $QUERY_BIN"
    echo "Please run ./run_1m.sh first to set up SIFT1M!"
    exit 1
fi
echo "✓ Query binary found"

# Check if groundtruth binary exists
if [ ! -f "$GT_BIN" ]; then
    echo "Error: Groundtruth binary not found: $GT_BIN"
    echo "Please run ./run_1m.sh first to set up SIFT1M!"
    exit 1
fi
echo "✓ Groundtruth binary found"

# Check if concurrent executor exists
if [ ! -f "./bin/concurrent_fresh_diskann" ]; then
    echo "Error: concurrent_fresh_diskann not found"
    echo "Please run ./run_1m.sh first to compile the executor!"
    exit 1
fi
echo "✓ Concurrent executor found"

# Check workloads
WORKLOAD_MISSING=0
if [ ! -f "$PROJECT_ROOT/workload_query_only_10k_sift1m.jsonl" ]; then
    echo "Warning: workload_query_only_10k_sift1m.jsonl not found"
    WORKLOAD_MISSING=1
fi
if [ ! -f "$PROJECT_ROOT/workload_query_heavy_sift1m.jsonl" ]; then
    echo "Warning: workload_query_heavy_sift1m.jsonl not found"
    WORKLOAD_MISSING=1
fi
if [ ! -f "$PROJECT_ROOT/workload_balanced_sift1m.jsonl" ]; then
    echo "Warning: workload_balanced_sift1m.jsonl not found"
    WORKLOAD_MISSING=1
fi
if [ ! -f "$PROJECT_ROOT/workload_insert_heavy_sift1m.jsonl" ]; then
    echo "Warning: workload_insert_heavy_sift1m.jsonl not found"
    WORKLOAD_MISSING=1
fi

if [ $WORKLOAD_MISSING -eq 1 ]; then
    echo ""
    echo "Error: Some workload files are missing!"
    echo "Please run ./run_1m.sh first to generate workloads!"
    exit 1
fi
echo "✓ All workload files found"
echo ""

echo "════════════════════════════════════════════════════"
echo "Running Benchmarks with L=$SEARCH_L, k=$K"
echo "════════════════════════════════════════════════════"
echo ""

# Results file
RESULTS_FILE="benchmark_results_sift1m_$(date +%Y%m%d_%H%M%S).txt"

echo "SIFT1M Concurrent FreshDiskANN Benchmark Results" > $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "Configuration: L=$SEARCH_L, k=$K, workers=$WORKERS, batch=$BATCH, N=1,000,000" >> $RESULTS_FILE
echo "=================================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# 1. Query-Only
echo "1. Query-Only (10,000 queries)"
echo "-----------------------------------"
./bin/concurrent_fresh_diskann \
    $GRAPH_FILE \
    $PROJECT_ROOT/workload_query_only_10k_sift1m.jsonl \
    $QUERY_BIN \
    $GT_BIN \
    --searchL $SEARCH_L \
    --k $K \
    --workers $WORKERS \
    --batch $BATCH 2>&1 | tee -a $RESULTS_FILE | grep -E "Insert avg|Delete avg|Query avg|Query QPS|recall@|Throughput"
echo ""

# 2. Query-Heavy
echo "2. Query-Heavy (300I + 200D + 9,500Q)"
echo "-----------------------------------"
./bin/concurrent_fresh_diskann \
    $GRAPH_FILE \
    $PROJECT_ROOT/workload_query_heavy_sift1m.jsonl \
    $QUERY_BIN \
    $GT_BIN \
    --searchL $SEARCH_L \
    --k $K \
    --workers $WORKERS \
    --batch $BATCH 2>&1 | tee -a $RESULTS_FILE | grep -E "Insert avg|Delete avg|Query avg|Query QPS|recall@|Throughput"
echo ""

# 3. Balanced
echo "3. Balanced (2,500I + 1,500D + 6,000Q)"
echo "-----------------------------------"
./bin/concurrent_fresh_diskann \
    $GRAPH_FILE \
    $PROJECT_ROOT/workload_balanced_sift1m.jsonl \
    $QUERY_BIN \
    $GT_BIN \
    --searchL $SEARCH_L \
    --k $K \
    --workers $WORKERS \
    --batch $BATCH 2>&1 | tee -a $RESULTS_FILE | grep -E "Insert avg|Delete avg|Query avg|Query QPS|recall@|Throughput"
echo ""

# 4. Insert-Heavy
echo "4. Insert-Heavy (5,000I + 1,000D + 4,000Q)"
echo "-----------------------------------"
./bin/concurrent_fresh_diskann \
    $GRAPH_FILE \
    $PROJECT_ROOT/workload_insert_heavy_sift1m.jsonl \
    $QUERY_BIN \
    $GT_BIN \
    --searchL $SEARCH_L \
    --k $K \
    --workers $WORKERS \
    --batch $BATCH 2>&1 | tee -a $RESULTS_FILE | grep -E "Insert avg|Delete avg|Query avg|Query QPS|recall@|Throughput"
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
echo "  • Dataset: SIFT1M (1,000,000 points, 128 dimensions)"
echo "  • Graph: $GRAPH_FILE"
echo "  • Parameters: L=$SEARCH_L, k=$K, workers=$WORKERS, batch=$BATCH"
echo "  • Workloads (10,000 ops each):"
echo "    - Query-Only: 10,000 queries"
echo "    - Query-Heavy: 300I + 200D + 9,500Q"
echo "    - Balanced: 2,500I + 1,500D + 6,000Q"
echo "    - Insert-Heavy: 5,000I + 1,000D + 4,000Q"
echo "  • Results: $SCRIPT_DIR/$RESULTS_FILE"
echo ""
