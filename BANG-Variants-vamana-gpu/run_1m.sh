#!/bin/bash

# SIFT1M Setup and Prerequisites Script
# Run this BEFORE running run_sift1m_benchmarks.sh
# Usage: bash run_1m.sh

set -e  # Exit on error


echo "║   SIFT1M Setup - Prerequisites & Preparation       ║"

echo ""

# Configuration - Use relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$SCRIPT_DIR/data"
BUILD_DIR="$SCRIPT_DIR/build"

# SIFT1M Files
BASE_DATA_FVECS="$PROJECT_ROOT/data/sift/sift_base.fvecs"
QUERY_FVECS="$PROJECT_ROOT/data/sift/sift_query.fvecs"
GT_IVECS="$PROJECT_ROOT/data/sift/sift_groundtruth.ivecs"

# Converted binary files
BASE_DATA_BIN="$DATA_DIR/sift1m_base.bin"
QUERY_BIN="$DATA_DIR/sift1m_query.bin"
GT_BIN="$DATA_DIR/sift1m_groundtruth.bin"
RANDOM_GRAPH="$DATA_DIR/sift1m_randomgraph.bin"

# Graph output
GRAPH_FILE="$BUILD_DIR/vamana_sift1m_alpha1.2.out"

cd "$SCRIPT_DIR"


echo "Step 1: Verify SIFT1M Source Data"


# Check if SIFT1M source data exists
echo "Checking SIFT1M source data..."
if [ ! -f "$BASE_DATA_FVECS" ]; then
    echo "Error: SIFT1M base data not found at $BASE_DATA_FVECS"
    exit 1
fi
echo " SIFT1M base data found"

if [ ! -f "$QUERY_FVECS" ]; then
    echo "Error: SIFT1M query data not found at $QUERY_FVECS"
    exit 1
fi
echo " SIFT1M query data found"

if [ ! -f "$GT_IVECS" ]; then
    echo "Error: SIFT1M groundtruth not found at $GT_IVECS"
    exit 1
fi
echo " SIFT1M groundtruth found"
echo ""

echo "Step 2: Convert Data to Binary Format"


# Convert base data if needed
if [ ! -f "$BASE_DATA_BIN" ]; then
    echo "Converting SIFT1M base data from fvecs to bin format..."
    echo "This will take a few minutes..."
    python3 scripts/vecs_to_binary.py "$BASE_DATA_FVECS" "$BASE_DATA_BIN" 4 1000000 128
    echo " Base data conversion complete"
else
    echo " Base data binary already exists: $BASE_DATA_BIN"
fi

# Convert query data if needed
if [ ! -f "$QUERY_BIN" ]; then
    echo "Converting SIFT1M query data from fvecs to bin format..."
    python3 scripts/vecs_to_binary.py "$QUERY_FVECS" "$QUERY_BIN" 4 10000 128
    echo " Query data conversion complete"
else
    echo " Query data binary already exists: $QUERY_BIN"
fi

# Convert groundtruth if needed
if [ ! -f "$GT_BIN" ]; then
    echo "Converting SIFT1M groundtruth from ivecs to bin format..."
    python3 scripts/vecs_to_binary.py "$GT_IVECS" "$GT_BIN" 4 10000 100
    echo " Groundtruth conversion complete"
else
    echo " Groundtruth binary already exists: $GT_BIN"
fi
echo ""


echo "Step 3: Generate Random Graph (if needed)"


# Check/Generate random graph
if [ ! -f "$RANDOM_GRAPH" ]; then
    echo "Generating random graph for SIFT1M..."
    echo "This will create a file with 1M nodes and random edges..."

    python3 << 'EOF'
import struct
import random

N = 1000000
R = 64  # Degree (edges per node)
output_file = "data/sift1m_randomgraph.bin"

print(f"Generating random graph: N={N}, R={R}")

with open(output_file, "wb") as f:
    # Write header: N and R
    f.write(struct.pack('<I', N))
    f.write(struct.pack('<I', R))

    # For each node, write R random neighbors
    for i in range(N):
        neighbors = random.sample(range(N), R)
        for neighbor in neighbors:
            f.write(struct.pack('<I', neighbor))

        if (i + 1) % 100000 == 0:
            print(f"  {i + 1:,} nodes processed...")

print(f" Random graph created: {output_file}")
EOF

    echo " Random graph generation complete"
else
    echo " Random graph already exists: $RANDOM_GRAPH"
fi
echo ""


echo "Step 4: Update Configuration for N=1,000,000"

sed -i 's/#define N [0-9]*/#define N 1000000/' src/vamana.h
sed -i 's/#define NUM_QUERIES [0-9]*/#define NUM_QUERIES 10000/' src/vamana.h
grep "#define N " src/vamana.h | head -1
echo " Configuration updated"
echo ""

echo "Step 5: Compile Vamana Graph Builder"

echo "Compiling vamana binary..."
make compile
echo " Vamana binary compiled"
echo ""


echo "Step 6: Build Vamana Graph for SIFT1M"

if [ -f "$GRAPH_FILE" ]; then
    echo "Graph file already exists: $GRAPH_FILE"
    read -p "Rebuild graph? This will take several minutes. (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Using existing graph file"
    else
        echo "Building Vamana graph (this may take 5-10 minutes)..."
        ./bin/vamana "$RANDOM_GRAPH" "$BASE_DATA_BIN" "$GRAPH_FILE"
        echo " Graph construction complete"
    fi
else
    echo "Building Vamana graph (this may take 5-10 minutes)..."
    ./bin/vamana "$RANDOM_GRAPH" "$BASE_DATA_BIN" "$GRAPH_FILE"
    echo " Graph construction complete"
fi
echo ""


echo "Step 7: Compile Concurrent Executor"

echo "Building concurrent FreshDiskANN executor..."
make compile-concurrent
echo "✓ Concurrent executor compiled"
echo ""


echo "Step 8: Generate Workloads"

cd "$PROJECT_ROOT"

python3 << 'EOF'
import json
import random

# 1. Query-Only (10,000 queries)
with open('workload_query_only_10k_sift1m.jsonl', 'w') as f:
    for i in range(10000):
        f.write(json.dumps({"type": "query", "query_id": random.randint(0, 9999), "k": 10}) + '\n')
print(" Query-Only: 10,000 queries")

# 2. Query-Heavy (300I + 200D + 9,500Q = 10,000 ops)
ops = [{"type": "insert", "point_id": random.randint(0, 999999)} for _ in range(300)]
ops += [{"type": "delete", "point_id": random.randint(0, 999999)} for _ in range(200)]
ops += [{"type": "query", "query_id": random.randint(0, 9999), "k": 10} for _ in range(9500)]
random.shuffle(ops)
with open('workload_query_heavy_sift1m.jsonl', 'w') as f:
    for op in ops:
        f.write(json.dumps(op) + '\n')
print(" Query-Heavy: 300I + 200D + 9,500Q")

# 3. Balanced (2,500I + 1,500D + 6,000Q = 10,000 ops)
ops = [{"type": "insert", "point_id": random.randint(0, 999999)} for _ in range(2500)]
ops += [{"type": "delete", "point_id": random.randint(0, 999999)} for _ in range(1500)]
ops += [{"type": "query", "query_id": random.randint(0, 9999), "k": 10} for _ in range(6000)]
random.shuffle(ops)
with open('workload_balanced_sift1m.jsonl', 'w') as f:
    for op in ops:
        f.write(json.dumps(op) + '\n')
print(" Balanced: 2,500I + 1,500D + 6,000Q")

# 4. Insert-Heavy (5,000I + 1,000D + 4,000Q = 10,000 ops)
ops = [{"type": "insert", "point_id": random.randint(0, 999999)} for _ in range(5000)]
ops += [{"type": "delete", "point_id": random.randint(0, 999999)} for _ in range(1000)]
ops += [{"type": "query", "query_id": random.randint(0, 9999), "k": 10} for _ in range(4000)]
random.shuffle(ops)
with open('workload_insert_heavy_sift1m.jsonl', 'w') as f:
    for op in ops:
        f.write(json.dumps(op) + '\n')
print(" Insert-Heavy: 5,000I + 1,000D + 4,000Q")
EOF

echo " All workloads generated"
echo ""

cd "$SCRIPT_DIR"


echo " SIFT1M Setup Complete!"

echo ""
echo "Summary of what was prepared:"
echo "  • Dataset: SIFT1M (1,000,000 points, 128 dimensions)"
echo "  • Binary files:"
echo "    - Base data: $BASE_DATA_BIN"
echo "    - Query data: $QUERY_BIN"
echo "    - Groundtruth: $GT_BIN"
echo "  • Graph: $GRAPH_FILE"
echo "  • Workloads (in $PROJECT_ROOT):"
echo "    - workload_query_only_10k_sift1m.jsonl (10,000 queries)"
echo "    - workload_query_heavy_sift1m.jsonl (300I + 200D + 9,500Q)"
echo "    - workload_balanced_sift1m.jsonl (2,500I + 1,500D + 6,000Q)"
echo "    - workload_insert_heavy_sift1m.jsonl (5,000I + 1,000D + 4,000Q)"
echo "  • Binaries compiled:"
echo "    - bin/vamana (graph builder)"
echo "    - bin/concurrent_fresh_diskann (benchmark executor)"
echo ""
echo "You can now run benchmarks with:"
echo "  ./run_sift1m_benchmarks.sh"
echo ""
