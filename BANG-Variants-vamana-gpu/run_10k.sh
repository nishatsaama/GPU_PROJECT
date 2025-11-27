#!/bin/bash

# SIFT10K Setup and Prerequisites Script
# Run this BEFORE running run_sift10k_benchmarks.sh
# Usage: bash run_10k.sh

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════╗"
echo "║   SIFT10K Setup - Prerequisites & Preparation      ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Configuration - Use relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$SCRIPT_DIR/data"
BUILD_DIR="$SCRIPT_DIR/build"

# SIFT10K Files (using siftsmall dataset)
BASE_DATA_BIN="$DATA_DIR/sift10k_base.bin"
QUERY_BIN="$DATA_DIR/siftsmall_query.bin"
GT_BIN="$DATA_DIR/siftsmall_groundtruth.bin"
RANDOM_GRAPH="$DATA_DIR/sift10k_randomgraph.bin"

# Graph output
GRAPH_FILE="$BUILD_DIR/vamana_alpha1.2.out"

cd "$SCRIPT_DIR"

echo "════════════════════════════════════════════════════"
echo "Step 1: Check/Prepare SIFT10K Data"
echo "════════════════════════════════════════════════════"

# Check if data files exist
if [ ! -f "$BASE_DATA_BIN" ]; then
    echo "Error: SIFT10K base data not found at $BASE_DATA_BIN"
    echo "Please ensure sift10k_base.bin is in the data directory"
    exit 1
fi
echo "✓ SIFT10K base data found"

if [ ! -f "$QUERY_BIN" ]; then
    echo "Error: Query data not found at $QUERY_BIN"
    exit 1
fi
echo "✓ Query data found"

if [ ! -f "$GT_BIN" ]; then
    echo "Error: Groundtruth not found at $GT_BIN"
    exit 1
fi
echo "✓ Groundtruth found"
echo ""

echo "════════════════════════════════════════════════════"
echo "Step 2: Generate/Check Random Graph"
echo "════════════════════════════════════════════════════"

# Check/Generate random graph
if [ ! -f "$RANDOM_GRAPH" ]; then
    echo "Generating random graph for SIFT10K..."

    python3 << 'EOF'
import struct
import random

N = 10000
R = 60  # Degree (edges per node)
output_file = "data/sift10k_randomgraph.bin"

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

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1:,} nodes processed...")

print(f"✓ Random graph created: {output_file}")
EOF

    echo "✓ Random graph generation complete"
else
    echo "✓ Random graph already exists: $RANDOM_GRAPH"
fi
echo ""

echo "════════════════════════════════════════════════════"
echo "Step 3: Update Configuration for N=10,000"
echo "════════════════════════════════════════════════════"
sed -i 's/#define N [0-9]*/#define N 10000/' src/vamana.h
sed -i 's/#define NUM_QUERIES [0-9]*/#define NUM_QUERIES 10000/' src/vamana.h
grep "#define N " src/vamana.h | head -1
echo "✓ Configuration updated"
echo ""

echo "════════════════════════════════════════════════════"
echo "Step 4: Compile Vamana Graph Builder"
echo "════════════════════════════════════════════════════"
echo "Compiling vamana binary..."
make compile
echo "✓ Vamana binary compiled"
echo ""

echo "════════════════════════════════════════════════════"
echo "Step 5: Build Vamana Graph for SIFT10K"
echo "════════════════════════════════════════════════════"
if [ -f "$GRAPH_FILE" ]; then
    echo "Graph file already exists: $GRAPH_FILE"
    read -p "Rebuild graph? This will take a minute or two. (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Using existing graph file"
    else
        echo "Building Vamana graph..."
        ./bin/vamana "$RANDOM_GRAPH" "$BASE_DATA_BIN" "$GRAPH_FILE"
        echo "✓ Graph construction complete"
    fi
else
    echo "Building Vamana graph..."
    ./bin/vamana "$RANDOM_GRAPH" "$BASE_DATA_BIN" "$GRAPH_FILE"
    echo "✓ Graph construction complete"
fi
echo ""

echo "════════════════════════════════════════════════════"
echo "Step 6: Compile Concurrent Executor"
echo "════════════════════════════════════════════════════"
echo "Building concurrent FreshDiskANN executor..."
make compile-concurrent
echo "✓ Concurrent executor compiled"
echo ""

echo "════════════════════════════════════════════════════"
echo "Step 7: Generate Workloads"
echo "════════════════════════════════════════════════════"
cd "$PROJECT_ROOT"

python3 << 'EOF'
import json
import random

# 1. Query-Only (400 queries)
with open('workload_query_only_400.jsonl', 'w') as f:
    for i in range(400):
        f.write(json.dumps({"type": "query", "query_id": random.randint(0, 99), "k": 10}) + '\n')
print("✓ Query-Only: 400 queries")

# 2. Query-Heavy (12I + 8D + 380Q = 400 ops)
ops = [{"type": "insert", "point_id": random.randint(0, 9999)} for _ in range(12)]
ops += [{"type": "delete", "point_id": random.randint(0, 9999)} for _ in range(8)]
ops += [{"type": "query", "query_id": random.randint(0, 99), "k": 10} for _ in range(380)]
random.shuffle(ops)
with open('workload_query_heavy.jsonl', 'w') as f:
    for op in ops:
        f.write(json.dumps(op) + '\n')
print("✓ Query-Heavy: 12I + 8D + 380Q")

# 3. Balanced (100I + 60D + 240Q = 400 ops)
ops = [{"type": "insert", "point_id": random.randint(0, 9999)} for _ in range(100)]
ops += [{"type": "delete", "point_id": random.randint(0, 9999)} for _ in range(60)]
ops += [{"type": "query", "query_id": random.randint(0, 99), "k": 10} for _ in range(240)]
random.shuffle(ops)
with open('workload_balanced.jsonl', 'w') as f:
    for op in ops:
        f.write(json.dumps(op) + '\n')
print("✓ Balanced: 100I + 60D + 240Q")

# 4. Insert-Heavy (200I + 40D + 160Q = 400 ops)
ops = [{"type": "insert", "point_id": random.randint(0, 9999)} for _ in range(200)]
ops += [{"type": "delete", "point_id": random.randint(0, 9999)} for _ in range(40)]
ops += [{"type": "query", "query_id": random.randint(0, 99), "k": 10} for _ in range(160)]
random.shuffle(ops)
with open('workload_insert_heavy.jsonl', 'w') as f:
    for op in ops:
        f.write(json.dumps(op) + '\n')
print("✓ Insert-Heavy: 200I + 40D + 160Q")
EOF

echo "✓ All workloads generated"
echo ""

cd "$SCRIPT_DIR"

echo "════════════════════════════════════════════════════"
echo "✓ SIFT10K Setup Complete!"
echo "════════════════════════════════════════════════════"
echo ""
echo "Summary of what was prepared:"
echo "  • Dataset: SIFT10K (10,000 points, 128 dimensions)"
echo "  • Binary files:"
echo "    - Base data: $BASE_DATA_BIN"
echo "    - Query data: $QUERY_BIN"
echo "    - Groundtruth: $GT_BIN"
echo "  • Graph: $GRAPH_FILE"
echo "  • Workloads (in $PROJECT_ROOT):"
echo "    - workload_query_only_400.jsonl (400 queries)"
echo "    - workload_query_heavy.jsonl (12I + 8D + 380Q)"
echo "    - workload_balanced.jsonl (100I + 60D + 240Q)"
echo "    - workload_insert_heavy.jsonl (200I + 40D + 160Q)"
echo "  • Binaries compiled:"
echo "    - bin/vamana (graph builder)"
echo "    - bin/concurrent_fresh_diskann (benchmark executor)"
echo ""
echo "You can now run benchmarks with:"
echo "  ./run_sift10k_benchmarks.sh"
echo ""
