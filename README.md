# SIFT Concurrent FreshDiskANN Benchmarks

Complete guide for running benchmarks and creating custom datasets.

---

## Section 1: Running Benchmarks on SIFT Datasets

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- SIFT dataset files in `../data/sift/`:
  - `sift_base.fvecs` (for SIFT1M)
  - `sift_query.fvecs`
  - `sift_groundtruth.ivecs`

### Quick Start

#### SIFT10K Benchmarks

```bash
# Step 1: Run setup (only needed once)
./run_10k.sh

# Step 2: Run benchmarks (can run multiple times)
./run_sift10k_benchmarks.sh
```

#### SIFT1M Benchmarks(download the dataset in data folder)

```bash
# Step 1: Run setup (only needed once - takes ~10-15 minutes)
./run_1m.sh

# Step 2: Run benchmarks (can run multiple times)
./run_sift1m_benchmarks.sh
```

### What the Setup Scripts Do

**`run_10k.sh`** (SIFT10K Setup):
1. Verifies SIFT10K data files exist
2. Generates random graph (10K nodes, degree 60)
3. Updates configuration (`N=10000`)
4. Compiles Vamana graph builder
5. Builds Vamana graph
6. Compiles concurrent executor
7. Generates 4 workloads (400 operations each):
   - Query-Only: 400 queries
   - Query-Heavy: 12I + 8D + 380Q
   - Balanced: 100I + 60D + 240Q
   - Insert-Heavy: 200I + 40D + 160Q

**`run_1m.sh`** (SIFT1M Setup):
1. Verifies SIFT1M source data exists
2. Converts data from `.fvecs`/`.ivecs` to `.bin` format
3. Generates random graph (1M nodes, degree 60)
4. Updates configuration (`N=1000000`)
5. Compiles Vamana graph builder
6. Builds Vamana graph (~5-10 minutes)
7. Compiles concurrent executor
8. Generates 4 workloads (10,000 operations each):
   - Query-Only: 10,000 queries
   - Query-Heavy: 300I + 200D + 9,500Q
   - Balanced: 2,500I + 1,500D + 6,000Q
   - Insert-Heavy: 5,000I + 1,000D + 4,000Q

### Understanding the Output

Each benchmark reports:
- **Insert avg**: Average time per insert (ms)
- **Delete avg**: Average time per delete (ms)
- **Query avg**: Average time per query (ms)
- **Query QPS**: Queries per second
- **5-recall@5**: Recall at k=5 (% of correct results)
- **10-recall@10**: Recall at k=10 (% of correct results)
- **Throughput**: Overall operations/second (all types)

Example output:
```
1. Query-Only (400 queries)
-----------------------------------
  Query avg:       0.033 ms
  Query QPS:       30442.6
  5-recall@5:      99.20%
  10-recall@10:    98.22%
Throughput: 30304.2 ops/sec
```

### Key Parameters

Edit these in the benchmark scripts:

```bash
SEARCH_L=20    # Worklist size (10-150). Higher = better accuracy, slower
K=10           # Number of neighbors to return
WORKERS=8      # Parallel worker threads
BATCH=50       # Batch size for GPU processing
```

**Parameter Guide:**
- **L (Search L)**: Beam width during graph search
  - L=10: Fast, ~94% recall
  - L=20: Balanced, ~99% recall (default)
  - L=100: Slow, ~100% recall
- **k**: Number of nearest neighbors to return (typically 10)
- **workers**: Number of parallel threads (typically 8)
- **batch**: Batch size for better GPU utilization (typically 50)

### Results Location

- Results saved to timestamped files in the `BANG-Variants-vamana-gpu/` directory
- Format: `benchmark_results_[sift1m_]YYYYMMDD_HHMMSS.txt`
- Full output (not just grep'd metrics) is saved to these files

---

## Section 2: Creating Custom Workloads

You can create your own workload patterns to test different scenarios beyond the default workloads.

### Workload File Format

Workloads are stored in `.jsonl` (JSON Lines) format. Each line is a JSON object representing one operation:

**Query Operation:**
```json
{"type": "query", "query_id": 42, "k": 10}
```

**Insert Operation:**
```json
{"type": "insert", "point_id": 1234}
```

**Delete Operation:**
```json
{"type": "delete", "point_id": 5678}
```

### Workload Generation Script

Create custom workloads with different operation mixes:

```python
# save as: generate_custom_workload.py
import json
import random

# Configuration
DATASET = "SIFT10K"  # Options: "SIFT10K" or "SIFT1M"

# Set ranges based on dataset
if DATASET == "SIFT10K":
    N = 10000           # Dataset size
    NUM_QUERIES = 100   # Number of queries available
    WORKLOAD_SIZE = 400 # Total operations
    OUTPUT_PREFIX = "workload_10k_"
else:  # SIFT1M
    N = 1000000
    NUM_QUERIES = 10000
    WORKLOAD_SIZE = 10000
    OUTPUT_PREFIX = "workload_1m_"

# Define your custom workload patterns
# Format: (num_inserts, num_deletes, num_queries, name)
workloads = [
    # Standard patterns
    (0, 0, WORKLOAD_SIZE, "query_only"),
    (int(0.03*WORKLOAD_SIZE), int(0.02*WORKLOAD_SIZE), int(0.95*WORKLOAD_SIZE), "query_heavy"),
    (int(0.25*WORKLOAD_SIZE), int(0.15*WORKLOAD_SIZE), int(0.60*WORKLOAD_SIZE), "balanced"),
    (int(0.50*WORKLOAD_SIZE), int(0.10*WORKLOAD_SIZE), int(0.40*WORKLOAD_SIZE), "insert_heavy"),

    # Custom patterns - add your own!
    (int(0.10*WORKLOAD_SIZE), int(0.50*WORKLOAD_SIZE), int(0.40*WORKLOAD_SIZE), "delete_heavy"),
    (int(0.33*WORKLOAD_SIZE), int(0.33*WORKLOAD_SIZE), int(0.34*WORKLOAD_SIZE), "equal_mix"),
    (int(0.70*WORKLOAD_SIZE), int(0.20*WORKLOAD_SIZE), int(0.10*WORKLOAD_SIZE), "write_heavy"),
]

# Generate workloads
for num_insert, num_delete, num_query, name in workloads:
    # Adjust to exact workload size
    total = num_insert + num_delete + num_query
    if total != WORKLOAD_SIZE:
        num_query = WORKLOAD_SIZE - num_insert - num_delete

    ops = []
    ops += [{"type": "insert", "point_id": random.randint(0, N-1)} for _ in range(num_insert)]
    ops += [{"type": "delete", "point_id": random.randint(0, N-1)} for _ in range(num_delete)]
    ops += [{"type": "query", "query_id": random.randint(0, NUM_QUERIES-1), "k": 10} for _ in range(num_query)]

    # Shuffle for realistic interleaving
    random.shuffle(ops)

    # Write to file
    filename = f"{OUTPUT_PREFIX}{name}.jsonl"
    with open(filename, 'w') as f:
        for op in ops:
            f.write(json.dumps(op) + '\n')

    print(f"✓ {name}: {num_insert}I + {num_delete}D + {num_query}Q → {filename}")

print(f"\nGenerated {len(workloads)} workloads for {DATASET}")
```

Run it:
```bash
# For SIFT10K workloads
python3 generate_custom_workload.py

# Or edit DATASET="SIFT1M" in the script for SIFT1M
```

### Testing Custom Workloads

Once you have generated workload files, test them using the concurrent executor:

**For SIFT10K:**
```bash
# Make sure you've run setup first
./run_10k.sh

# Test your custom workload
./bin/concurrent_fresh_diskann \
    build/vamana_alpha1.2.out \
    workload_10k_delete_heavy.jsonl \
    data/siftsmall_query.bin \
    data/siftsmall_groundtruth.bin \
    --searchL 20 \
    --k 10 \
    --workers 8 \
    --batch 50
```

**For SIFT1M:**
```bash
# Make sure you've run setup first
./run_1m.sh

# Test your custom workload
./bin/concurrent_fresh_diskann \
    build/vamana_sift1m_alpha1.2.out \
    workload_1m_delete_heavy.jsonl \
    data/sift1m_query.bin \
    data/sift1m_groundtruth.bin \
    --searchL 20 \
    --k 10 \
    --workers 8 \
    --batch 50
```

### Batch Testing Multiple Workloads

Create a script to test multiple custom workloads:

```bash
#!/bin/bash
# save as: test_custom_workloads.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Choose dataset: SIFT10K or SIFT1M
DATASET="SIFT10K"

if [ "$DATASET" == "SIFT10K" ]; then
    GRAPH_FILE="$SCRIPT_DIR/build/vamana_alpha1.2.out"
    QUERY_FILE="$SCRIPT_DIR/data/siftsmall_query.bin"
    GT_FILE="$SCRIPT_DIR/data/siftsmall_groundtruth.bin"
    WORKLOAD_PREFIX="workload_10k_"
else
    GRAPH_FILE="$SCRIPT_DIR/build/vamana_sift1m_alpha1.2.out"
    QUERY_FILE="$SCRIPT_DIR/data/sift1m_query.bin"
    GT_FILE="$SCRIPT_DIR/data/sift1m_groundtruth.bin"
    WORKLOAD_PREFIX="workload_1m_"
fi

SEARCH_L=20
K=10
WORKERS=8
BATCH=50

cd "$SCRIPT_DIR"

RESULTS_FILE="custom_workload_results_$(date +%Y%m%d_%H%M%S).txt"

echo "Custom Workload Benchmark Results - $DATASET" > $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "Configuration: L=$SEARCH_L, k=$K, workers=$WORKERS, batch=$BATCH" >> $RESULTS_FILE
echo "=================================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Test all workloads matching the prefix
for workload in ${PROJECT_ROOT}/${WORKLOAD_PREFIX}*.jsonl; do
    workload_name=$(basename $workload .jsonl)

    echo "Testing: $workload_name"
    echo "-----------------------------------"

    ./bin/concurrent_fresh_diskann \
        $GRAPH_FILE \
        $workload \
        $QUERY_FILE \
        $GT_FILE \
        --searchL $SEARCH_L \
        --k $K \
        --workers $WORKERS \
        --batch $BATCH 2>&1 | tee -a $RESULTS_FILE | grep -E "Insert avg|Delete avg|Query avg|Query QPS|recall@|Throughput"

    echo ""
done

echo "✓ All custom workloads tested!"
echo "Results saved to: $RESULTS_FILE"
```

Make it executable and run:
```bash
chmod +x test_custom_workloads.sh
./test_custom_workloads.sh
```

### Custom Workload Ideas

**Delete-Heavy (10I + 50D + 40Q per 100 ops):**
- Tests graph stability under high deletion rate
- Useful for cache eviction scenarios

**Equal Mix (33I + 33D + 34Q per 100 ops):**
- Balanced stress test
- Realistic for dynamic systems

**Write-Heavy (70I + 20D + 10Q per 100 ops):**
- Tests insertion throughput
- Useful for bulk loading scenarios

**Bursty Pattern:**
```python
# Generate bursts of operations
ops = []
for _ in range(10):  # 10 bursts
    # Burst of inserts
    ops += [{"type": "insert", "point_id": random.randint(0, N-1)} for _ in range(20)]
    # Burst of queries
    ops += [{"type": "query", "query_id": random.randint(0, NUM_QUERIES-1), "k": 10} for _ in range(20)]
```

**Time-Series Pattern:**
```python
# Simulate time-series ingestion with queries
ops = []
for i in range(WORKLOAD_SIZE // 11):
    # Insert 10 new points
    ops += [{"type": "insert", "point_id": i*10 + j} for j in range(10)]
    # Query the latest data
    ops.append({"type": "query", "query_id": random.randint(0, NUM_QUERIES-1), "k": 10})
```

---
