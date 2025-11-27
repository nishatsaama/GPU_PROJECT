#!/usr/bin/env python3
"""
Convert binary format (.bin) to fvecs format (.fvecs)

Binary format: Raw float32 array (N * D * 4 bytes)
Fvecs format: For each vector: [dim(int32), vector_data(D * float32)]

Usage:
    python3 bin_to_fvecs.py <input.bin> <output.fvecs> <num_vectors> <dimension>

Example:
    python3 bin_to_fvecs.py data/sift10k_base.bin data/sift10k_base.fvecs 10000 128
"""
import struct
import sys
import os

def convert_bin_to_fvecs(input_bin, output_fvecs, num_vectors, dimension):
    """Convert .bin to .fvecs format."""

    # Validate input file
    if not os.path.exists(input_bin):
        print(f"Error: Input file not found: {input_bin}")
        return False

    # Check file size
    expected_size = num_vectors * dimension * 4  # float32 = 4 bytes
    actual_size = os.path.getsize(input_bin)

    if actual_size != expected_size:
        print(f"Warning: File size mismatch!")
        print(f"  Expected: {expected_size} bytes ({num_vectors} × {dimension} × 4)")
        print(f"  Actual:   {actual_size} bytes")
        print(f"  Difference: {actual_size - expected_size} bytes")

        # Try to auto-correct num_vectors
        if actual_size % (dimension * 4) == 0:
            corrected_num = actual_size // (dimension * 4)
            print(f"  Auto-corrected to {corrected_num} vectors")
            num_vectors = corrected_num
        else:
            print(f"Error: File size not divisible by dimension × 4")
            return False

    print(f"Converting {input_bin} to {output_fvecs}")
    print(f"  Vectors: {num_vectors}")
    print(f"  Dimension: {dimension}")
    print(f"  Input size: {actual_size / 1024 / 1024:.2f} MB")

    # Read binary file
    with open(input_bin, 'rb') as f_in:
        data = f_in.read()

    # Convert to fvecs format
    with open(output_fvecs, 'wb') as f_out:
        offset = 0
        for i in range(num_vectors):
            # Write dimension header (int32)
            f_out.write(struct.pack('i', dimension))

            # Write vector data (D × float32)
            vector_data = data[offset:offset + dimension * 4]
            f_out.write(vector_data)

            offset += dimension * 4

            # Progress update
            if (i + 1) % 1000 == 0:
                progress = (i + 1) / num_vectors * 100
                print(f"  Progress: {i + 1}/{num_vectors} ({progress:.1f}%)", end='\r')

    output_size = os.path.getsize(output_fvecs)
    print(f"\n  Output size: {output_size / 1024 / 1024:.2f} MB")
    print(f"✓ Conversion complete!")

    return True

def validate_fvecs(fvecs_file, expected_dim):
    """Validate .fvecs file format."""
    print(f"\nValidating {fvecs_file}...")

    with open(fvecs_file, 'rb') as f:
        # Read first vector
        dim_bytes = f.read(4)
        if len(dim_bytes) < 4:
            print("Error: File too short")
            return False

        dim = struct.unpack('i', dim_bytes)[0]

        if dim != expected_dim:
            print(f"Error: Dimension mismatch (expected {expected_dim}, got {dim})")
            return False

        # Read first vector data
        vec_data = f.read(dim * 4)
        if len(vec_data) < dim * 4:
            print("Error: Incomplete vector data")
            return False

        # Check file size
        f.seek(0, 2)
        file_size = f.tell()
        bytes_per_vector = 4 + dim * 4

        if file_size % bytes_per_vector != 0:
            print(f"Warning: File size not aligned to vector size")
            print(f"  File size: {file_size}")
            print(f"  Bytes per vector: {bytes_per_vector}")
            return False

        num_vectors = file_size // bytes_per_vector
        print(f"✓ Valid .fvecs file: {num_vectors} vectors, dim={dim}")

        # Show first vector sample
        first_vec = struct.unpack(f'{dim}f', vec_data)
        print(f"  First vector sample: [{first_vec[0]:.3f}, {first_vec[1]:.3f}, {first_vec[2]:.3f}, ...]")

    return True

def main():
    if len(sys.argv) != 5:
        print(__doc__)
        print("\nQuick examples:")
        print("  # Convert SIFT 10K base vectors")
        print("  python3 bin_to_fvecs.py data/sift10k_base.bin data/sift10k_base.fvecs 10000 128")
        print("")
        print("  # Convert SIFT 10K queries")
        print("  python3 bin_to_fvecs.py data/siftsmall_query.bin data/siftsmall_query.fvecs 100 128")
        print("")
        print("  # Convert SIFT 1M base vectors")
        print("  python3 bin_to_fvecs.py data/sift1m_base.bin data/sift1m_base.fvecs 1000000 128")
        sys.exit(1)

    input_bin = sys.argv[1]
    output_fvecs = sys.argv[2]
    num_vectors = int(sys.argv[3])
    dimension = int(sys.argv[4])

    # Convert
    success = convert_bin_to_fvecs(input_bin, output_fvecs, num_vectors, dimension)

    if not success:
        sys.exit(1)

    # Validate
    validate_fvecs(output_fvecs, dimension)

    print(f"\nYou can now use this with workload_generator.py:")
    print(f"  cd ..")
    print(f"  python3 workload_generator.py \\")
    print(f"      --scenario e_commerce \\")
    print(f"      --dataset {output_fvecs} \\")
    print(f"      --queries <query.fvecs> \\")
    print(f"      --output workload.jsonl \\")
    print(f"      --max_events 10000")

if __name__ == '__main__':
    main()
