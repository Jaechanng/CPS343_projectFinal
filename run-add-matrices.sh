#!/usr/bin/env bash
set -u

BIN="./add-matrices"
SRC="add-matrices.cu"
OUT="add-matrices-result.txt"

SIZES=(128 256 512 1024 2048 4096 8192)
CONFIGS=(
    "1 128"
    "2 256"
    "4 512"
    "8 1024"
)
RUNS_PER_CONFIG=3

if [[ ! -x "$BIN" || "$SRC" -nt "$BIN" ]]; then
    if ! command -v nvcc > /dev/null 2>&1; then
        echo "Error: $BIN does not exist and nvcc is not in PATH."
        echo "Compile $SRC first, or load CUDA so this script can build it."
        exit 1
    fi

    echo "Building $BIN..."
    nvcc -arch=compute_61 -code=sm_61,sm_75,sm_80 -O3 -o "$BIN" "$SRC" || exit 1
fi

{
    echo "add-matrices multiple-run output"
    echo "Generated: $(date)"
    echo "All matrix dimensions are greater than 100, so sample matrix values are not printed."
    echo
} > "$OUT"

run_id=1
for size in "${SIZES[@]}"; do
    for config in "${CONFIGS[@]}"; do
        read -r streams rows_per_chunk <<< "$config"

        if (( rows_per_chunk > size )); then
            continue
        fi

        for ((trial = 1; trial <= RUNS_PER_CONFIG; trial++)); do
            echo "Run $run_id: ${size}x${size}, streams=$streams, rows_per_chunk=$rows_per_chunk"

            {
                echo "============================================================"
                echo "Run $run_id"
                echo "Command: $BIN $size $size $streams $rows_per_chunk"
                echo "Matrix: ${size} rows x ${size} cols"
                echo "Streams: $streams"
                echo "Rows per chunk: $rows_per_chunk"
                echo "Trial: $trial"
                echo
                "$BIN" "$size" "$size" "$streams" "$rows_per_chunk"
                echo
            } >> "$OUT" 2>&1

            run_id=$((run_id + 1))
        done
    done
done

echo "Saved output text to $OUT"
