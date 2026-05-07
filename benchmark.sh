#!/bin/bash

MATRIX_DIR="/home/cps343/matrix"
OUT="dot-matrices_result.txt"

PROGRAMS=(
    "./dot-matrices_cuda_stream"
    "./dot-matrices_cuda"
)

MATRICES=(
    "A-500x500.dat"
    "A-1000x1000.dat"
    "A-5000x5000.dat"
    "A-10000x10000.dat"
)

{
    echo "======================================"
    echo " CUDA Benchmark (Raw Output)"
    echo " Generated: $(date)"
    echo "======================================"

    for prog in "${PROGRAMS[@]}"; do
        for matrix in "${MATRICES[@]}"; do

            MATRIX_PATH="${MATRIX_DIR}/${matrix}"

            echo ""
            echo "--------------------------------------"
            echo "Program : $prog"
            echo "Matrix  : $matrix"
            echo "--------------------------------------"

            echo ""
            echo "[P1000 ]"
            srun --gres=gpu:P1000 "$prog" "$MATRIX_PATH"

            echo ""
            echo "[RTX3000]"
            srun --gres=gpu:RTX3000 "$prog" "$MATRIX_PATH"

            echo ""
            echo "[RTXA5000 - ml partition]"
            srun -p ml --gres=gpu:RTXA5000 "$prog" "$MATRIX_PATH"

            echo ""
        done
    done

    echo "======================================"
    echo " Done"
    echo "======================================"
} 2>&1 | tee "$OUT"
