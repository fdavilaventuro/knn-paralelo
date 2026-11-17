#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_FILE="$ROOT_DIR/data/resultados_benchmark.csv"

cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/data"

make clean
make

printf 'Modelo,Tamano_Dataset,Procesos,Tiempo_Segundos\n' > "$RESULTS_FILE"

for N in 5000 20000 50000; do
    python3 scripts/preprocesamiento.py "$N"

    TIME="$(./src/secuencial.out)"
    printf 'Secuencial,%s,1,%s\n' "$N" "$TIME" >> "$RESULTS_FILE"

    for p in 2 4 8 16; do
        TIME="$(mpirun -np "$p" ./src/modelo1.out)"
        printf 'Modelo1,%s,%s,%s\n' "$N" "$p" "$TIME" >> "$RESULTS_FILE"

        TIME="$(mpirun -np "$p" ./src/modelo2.out)"
        printf 'Modelo2,%s,%s,%s\n' "$N" "$p" "$TIME" >> "$RESULTS_FILE"
    done
done

python3 scripts/graficar_speedup.py
printf 'Benchmark y gráficas completados: %s\n' "$RESULTS_FILE"