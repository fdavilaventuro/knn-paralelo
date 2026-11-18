#!/usr/bin/env bash

set -euo pipefail

# ==========================================
# PARÁMETROS DE CONFIGURACIÓN DEL BENCHMARK
# ==========================================
TAMANOS_N=(5000 20000 50000)
CORES=(2 4 8 16 32)
MPI_FLAGS="--use-hwthread-cpus"
# ==========================================

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_FILE="$ROOT_DIR/data/resultados_benchmark.csv"

cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/data"

make clean
make

printf 'Modelo,Tamano_Dataset,Procesos,Tiempo_Total,Tiempo_Comp,Tiempo_Comm\n' > "$RESULTS_FILE"

for N in "${TAMANOS_N[@]}"; do
    echo ">>> Generando dataset N = $N..."
    python3 scripts/preprocesamiento.py "$N"

    echo "    Ejecutando Secuencial (p = 1)..."
    TIME="$(./src/secuencial.out)"
    printf 'Secuencial,%s,1,%s\n' "$N" "$TIME" >> "$RESULTS_FILE"

    for p in "${CORES[@]}"; do
        echo "    Ejecutando p = $p procesos..."
        
        TIME="$(mpirun $MPI_FLAGS -np "$p" ./src/modelo1.out)"
        printf 'Modelo1,%s,%s,%s\n' "$N" "$p" "$TIME" >> "$RESULTS_FILE"

        TIME="$(mpirun $MPI_FLAGS -np "$p" ./src/modelo2.out)"
        printf 'Modelo2,%s,%s,%s\n' "$N" "$p" "$TIME" >> "$RESULTS_FILE"
    done
done

echo ">>> Generando gráficas..."
python3 scripts/graficar_speedup.py
printf 'Benchmark y gráficas completados: %s\n' "$RESULTS_FILE"