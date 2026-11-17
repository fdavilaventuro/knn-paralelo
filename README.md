# KNN paralelo con MPI

Implementación en C++ de K-Nearest Neighbors (KNN) sobre memoria
distribuida, utilizando MPI para comparar dos arquitecturas de
paralelización:

- **Modelo 1 — Scatterv Train:** el conjunto de entrenamiento se
  distribuye entre los procesos mediante `MPI_Scatterv`.
- **Modelo 2 — Bcast Train:** el conjunto de entrenamiento se replica en
  todos los procesos mediante `MPI_Bcast` y se distribuye el trabajo de
  evaluación.

Los datasets CSV se almacenan localmente en `data/`, una carpeta excluida
del control de versiones.

## Requisitos

- Compilador C++ con soporte para C++11 o superior.
- Implementación de MPI, como Open MPI o MPICH.
- Python 3 para las utilidades de preparación y análisis.
- GNU Make.

## Entorno Python

### Linux y macOS

Desde la raíz del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows PowerShell

Desde la raíz del proyecto:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Si la política de ejecución de PowerShell bloquea la activación, puede usar
`cmd.exe`:

```bat
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

### Windows con WSL 2

Desde Ubuntu en WSL:

```bash
cd /mnt/c/Users/<usuario>/Documents/GitHub/knn-paralelo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

En todos los sistemas, active `.venv` antes de ejecutar los scripts de Python.

## Scripts de Python

Ejecute los scripts desde la raíz del proyecto:

```bash
python scripts/<nombre_del_script>.py
```

Para preparar los datasets y evaluar los resultados:

```bash
python scripts/preprocesamiento.py 100000
python scripts/postprocesamiento.py
```

El primer comando recibe `N`, el número total deseado de muestras, y genera
datos aumentados con ruido gaussiano. El segundo comando imprime el `accuracy`
de cada modelo y genera
`grafico_resultados.png` en la raíz del proyecto.

Consulte `scripts/` para conocer los nombres y argumentos disponibles.

## Compilación y ejecución

El proyecto requiere GNU Make, un compilador MPI (`mpic++`) y `mpirun`.
Los comandos son iguales en cualquier terminal compatible:

```bash
make
make secuencial
make modelo1
make modelo2
make clean
```

Los targets generan `src/secuencial.out`, `src/modelo1.out` y
`src/modelo2.out`.

### Linux

En Debian o Ubuntu, instale las herramientas con:

```bash
sudo apt update
sudo apt install build-essential make openmpi-bin libopenmpi-dev
```

### macOS

Con Homebrew:

```bash
brew install make open-mpi
```

Ejecute los comandos `make` desde Terminal.

### Windows

La opción recomendada es WSL 2 con Ubuntu. Dentro de Ubuntu, instale las
dependencias con los comandos de Linux y acceda al repositorio, por ejemplo:

```bash
cd /mnt/c/Users/<usuario>/Documents/GitHub/knn-paralelo
make
```

También puede utilizar MSYS2/MinGW si dispone de GNU Make y una implementación
de MPI configurados en el `PATH`.

Para ejecutar un binario MPI directamente:

```bash
mpirun -np <procesos> <ejecutable> [argumentos]
```

Por ejemplo:

```bash
mpirun -np 4 ./src/modelo1.out
```

Use el target correspondiente del `Makefile` si requiere rutas o parámetros
específicos del dataset.

Para ejecutar el benchmark completo y generar automáticamente las gráficas:

```bash
bash run_benchmark.sh
```

El CSV se guarda en `data/resultados_benchmark.csv` y la figura de tiempo,
speedup y eficiencia en `data/analisis_rendimiento.png`. Si solo se desea
regenerar la figura a partir de un CSV existente, ejecute:

```bash
python scripts/graficar_speedup.py
```