CXX := mpic++
CXXFLAGS := -O3 -std=c++17 -Wall

TARGETS := src/secuencial.out src/modelo1.out src/modelo2.out

.PHONY: all clean secuencial modelo1 modelo2

all: $(TARGETS)

src/%.out: src/%.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

secuencial: src/secuencial.out

modelo1: src/modelo1.out

modelo2: src/modelo2.out

clean:
	$(RM) $(TARGETS)
