.PHONY: all clean

SRCDIR := src
INCDIR := include
BUILDDIR := build

SRC := $(wildcard $(SRCDIR)/*.c)
OBJ_CPU := $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%_cpu.o,$(SRC))
OBJ_GPU := $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%_gpu.o,$(SRC))

SRC_CU := $(wildcard $(SRCDIR)/*.cu)
OBJ_CU := $(patsubst $(SRCDIR)/%.cu,$(BUILDDIR)/%_cu.o,$(SRC_CU))

all: main-cpu main-gpu

main-cpu: $(OBJ_CPU)
	gcc -o $@ $^ -lm

main-gpu: $(OBJ_GPU) $(OBJ_CU)
	gcc -o $@ $^ -lm -L/opt/cuda/lib64/ -lcudart

$(BUILDDIR)/%_cpu.o: $(SRCDIR)/%.c | $(BUILDDIR)
	gcc -c $< -o $@ -I$(INCDIR) -DMATRIX_CPU

$(BUILDDIR)/%_gpu.o: $(SRCDIR)/%.c | $(BUILDDIR)
	gcc -c $< -o $@ -I$(INCDIR) -DMATRIX_GPU

$(BUILDDIR)/%_cu.o: $(SRCDIR)/%.cu | $(BUILDDIR)
	nvcc -c $< -o $@ -I$(INCDIR)

$(BUILDDIR):
	mkdir -p $@

clean:
	rm -rf main-cpu main-gpu $(BUILDDIR)
