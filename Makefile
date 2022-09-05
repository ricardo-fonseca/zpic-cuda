NVCC = nvcc
CFLAGS = -O3
LDFLAGS =

# The target GPU architecture must be set to (at least) compute_60
# to support double precision atomics
CFLAGS += --gpu-architecture=compute_60
 
SOURCE = field.cu vector_field.cu particles.cu \
        emf.cu laser.cu current.cu \
        udist.cu density.cu species.cu \
        main.cu

# Add ZDF library
SOURCE += zdf.c

TARGET = zpic-cuda

# Add CUDA object files
OBJ = $(SOURCE:.cu=.o)

# Add C object files
OBJ := $(OBJ:.c=.o)

.PHONY: all
all: $(SOURCE) $(TARGET)

$(TARGET) : $(OBJ)
	$(NVCC) $(CFLAGS) $(OBJ) $(LDFLAGS) -o $@

%.o : %.cu
	$(NVCC) -c $(CFLAGS) $< -o $@

%.o : %.c
	$(NVCC) -c $(CFLAGS) $< -o $@

.PHONY: clean
clean :
	@touch $(TARGET) $(OBJ)
	rm -f $(TARGET) $(OBJ)



