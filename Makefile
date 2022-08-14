NVCC = nvcc
CFLAGS = -O3
LDFLAGS =
 
SOURCE = field.cu particles.cu tile_vfld.cu tile_zdf.cu \
         emf.cu laser.cu current.cu species.cu \
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



