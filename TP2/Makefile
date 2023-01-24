CC = gcc
BIN = bin/
SRC = src/
INCLUDE= include/
EXEC = k_means
THREADS = 5

CFLAGS = -O2 -fno-omit-frame-pointer -g -ftree-vectorize -msse4

TARGET = k_means

all: $(TARGET)
$(TARGET):$(SRC)$(TARGET).c
	$(CC) $(CFLAGS) -o $(BIN)$(TARGET) $(SRC)$(TARGET).c -lm -fopenmp

clean:
	rm -r $(BIN)*


run:
	./$(BIN)$(TARGET)

perf: ./$(BIN)$(TARGET)
	srun --cpus-per-task=$(THREADS) --partition=cpar perf stat -e L1-dcache-load-misses -M cpi ./$(BIN)$(TARGET)

runseq:
	srun --cpus-per-task=$(THREADS) --partition=cpar perf stat -e L1-dcache-load-misses -M cpi ./$(BIN)$(EXEC) 10000000	$(CP_CLUSTERS)

runpar:
	srun --cpus-per-task=$(THREADS) --partition=cpar perf stat -e L1-dcache-load-misses -M cpi ./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS) $(THREADS)
