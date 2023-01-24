CC = gcc
MPICC=mpicc
BIN = bin/
SRC = src/
INCLUDE= include/
EXEC = k_means
THREADS = 4
CP_CLUSTERS = 4
PROCESSOS = 4


CFLAGS = -O2 -fno-omit-frame-pointer -g -ftree-vectorize -msse4

TARGET = k_means

all: $(TARGET)
$(TARGET):$(SRC)$(TARGET).c
	$(MPICC) $(CFLAGS) -o $(BIN)$(TARGET) $(SRC)$(TARGET).c -lm 

clean:
	rm -r $(BIN)*


run:
	./$(BIN)$(TARGET)

perf: ./$(BIN)$(TARGET)
	srun --cpus-per-task=$(THREADS) --partition=cpar perf stat -e L1-dcache-load-misses -M cpi mpirun --oversubscribe -np $(PROCESSOS) ./$(BIN)$(TARGET)

runseq:
	srun --cpus-per-task=$(THREADS) --partition=cpar perf stat -e L1-dcache-load-misses -M cpi mpirun --oversubscribe -np $(PROCESSOS) ./$(BIN)$(EXEC) 10000000	$(CP_CLUSTERS)

runpar:
	srun --cpus-per-task=$(THREADS) --partition=cpar perf stat -e L1-dcache-load-misses -M cpi mpirun --oversubscribe -np $(PROCESSOS) ./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS) $(THREADS) $(PROCESSOS)

newbash:
	srun --partition=cpar --cpus-per-task=$(THREADS) --pty bash

record: 
	srun --partition=cpar --cpus-per-task=$(THREADS) perf record ./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS) && perf report

stats:
	srun --partition=cpar --cpus-per-task=$(THREADS) perf stat -e L1-dcache-load-misses -M cpi -r 3 ./$(BIN)$(EXEC) 10000000 $(CP_CLUSTERS)
