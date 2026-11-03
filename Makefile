CC = gcc
FLAGS = -std=c18 -Wall -Wextra -pedantic -D_XOPEN_SOURCE=700 -D_DEFAULT_SOURCE -g
# Wextra is for aditional warnings not included in Wall
# pedantic forces the compiler to strictly enforce ISO C standard rules
# D_XOPEN_SOURCE=700 unlocks low-level POSIX.1-2008 / Single UNIX Specification
#	(SUSv4) system function declarations in C headers. Without this macro,
#	system functions like shm_open, mmap, sigaction, nanosleep, or ftruncate
#	might trigger "implicit declaration" compiler warnings or errors when
#	compiling with strict standards like -std=c18.
LDFLAGS = -lm # for tanh and softmax

EXE = code
OBJS = perceptron.o

$(EXE): $(OBJS)
	$(CC) $(FLAGS) $(OBJS) -o $(EXE) $(LDFLAGS)

perceptron.o: perceptron.c perceptron.h
	$(CC) $(FLAGS) -c perceptron.c -o perceptron.o

clean:
	rm -f $(OBJS) $(EXE)