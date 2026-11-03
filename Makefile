CC = gcc
FLAGS = -std=c18 -Wall -Wextra -pedantic -D_XOPEN_SOURCE=700 -D_DEFAULT_SOURCE -g
LDFLAGS = -lm

EXE = code
OBJS = perceptron.o

$(EXE): $(OBJS)
	$(CC) $(FLAGS) $(OBJS) -o $(EXE) $(LDFLAGS)

perceptron.o: perceptron.c perceptron.h
	$(CC) $(FLAGS) -c perceptron.c -o perceptron.o

clean:
	rm -f $(OBJS) $(EXE)