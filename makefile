
CC = gcc
CFLAGS = -Wall -Wextra -std=c99
LDFLAGS = -lm  # Add additional libraries if needed

# Define the source files and corresponding object files
SRCS = main.c neuron.c layer.c network.c data.c training_set.c math_util.c
OBJS = $(SRCS:.c=.o)

# Define the executable name
EXECUTABLE = neural_network

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(EXECUTABLE) $(LDFLAGS)

# Compile each source file into an object file
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXECUTABLE)