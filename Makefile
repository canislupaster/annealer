# Makefile for C++ project

# Compiler settings
CXX = clang++
CXXFLAGS = -std=c++23 -Wall -pthread -g

# Source files
SRCS = annealer.cpp parser.cpp annealer_old.cpp
HEADERS = parser.hh annealer_old.hpp

# Executable name
BUILD = build
TARGET = annealer

# Default target
all: $(TARGET)-fast

# Rule to create the executable
$(BUILD)/$(TARGET)-fast: $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS) -Ofast

# Rule to create the executable
$(BUILD)/$(TARGET)-debug: $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS) -O0 -fsanitize=address

# Run the program
run: $(BUILD)/$(TARGET)-fast
	./$(BUILD)/$(TARGET)-fast

# Clean up generated files
clean:
	rm -f $(BUILD)/$(TARGET)-*

# Phony targets
.PHONY: all clean run
