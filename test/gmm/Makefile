CXX = g++

CXXFLAGS = -I/usr/include/python3.8 \
           -Iinclude \
           -I/usr/include/python3.8/lib/site-packages/numpy/core/include \
           -L/usr/include/python3.8/libs \
           -lpython3.8 \
           -lm \
           -g

SRC_DIR = src
BUILD_DIR = build
LIB_DIR = lib

SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES = $(SRC_FILES:.cpp=.o)

OUTPUT = $(BUILD_DIR)/output

all: $(OUTPUT)

$(OUTPUT): $(SRC_FILES)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(SRC_FILES) -o $(OUTPUT) $(CXXFLAGS)

clean:
	rm -rf $(BUILD_DIR)/*.o $(OUTPUT)

run: all
	./$(OUTPUT)
