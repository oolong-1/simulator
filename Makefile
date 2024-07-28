CXX = g++
CXXFLAGS = -I/usr/include/opencv4 -I./include -std=c++11
LDFLAGS = -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

TARGET = test
SRC = src/main.cpp src/preprocess.cpp src/convolution.cpp src/activation.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

