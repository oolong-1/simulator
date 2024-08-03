CXX = g++
CXXFLAGS = -I/usr/include/opencv4 -I. -std=c++11 -Wall -Wextra -g
LDFLAGS = -lopencv_stitching -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect -lopencv_dnn_superres -lopencv_dpm -lopencv_highgui -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor -lopencv_quality -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_shape -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_superres -lopencv_optflow -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_ml -lopencv_videostab -lopencv_videoio -lopencv_viz -lopencv_ximgproc -lopencv_video -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core -lstdc++fs

TARGET = output/test
SRCS = run/main.cpp run/read.cpp npu/pre.cpp npu/cim.cpp npu/vpu.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LDFLAGS)

run/main.o: run/main.cpp
	$(CXX) $(CXXFLAGS) -c run/main.cpp -o run/main.o

run/read.o: run/read.cpp
	$(CXX) $(CXXFLAGS) -c run/read.cpp -o run/read.o

npu/pre.o: npu/pre.cpp
	$(CXX) $(CXXFLAGS) -c npu/pre.cpp -o npu/pre.o

npu/cim.o: npu/cim.cpp
	$(CXX) $(CXXFLAGS) -c npu/cim.cpp -o npu/cim.o

npu/vpu.o: npu/vpu.cpp
	$(CXX) $(CXXFLAGS) -c npu/vpu.cpp -o npu/vpu.o

clean:
	rm -f $(TARGET) $(OBJS)
