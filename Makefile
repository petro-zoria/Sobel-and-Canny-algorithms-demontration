CC      = gcc
CFLAGS  = -O2 -Wall
LDFLAGS = -lm
TARGET  = vision

# Camera device per OS (override on command line if needed)
# Linux:   DEVICE="/dev/video0"  FORMAT=v4l2
# macOS:   DEVICE="0"            FORMAT=avfoundation
# Windows: DEVICE="video=..."    FORMAT=dshow
DEVICE  ?= 0
FORMAT  ?= avfoundation
WIDTH   ?= 640
HEIGHT  ?= 480
FPS     ?= 15

all: $(TARGET)

$(TARGET): vision.c
	$(CC) $(CFLAGS) -o $(TARGET) vision.c $(LDFLAGS)

# Live edge detection — side-by-side window (original | processed)
live-sobel: $(TARGET)
	ffmpeg -f $(FORMAT) -i "$(DEVICE)" \
	       -vf scale=$(WIDTH):$(HEIGHT) \
	       -fflags nobuffer -flags low_delay \
	       -f image2pipe -vcodec ppm -r $(FPS) - \
	| ./$(TARGET) sobel \
	| ffplay -f image2pipe -vcodec pgm \
	         -fflags nobuffer -flags low_delay -framedrop \
	         -window_title "Vision: Sobel" -

live-canny: $(TARGET)
	ffmpeg -f $(FORMAT) -i "$(DEVICE)" \
	       -vf scale=$(WIDTH):$(HEIGHT) \
	       -fflags nobuffer -flags low_delay \
	       -f image2pipe -vcodec ppm -r $(FPS) - \
	| ./$(TARGET) canny \
	| ffplay -f image2pipe -vcodec pgm \
	         -fflags nobuffer -flags low_delay -framedrop \
	         -window_title "Vision: Canny" -

clean:
	rm -f $(TARGET)
