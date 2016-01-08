# README #

### Introduction ###

This code repository aims at real-time texture-less object instance detection and tracking. This first version was revised from OpenCV source code, which implemented the TPAMI12 paper "Gradient Response Maps for Real-time Detection of Texture-less Objects".

### Requirements ###

1. OS Platform: Windows
2. [OpenCV](http://opencv.org/): The version should be below 3.0, because the 2.X APIs are different from 3.X somehow. You can download this comipled [OpenCV2.4.6](http://pan.baidu.com/s/1mgWFNHu) here just for VC10_X86.
3. [freeglut](http://freeglut.sourceforge.net/): Freegult for MSVC can be downloaded from [freeglut 3.0.0 MSVC Package](http://files.transmissionzero.co.uk/software/development/GLUT/freeglut-MSVC.zip)
4. glog: You can download source code from its github page [here](https://github.com/google/glog), then compiled it. Or you can download this comipiled [glog](http://pan.baidu.com/s/1o6TTRL4) here just for  VC10_X86.
5. [protobuf](https://developers.google.com/protocol-buffers/): You can download source code from its github page [here](https://github.com/google/protobuf), then compiled it. Or you can download this comipiled [protobuf](http://pan.baidu.com/s/1o73fIHG) here just for  VC10_X86.
6. Before you can run the code project, you should configure the correct pathes of includes, libs and dlls for aforementioned third dependencies.

### Installation and Demo ###

1. Clone the Clone the LINE2D repository

	```
	git clone https://github.com/imbinwang/LINE2D.git
	```

2. We'll call the directory that you cloned LINE2D into LINE2D_ROOT. You should modify the path variable **configFile** in `LINE2D_ROOT/src/main.cpp` to the path of file `LINE2D_ROOT/data/HBLTS8/config.prototxt` and change the pathes in `LINE2D_ROOT/data/HBLTS8/config.prototxt` correspondly

	```
    // in main.cpp, change this path
    configFile = "D:\\project\\VSProject\\LINE2D4IKEA\\Bin_LINE2DIKEA\\data\\HBLTS8\\config.prototxt";
	```
	```
    // in LINE2D_ROOT/data/HBLTS8/config.prototxt, change these pathes
    img_dir: "D:\\project\\VSProject\\LINE2D4IKEA\\Bin_LINE2DIKEA\\data\\HBLTS8\\templates\\HBLTS8_70_cut"
	pose_path: "D:\\project\\VSProject\\LINE2D4IKEA\\Bin_LINE2DIKEA\\data\\HBLTS8\\poses\\HBLTS8_70_cut.txt"
    ...   
	```

3. Run and see the results in `LINE2D_ROOT/data/HBLTS8/snapshot`

### Beyond the demo ###

Before you try you own data, you should scan the file `LINE2D_ROOT/protobuf/Config.prototxt` and understand the meaning of each paramter. Then you can write a configura file for you own data like `LINE2D_ROOT/data/HBLTS8/config.prototxt`.


