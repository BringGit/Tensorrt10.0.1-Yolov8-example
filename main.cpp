#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <iostream>
#include <fstream>

#include <random>
#include <cuda.h>
#include "inference.h"
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace std;




int main()
{
	InferenceParams params;
	params.Async = true;
	params.modelFileName = "yolov8m.engine";
	params.buildType = BuilderFlag::kTF32;
	Inference inf = Inference(params);
	/*if (!inf.build())
	{
		return -1;
	}*/
	if (!inf.infer("E:/Project/DATA/bus.jpg"))
	{
		return -1;
	}

}
