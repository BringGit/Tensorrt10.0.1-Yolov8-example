
#pragma once
#include "NvInfer.h"
//#include <cuda_runtime_api.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include "NvOnnxParser.h"
#include <random>
using namespace nvinfer1;

#define ASSERT(condition)                                                   \
    do                                                                      \
    {                                                                       \
        if (!(condition))                                                   \
        {                                                                   \
            std::cout << "Assertion failure: " << #condition << std::endl;  \
            exit(EXIT_FAILURE);                                                       \
        }                                                                   \
    } while (0)



class Logger : public ILogger
{
	void log(Severity severity, const char* msg) noexcept override
	{
		if (severity <= Severity::kWARNING)
			std::cout << msg << std::endl;
	}
};

struct InferDeleter
{
	template <typename T>
	void operator()(T* obj) const
	{
		delete obj;
	}
};

struct Detection
{
	int class_id{ 0 };
	std::string className{};
	float confidence{ 0.0 };
	cv::Scalar color{};
	cv::Rect box{};
};

struct InferenceParams
{
	std::string modelFileName;
	bool Async;
	nvinfer1::BuilderFlag buildType;
};
void setAllDynamicRanges(nvinfer1::INetworkDefinition* network, float inRange = 2.0F, float outRange = 4.0F);
void enableDLA(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, int useDLACore, bool allowGPUFallback = true);
class Inference
{
public:
	Inference(InferenceParams Params);
	~Inference();
	bool build();
	bool infer(std::string imgpath);

private:

	bool loadEngine();

	bool constructNetwork(std::unique_ptr<nvinfer1::IBuilder>& builder,
		std::unique_ptr<nvinfer1::INetworkDefinition>& network, std::unique_ptr<nvinfer1::IBuilderConfig>& config,
		std::unique_ptr<nvonnxparser::IParser>& parser, nvinfer1::BuilderFlag buildType);
	void memcpyBuffers(void* dstPtr, void const* srcPtr, size_t byteSize, cudaMemcpyKind memcpyType, bool const async=false);
	bool preProcess(std::string imgpath, cv::Mat& img, std::vector<float>& factors);
	std::vector<Detection> postProcess(std::vector<float> factors);
	void drawBoxes(cv::Mat frame, std::vector<Detection> result,bool show);

	std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to deserialize the engine
	std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
	cudaStream_t mStream=0;
	InferenceParams mParams;
	
	nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
	nvinfer1::Dims mOutputDims;

	size_t input_size;
	size_t output_size;
	Logger logger;
	std::vector<void*> bindings;
	void* input_mem{ nullptr };
	void* output_mem{ nullptr };

	float* input_buff;
	float* output_buff;

	bool letterBoxForSquare = true;
};
