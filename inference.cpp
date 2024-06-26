#include "inference.h"


Inference::Inference(InferenceParams Params): mParams(Params), mRuntime(nullptr), mEngine(nullptr)
{
    if (mParams.modelFileName.find(".onnx") != std::string::npos)
    {
        build();
    }
    else
    {
        loadEngine();
    }

    input_size = mInputDims.d[0] * mInputDims.d[1] * mInputDims.d[2] * mInputDims.d[3];
    output_size = mOutputDims.d[0] * mOutputDims.d[1] * mOutputDims.d[2];
    input_buff = (float*)malloc(input_size * sizeof(float));
    output_buff = (float*)malloc(output_size * sizeof(float));
    cudaMalloc(&input_mem, input_size * sizeof(float));
    cudaMalloc(&output_mem, output_size * sizeof(float));
    if (mParams.Async)
    {
        cudaStreamCreate(&mStream);
    }
    else
    {
        bindings.emplace_back(input_mem);
        bindings.emplace_back(output_mem);
    }
    
}

Inference::~Inference()
{
    cudaFree(input_mem);
    cudaFree(output_mem);
    free(input_buff);
    free(output_buff);
   
}
bool Inference::loadEngine()
{
    std::ifstream input(mParams.modelFileName, std::ios::binary);
    if (!input)
    {
        return false;
    }
    input.seekg(0, input.end);
    const size_t fsize = input.tellg();
    input.seekg(0, input.beg);
    std::vector<char> bytes(fsize);
    input.read(bytes.data(), fsize);

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(logger));
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(bytes.data(), bytes.size()), InferDeleter());
    if (!mEngine)
        return false;
    
    int nbio = mEngine->getNbIOTensors();
    const char* inputname = mEngine->getIOTensorName(0);
    std::cout << "input name :" << inputname << std::endl;
    const char* outputname = mEngine->getIOTensorName(mEngine->getNbIOTensors() - 1);
    std::cout << "output name :" << outputname << std::endl;
    Dims input_shape = mEngine->getTensorShape(inputname);
    Dims output_shape = mEngine->getTensorShape(outputname);
    mInputDims = Dims4(input_shape.d[0], input_shape.d[1], input_shape.d[2], input_shape.d[3]);
    mOutputDims = Dims4(output_shape.d[0], output_shape.d[1], output_shape.d[2], output_shape.d[3]);
   

    return true;
}
bool Inference::build()
{
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder)
    {
        return false;
    }

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    if (!network)
    {
        return false;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser
        = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser, mParams.buildType);
    if (!constructed)
    {
        return false;
    }


    std::unique_ptr<IHostMemory> plan{ builder->buildSerializedNetwork(*network, *config) };
    if (!plan)
    {
        return false;
    }

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(logger));
    if (!mRuntime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    if (network->getNbInputs() != 1)
    {
        return false;
    }
    mInputDims = network->getInput(0)->getDimensions();
    mOutputDims = network->getOutput(0)->getDimensions();

    return true;
}

bool Inference::infer(std::string imgpath)
{

    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }


    context->setTensorAddress(mEngine->getIOTensorName(0), input_mem);
    context->setTensorAddress(mEngine->getIOTensorName(mEngine->getNbIOTensors() - 1), output_mem);

    cv::Mat frame;
    std::vector<float> factors;
    if (!preProcess(imgpath, frame, factors))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    memcpyBuffers(input_mem,input_buff, input_size * sizeof(float),cudaMemcpyHostToDevice, mParams.Async);

    bool status = false;
    if (mParams.Async)
    {
        status = context->enqueueV3(mStream);
    }
    else
    {
        status = context->executeV2(bindings.data());
    }
    
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    memcpyBuffers(output_buff, output_mem, output_size * sizeof(float), cudaMemcpyDeviceToHost,mParams.Async);

    if (mParams.Async)
    {
        cudaStreamSynchronize(mStream);
    }
    // Verify results
    std::vector<Detection>outputs = postProcess(factors);

    drawBoxes(frame, outputs, true);
    return true;
}


bool Inference::constructNetwork(std::unique_ptr<nvinfer1::IBuilder>& builder,
    std::unique_ptr<nvinfer1::INetworkDefinition>& network, std::unique_ptr<nvinfer1::IBuilderConfig>& config,
    std::unique_ptr<nvonnxparser::IParser>& parser, nvinfer1::BuilderFlag buildType)
{
    auto parsed = parser->parseFromFile(mParams.modelFileName.c_str(),
        static_cast<int32_t>(ILogger::Severity::kWARNING));
    if (!parsed)
    {
        return false;
    }
    
    config->setFlag(buildType);
    if (buildType == BuilderFlag::kINT8)
    {
        setAllDynamicRanges(network.get(), 127.0F, 127.0F);
    }
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 20);
    config->setMemoryPoolLimit(MemoryPoolType::kTACTIC_SHARED_MEMORY, 48 << 10);
    //enableDLA(builder.get(), config.get(), true);
    return true;
}

bool Inference::preProcess(std::string imgpath, cv::Mat& img, std::vector<float>& factors)
{
    img = cv::imread(imgpath);
    cv::Mat mat;
    int rh = img.rows;
    int rw = img.cols;
    int rc = img.channels();
    
    cv::cvtColor(img, mat, cv::COLOR_BGR2RGB);
    int maxImageLength = rw > rh ? rw : rh;
    if (letterBoxForSquare && (mInputDims.d[2] == mInputDims.d[3]))
    {
        factors.emplace_back(maxImageLength / 640.0);
        factors.emplace_back(maxImageLength / 640.0);
    }
    else
    {
        factors.emplace_back(img.rows / 640.0);
        factors.emplace_back(img.cols / 640.0);
    }
    cv::Mat maxImage = cv::Mat::zeros(maxImageLength, maxImageLength, CV_8UC3);
    maxImage = maxImage * 255;
    cv::Rect roi(0, 0, rw, rh);
    mat.copyTo(cv::Mat(maxImage, roi));
    cv::Mat resizeImg;
    int length = 640;
    cv::resize(maxImage, resizeImg, cv::Size(length, length), 0.0f, 0.0f, cv::INTER_LINEAR);
    resizeImg.convertTo(resizeImg, CV_32FC3, 1 / 255.0);
    rh = resizeImg.rows;
    rw = resizeImg.cols;
    rc = resizeImg.channels();
    
    for (int i = 0; i < rc; ++i) {
        cv::extractChannel(resizeImg, cv::Mat(rh, rw, CV_32FC1, input_buff + i * rh * rw), i);
    }
    return true;
}

std::vector<Detection> Inference::postProcess(std::vector<float> factors)
{
    const int outputSize = mOutputDims.d[1];
    //float* output = static_cast<float*>(output_buff);
    cv::Mat outputs(84, 8400, CV_32F, output_buff);

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    // Preprocessing output results
    std::vector<std::string> classes{ "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };
    int rows = outputs.size[0];
    int dimensions = outputs.size[1];
    bool yolov8 = false;

    // yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
    // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
    if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
    {
        yolov8 = true;
        rows = outputs.size[1];
        dimensions = outputs.size[0];

        outputs = outputs.reshape(1, dimensions);
        cv::transpose(outputs, outputs);
    }

    float* data = (float*)outputs.data;
    for (int i = 0; i < rows; ++i)
    {
        float* classes_scores = data + 4;

        cv::Mat scores(1, 80, CV_32FC1, classes_scores);
        cv::Point class_id;
        double maxClassScore;

        minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > 0.25)
        {
            confidences.push_back(maxClassScore);
            class_ids.push_back(class_id.x);

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w) * factors[0]);
            int top = int((y - 0.5 * h) * factors[1]);

            int width = int(w * factors[0]);
            int height = int(h * factors[1]);
            boxes.push_back(cv::Rect(left, top, width, height));
        }

        data += dimensions;
    }
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.5, nms_result);

    std::vector<Detection> detections{};
    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];

        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);
        result.color = cv::Scalar(dis(gen),
            dis(gen),
            dis(gen));

        result.className = classes[result.class_id];
        result.box = boxes[idx];

        detections.push_back(result);
    }

    return detections;
}


void Inference::memcpyBuffers(void* dstPtr, void const* srcPtr, size_t byteSize, cudaMemcpyKind memcpyType, bool const async)
{
        if (async)
        {
            cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, mStream);
            //cudaStreamSynchronize(mStream);
        }
        else
            cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType);
}

void Inference::drawBoxes(cv::Mat frame, std::vector<Detection> result, bool show)
{   
    std::cout << "Number of detections:" << result.size() << std::endl;
    for (int i = 0; i < result.size(); ++i)
    {
        Detection detection = result[i];

        cv::Rect box = detection.box;
        cv::Scalar color = detection.color;

        // Detection box
        cv::rectangle(frame, box, color, 2);

        // Detection box text
        std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

        cv::rectangle(frame, textBox, color, cv::FILLED);
        cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
    if (show)
    {
        float scale = 0.8;
        cv::resize(frame, frame, cv::Size(frame.cols * scale, frame.rows * scale));
        cv::imshow("Inference", frame);

        cv::waitKey(-1);
    }
}

void setAllDynamicRanges(nvinfer1::INetworkDefinition* network, float inRange, float outRange)
{
    // Ensure that all layer inputs have a scale.
    for (int i = 0; i < network->getNbLayers(); i++)
    {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbInputs(); j++)
        {
            nvinfer1::ITensor* input{ layer->getInput(j) };
            // Optional inputs are nullptr here and are from RNN layers.
            if (input != nullptr && !input->dynamicRangeIsSet())
            {
                ASSERT(input->setDynamicRange(-inRange, inRange));
            }
        }
    }

    // Ensure that all layer outputs have a scale.
    // Tensors that are also inputs to layers are ingored here
    // since the previous loop nest assigned scales to them.
    for (int i = 0; i < network->getNbLayers(); i++)
    {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbOutputs(); j++)
        {
            nvinfer1::ITensor* output{ layer->getOutput(j) };
            // Optional outputs are nullptr here and are from RNN layers.
            if (output != nullptr && !output->dynamicRangeIsSet())
            {
                // Pooling must have the same input and output scales.
                if (layer->getType() == nvinfer1::LayerType::kPOOLING)
                {
                    ASSERT(output->setDynamicRange(-inRange, inRange));
                }
                else
                {
                    ASSERT(output->setDynamicRange(-outRange, outRange));
                }
            }
        }
    }
}
void enableDLA(
    nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, int useDLACore, bool allowGPUFallback)
{
    if (useDLACore >= 0)
    {
        if (builder->getNbDLACores() == 0)
        {
            std::cerr << "Trying to use DLA core " << useDLACore << " on a platform that doesn't have any DLA cores"
                << std::endl;
            ASSERT("Error: use DLA core on a platfrom that doesn't have any DLA cores" && false);
        }
        if (allowGPUFallback)
        {
            config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        }
        if (!config->getFlag(nvinfer1::BuilderFlag::kINT8))
        {
            // User has not requested INT8 Mode.
            // By default run in FP16 mode. FP32 mode is not permitted.
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(useDLACore);
    }
}