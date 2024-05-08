import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def converter(onnx_filename, trt_filename, half):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)

    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    builder_config = builder.create_builder_config()
    builder_config.max_workspace_size = 3 << 30

    if half:
        builder_config.set_flag(trt.BuilderFlag.FP16)

    with open(onnx_filename, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print (parser.get_error(error))

    plan = builder.build_serialized_network(network, builder_config)
    
    with trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(plan)
    
    with open(trt_filename, 'wb') as f:
        f.write(engine.serialize())