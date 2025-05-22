# import tensorflow as tf
# print("GPU 可用:", tf.config.list_physical_devices('GPU'))
# print("CUDA 支持:", tf.test.is_built_with_cuda())
# print("cuDNN 支持:", tf.test.is_built_with_cudnn())
# import tensorflow as tf
# a = tf.constant(1.)
# b = tf.constant(2.)
# print(a+b)
# print(tf.__version__)
# print(tf.test.gpu_device_name())
# print('GPU:',tf.config.list_physical_devices('GPU'))
# print('CPU:',tf.config.list_physical_devices('CPU'))
# print(tf.test.is_gpu_available())


import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")