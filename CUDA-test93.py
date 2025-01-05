from nvidia.cudnn import cudnn

print("cuDNN version:", cudnn.getVersion())
print("cuDNN runtime version:", cudnn.getRuntimeVersion())