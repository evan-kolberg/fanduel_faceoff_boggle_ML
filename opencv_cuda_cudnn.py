import cv2


def print_cuda_device_info():
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            for i in range(count):
                device_info = cv2.cuda.DeviceInfo(i)
                print(f"Device {i}: {device_info.name()}")
                print(f"  Compute capability: {device_info.majorVersion()}.{device_info.minorVersion()}")
                print(f"  Total memory: {device_info.totalMemory() / (1024 * 1024)} MB")
        else:
            print("No CUDA devices found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print_cuda_device_info()
