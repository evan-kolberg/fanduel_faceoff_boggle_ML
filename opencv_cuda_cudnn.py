import numpy as np
import cv2 as cv
npTmp = np.random.random((1024, 1024)).astype(np.float32)
npMat1 = npMat2 = npMat3 = npDst = np.stack([npTmp,npTmp],axis=2)
cuMat1 = cuMat2 = cuMat3 = cuDst = cv.cuda_GpuMat(npMat1)
%timeit cv.cuda.gemm(cuMat1, cuMat2,1,cuMat3,1,cuDst,1)
