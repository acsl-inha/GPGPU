import numpy as np
import sys
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from time import time
import math



class Shared:
    def __init__(self, A, b, learning_rate, constrained=None):
        ## input matrix's shape
        self.length = A.shape[0]
        self.width = A.shape[1]
        
        ## in CPU
        self.A = np.float32(A)
        self.b = np.float32(b)
        self.theta = np.float32(np.zeros(self.width))
        self.out = np.float32(np.zeros(self.length))
        self.grad = np.float32(np.zeros(self.width))

        ## in GPU
        self.GPU_A = gpuarray.to_gpu(self.A.reshape(self.length*self.width))
        self.GPU_b = gpuarray.to_gpu(self.b)
        self.GPU_theta = gpuarray.to_gpu(self.theta)
        self.GPU_out = gpuarray.to_gpu(self.out)
        self.GPU_grad = gpuarray.to_gpu(self.grad)

        ## for initialize out vector and grad vector
        self.init_out = gpuarray.empty_like(self.GPU_out)
        self.init_grad = gpuarray.empty_like(self.GPU_grad)

        ## TPB: thread_per_block, BPG: block_per_grid        
        self.TPB, self.BPG = self.optimal_block_size(self.length)

        ## learning_rate
        self.learning_rate = learning_rate

        ## for constrained lstsq
        if not constrained:
            pass

        else:
            self.constrained_unpacking(constrained)

    def optimal_block_size(self, n):
        
        thread_per_block = int(math.sqrt(n / 2))

        block_per_grid = int(n / thread_per_block) + 1


        return thread_per_block, block_per_grid

    def constrained_unpacking(self, constrained):
        self.constrained = np.zeros((2,self.width))
        self.constrained[0,:] += sys.float_info.max
        self.constrained[1,:] -= sys.float_info.max

        for i in range(constrained.shape[0]):
            index = constrained[0,i]
            if not constrained[1,i]:
                self.constrained[0,index] = constrained[1,i]
            else:
                pass

            if not constrained[2,i]:
                self.constrained[1,index] = constrained[2,i]
            else:
                pass

        self.constrained = np.float32(self.constrained)

        self.GPU_constrained = gpuarray.to_gpu(self.constrained)

    def momentum(self):
        self.s = self.learning_rate
        self.beta = 1/3

    def nesterov(self):
        self.GPU_p_theta = gpuarray.empty_like(self.theta)
        self.alpha = self.learning_rate



class GetGradient:
    def __init__(self, shared):
       self.shared = shared
       
       self.kernel_function()

    def run(self):

        self.initialize()

        ## get out = np.dot(A, theta) - b
        self.first(self.shared.GPU_out,
                   self.shared.GPU_A,
                   self.shared.GPU_theta,
                   self.shared.GPU_b,
                   np.int32(self.shared.length),
                   np.int32(self.shared.width),
                   block=(self.shared.TPB,1,1),
                   grid=(self.shared.BPG,1,1))

        ## get grad = np.dot(A.T, out)
        self.second(self.shared.GPU_grad,
                   self.shared.GPU_A,
                   self.shared.GPU_out,
                   np.int32(self.shared.TPB),
                   np.int32(self.shared.BPG),
                   np.int32(self.shared.length),
                   np.int32(self.shared.width),
                   block=(self.shared.BPG,1,1),
                   grid=(self.shared.width,1,1))
                   
                   

    def kernel_function(self):
        ## block=(thread_per_block,1,1), grid=(block_per_grid,1,1)
        first_ker_function = \
        """
        #define x (threadIdx.x + blockIdx.x * blockDim.x)

        __global__ void first(float* out, float* A, float* theta, float* b, int length, int width) {
            
            if (x < length) {
                for (int j = 0; j < width; j++) {
                    int index1 = x * width + j;

                    out[x] += A[index1] * theta[j];
                    }

                out[x] -= b[x];
            }
        }
        """
        first_ker = SourceModule(first_ker_function)



        ## block=(block_per_grid,1,1), grid=(width,1,1)
        second_ker_function = \
        """
        #define tx (threadIdx.x)
        #define bx (blockIdx.x)

        __global__ void second(float* grad, float* A, float* out, int thread_per_block, int block_per_grid, int length, int width) {

            __shared__ float grad_jerk[1000];

            grad_jerk[tx] = 0;

            __syncthreads();
            
            for (int i = 0; i < thread_per_block; i++) {
                int index1 = tx * thread_per_block + i;
                int index2 = index1 * width + bx;
                
                grad_jerk[tx] += A[index2] * out[index1];
            }

            __syncthreads();

            if (tx == 0) {
                for (int i = 0; i < block_per_grid; i++) {
                    grad[bx] += grad_jerk[i];
                }
            }
            else {
                grad_jerk[1000-tx] = 0;
            }

            __syncthreads();
        }
        """
        second_ker = SourceModule(second_ker_function)

        self.first = first_ker.get_function("first")
        self.second = second_ker.get_function("second")

    def initialize(self):
        self.shared.GPU_out[:] = self.shared.init_out[:]



class Optimizer:
    def __init__(self, shared):
        self.shared = shared

    def run(self):
        return NotImplementedError()

    def kernel_function(self):

        ## block=(width,1,1), grid=(1,1,1)
        constrained_projection_ker_function = \
        """
        #define x (threadIdx.x)

        __global__ void constrained_projection (float* theta, float* constrained, int width) {
            int upper = x;
            int downer = x + width;

            if (theta[x] > constrained[upper]){
                if (theta[x] > constrained[downer]) {
                    theta[x] = constrained[upper];
                }
                else {}
            }
            else {
                if (theta[x] < constrained[downer]) {
                    theta[x] = constrained[downer];
                }
                else {}
            }
        }
        """
        constrained_projection_ker = SourceModule(constrained_projection_ker_function)

        self.constrained_projection = constrained_projection_ker.get_function("constrained_projection")

    def initialize(self):
        return NotImplementedError()



class GradientMethod(Optimizer):
    def __init__(self, shared):
        super().__init__(shared)
        super().kernel_function()

        self.kernel_function()

    def run(self):

        self.gradient_method(self.shared.GPU_theta,
                             self.shared.GPU_grad,
                             np.float32(self.shared.learning_rate),
                             block=(self.shared.width,1,1),
                             grid=(1,1,1))

        self.initialize()
        
    def kernel_function(self):

        ## block=(width,1,1), grid=(1,1,1)
        gradient_method_ker_function = \
        """
        #define x (threadIdx.x)

        __global__ void gradient_method (float* theta, float* grad, float learning_rate) {
            theta[x] -= learning_rate * grad[x];

            __syncthreads();
        }
        """
        gradient_method_ker = SourceModule(gradient_method_ker_function)

        self.gradient_method = gradient_method_ker.get_function("gradient_method")

    def initialize(self):
        self.shared.GPU_grad[:] = self.shared.init_grad[:]



class MomentumMethod(Optimizer):
    def __init__(self, shared):
        super().__init__(shared)
        super().kernel_function()

        self.kernel_function()

    def run(self):

        self.momentum_method(self.shared.GPU_theta,
                             self.shared.GPU_grad,
                             np.float32(self.shared.s),
                             block=(self.shared.width,1,1),
                             grid=(1,1,1))
                             
        self.initialize()

    def kernel_function(self):

        ## block=(width,1,1), grid=(1,1,1)
        momentum_method_ker_function = \
        """
        #define x (threadIdx.x)

        __global__ void momentum_method (float* theta, float* grad, float s) {
            theta[x] -= s * grad[x];

            __syncthreads();
        }
        """
        momentum_method_ker = SourceModule(momentum_method_ker_function)
        
        ## block=(width,1,1), grid=(1,1,1)
        momentum_ker_function = \
        """
        #define x (threadIdx.x)

        __global__ void momentum (float* grad, float beta) {

            grad[x] *= beta;
        }
        """
        momentum_ker = SourceModule(momentum_ker_function)

        self.momentum_method = momentum_method_ker.get_function("momentum_method")
        self.momentum = momentum_ker.get_function("momentum")

    def initialize(self):
        self.momentum(self.shared.GPU_grad,
                      np.float32(self.shared.beta),
                      block=(self.shared.width,1,1),
                      grid=(1,1,1))
                      


class NesterovMethod(Optimizer):
    def __init__(self, shared):
        super.__init__(shared)
        super().kernel_function()

        self.kernel_function()

    def run(self, iter):

        self.nesterov_method(self.shared.GPU_theta,
                             self.shared.GPU_grad,
                             np.float32(self.shared.learning_rate),
                             block=(self.shared.width,1,1),
                             grid=(1,1,1))

        self.initialize(iter)

    def kernel_function(self):

        ## block=(width,1,1), grid=(1,1,1)
        ## theta = y - alpha*grad(y)
        nesterov_method_ker_function = \
        """
        #define x (threadIdx.x)
        __global__ void nesterov_method (float* theta, float* grad, float learning_rate) {

            theta[x] -= learning_rate * grad[x];
        }
        """
        nesterov_method_ker = SourceModule(nesterov_method_ker_function)

        ## block=(width,1,1), grid=(1,1,1)
        ## y = theta + u * (theta - p_theta)
        nesterov_ker_function = \
        """
        #define x (threadIdx.x)

        __global__ void nesterov (float* theta, float* p_theta, int iter) {
            
            float u = 1 - 3 / (5 + iter);

            theta[x] += u * (theta[x] - p_theta[x]);

            __syncthreads();

            p_theta[x] = theta[x];

            __syncthreads();
        }
        """
        nesterov_ker = SourceModule(nesterov_ker_function)

        self.nesterov_method = nesterov_method_ker.get_function("nesterov_method")
        self.nesterov = nesterov_ker.get_function("nesterov")
        
    def initialize(self, iter):
        
        self.nesterov(self.shared.GPU_theta,
                      self.shared.GPU_p_theta,
                      np.int32(iter),
                      block=(self.shared.width,1,1),
                      grid=(1,1,1))



class LeastSquare:
    def __init__(self, A, b, learning_rate, epoches=10, iteration=5, optimize_method="GD", constrained=None):
        ## shared
        self.shared = Shared(A, b, learning_rate)

        ## gradient
        self.get_gradient = GetGradient(self.shared)
        
        ## optimizer
        if optimize_method == "GD":
            self.optimizer = GradientMethod(self.shared)
        
        elif optimize_method == "momentum":
            self.optimizer = MomentumMethod(self.shared)
            self.shared.momentum()

        elif optimize_method == "Nesterov":
            self.optimizer = NesterovMethod(self.shared)
            self.shared.nesterov()
            
        else:
            return NotImplementedError()

        ## epoches, iteration
        self.epoches = epoches
        self.iteration = iteration

        ## constrained


        ## error log
        self.error = np.zeros(epoches)

    def solve(self):
        for epoch in range(self.epoches):
            for iter in range(self.iteration):
                ## get gradient
                self.get_gradient.run()

                ## optimize
                self.optimizer.run()

    def record_error(self, epoch):

        self.error[epoch] = np.linalg.norm(self.shared.GPU_out.get())