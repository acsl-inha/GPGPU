# GPGPU
Written by Daegeun

## folders
### 1. Numba
CUDA를 지원하는 python library로 CUDA 공부 처음에 사용해 보았던 라이브러리 입니다.

### 2. PyCUDA
CUDA를 지원하는 Python library로 마지막에 정착한 라이브러리로 밑에 python file들은 모두 PyCUDA를 이용해서 작성되어 있습니다. 
이 폴더에 PyCUDA를 공부하고 python file들을 만들면서 만든 .ipynb 파일들이 들어있습니다.

### 3. lstsq
CUDA를 활용해 만든 첫번쨰 파일로 least square 문제를 경사하강법을 이용해 푸는 프레임워크 입니다.

### 4. minimum energy control
CUDA를 활용해 만든 두번째 파일로 
> $\quad min \qquad \qquad \qquad \ \begin{Vmatrix} u \end{Vmatrix}$ <br> 
> $subject\ to.\quad Gu = x_{des} - Q - A^nx_0$ <br> 

의 문제를 푸는 프레임워크 입니다.

## files
### 1. constraints_for_input.py
3차원 input, $u_x$, $u_y$, $u_z$ 에 대해 u의 크기가 upper_boundary와 lower_boundary에 있도록 사영시켜주는 함수를 담고 있습니다.

kernel_function의 project_function을 살펴보세요

### 2. evaluate.py
학습의 중간중간 학습량과 학습에 따른 error의 변화를 계산하며 학습률과 학습 종료 조건을 조정하는 함수를 담고 있습니다.

kernel_function의 'get_error_vector' ~ 'get_norm_of_gradient'을 살펴보세요.

### 3. get_gradient.py
gradient를 계산하는 함수를 담고 있습니다. $\nabla f = \lVert A^T(Ax - b) \rVert$를 통해 계산합니다.

### 4. kernel_function.py
minimum energy control 문제를 풀기 위해 사용되는 모든 GPU kernel function들을 담고 있는 파일입니다.

### 5. minimum_energy_control.py
minimum energy control 문제의 문제가 정의되어 있는 파일입니다. 각 state transition matrix등의 matrix들을 정의하는 역할을 합니다.

kernel_function의 'get_gradient' ~ 'get_mva_bias'을 살펴보세요.

### 6. optimizer.py
경사하강법의 학습을 담당하고 있는 파일입니다. 
>$opt_theta = theta - learning_rate * gradient$<br>

라는 간단한 수식을 담고있습니다.

kernel_function의 'basic_optimizer'을 살펴보세요.

### 7. parameter_tuning.py
경사하강법을 가속하는 알고리즘(momentum, nesterov등)들의 parameter들을 튜닝해주는 파일입니다. 불안정해서 사용할 때 주의해주세요.

### 8. shared.py
아마 least square 문제를 푸는 도중 메모리 접근을 위해 포인터 클래스를 만든 것일겁니다.
class를 포인터로 쓰는건 상당히 유용합니다.
python으로 C처럼 별에 별거를 만들수 있습니다.

### 9. solver.py
크게 중요성이 있지는 않습니다. 앞서 나온 파일들의 함수를 조합하여 문제를 풀 뿐인 __instance__.solve() 가 다인 클래스를 담고 있는 파일입니다.

