# GPGPU
Written by Daegeun

## folders
### 1. Numba
CUDA를 지원하는 Python library로 CUDA 공부 처음에 사용해 보았던 라이브러리 입니다.

### 2. PyCUDA
CUDA를 지원하는 Python library로 마지막에 정착한 라이브러리로 밑에 python file들은 모두 PyCUDA를 이용해서 작성되어 있습니다. 
이 폴더에 PyCUDA를 공부하고 python file들을 만들면서 만든 .ipynb 파일들이 들어있습니다.

### 3. lstsq
CUDA를 활용해 만든 첫번쨰 파일로 least square 문제를 경사하강법을 이용해 푸는 프레임워크 입니다.

### 4. minimum energy control
CUDA를 활용해 만든 두번째 파일로 
> $\quad min \qquad \qquad \begin{Vmatrix} u \end{Vmatrix}$ <br> 
> $subject\ to.\quad Gu = x_{des} - Q - A^nx_0$ <br> 

의 문제를 푸는 프레임워크 입니다.

