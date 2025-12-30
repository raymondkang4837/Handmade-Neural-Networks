# src/core/tensor.py

import numpy as np

class Tensor:
    '''
    PyTorch와 같은 역할을 수행하도록 최대한 비슷하게 구현
    1. 데이터 구조를 정의
    2. 연산을 수행할 때 연산 그래프 생성
    3. 자동 미분 수행

    data : numpy 배열
    grad : gradient (dL/dTensor)
    requires_grad : 역전파 계산 필요 여부
    _prev : 이 tensor를 만든 연산의 부모
    _backward() : 연산별 미분 규칙

    '''
    def __init__(self, data, requires_grad=False, name=None):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=float)

        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self._backward = lambda : None
        self._prev = set()
        self.name = name

    # 텐서 정보 출력
    def __repr__(self): 
        output = "Tensor"
        w1 = f"data={self.data}"
        w2 = f"grad={self.grad}"
        w3 = f"requires_grad={self.requires_grad}"
        w4 = f"name={self.name}"
        return output + f"({w1}, {w2}, {w3}, {w4})"
    
    # Backward : autograd, 그래프를 만들고 이에 대한 Chain rule을 적용하는 엔진
    def backward(self, grad=None):
        if not self.requires_grad:
            return # 상수 처리
        
        if grad is None: # grad 가 None이면 1로 시작
            
            if self.data.size != 1: # grad는 스칼라여야 해
                raise RuntimeError(
                    "grad must be specified for non-scalar tensor"
                )
            grad = np.ones_like(self.data) 

        # Topological order
        topo = [] # 계산 그래프의 노드들을 정렬해서 넣어둘 리스트
        visited = set() # DFS 중 중복 방문 방지

        def build(v): # DFS로 그래프 탐색 
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
                

        build(self) # TOPO 리스트를 입력 -> 출력 순으로 연산을 쌓음

        # 마지막 노드에 초기 grad 설정
        self.grad = grad

        # 역순으로 backward 호출
        for t in reversed(topo):
            t._backward()

    
    # 유틸 함수
    @staticmethod
    def _ensure_tensor(other):
        if isinstance(other, Tensor):
            return other
        return Tensor(other)
    
    # --------------
    # 기본 연산들
    # ---------------
    def __add__(self, other):
        other = Tensor._ensure_tensor(other)
        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad)
    
        def _backward():
            if self.requires_grad:
                # broadcasting 으로 grad shape 맞춰
                grad_self = out.grad

                while grad_self.ndim > self.data.ndim:
                    grad_self = grad_self.sum(axis=0)

                for i, dim in enumerate(self.data.shape):
                    if dim == 1:
                        grad_self = grad_self.sum(axis=i, keepdims=True)
                self.grad = (self.grad + grad_self) if self.grad is not None else grad_self

            if other.requires_grad:
                grad_other = out.grad
                while grad_other.ndim > other.data.ndim:
                    grad_other = grad_other.sum(axis=0)

                for i, dim in enumerate(other.data.shape):
                    if dim == 1:
                        grad_other = grad_other.sum(axis=i, keepdims=True)
                other.grad = (other.grad + grad_other) if other.grad is not None else grad_other

            out._backward = _backward
            out._prev = {self, other}
            return out
        
    def __radd__(self, other):
        return self + other
        
    def __sub__(self, other):
        other = Tensor._ensure_tensor(other)
        return self + (-other)
        
    def __rsub__(self, other):
        other = Tensor._ensure_tensor(other)
        return other - self
        
    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = -out.grad
                self.grad = (self.grad + grad_self) if self.grad is not None else grad_self

            out._backward = _backward
            out._prev = {self}
            return out
            
    def __mul__(self, other):
        other = Tensor._ensure_tensor(other)
        out = Tensor(self.data * other.data,
                requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = out.grad * other.data
                # broadcasting 보정
                while grad_self.ndim > self.data.ndim:
                    grad_self = grad_self.sum(axis=0)
                for i, dim in enumerate(self.data.shape):
                    if dim == 1:
                        grad_self = grad_self.sum(axis=i, keepdims=True)
                self.grad = (self.grad + grad_self) if self.grad is not None else grad_self

            if other.requires_grad:
                grad_other = out.grad * self.data
                while grad_other.ndim > other.data.ndim:
                    grad_other = grad_other.sum(axis=0)
                for i, dim in enumerate(other.data.shape):
                    if dim == 1:
                        grad_other = grad_other.sum(axis=i, keepdims=True)
                other.grad = (other.grad + grad_other) if other.grad is not None else grad_other

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        """
        행렬곱: (n, m) @ (m, k) -> (n, k)
        """
        other = Tensor._ensure_tensor(other)
        out = Tensor(self.data @ other.data,
                     requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = out.grad @ other.data.T
                self.grad = (self.grad + grad_self) if self.grad is not None else grad_self
            if other.requires_grad:
                grad_other = self.data.T @ out.grad
                other.grad = (other.grad + grad_other) if other.grad is not None else grad_other

        out._backward = _backward
        out._prev = {self, other}
        return out

    def T(self):
        out = Tensor(self.data.T, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = out.grad.T
                self.grad = (self.grad + grad_self) if self.grad is not None else grad_self

        out._backward = _backward
        out._prev = {self}
        return out
    
    # -------------------
    # 비선형, Reduction
    # -------------------
    def relu(self):
        out = Tensor(np.maximum(self.data, 0), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = out.grad * (self.data > 0)
                self.grad = (self.grad + grad_self) if self.grad is not None else grad_self

        out._backward = _backward
        out._prev = {self}
        return out

    def sigmoid(self):
        sig = 1 / (1 + np.exp(-self.data))
        out = Tensor(sig, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = out.grad * sig * (1 - sig)
                self.grad = (self.grad + grad_self) if self.grad is not None else grad_self

        out._backward = _backward
        out._prev = {self}
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad_self = out.grad
                # grad를 입력 shape로 브로드캐스트
                if axis is not None and not keepdims:
                    grad_self = np.expand_dims(grad_self, axis=axis)
                grad_self = np.broadcast_to(grad_self, self.data.shape)
                self.grad = (self.grad + grad_self) if self.grad is not None else grad_self

        out._backward = _backward
        out._prev = {self}
        return out

    def mean(self, axis=None, keepdims=False):
        denom = self.data.size if axis is None else self.data.shape[axis]
        out = self.sum(axis=axis, keepdims=keepdims) * (1.0 / denom)
        return out
    
    # -----------------------------
    #  기타
    # -----------------------------
    def detach(self):
        """그래프 끊고 순수 data만 사용하고 싶을 때."""
        return Tensor(self.data.copy(), requires_grad=False)