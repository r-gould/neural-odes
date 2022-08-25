from operator import add, mul, truediv, neg

class OperableList(list):

    def __add__(self, right_inp):
        
        if isinstance(right_inp, OperableList):
            if len(self) != len(right_inp):
                raise ValueError("Input lengths do not match.")
            return OperableList(map(add, self, right_inp))

        elif isinstance(right_inp, list):
            return self + OperableList(right_inp)

        elif isinstance(right_inp, (int, float)):
            right_list = [right_inp for _ in range(len(self))]
            return self + OperableList(right_list)
        
        return NotImplemented()

    def __radd__(self, left_inp):
        return self + left_inp

    def __sub__(self, right_inp):
        return self + (-right_inp)

    def __rsub__(self, left_inp):
        return -self + left_inp

    def __mul__(self, right_inp):

        if isinstance(right_inp, OperableList):
            if len(self) != len(right_inp):
                raise ValueError("Input lengths do not match.")
            return OperableList(map(mul, self, right_inp))
        
        elif isinstance(right_inp, list):
            return self * OperableList(right_inp)

        elif isinstance(right_inp, (int, float)):
            right_list = [right_inp for _ in range(len(self))]
            return self * OperableList(right_list)
        
        return NotImplemented()

    def __rmul__(self, left_inp):
        return self * left_inp

    def __truediv__(self, right_inp):

        if isinstance(right_inp, OperableList):
            if len(self) != len(right_inp):
                raise ValueError("Input lengths do not match.")
            return OperableList(map(truediv, self, right_inp))
        
        elif isinstance(right_inp, list):
            return self / OperableList(right_inp)

        elif isinstance(right_inp, (int, float)):
            right_list = [right_inp for _ in range(len(self))]
            return self / OperableList(right_list)
        
        return NotImplemented()

    def __neg__(self):
        return OperableList(map(neg, self))