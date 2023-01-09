class banana:
    def __init__(self, value:int):
        self.value = 2

    def __sum__(self, other):
        self.value += other.value


A = banana(1)
B = banana(2)
