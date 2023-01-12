from utils import exec_time


@exec_time
def prova():
    a = 1290381203 % 3 == 0


@exec_time
def prova1():
    a = 1290381203 // 3 == 0


if __name__ == "__main__":
    prova()
    prova1()
