from multiprocessing import Lock, Process, Queue, current_process
import time
import queue  # imported for using queue.Empty exception


def do_job(input, output):
    classe = stupid_class()
    while True:
        data = input.get()
        data += 1
        print("in process")
        output.put(classe)


class stupid_class:
    def __init__(self):
        a = 1


def main():
    number_of_processes = 4
    input = [Queue() for _ in range(number_of_processes)]
    output = [Queue() for _ in range(number_of_processes)]

    # creating processes
    processes = []
    for w in range(number_of_processes):
        p = Process(target=do_job, args=(input[w], output[w]))
        processes.append(p)
        p.start()

    for i in range(20):
        for out, inp in zip(output, input):
            inp.put(i)
            print(out.get())

    # # completing process
    # for p in processes:
    #     p.join()

    # # print the output
    # while not tasks_that_are_done.empty():
    #     print(tasks_that_are_done.get())

    # return True


if __name__ == "__main__":
    main()
