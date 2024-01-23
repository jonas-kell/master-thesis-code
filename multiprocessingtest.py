import multiprocessing as mp

NCORE = 4


def fast_generator1():
    for line in range(200):
        yield line  # initial production


def slow_generator(lines):
    def gen_to_queue(input_q, lines):
        # This function simply consume our generator and write it to the input queue
        for line in lines:
            input_q.put(line)
        for _ in range(NCORE):  # Once generator is consumed, send end-signal
            input_q.put(None)

    def process(input_q, output_q):
        while True:
            line = input_q.get()
            if line is None:
                output_q.put(None)
                break
            output_q.put(line - 1)  # heavy computational task

    input_q = mp.Queue(maxsize=NCORE * 2)
    output_q = mp.Queue(maxsize=NCORE * 2)

    # Here we need 3 groups of worker :
    # * One that will consume the input generator and put it into a queue. It will be `gen_pool`. It's ok to have only 1 process doing this, since this is a very light task
    # * One that do the main processing. It will be `pool`.
    # * One that read the results and yield it back, to keep it as a generator. The main thread will do it.
    gen_pool = mp.Pool(1, initializer=gen_to_queue, initargs=(input_q, lines))
    pool = mp.Pool(NCORE, initializer=process, initargs=(input_q, output_q))

    finished_workers = 0
    while True:
        line = output_q.get()
        if line is None:
            finished_workers += 1
            if finished_workers == NCORE:
                break
        else:
            yield line


def fast_generator2(lines):
    for line in lines:
        yield line + 1  # light computational task


if __name__ == "__main__":
    lines = fast_generator1()
    lines = slow_generator(lines)
    lines = fast_generator2(lines)
    for line in lines:
        print(line)
