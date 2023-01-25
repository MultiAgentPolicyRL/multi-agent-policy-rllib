if __name__ == "__main__":
    # Distribution algorithm
    rollout_fragment_length = 200
    batch_size = 6000
    num_workers = 12

    # controllo se e' la batch e' divisibile per num_workers:
    if batch_size % rollout_fragment_length != 0:
        ValueError(f"batch_size % rollout_fragment_length must be == 0")

    # 30
    batch_iterations = batch_size // rollout_fragment_length

    # 30//12 =  2
    iterations_per_worker = batch_iterations // num_workers

    remaining_iterations = batch_iterations - (iterations_per_worker * num_workers)

    workers_iterations = []

    for i in range(num_workers):
        if remaining_iterations > 0:
            iter = iterations_per_worker + 1
            remaining_iterations -= 1
        else:
            iter = iterations_per_worker

        workers_iterations.append(iter)

    print(sum(workers_iterations))
