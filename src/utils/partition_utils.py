import torch

def partitions(elementCount, windowSize, index):
    # Initialize the starting point for the first window
    start = 0
    end = index % windowSize
    if end == 0:
        end = windowSize
    output = [[start, end]]

    start = end

    while start < elementCount:
        end = start + windowSize

        # Ensure the end doesn't exceed the elementCount
        if end > elementCount:
            end = elementCount

        output.append([start, end])

        # Update the start for the next window
        start = end

def partitions_wrap_around(elementCount, windowSize, index):
    # Initialize the starting point for the first window
    start = 0
    end = index % windowSize
    if end == 0:
        end = windowSize
    output = [[start, end]]

    start = end

    while start < elementCount:
        end = start + windowSize

        # Ensure the end doesn't exceed the elementCount
        if end > elementCount:
            end = elementCount

        output.append([start, end])

        # Update the start for the next window
        start = end
    # get the start and end intervals
    start_interval = output[0]
    end_interval = output[-1]

    # if the length of the intervals combines is less than the window size
    # then combine the intervals in a tuple
    start_length = start_interval[1] - start_interval[0]
    end_length = end_interval[1] - end_interval[0]
    if (start_length + end_length) <= windowSize:
        # remove the start and end intervals
        output = output[1:-1]
        # add a new interval that has both
        output.append((start_interval, end_interval))

    return output


def create_initial_windows(elementCount, windowSize):
    output = []
    start = 0
    while start < elementCount:
        end = start + windowSize

        if end > elementCount:
            end = elementCount

        # Ensure the end doesn't exceed the elementCount
        if end > elementCount:
            end = elementCount

        output.append([start, end])

        # Update the start for the next window
        start = end

    return output

def partition_wrap_around_2(elementCount, windowSize, index, offset=1):
    # Initialize the starting point for the first window
    initial_windows = create_initial_windows(elementCount, windowSize)
    actual_offset = (index * offset) % windowSize

    # for each window shift the start and end 
    # by the half length
    shifted_windows = []
    for i, window in enumerate(initial_windows):
        window[0] += actual_offset
        window[1] += actual_offset

        # ensure the end doesn't exceed the element count
        if i == len(initial_windows) - 1:
            window[1] = elementCount
            initial_window = [0, actual_offset]
            shifted_windows.append((initial_window, window))

        else:
            shifted_windows.append(window)

    return shifted_windows

def shifted_passes(elementCount, windowSize):
    even_pass = partition_wrap_around_2(elementCount, windowSize, 0)
    odd_pass = partition_wrap_around_2(elementCount, windowSize, 1)

    return [even_pass, odd_pass]

def shifted_passes_2(elementCount, windowSize):
    passes = []
    for i in range(windowSize):
        p = partitions_wrap_around(elementCount, windowSize, i)
        passes.append(p)

    return passes

def circular_shift(element_count, offset):
    offset = offset % element_count
    initial_indices = list(range(element_count))
    # circular shift the indices by the offset
    shifted_indices = initial_indices[offset:] + initial_indices[:offset]

    return shifted_indices

# create a list of start and end indices up to element 
# where the length is max window_size and after the 
# first element include last_n elements
def last_n_indices(element_count, window_size, last_n):
    assert(last_n <= window_size)
    last_n = min(last_n, window_size)

    output = []

    output.append([0, window_size])

    start = window_size
    increment_amount = window_size - last_n

    while start < element_count:
        # Determine end index for the window based on window_size
        end = start + increment_amount
        
        # Adjust end index if it exceeds element_count
        end = min(end, element_count)
        
        output.append([start, end])
        
        # Update start for the next iteration (moving it by window_size - last_n)
        start = end
        
    return output     
        
def peel_next_and_new(tensor, last_n, last_count):
    # slice the latents portion after last_n_count 
    new_latents = tensor[:, :, last_n:]
    last_n_latents = tensor[:, :, -last_n:]
    return new_latents, last_n_latents

if __name__ == "__main__":
    element_count = 30
    window_size = 24
    index = 1
    tensor = torch.tensor([[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]])
    result = peel_next_and_new(tensor, 7, 3)

    new_tensor = torch.tensor([[[11, 12, 13]]])
    result = torch.concat([result[1], new_tensor], dim=2)
    print(result)