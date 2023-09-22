import torch

def partitions(elementCount, windowSize, index, offset):
    # Initialize the starting point for the first window
    start = 0
    end = (index * offset) % windowSize
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

    return output

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

def convolution_partitions(element_count, window_size, offset):
    # create a window of size window_size
    # and shift it by offset
    # repeat this until the window reaches the end
    # of the element_count
    output = []
    start = 0
    end = window_size
    while end <= element_count:
        output.append([start, end])
        start += offset
        end += offset

    # add the last window to the output list
    # if the end of the window exceeds the element_count
    if end > element_count:
        output[-1] = [output[-1][0], element_count]

    return output

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

def indices_for_averaging(frame_count, window_size, offset):
    # Initialize the list to store the (start, end) indices
    result = []

    # Start from the first frame
    start = 0
    end = start + window_size
    result.append((start, end))
    # Continue generating windows until the last end index reaches `frame_count - 1`
    while end < frame_count:
        
        # Move to the next window using the offset
        start += offset
        end += offset

        result.append((start, end))

    # Handle the edge case where the last window's end is beyond `frame_count - 1`
    if end > frame_count and len(result) > 0:
        result[-1] = (result[-1][0], frame_count)
    return result

if __name__ == "__main__":
    element_count = 40
    window_size = 16
    index = 8
    offset = 12
    result = indices_for_averaging(element_count, window_size, offset)
    print(result)