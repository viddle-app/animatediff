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

if __name__ == "__main__":
    elementCount = 30
    windowSize = 9
    index = 1
    result = partition_wrap_around_2(elementCount, windowSize, index)
    print(result)