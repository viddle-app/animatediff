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

if __name__ == "__main__":
    elementCount = 10
    windowSize = 3
    index = 3
    result = partitions(elementCount, windowSize, index)
    print(result)