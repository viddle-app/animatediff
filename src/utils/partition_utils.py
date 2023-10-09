from typing import Any, List
import torch
from torch import FloatTensor
from typing import List
import numpy as np

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

# need explain to chatgpt how to do this
# def partition_wrap_around_2(elementCount, windowSize, index, offset=1):


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


class Range:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __str__(self):
        return f"Range({self.start}, {self.end})"
    
    __repr__ = __str__

class Slice3D:
    def __init__(self, 
                 ts:Range, 
                 xs:Range, 
                 ys:Range,
    ): 
        self.ts = ts
        self.xs = xs
        self.ys = ys

    def __str__(self):
        return f"Slice3D(ts={self.ts}, xs={self.xs}, ys={self.ys})"

    __repr__ = __str__

class Offset3D:
    def __init__(self,
                 t: int, x: int, y: int):
        self.t = t
        self.x = x
        self.y = y

    def __str__(self):
        return f"Offset3D(t={self.t}, x={self.x}, y={self.y})"
    
    __repr__ = __str__
        

# Assuming FloatTensor is a pre-defined type or class
# If FloatTensor is a typo or if it's just an example, you can replace it with the correct type
class WeightedSlice3D:
    def __init__(self, 
                 ts:Range, 
                 xs:Range, 
                 ys:Range,
                 weights,  # type FloatTensor is removed for clarity unless you define it
    ): 
        self.ts = ts
        self.xs = xs
        self.ys = ys
        self.weights = weights

    def __str__(self):
        return f"WeightedSlice3D(ts={self.ts}, xs={self.xs}, ys={self.ys}, weights={self.weights})"
    
    __repr__ = __str__

def compute_slices(region: Range, window: Range, offset: int, max_limit: int, wrap_around: bool):
    slices = []
    
    # Starting position
    pos = region.start
    
    # Keep track of the positions we have already processed to avoid infinite loops
    processed_positions = set()
    
    while pos < region.end and pos not in processed_positions:
        processed_positions.add(pos)
        
        end_pos = pos + window.end - window.start
        
        if wrap_around and end_pos > max_limit:
            # Add the portion until max_limit
            slices.append(Range(pos, max_limit))
            
            # Calculate the overflow and wrap around
            overflow = end_pos - max_limit
            slices.append(Range(region.start, region.start + overflow))
            
            # Next starting position considering the offset and wrap-around
            pos = region.start + overflow
        else:
            slices.append(Range(pos, min(end_pos, region.end)))
            pos += offset  # Move to the next starting position by the offset value
            
    return slices





def multi(whole_region: Slice3D, window: Slice3D, offset: Offset3D, wrap_around: bool):
    ts_slices = compute_slices(whole_region.ts, window.ts, offset.t, whole_region.ts.end, wrap_around)
    xs_slices = compute_slices(whole_region.xs, window.xs, offset.x, whole_region.xs.end, wrap_around)
    ys_slices = compute_slices(whole_region.ys, window.ys, offset.y, whole_region.ys.end, wrap_around)
    
    # Forming the slices
    slices = []
    for t in ts_slices:
        for x in xs_slices:
            for y in ys_slices:
                slices.append(Slice3D(t, x, y))
    
    return slices


def compute_weights(slice_range: Range, window: Range, offset: Range, max_limit: int):
    # Calculate the overlap range
    overlap_start = max(slice_range.start, window.start + offset.start)
    overlap_end = min(slice_range.end, window.end + offset.start)
    
    overlap_length = max(0, overlap_end - overlap_start)
    
    # Initialize weights to 1
    weights = np.ones(slice_range.end - slice_range.start)
    
    if overlap_length > 0:
        # Linearly interpolate weights for the overlap region
        weights[overlap_start - slice_range.start: overlap_end - slice_range.start] = \
            np.linspace(0, 1, overlap_length)
    
    return weights

def multi_with_weights(whole_region: Slice3D, window: Slice3D, offset: Slice3D, wrap_around: bool):
    ts_slices = compute_slices(whole_region.ts, window.ts, offset.ts, whole_region.ts.end, wrap_around)
    xs_slices = compute_slices(whole_region.xs, window.xs, offset.xs, whole_region.xs.end, wrap_around)
    ys_slices = compute_slices(whole_region.ys, window.ys, offset.ys, whole_region.ys.end, wrap_around)
    
    # Forming the weighted slices
    weighted_slices = []
    for t in ts_slices:
        for x in xs_slices:
            for y in ys_slices:
                ts_weights = compute_weights(t, window.ts, offset.ts, whole_region.ts.end)
                xs_weights = compute_weights(x, window.xs, offset.xs, whole_region.xs.end)
                ys_weights = compute_weights(y, window.ys, offset.ys, whole_region.ys.end)
                
                # Form a 3D weight tensor by broadcasting the individual weights
                weights = ts_weights[:, None, None] * xs_weights[None, :, None] * ys_weights[None, None, :]
                weighted_slices.append(WeightedSlice3D(t, x, y, weights))
    
    return weighted_slices

def partition_sliding(elementCount, windowSize, offset, index):
    actual_offset = offset - (index % offset)

    start = 0
    end = windowSize - actual_offset
    partitions = [[start, end]]

    start = end
    while start < elementCount:
        end = start + windowSize
        partitions.append([start, min(end, elementCount)])
        start = end

    # if the last entry is not greater than window size - actual_offset
    # then combine the last two entries
    if partitions[-1][1] - partitions[-1][0] <= windowSize - actual_offset:
        partitions[-2][1] = partitions[-1][1]
        partitions = partitions[:-1]

    return partitions

def partition_sliding_2(elementCount, windowSize, offset):
    if windowSize <= 0 or offset <= 0 or elementCount <= 0:
        raise ValueError("Inputs must be positive integers.")

    # Calculate how many steps to move forward after each window
    step = windowSize - offset

    # Initialize starting point of the window
    start = 0
    end = start + windowSize
    windows = []

    # Function to calculate the weights for a window
    def calculate_weights(window_idx, total_windows):
        weights = [1.0] * windowSize

        # If it's not the first window, handle the overlap at the beginning
        if window_idx > 0:
            for i in range(offset):
                blend_weight = i / float(offset)
                weights[i] = blend_weight

        # If it's not the last window, handle the overlap at the end
        if window_idx < total_windows - 1:
            for i in range(windowSize - offset, windowSize):
                blend_weight = 1.0 - (i - (windowSize - offset)) / float(offset)
                weights[i] = blend_weight

        return weights

    # First window
    windows.append([start, end, calculate_weights(0, 2)])
    start += step

    window_idx = 1
    while end + step < elementCount:
        end = start + windowSize
        windows.append([start, end, calculate_weights(window_idx, 3)])

        # Move the window forward
        start += step
        window_idx += 1

    # Last window, might be truncated
    end = min(start + windowSize, elementCount)
    windows.append([start, end, calculate_weights(window_idx, window_idx)])

    return windows



if __name__ == "__main__":
    # element_count = 64
    # window_size = 16
    # index = 0
    # offset = 4
    # 
    # result = partition_sliding_2(element_count, window_size, offset)
    # print(result)

    wrap_around = True
    whole_region = Slice3D(Range(0, 2), Range(0, 2), Range(0, 2))
    window = Slice3D(Range(0, 1), Range(0, 1), Range(0, 1))
    offset = Offset3D(1, 1, 1)
# 
    result = multi(whole_region, window, offset, wrap_around)
    print(result)