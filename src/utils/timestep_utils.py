
def sliding_window(timesteps, frames_count):
    duplicated = [[t] * frames_count for t in timesteps]

    extended = []
    for i, dup in enumerate(duplicated):
        padding_at_beginning = i
        padding_at_end = len(duplicated) - i - 1

        none_padded_dup = [None] * padding_at_beginning + dup + [None] * padding_at_end

        extended.append(none_padded_dup)

    transposed = [list(row) for row in zip(*extended)]

    # filter out None
    transposed = [[t for t in row if t is not None][::-1] for row in transposed]

    return transposed

if __name__ == "__main__":
  timesteps = [1, 2, 3, 4]
  total_count = 3

  result = sliding_window(timesteps, total_count)
  for r in result:
      print(r)