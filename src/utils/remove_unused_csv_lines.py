import csv
import os

# Define input and output CSV file names
input_csv = "data/output.csv"
output_csv = "output.csv"
video_folder = "data/videos"

# Create a list to store valid lines
valid_lines = []

# Read the CSV file
with open(input_csv, 'r') as infile:
    reader = csv.reader(infile)
    for row in reader:
        # Build the video file name from the CSV line
        video_file_name = row[0] + ".mp4"
        video_path = os.path.join(video_folder, video_file_name)

        # Check if the video file exists
        if os.path.exists(video_path):
            valid_lines.append(row)

# Write the valid lines to the new CSV file
with open(output_csv, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(valid_lines)

print(f"Filtered data written to {output_csv}")
