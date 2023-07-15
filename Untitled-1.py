input_file ="seq_355.fasta"
output_file = "filtered_seq.fasta"

with open(input_file, "r") as f:
    lines = f.readlines()

filtered_lines = []
keep_sequence = False

for line in lines:
    print(line)
    if line.startswith(">Q"):
        keep_sequence = True
    elif line.startswith(">"):
        keep_sequence = False

    if keep_sequence:
        filtered_lines.append(line)

with open(output_file, "w") as f:
    f.writelines(filtered_lines)

print("Filtered sequences have been written to", output_file)

# # # import os

# # # fasta_file = "seq_355.fasta"
# # # pssm_folder = "pssm_raw"
# # # output_fasta = "filtered_seq.fasta"
# # # output_folder = "filtered_pssm"

# # # # Create the output folder if it doesn't exist
# # # os.makedirs(output_folder, exist_ok=True)

# # # sequences_to_keep = set()

# # # with open(fasta_file, "r") as f:
# # #     lines = f.readlines()

# # # for i in range(0, len(lines), 2):
# # #     header = lines[i].strip()
# # #     sequence = lines[i + 1].strip()

# # #     if header.startswith(">Q"):
# # #         sequences_to_keep.add(header)
# # #         sequences_to_keep.add(sequence)

# # # with open(output_fasta, "w") as f:
# # #     for header, sequence in zip(lines[::2], lines[1::2]):
# # #         if header.strip() in sequences_to_keep:
# # #             f.write(header)
# # #             f.write(sequence)

# # # pssm_files = os.listdir(pssm_folder)

# # # for pssm_file in pssm_files:
# # #     if pssm_file.startswith("O"):
# # #         pssm_path = os.path.join(pssm_folder, pssm_file)
# # #         output_pssm_path = os.path.join(output_folder, pssm_file)
# # #         with open(pssm_path, "rb") as src_file, open(output_pssm_path, "wb") as dst_file:
# # #             dst_file.write(src_file.read())

# # # print("Filtered sequences have been written to", output_fasta)
# # # print("Filtered PSSM files have been saved in the", output_folder, "folder.")

# # import os

# # pssm_folder = "pssm_raw"
# # input_file = "seq_355.fasta"
# # output_file = "filtered_seq.fasta"

# # # Get the list of file names in the pssm folder
# # pssm_files = os.listdir(pssm_folder)

# # # Read the input file
# # with open(input_file, "r") as f:
# #     lines = f.readlines()

# # # Filter the sequences based on file names
# # filtered_sequences = []
# # current_sequence = []
# # preserve_sequence = False

# # for line in lines:
# #     if line.startswith(">"):
# #         sequence_name = line[1:].strip()

# #         # Check if the file name is not in the pssm folder
# #         if sequence_name not in [pssm_file.split(".")[0] for pssm_file in pssm_files]:
# #             preserve_sequence = True
# #         else:
# #             preserve_sequence = False

# #         if current_sequence and preserve_sequence:
# #             filtered_sequences.extend(current_sequence)

# #         current_sequence = [line] if preserve_sequence else []
# #     else:
# #         if preserve_sequence:
# #             current_sequence.append(line)

# # # Add the last sequence if it needs to be preserved
# # if current_sequence and preserve_sequence:
# #     filtered_sequences.extend(current_sequence)

# # # Write the filtered sequences to the output file
# # with open(output_file, "w") as f:
# #     f.writelines(filtered_sequences)

# # print("Filtered sequences saved to", output_file)

# import os

# pssm_folder = "pssm_raw"
# input_file = "seq_355.fasta"
# output_file = "filtered_seq.fasta"

# # Get the list of file names in the pssm folder
# pssm_files = [os.path.splitext(filename)[0] for filename in os.listdir(pssm_folder)]

# # Read the input file
# with open(input_file, "r") as f:
#     lines = f.readlines()

# # Filter the sequences based on file names
# filtered_sequences = []
# current_sequence = []
# preserve_sequence = False

# for line in lines:
#     if line.startswith(">"):
#         sequence_name = line[1:].strip()

#         # Check if the file name is not in the pssm folder
#         if sequence_name not in pssm_files:
#             preserve_sequence = True
#         else:
#             preserve_sequence = False

#         if current_sequence and preserve_sequence:
#             filtered_sequences.extend(current_sequence)

#         current_sequence = [line] if preserve_sequence else []
#     else:
#         if preserve_sequence:
#             current_sequence.append(line)

# # Add the last sequence if it needs to be preserved
# if current_sequence and preserve_sequence:
#     filtered_sequences.extend(current_sequence)

# # Write the filtered sequences to the output file
# with open(output_file, "w") as f:
#     f.writelines(filtered_sequences)

# print("Filtered sequences saved to", output_file)




# import os

# pssm_folder = "pssm_raw"
# input_file = "seq_355.fasta"
# output_file = "filtered_seq.fasta"

# # Get the list of file names in the pssm folder
# pssm_files = [os.path.splitext(filename)[0] for filename in os.listdir(pssm_folder)]

# # Read the input file
# with open(input_file, "r") as f:
#     lines = f.readlines()

# # Filter the sequences based on file names
# filtered_sequences = []
# current_sequence = []
# preserve_sequence = False

# for line in lines:
#     if line.startswith(">"):
#         sequence_name = line[1:].strip()

#         # Check if the file name is not in the pssm folder
#         if sequence_name not in pssm_files:
#             preserve_sequence = True
#         else:
#             preserve_sequence = False

#         if current_sequence and preserve_sequence:
#             filtered_sequences.extend(current_sequence)

#         current_sequence = [line] if preserve_sequence else []
#     else:
#         if preserve_sequence:
#             current_sequence.append(line)

# # Add the last sequence if it needs to be preserved
# if current_sequence and preserve_sequence:
#     filtered_sequences.extend(current_sequence)

# # Write the filtered sequences to the output file
# with open(output_file, "w") as f:
#     f.writelines(filtered_sequences)

# print("Filtered sequences saved to", output_file)

