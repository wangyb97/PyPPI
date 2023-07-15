import threading
from multiprocessing import Pool
import os

# getSplitFile
script = '''
data_path=${TMP_DIR}/splitFile
split_program="${PRO_DIR}/utils/split.sh"
mkdir -p ${data_path}
${split_program} ${INPUT_FN} ${data_path}/
'''
os.system("bash -c '{}'".format(script))

# Defines paths to input .fasta files, output .pssm files and the uniref90 database.
TMP_DIR = os.environ.get('TMP_DIR')
fasta_path = TMP_DIR + '/splitFile/'
output_path = TMP_DIR + '/PSSM_raw/'
pssm_database_path=os.environ.get('pssm_database_path')

cmd = "mkdir -p " + output_path
os.system(cmd)

file_list = os.listdir(fasta_path)
print(file_list) # like ['2hrkB.fasta', '1ktzA.fasta']

new_file_list = []
for file_name in file_list:
    file_name_without_ext = os.path.splitext(file_name)[0]
    file_name_parts = file_name_without_ext.split('.')
    new_file_name = file_name_parts[0]
    new_file_list.append(new_file_name)
# print(new_file_list) # like ['2hrkB', '1ktzA']
print(len(new_file_list))


def get_pssm(i):
    # os.system("psiblast -query " + fasta_path + i +".fasta -db "+database_path+" -num_iterations 3 -out_ascii_pssm " +output_path+ i +".pssm -num_threads 300")
    os.system("psiblast -query " + fasta_path + i + ".fasta -db " + pssm_database_path +
              " -num_iterations 3 -out_ascii_pssm " + output_path + i + ".pssm -num_threads 3")


def process_file(i):
    get_pssm(i)
    return i


if __name__ == '__main__':
    # it is good practice to include the if __name__ == '__main__': condition,
    # when using the multiprocessing module in Python.
    pool = Pool(processes=2)  # Adjust the number of processes as per your requirement
    result = pool.map(process_file, new_file_list)
    print(len(result))

# count = 0
# for i in new_file_list:
#     get_pssm(i, count)
#     count = count+1

# if __name__ == "__main__":
#     pool = multiprocessing.Pool(3)
#     pool.map(get_pssm, range(3))
#
#     pool.close()
#     pool.join()

# count = 0
# thread_list = []
# for i in new_file_list: # the final extraction is done using a dictionary to search
#     thread = threading.Thread(target=get_pssm,
#                         args=(i, count))
#     # thread = multiprocessing.Process(target=get_pssm,
#     #                     args=(i, count))
#     thread_list.append(thread)
#     count+=1
# for i in thread_list:
#     i.start()
# for i in thread_list:
#     i.join()
