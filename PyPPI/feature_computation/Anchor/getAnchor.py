import os

if os.path.isfile('../txtFeatures/Anchor.txt'):
    print('The Anchor file has already been generated, please do not generate it again!')
    exit(1)

# Define the shell script
script = '''
echo split start!

data_path=${TMP_DIR}/splitFile

split_program="${PRO_DIR}/utils/split.sh"
mkdir -p ${data_path}
${split_program} ${INPUT_FN} ${data_path}/

echo split over!
'''
os.system("bash -c '{}'".format(script))


TMP_DIR = os.environ.get('TMP_DIR')
data_path=TMP_DIR + '/splitFile'
data_list = os.listdir(data_path)
# print('data_list:',data_list)

for i in reversed(data_list):
    a = "python './iupred2a/iupred2a.py' -a " + '\''+data_path + '/' + i+'\'' +' long'
    os.system(a)

