# -- coding: utf-8 --
import argparse
import pickle


def load_dataset(pkl_path):
    with open(pkl_path, "rb") as f:
        files_lst = pickle.load(f)
        return files_lst


# 参数
parser = argparse.ArgumentParser()
parser.add_argument('--original_pkl', '-op', type=str,
                    help='original path of .pkl file', default="./datapaths/datapath_MHS_fingersafe_test.pkl")
parser.add_argument('--original_type', '-ot', type=str,
                    help='adv type of original data', default="fingersafe")
parser.add_argument('--new_pkl', '-np', type=str,
                    help='new path of data',
                    default="./datapaths/datapath_MHS_fingersafe_gamma_10_test.pkl")
parser.add_argument('--new_type', '-nt', type=str,
                    help='adv type of new data', default="fingersafe_gamma_10")
args = parser.parse_args()

# create new .pkl file from the order of certain corresponding old .pkl file
original_type = args.original_type
original_pkl = args.original_pkl
new_type = args.new_type
new_pkl = args.new_pkl
datapath = load_dataset(original_pkl)  # return a dict
print(datapath[1][1])
# replace string in datapath to new path, e.g., pgd -> min
for key_class in range(len(datapath)):
    for key_pic in range(len(datapath[1])):
        original_path = datapath[key_class + 1][key_pic + 1]  # string
        new_path = original_path.replace(original_type, new_type)
        datapath[key_class + 1][key_pic + 1] = new_path  # replace datapaths

# write to new pkl
f = open(new_pkl, 'wb')
pickle.dump(datapath, f)
print(new_pkl)
print(datapath[1][1])
