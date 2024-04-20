import os
import random
import shutil
from itertools import islice

output_folder_path = "Dataset/SplitData"
input_folder_path = "Dataset/all"
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["fake", "real"]

try:
    shutil.rmtree(output_folder_path)
    print("Folder deleted successfully")
except OSError as e:
    os.mkdir(output_folder_path)

# -------- Directories to create ---------
os.makedirs(f"{output_folder_path}/Train/images", exist_ok=True)
os.makedirs(f"{output_folder_path}/Train/labels", exist_ok=True)

os.makedirs(f"{output_folder_path}/Val/images", exist_ok=True)
os.makedirs(f"{output_folder_path}/Val/labels", exist_ok=True)

os.makedirs(f"{output_folder_path}/Test/images", exist_ok=True)
os.makedirs(f"{output_folder_path}/Test/labels", exist_ok=True)

# -------- Get the names ---------
listNames = os.listdir(input_folder_path)
uniqueNames = []

for name in listNames:
    uniqueNames.append(name.split(".")[0])
    uniqueNames = list(set(uniqueNames))

# ------ Shuffle --------
random.shuffle(uniqueNames)

# -------- Number of images in each folder ---------
lenData = len(uniqueNames)
lenTrain = int(lenData * splitRatio["train"])
lenVal = int(lenData * splitRatio["val"])
lenTest = int(lenData * splitRatio["test"])

# --------  Put remaining images in Training -----------
if lenData != lenTrain + lenTest + lenVal:
    remaining = lenData - (lenTrain + lenTest + lenVal)
    lenTrain += remaining

# --------  Split the list -----------
lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]
print(
    f"Total Images:{lenData} \nSplit: {len(Output[0])} {len(Output[1])} {len(Output[2])}"
)

# --------  Copy the files  -----------

sequence = ["train", "val", "test"]
for i, out in enumerate(Output):
    for fileName in out:
        shutil.copy(
            f"{input_folder_path}/{fileName}.jpg",
            f"{output_folder_path}/{sequence[i]}/images/{fileName}.jpg",
        )
        shutil.copy(
            f"{input_folder_path}/{fileName}.txt",
            f"{output_folder_path}/{sequence[i]}/labels/{fileName}.txt",
        )

print("Split Process Completed...")


# -------- Creating Data.yaml file  -----------

dataYaml = f"path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}"


f = open(f"{output_folder_path}/data.yaml", "a")
f.write(dataYaml)
f.close()

print("Data.yaml file Created...")
