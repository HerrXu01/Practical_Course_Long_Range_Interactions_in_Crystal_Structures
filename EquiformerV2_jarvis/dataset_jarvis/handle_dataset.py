from pathlib import Path
import lmdb
import pickle


# path = Path("/nfs/homedirs/xzh/equiformer_v2/datasets/oc20/s2ef/2M/train")
# path = Path("/nfs/homedirs/xzh/equiformer_v2/dataset_jarvis/s2ef")
path = Path("/nfs/homedirs/xzh/equiformer_v2/dataset_jarvis/jarvis_lmdb")

print(path.is_file())
db_paths = sorted(path.glob("*.lmdb"))
print(db_paths)
metadata_path = path / "metadata.npz"
print(metadata_path)

env = lmdb.open(str(db_paths[0]), subdir=False, readonly=True)
txn = env.begin()
i = 0

# print(txn.get("length".encode("ascii")))
# print(pickle.loads(txn.get("length".encode("ascii"))))

print("first")
element_1 = pickle.loads(txn.get("1".encode("ascii")))
print(element_1)
print(type(element_1))
print({k: v for k, v in element_1.__dict__.items() if v is not None})


'''

for key, value in txn.cursor():
    print(key)
    print(value)

'''