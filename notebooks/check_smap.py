import h5py
import glob


files = glob.glob(
    "raw_data/soil_moisture/*.h5"
)

print("SMAP files:", len(files))

file = files[0]

print("\nReading:")
print(file)


def show_structure(name, obj):

    if isinstance(obj, h5py.Dataset):

        print(
            "DATASET:",
            name,
            "| shape:",
            obj.shape,
            "| dtype:",
            obj.dtype
        )


with h5py.File(file, "r") as f:

    print("\n========================================")
    print("SMAP DATASET STRUCTURE")
    print("========================================")

    f.visititems(show_structure)