# Day 1 Optional Challenge Solution: Manual Strides Calculator

def compute_strides(shape, itemsize=4):
    """
    Computes standard row-major strides for a given shape.
    itemsize: size of one element in bytes (e.g. 4 for float32).
    """
    strides = []
    current_stride = itemsize
    for dim in reversed(shape):
        strides.append(current_stride)
        current_stride *= dim
    return list(reversed(strides))

def get_element(flat_tensor, strides, indices):
    """
    Retrieves an element from a flattened 1D list using strides and indices.
    """
    offset = 0
    for stride, idx in zip(strides, indices):
        offset += idx * stride
    # Convert byte offset back to index (assuming 4-byte elements)
    element_index = offset // 4
    return flat_tensor[element_index]

# Verification
shape = (2, 3, 4)
strides = compute_strides(shape, itemsize=4)
print("Computed Strides (bytes):", strides)

# Simulated flattened data: 0 to 23
flat_data = list(range(24))

# Retrieve element at index (1, 2, 3)
indices = (1, 2, 3)
val = get_element(flat_data, strides, indices)
print(f"Value at indices {indices}: {val} (Expected: 23)")
