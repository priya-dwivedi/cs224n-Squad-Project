import numpy as np

a = np.random.randint(10, size=10)

print(a)

# b = a[6:10]
#
# print(b)
#
# pos = np.argmax(b)
#
# print(pos)

d = dict(enumerate(a,2))

print(d)


kernel_sizes = [2,4,6,10,15, 20,25, 50]

for i, filter_size in enumerate(kernel_sizes):

    print("Filter size", filter_size)