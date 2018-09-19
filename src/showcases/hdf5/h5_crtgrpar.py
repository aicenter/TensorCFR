#!/usr/bin/env python3

#
# This example creates HDF5 file group.h5 and group MyGroup in it.
# Absolute and relative paths are used to create groups in MyGroup.
#
import h5py

#
# Use 'w' to remove existing file and create a new one; use 'w-' if
# create operation should fail when the file already exists.
#
print("Creating HDF5 file group.h5...")
file = h5py.File('group.h5', 'w')
#
# Create a group with the name "MyGroup"
#
print("Creating group MyGroup in the file...")
group = file.create_group("MyGroup")
#
# Create group "Group_A" in group MyGroup
#
print("Creating group Group_A in MyGroup using absolute path...")
group_a = file.create_group("/MyGroup/Group_A")
#
# Create group "Group_B" in group MyGroup
#
print("Creating group Group_B in MyGroup using relative path...")
group_b = group.create_group("Group_B")
#
# Print the contents of MyGroup group
#
print("Printing members of MyGroup group:", group.keys())
#
# Close the file before exiting; H5Py will close the groups we created.
#
file.close()
