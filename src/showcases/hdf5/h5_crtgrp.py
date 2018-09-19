#!/usr/bin/env python3

#
# This example creates an HDF5 file group.h5 and a group MyGroup in it
# using H5Py interfaces to the HDF5 library.
#
import h5py

# Use 'w' to remove existing file and create a new one; use 'w-' if
# create operation should fail when the file already exists.
#
print("Creating an HDF5 file with the name group.h5...")
file = h5py.File('group.h5', 'w')
#
# Show the Root group which is created when the file is created.
#
print("When an HDF5 file is created, it has a Root group with the name '", file.name, "'.")
#
# Create a group with the name "MyGroup"
#
print("Creating a group MyGroup in the file...")
group = file.create_group("MyGroup")
#
# Print the content of the Root group
#
print("An HDF5 group is a container for other objects; ...")
print("Show the members of the Root group using dictionary key method:", file.keys())
#
# Another way to show the content of the Root group.
print("Show the members of the Root group using the list function:", list(file))
#
# Close the file before exiting; H5Py will close the group.
#
file.close()
