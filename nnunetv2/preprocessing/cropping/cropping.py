from typing import List

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_opening, ball
from skimage.filters import threshold_triangle

# Hello! crop_to_nonzero is the function you are looking for. Ignore the rest.
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, crop_to_bbox, bounding_box_to_slice
from acvl_utils.morphology.morphology_helper import remove_all_but_largest_component, generate_ball


def create_nonzero_mask(data):
    """

    :param data:
    :return: the mask is True where the data is nonzero
    """
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = data[0] != 0
    for c in range(1, data.shape[0]):
        nonzero_mask |= data[c] != 0
    return binary_fill_holes(nonzero_mask)


def get_nonzero_bbox(data: np.ndarray) -> List[List[int]]:
    """
    data: (c, x, y(, z)). The bounding box of the voxels that are nonzero in any channel, half open ([start, end) per
    spatial axis); an image without nonzero voxels gives the full extent (like acvl_utils' get_bbox_from_mask). This is
    also the box of create_nonzero_mask(data) although that mask has its holes filled: a zero voxel outside the box
    reaches the image border along a straight line of zero voxels, so it is not a hole.

    acvl_utils' get_bbox_from_mask scans slices from both ends along every axis, and the slices along all but the
    outermost axis in memory are strided, each reading (nearly) every cache line: a box a few dozen voxels away from
    the borders (zero background, as in MRI) cost many full passes. Here only the outermost axis is scanned. If the
    remaining slab is nonzero on all of its other borders (CT, where air is nonzero), the box spans those axes and
    the faces are all that is read; otherwise one reduction onto a plane gives them. np.any casts in small buffers, so
    no full-size mask is allocated.
    """
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    # scan the spatial axis that is outermost in memory (preprocessing passes transposed views)
    order = sorted(range(data.ndim - 1), key=lambda ax: -abs(data.strides[ax + 1]))
    arr = data.transpose(0, *[ax + 1 for ax in order])
    spatial = arr.shape[1:]

    start = 0
    while start < spatial[0] and not arr[:, start].any():
        start += 1
    if start == spatial[0]:
        return [[0, s] for s in data.shape[1:]]
    end = spatial[0]
    while not arr[:, end - 1].any():
        end -= 1
    bbox = [[start, end]]

    slab = arr[:, start:end]
    if all(slab[(slice(None),) * (ax + 1) + (0,)].any() and slab[(slice(None),) * (ax + 1) + (-1,)].any()
           for ax in range(1, len(spatial))):
        bbox += [[0, s] for s in spatial[1:]]
    else:
        plane = np.zeros(spatial[1:], dtype=bool)
        for c in range(arr.shape[0]):
            plane |= np.any(slab[c], axis=0)
        for ax in range(plane.ndim):
            idx = np.flatnonzero(plane.any(axis=tuple(a for a in range(plane.ndim) if a != ax)))
            bbox.append([int(idx[0]), int(idx[-1]) + 1])
    # back to the axis order of data
    return [bbox[order.index(ax)] for ax in range(len(order))]


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    # Crop first, then build the nonzero mask (binary_fill_holes) of the cropped image only. Same mask: everything
    # outside the box is zero and connected to the image border, so cropping does not change which voxels are holes.
    bbox = get_nonzero_bbox(data)
    slicer = (slice(None), ) + bounding_box_to_slice(bbox)
    data = data[slicer]
    nonzero_mask = create_nonzero_mask(data)[None]
    if seg is not None:
        seg = seg[slicer]
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    else:
        seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label))
    return data, seg, bbox


def create_brain_mask(data):
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    brain_mask = None
    for c in range(0, data.shape[0]):
        # Threshold with triangle method. Otsu usually works well also but triangler might be better for PET
        thresh = threshold_triangle(data[c])
        brain_mask_c = data[c] > thresh
        # Clean
        brain_mask_c = binary_opening(brain_mask_c, ball(3))
        # Retrieve largest region
        brain_mask_c = remove_all_but_largest_component(brain_mask_c)
        if brain_mask is None:
            brain_mask = brain_mask_c
        else:
            brain_mask |= brain_mask_c
    return brain_mask


def crop_to_brain(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_brain_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)
    nonzero_mask = nonzero_mask[slicer][None]
    
    slicer = (slice(None), ) + slicer
    data = data[slicer]
    if seg is not None:
        seg = seg[slicer]
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    else:
        seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label))
    return data, seg, bbox

