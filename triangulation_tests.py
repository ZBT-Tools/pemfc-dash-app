"""
IMplementation of "Explore"-Function to  identify parameter limits and gradient-based refinement

Delaunay Tests

Sources:
    https://docs.scipy.org/doc/scipy/tutorial/spatial.html
"""
import copy
import math

from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


def testfunction(xy):
    val = xy[0] ** 2 + xy[1] ** 2
    #if val < 5:
    #    return -9999
    #if val > 100:
    #    return -99999999
    #else:
    #    return val
    # return math.sin(xy[0]) + math.cos(xy[1])
    # return xy[0]**2 +  xy[1]**2

    x, y = xy
    a = 0.2
    val = x + np.exp(-((x ** 2 + y ** 2 - 0.75 ** 2) ** 2) / a ** 4)

    if (math.sqrt(x ** 2 + y **2)>0.75) and (y >0) and (x>0):
        return -99999999
    elif math.sqrt(x ** 2 + y **2)<0.2:
        return -99999999
    else:
        return val




def initial_sampling(n_x: int = 5, n_y: int = 5) -> np.ndarray:
    """
    returns np.ndarray. Each row represents one coordinate set:
    coords =  [ [x1,y1],
                [x2,y2],
                    ...]
    """
    x = (np.linspace(-1, 1, n_x))
    y = (np.linspace(-1, 1, n_y))
    coords = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    return coords


def calc_z(coords):
    return np.apply_along_axis(testfunction, 1, coords)


def calc_refinement_coords(tri: Delaunay, point_results: np.ndarray, n: int = 5) -> (list, float):
    """
    - Calculation of max absolute differences for each triangle,

    Input:
    tri: Delaunay()-Object
    point_results: (tri.points with additional z column): n_p(index),x,y,z

    Simple refinement logic:
    Refine at location of highest value difference.
    Remove points of triangles, consist only of error values AND point difference below defined
    distance limit.

    """
    error_val = -99999999
    distance_limit = 0.05

    local_tri_simplices = copy.deepcopy(tri.simplices)
    local_point_results = copy.deepcopy(tri.simplices)

    # Part 1: Identify triangles with error values on all three points.
    # Those will be removed from refinement if
    # Coo


    # Create array with tri.simplices array and values
    local_tri_values = np.c_[
        tri.simplices, [[point_results[p1, 2], point_results[p2, 2], point_results[p3, 2]]
                        for p1, p2, p3 in tri.simplices]]

    # Identify triangles only consist of


    # Add column to tri with height difference
    simpl_diff = np.c_[
        tri.simplices, [max([point_results[p1, 2], point_results[p2, 2], point_results[p3, 2]]) - \
                        min([point_results[p1, 2], point_results[p2, 2], point_results[p3, 2]])
                        for p1, p2, p3 in tri.simplices]]

    # Sorted indices based on difference
    max_idxs = np.argsort(simpl_diff[:, 3])[::-1]

    # Define refinement points:
    # looping through tri.simplices in order of max_idxs
    # Only refine between points with distance > limit.
    # This ensures refinement and also consinders None-Value locations

    new_coords = np.empty((0, 2), float)
    ct = 0
    while len(new_coords[:, 0]) < n:
        old_pts = tri.simplices[max_idxs[ct]]
        old_pts_coords = [tri.points[pt] for pt in old_pts]

        p1 = old_pts_coords[0]
        p2 = old_pts_coords[1]
        p3 = old_pts_coords[2]

        if math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) > distance_limit:
            p12 = np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2])
            new_coords = np.append(new_coords, np.array([p12]), axis=0)
        else:
            pass

        if math.sqrt((p1[0] - p3[0]) ** 2 + (p1[1] - p3[1]) ** 2) > distance_limit:
            p13 = np.array([(p1[0] + p3[0]) / 2, (p1[1] + p3[1]) / 2])
            new_coords = np.append(new_coords, np.array([p13]), axis=0)
        else:
            pass

        if math.sqrt((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2) > distance_limit:
            p23 = np.array([(p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2])
            new_coords = np.append(new_coords, np.array([p23]), axis=0)
        else:
            pass

        # Drop identical entries
        new_coords = np.unique(new_coords, axis=0)

        ct += 1


    return new_coords


if __name__ == "__main__":
    """
    

    """

    # Initial Calculation Step
    # ------------------------------

    # Calculation of initial sampling points (default: 5x5 points)
    coords = initial_sampling()
    # Delaunay Triangulation
    tri = Delaunay(coords)
    # Calculation of z values:
    z = calc_z(coords)
    # Combine coordinates and results
    results = np.c_[coords, z]

    # Delaunay-Object, herein tri, has points-Attribute with point index and coordinates.
    # Create array with point index, coordinates and results in the order of tri.points
    coords_results = \
        np.c_[tri.points,
        [results[(results[:, 0] == k) & (results[:, 1] == v), 2][0] for k, v in tri.points]]

    # Refinement Step
    # -----------------
    # For each triangle calc max diff, find n largest differences
    # Information about variables:
    #   Points forming trinangle: tri.simplices, index is trangle number  n_t(index),p1,p2,p3
    #   Coordinates of points in tri.points                               n_p(index), x,y
    #   coords_results (tri.points with additional z column)              n_p(index),x,y,z
    for _ in range(200):
        print(_)
        try:
            # Identify triangles to refine, drop faults, calc new coords
            coords_new = calc_refinement_coords(tri, coords_results)

            # Calculation of new z values:
            z_new = calc_z(coords_new)
            # Combine new coordinates and new results
            results_new = np.c_[coords_new, z_new]

            # Combine coords
            coords = np.append(coords, coords_new, axis=0)

            # Delaunay Triangulation
            tri = Delaunay(coords)

            # Combine old and new results
            results = np.append(results, results_new, axis=0)

            # Delaunay-Object, herein tri, has points-Attribute with point index and coordinates.
            # Create array with point index, coordinates and results in the order of tri.points
            coords_results = \
                np.c_[tri.points,
                [results[(results[:, 0] == k) & (results[:, 1] == v), 2][0] for k, v in tri.points]]

            if _ % 25 == 0:
                # # Plot Triangles
                # # ----------------------------------------------------------------------------------------------
                # fig1, ax1 = plt.subplots()
                # plt.triplot(coords[:, 0], coords[:, 1], tri.simplices)
                # plt.plot(coords[:, 0], coords[:, 1], 'o')
                #
                # for j, p in enumerate(coords):
                #     plt.text(p[0] - 0.03, p[1] + 0.03, j, ha='right')  # label the points
                # for j, s in enumerate(tri.simplices):
                #     p = coords[s].mean(axis=0)
                #     plt.text(p[0], p[1], '#%d' % j, ha='center')  # label triangles
                # plt.show(block=False)

                # Plot Contour
                # ----------------------------------------------------------------------------------------------
                fig1, ax1 = plt.subplots()

                ax1.set_aspect('equal')
                levels = np.linspace(-1.5, 2, 50)
                tcf = ax1.tricontourf(results[:, 0], results[:, 1], results[:, 2], levels)
                plt.triplot(coords[:, 0], coords[:, 1], tri.simplices,color ='white',linewidth=0.2)
                plt.show(block=False)
            else:
                pass



        except IndexError:
            break




