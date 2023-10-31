"""
Gradient-based refinement for two parameter

Motivation:
Implementation of "Explore"-Function to pemfc Dash App for identification of parameter
relations and limits

Description:

Sources:
    https://docs.scipy.org/doc/scipy/tutorial/spatial.html
"""
import copy
import math
from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt


def testfunction(xy):
    """
    Source: https://github.com/python-adaptive/adaptive
    """
    x, y = xy
    a = 0.2
    val = x + np.exp(-((x ** 2 + y ** 2 - 0.75 ** 2) ** 2) / a ** 4)

    if (math.sqrt(x ** 2 + y ** 2) > 0.75) and (y > 0) and (x > 0):
        return -99999999
    elif math.sqrt(x ** 2 + y ** 2) < 0.2:
        return -99999999
    else:
        return val


def initial_sampling(n_x: int, n_y: int, bounds_x: list, bounds_y: list) -> np.ndarray:
    """
    Equally distributed initial samling coordinates
    returns np.ndarray.
    Each row represents one coordinate set:
    coords =  [ [x1,y1],
                [x2,y2],
                    ...]
    """
    x = (np.linspace(bounds_x[0], bounds_x[1], n_x))
    y = (np.linspace(bounds_y[0], bounds_y[1], n_y))
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


class Refinement2D:
    def __init__(self, name, vorname, geb_datum, gewicht):
        self.name = name
        self.vorname = vorname
        self.geb_datum = geb_datum
        self.gewicht = gewicht

if __name__ == "__main__":

    # Initialization
    # ------------------------------
    # Calculation of initial sampling points
    coords = initial_sampling(n_x=5, bounds_x=[-1, 1], n_y=5, bounds_y=[-1, 1])

    # Initial Calculation Step
    # ------------------------------
    # Delaunay Triangulation
    tri = Delaunay(coords)
    # Calculation of z values:
    z = calc_z(coords)
    # Combine coordinates and results
    results = np.c_[coords, z]

    # Delaunay-Object, here "tri", has points-Attribute with point index and coordinates.
    # Create array with point index, coordinates and results in the order of tri.points
    coords_results = \
        np.c_[tri.points,
        [results[(results[:, 0] == k) & (results[:, 1] == v), 2][0] for k, v in tri.points]]

    # Refinement Step
    # -----------------
    # 1.) Create new points based on refinement algorithm
    # 2.) Perform calculation for new points and add to dataset
    # Information about variables:
    #   Points forming trinangle: tri.simplices, index is trangle number  n_t(index),p1,p2,p3
    #   Coordinates of points in tri.points                               n_p(index), x,y
    #   coords_results (tri.points with additional z column)              n_p(index),x,y,z
    for _ in range(200):
        print(_)
        try:
            # 1.) Create new points based on refinement algorithm
            coords_new = calc_refinement_coords(tri, coords_results)

            # 2.) Perform calculation for new points and add to dataset
            # Merge prior and newly added points
            coords = np.append(coords, coords_new, axis=0)

            # Calculation of new z values:
            z_new = calc_z(coords_new)
            # Combine point coordinates and results
            results_new = np.c_[coords_new, z_new]

            # Merge prior and newly added results
            results = np.append(results, results_new, axis=0)

            # Update Delaunay Triangulation
            tri = Delaunay(coords)

            # Create array with point index, coordinates and results in the order of tri.points
            coords_results = \
                np.c_[tri.points,
                [results[(results[:, 0] == k) & (results[:, 1] == v), 2][0] for k, v in tri.points]]

            if _ % 25 == 0:
                # # Plot Triangles
                # # --------------------------------------------------------------------------------
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
                # ----------------------------------------------------------------------------------
                fig1, ax1 = plt.subplots()

                ax1.set_aspect('equal')
                levels = np.linspace(-1.5, 2, 50)
                tcf = ax1.tricontourf(results[:, 0], results[:, 1], results[:, 2], levels)
                plt.triplot(coords[:, 0], coords[:, 1], tri.simplices, color='white', linewidth=0.2)
                plt.show(block=False)
            else:
                pass

        except IndexError:  # IndexError ends calculation as soon as refinement criteria was reached
            break
