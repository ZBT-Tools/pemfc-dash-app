"""
Delaunay Tests
"""
from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


def testfunction(xy):
    val = (1 - xy[0]) ** 2 + (2 - xy[1]) ** 2
    # if 2 < val < 50:
    #     return (1 - xy[0]) ** 2 + (2 - xy[1]) ** 2
    # else:
    #     return None
    return (1 - xy[0]) ** 2 + (2 - xy[1]) ** 2


def initial_points(n_x=5, n_y=5):
    x = (np.linspace(0, 10, n_x))
    y = (np.linspace(0, 10, n_y))
    coords = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    return coords


def calc_z(coords):
    return np.apply_along_axis(testfunction, 1, coords)


def calc_max_diff_triangles(tri, point_data, n=5):
    """
    Calculation of max absolute difference
    """
    # Add column to tri with height difference
    tri = np.c_[tri, [max([point_data[p1, 2], point_data[p2, 2], point_data[p3, 2]]) - \
                      min([point_data[p1, 2], point_data[p2, 2], point_data[p3, 2]])
                      for p1, p2, p3 in tri]]
    max_idxs = np.argpartition(tri[:,3], len(tri[:,3]) - n)[-n:]
    return max_idxs

def calc_new_points(tri, points, max_idxs):
    for idx in max_idxs:
        n_pts = tri[idx]
        pts_coords = [points[pt] for pt in n_pts]

        p1 = pts_coords[0]
        p2 = pts_coords[1]
        p3 = pts_coords[2]

        p12 = np.array([(p1[0]+p2[0])/2, (p1[1]+p2[1])/2])
        p13 = np.array([(p1[0] + p3[0]) / 2, (p1[1] + p3[1]) / 2])
        p23 = np.array([(p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2])





if __name__ == "__main__":
    """
    https://docs.scipy.org/doc/scipy/tutorial/spatial.html
    """

    coords = initial_points()
    tri = Delaunay(coords)

    # Perform calculation:
    z = calc_z(coords)

    results = np.c_[coords, z]

    # Coordinates and results in the order of tri.points
    coords_results = \
        np.c_[tri.points,
        [results[(results[:, 0] == k) & (results[:, 1] == v), 2][0] for k, v in tri.points]]

    # Refinement loop
    # -----------------
    # For each triangle calc max diff, find n largest
    # Points forming trinangle: tri.simplices, index is trangle number  n_t,p1,p2,p3
    # Coordinates of points in tri.points                               n_p, x,y
    # coords_results (tri.points with additional z column)              x,y,z

    # Identify n triangles to refine
    triangles_indices_to_refine = calc_max_diff_triangles(tri.simplices,coords_results)

    # Calculate new coordinates for to be refined triangles
    calc_new_points(tri.simplices,tri.points,triangles_indices_to_refine)

    # Add new points to overall data

    # Plot Triangles
    # ----------------------------------------------------------------------------------------------
    plt.triplot(coords[:, 0], coords[:, 1], tri.simplices)
    plt.plot(coords[:, 0], coords[:, 1], 'o')

    for j, p in enumerate(coords):
        plt.text(p[0] - 0.03, p[1] + 0.03, j, ha='right')  # label the points
    for j, s in enumerate(tri.simplices):
        p = coords[s].mean(axis=0)
        plt.text(p[0], p[1], '#%d' % j, ha='center')  # label triangles
    plt.show()

    # Plot Contour
    # ----------------------------------------------------------------------------------------------
    fig1, ax1 = plt.subplots()
    ax1.set_aspect('equal')
    tcf = ax1.tricontourf(results[:, 0], results[:, 1], results[:, 2])
    plt.show()
