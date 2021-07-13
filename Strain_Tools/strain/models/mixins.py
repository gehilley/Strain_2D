import numpy as np
from scipy.spatial import Delaunay
from numpy.linalg import inv

class DelaunayMixin(object):

    def _configure_network_with_flat_delaunay(self, myVelfield):
        elon = [x.elon for x in myVelfield];
        nlat = [x.nlat for x in myVelfield];
        z = np.array([elon, nlat]);
        z = z.T;
        tri = Delaunay(z);

        self._triangle_vertices = z[tri.simplices];
        trishape = np.shape(self._triangle_vertices);  # 516 x 3 x 2, for example

        # We are going to solve for the velocity gradient tensor at the centroid of each triangle.
        centroids = [];
        for i in range(trishape[0]):
            xcor_mean = np.mean([self._triangle_vertices[i, 0, 0], self._triangle_vertices[i, 1, 0], self._triangle_vertices[i, 2, 0]]);
            ycor_mean = np.mean([self._triangle_vertices[i, 0, 1], self._triangle_vertices[i, 1, 1], self._triangle_vertices[i, 2, 1]]);
            centroids.append([xcor_mean, ycor_mean]);
        self._xcentroid = [x[0] for x in centroids];
        self._ycentroid = [x[1] for x in centroids];
        self._inv_Design_Matrix = np.zeros((trishape[0], 6, 6))
        self._index = np.zeros((trishape[0],3), dtype = np.int32)
        for i in range(trishape[0]):
            # Get the velocities of each vertex (VE1, VN1, VE2, VN2, VE3, VN3)
            # Get velocities for Vertex 1 (triangle_vertices[i,0,0] and triangle_vertices[i,0,1])
            xindex1 = np.where(elon == self._triangle_vertices[i, 0, 0])
            yindex1 = np.where(nlat == self._triangle_vertices[i, 0, 1])
            self._index[i, 0]= np.intersect1d(xindex1, yindex1);
            xindex2 = np.where(elon == self._triangle_vertices[i, 1, 0])
            yindex2 = np.where(nlat == self._triangle_vertices[i, 1, 1])
            self._index[i, 1] = np.intersect1d(xindex2, yindex2);
            xindex3 = np.where(elon == self._triangle_vertices[i, 2, 0])
            yindex3 = np.where(nlat == self._triangle_vertices[i, 2, 1])
            self._index[i, 2] = np.intersect1d(xindex3, yindex3);

            # Get the distance between centroid and vertex (in km)
            dE1 = (self._triangle_vertices[i, 0, 0] - self._xcentroid[i]) * 111.0 * np.cos(np.deg2rad(self._ycentroid[i]));
            dE2 = (self._triangle_vertices[i, 1, 0] - self._xcentroid[i]) * 111.0 * np.cos(np.deg2rad(self._ycentroid[i]));
            dE3 = (self._triangle_vertices[i, 2, 0] - self._xcentroid[i]) * 111.0 * np.cos(np.deg2rad(self._ycentroid[i]));
            dN1 = (self._triangle_vertices[i, 0, 1] - self._ycentroid[i]) * 111.0;
            dN2 = (self._triangle_vertices[i, 1, 1] - self._ycentroid[i]) * 111.0;
            dN3 = (self._triangle_vertices[i, 2, 1] - self._ycentroid[i]) * 111.0;

            Design_Matrix = np.array([[1, 0, dE1, dN1, 0, 0], [0, 1, 0, 0, dE1, dN1], [1, 0, dE2, dN2, 0, 0], [0, 1, 0, 0, dE2, dN2],
                                      [1, 0, dE3, dN3, 0, 0], [0, 1, 0, 0, dE3, dN3]]);

            # Invert to get the components of the velocity gradient tensor.
            self._inv_Design_Matrix[i,:,:] = np.reshape(inv(Design_Matrix), (1, 6, 6))