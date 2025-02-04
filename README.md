# Exact 3D Greens Function Integrations on Triangles

![Integration Results](./thumbnail.jpg)
*This demo computes exact integrations of 3D Green's functions and compares them with numerical solutions.*

### Table of Contents

- [Introduction](#introduction)
- [Demo](#demo)
- [Results](#results)
- [Discussion](#discussion)
- [Citation](#citation)
- [License](#license)

## Introduction

We compute analytically the Green's and its gradient and product with linear functions on a triangle T in R3.

Let x be an evaluation point, r(x,y) = |y-x| and Let h(x) be a linear function on the triangle with value 1 at the first triangle vertex and 0 on the other vertices.

We compute the following integrals 

$\int_T \ G(x,y) dA_y = \int_T \ \frac{1}{|y-x|} dA_y$

$\int_T \ h(x) \ G(x,y) dA_y = \int_T \ h(x) \ \frac{1}{|y-x|} dA_y$

$\int_T \ \partial_{n_y} G(x,y) dA_y = \int_T \ \langle n_y , \nabla \frac{1}{|y-x|} \rangle dA_y$

$\int_T \ h(x) \ \partial_{n_y} G(x,y) dA_y = \int_T \ h(x) \ \langle n_y , \nabla \frac{1}{|y-x|} \rangle dA_y$

$\int_T \ \nabla G(x,y) dA_y = \int_T \ \nabla \frac{1}{|y-x|} dA_y$

$\int_T \ h(x) \ \nabla G(x,y) dA_y = \int_T \ h(x) \ \nabla \frac{1}{|y-x|} dA_y$

The analytic computation is performed based on [Graglia]'s results.

The following reference was the base for the formulae in this code:
[Graglia] R. D. Graglia, "Numerical Integration of the Linear Shape Functions Times the 3-D Green's Function
or Its Gradient on a Plane Triangle," IEEE Transactions on Antennas and Propagation, 
vol. 41, no. 10, pp. 1448–1455, Oct. 1993.

## Demo

Simply run the *greens_triangle_integration_demo.py*, and pip install missing packages if prompted.

An abitrary triangle is generated by the vertices triangle[0], triangle[1], triangle[2].
A wide selection of test evaluation points are generated.
The a numerical integration is performed via scipy.integrate.dblquad.

The test evaluation points are lines going through:
- a point further outside of the triangle plane
- a point in the triangle plane but not in the triangle or along an extended edge
- a point on the extended edge of a triangle
- a point on the extended edge of a triangle but off the other direction
- a point in the triangle but not on an edge
- a point on an edge but not a vertex
- a triangle vertex (the first one, where h=1) 

The formulas have been adjusted to not require a transformation into the local triangle frame.
Additionally, an orientation dependant singularity issue on the extended edge of a triangle has been fixed.

Important note: When an evaluation point is at the boundary of the triangle, the tangential part of the gradient integrals become singular.

## Results

Running the code will show you the following results:

![Integration Results](./integration_results/1_outside_of_the_triangle_plane.jpg)
*Figure 1: Integration results for a line through a point outside of the triangle plane.*

![Integration Results](./integration_results/2_in_the_triangle_plane_but_not_in_the_triangle_or_along_an_extended_edge.jpg)
*Figure 2: Integration results for a line through a point in the triangle plane but not in the triangle or along an extended edge.*

![Integration Results](./integration_results/3_on_the_extended_edge_of_a_triangle.jpg)
*Figure 3: Integration results for a line through a point on the extended edge of a triangle.*

![Integration Results](./integration_results/4_on_the_extended_edge_of_a_triangle_but_off_the_other_direction.jpg)
*Figure 4: Integration results for a line through a point on the extended edge of a triangle but off the other direction. [Graglia]'s formulae would fail in this case.*

![Integration Results](./integration_results/5_in_the_triangle_but_not_on_an_edge.jpg)
*Figure 5: Integration results for a line through a point in the triangle but not on an edge.*

![Integration Results](./integration_results/6_through_a_point_on_an_edge.jpg)
*Figure 6: Integration results for a line through a point on an edge but not a vertex. There is a singularity.*

![Integration Results](./integration_results/7_through_a_triangle_vertex.jpg)
*Figure 7: Integration results for a line through a triangle vertex (the first one, where h=1). There is a singularity.*

## Discussion
The analtic integration results are robust to the many possible cases and return the same result as the numerical integration based on scipy's dblquad. Away from singularities the analytic integration is around 10e5 times faster, while at singularities this reaches 10e6 speed-up using the default settings of dblquad.

However, the singularity at the tangential component when integrating $\nabla G(x,y)$ when $x \in \partial T$ seems to be unresolvable. Luckily, the normal gradient $\partial_n G(x,y)$ does not have this problem.

This code was created for dealing with boundary integral equations.

## Citation

If you use this code in your research, please cite it as follows:

```bibtex
@misc{padilla2025exact3Dgreens,
    author = {Marcel Padilla},
    title = {Exact 3D Greens Function Integrations on Triangles},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/marcelpadilla/exact-3D-greens-function-integrations-on-triangles}},
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/marcelpadilla/exact-3D-greens-function-integrations-on-triangles/blob/main/LICENSE) file for details. Enjoy using it for whatever projects you have. Just please cite it.
