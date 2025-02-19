#!/usr/bin/env python3
import numpy as np


# -----------------------------------------------------------------------------
### Settings ###
accuracy_treshold = 1e-6
# -----------------------------------------------------------------------------
    

def integrate(evaluation_points, triangle, values_of_interest):
    """
    We compute analytically the integrals of the 3D Green's functions, its gradient and their product with linear functions on a triangle T in R3.

    Let x be an evaluation point, r(x,y) = |y-x| and Let h(x) be a linear function on the triangle with value 1 at the first triangle vertex and 0 on the other vertices.

    We compute the following integrals 
    ∫ 1/r
    ∫ h ⋅ 1/r
    ∫ <n, ∇(1/r)>
    ∫ h ⋅ <n, ∇(1/r)>
    ∫ ∇(1/r)
    ∫ h ⋅ ∇(1/r)
    over a single triangle T in R3. Note that we are not dividing by 4pi.

    Input:
        - A set of evaluation points in R3, in form of a numpy array.
        - A triangle T in R3.
        - A set of strings specifying what to return. They can be the following:
            - 'G' : the integral of 1/r
            - 'h_G' : the integral of h ⋅ 1/r
            - 'n_grad_G' : the integral of <n, ∇(1/r)>
            - 'h_n_grad_G' : the integral of h ⋅ <n, ∇(1/r)>
            - 'grad_G' : the integral of ∇(1/r)
            - 'h_grad_G' : the integral of h ⋅ ∇(1/r)
            - when using h, an array of dimension (nr_evaluation_points, 3) is returned, where the columns are the integrals with h0, h1 and h2, respectively. h0(trianlge[0])=1, h0(trianlge[1])=h0(trianlge[2])=0, and so on.
            - Use integration_results['h_G'][:,0] and integration_results['h_grad_G'][:,:,0] to access all h0 integrations.
    Output:
        - The analytic integral values on the evaluation points, in form of numpy arrays, stored in a dictionary that can be accessed by the strings mentioned in the input.

    Important note: When an evaluation point is at the boundary of the triangle, the tangential part of the gradient integrals become singular.

    For more detailed information, instructions and a demo see:
    https://github.com/marcelpadilla/exact-3D-greens-function-integrations-on-triangles
    """
    
    nr_evaluation_points = evaluation_points.shape[0]
    
    # Check if the triangle is degenerate
    if np.linalg.matrix_rank(triangle) < 3:
        raise ValueError("The triangle is degenerate (vertices are collinear).")
    
    # dictionary to store the results
    integration_results = {}
    
    # remap origin
    evaluation_points = evaluation_points - triangle[0]
    triangle = triangle - triangle[0]

    # Lengths and edge tangent vectors
    unit_tangents = np.array([
        triangle[2] - triangle[1],
        triangle[0] - triangle[2],
        triangle[1] - triangle[0]
    ])
    l0, l1, l2 = np.linalg.norm(unit_tangents, axis=1)
    unit_tangents = normalize_rows(unit_tangents)

    # normal and edge normals
    normal = get_normal_from_triangle(triangle)
    edge_normals = np.array([
        np.cross(unit_tangents[0], normal),
        np.cross(unit_tangents[1], normal),
        np.cross(unit_tangents[2], normal)
    ])
    
    # triangle frame basis vectors
    u_unit = unit_tangents[2]
    v_unit = -edge_normals[2]
    w_unit = normal

    # evaluation points coordinates in triangle frame
    u_eval = np.dot( evaluation_points, u_unit )
    v_eval = np.dot( evaluation_points, v_unit )
    w_eval = np.dot( evaluation_points, w_unit )
    
    # parametrization of linear map
    u2 = np.dot( triangle[2] , u_unit )
    v2 = np.dot( triangle[2] , v_unit )
    form = np.array([
        [1, -1/l2, ( (u2 / l2) - 1 )/v2],
        [0,  1/l2, -u2 / ( l2*v2 )],
        [0,  0,  1/v2]
    ])
    h1, h2, h3 = form @ [np.ones_like(u_eval), u_eval, v_eval]
    h_eval = form @ np.stack([np.ones_like(u_eval), u_eval, v_eval], axis=0)
    
    # Project observation points to the plane of the triangel
    P_plane = evaluation_points - np.dot( evaluation_points, w_unit )[:, np.newaxis] * w_unit
    
    # Project observation points to the edges of the triangle
    P_edge = np.zeros((nr_evaluation_points, 3, 3))
    for i in range(3):
        P_edge[:, i, :] = triangle[(i + 1) % 3] + unit_tangents[i] * np.dot(evaluation_points - triangle[(i + 1) % 3], unit_tangents[i])[:, np.newaxis]
    

    # parametrizations of integration
    splus = np.zeros((nr_evaluation_points, 3))
    sminus = np.zeros((nr_evaluation_points, 3))
    timer_start = np.zeros((nr_evaluation_points, 3))
    for i in range(3):
        splus[:, i] = np.dot( triangle[(i + 2) % 3] - P_edge[:, i, :] , unit_tangents[i])
        sminus[:, i] = np.dot( triangle[(i + 1) % 3] - P_edge[:, i, :] , unit_tangents[i])
        timer_start[:, i] = np.dot( P_edge[:, i, :] - P_plane , edge_normals[i])
    
    # distances
    R0 = np.linalg.norm(evaluation_points[:, np.newaxis, :] - P_edge, axis=2)
    Rplus = np.linalg.norm(evaluation_points[:, np.newaxis, :] - triangle[np.array([2, 0, 1])], axis=2)
    Rminus = np.linalg.norm(evaluation_points[:, np.newaxis, :] - triangle[np.array([1, 2, 0])], axis=2)
    
    # apply in plane treshhold 
    threshhold = accuracy_treshold * min([l0, l1, l2])
    timer_start[np.abs(timer_start) < threshhold] = 0.0
    w_eval[np.abs(w_eval) < threshhold] = 0.0
    R0[np.abs(R0) < threshhold] = 0.0
    w0_tile = np.tile(w_eval, (3,1)).T

    # side functions
    with np.errstate(divide='ignore', invalid='ignore'):
        f2 = np.log((Rplus + splus) / (Rminus + sminus))
        # the following line addresses a singularity issue on the extended edge of a triangle (a 0/0 error)
        f2 = np.where( (np.isnan(f2)) | (f2 < 0), np.log((Rminus - sminus) / (Rplus - splus)), f2)
        # next we acknowledge that on the triangle boundary we have an unresolved singularity that we simply remove.
        # It is only noticible in the tangential component of the gradient.
        f2 = np.where( (np.isnan(f2)) | np.isinf(f2), 0, f2)
    f3 = (splus * Rplus - sminus * Rminus) + R0**2 * f2
    with np.errstate(divide='ignore', invalid='ignore'):
        betas = (np.arctan(timer_start * splus / (R0**2 + np.abs(w0_tile) * Rplus))
                - np.arctan(timer_start * sminus / (R0**2 + np.abs(w0_tile) * Rminus)))
        betas = np.where( R0 < threshhold, 0, betas)
        beta = np.sum( betas, axis=1 )
    
    ### Basic integral of Green's function G
    
    # eq. (19): integral of 1/r
    if 'G' in values_of_interest or 'h_G' in values_of_interest:
        Int_G = np.sum(timer_start*f2 , axis=1) - np.abs(w_eval)*beta
        integration_results['G'] = Int_G

    # eq. (20): integral of u or v times (1/r)
    if 'h_G' in values_of_interest:
        sum_vector = 0.5 * f3 @ edge_normals
        Iua = np.dot( sum_vector , u_unit  )
        Iva = np.dot( sum_vector , v_unit  )

        # eq. (24): integral of linear map times (1/r)
        Iu = u_eval * Int_G + Iua
        Iv = v_eval * Int_G + Iva 
        
        # compute for the 3 linear basis functions on the triangle
        Int_h_G = ( form @ [Int_G, Iu, Iv] ).T
        integration_results['h_G'] = Int_h_G
            
    ### Integral of the gradient of the Green's function G
    
    # eq. (34): integral of grad(1/r)
    if 'grad_G' in values_of_interest or 'h_grad_G' in values_of_interest:
        Int_grad_G = - f2 @ edge_normals - np.sign(w_eval)[:, np.newaxis] * beta[:, np.newaxis] * w_unit
        integration_results['grad_G'] = Int_grad_G
    
    # eq. (36): integrals of u or v times grad(1/r)
    if 'h_grad_G' in values_of_interest:
        part1u = (w_eval * np.dot((f2 @ edge_normals), u_unit))[:, None] * w_unit
        part1v = (w_eval * np.dot((f2 @ edge_normals), v_unit))[:, None] * w_unit
        part2u = (-np.abs(w_eval) * beta)[:, None] * u_unit
        part2v = (-np.abs(w_eval) * beta)[:, None] * v_unit
        u_dot_s = np.sum(u_unit * unit_tangents, axis=1)
        v_dot_s = np.sum(v_unit * unit_tangents, axis=1)
        f = ( (f2 * timer_start)[:, :, None] * unit_tangents[None, :, :]) - ((Rplus - Rminus)[:, :, None] * edge_normals[None, :, :])
        part3u = (f * u_dot_s[None, :, None]).sum(axis=1)
        part3v = (f * v_dot_s[None, :, None]).sum(axis=1)
        Igradua = part1u + part2u + part3u
        Igradva = part1v + part2v + part3v
        
        # eq. (40): integral of linear map times grad(1/r)
        grad_G_u = u_eval[:,np.newaxis] * Int_grad_G + Igradua
        grad_G_v = v_eval[:,np.newaxis] * Int_grad_G + Igradva
        
        # compute for the 3 linear basis functions on the triangle
        Int_h_grad_G = np.tensordot(form, [Int_grad_G, grad_G_u, grad_G_v], axes=1).transpose(1, 2, 0)
        integration_results['h_grad_G'] = Int_h_grad_G

    ### Integral of the normal gradient of the Green's function G

    # eq. (34): integral of <n , grad(1/r) >
    if 'n_grad_G' in values_of_interest:
        Int_n_grad_G = -np.sign(w_eval) * beta
        integration_results['n_grad_G'] = Int_n_grad_G
    
    # eq. (40): integrals of linear map times <n , grad(1/r) >
    if 'h_n_grad_G' in values_of_interest:
        Int_h_n_grad_G = ( np.tensordot(form[:, 1:], [
            (w_eval * np.dot(f2 @ edge_normals, u_unit)),
            (w_eval * np.dot(f2 @ edge_normals, v_unit))
        ], axes=1) -np.sign(w_eval) * beta * h_eval ).T
        integration_results['h_n_grad_G'] = Int_h_n_grad_G

    # final return
    return integration_results


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def get_normal_from_triangle(triangle):
    """
    Computes the normal vector of a triangle defined by vertices V. This function is used to be consistent.
    """
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    if np.linalg.norm(normal) < 1e-14:
        raise ValueError("The triangle is degenerate (vertices are collinear).")
    return normalize(normal)
    
def normalize(v):
    """ Return the unit (normalized) vector of v. """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-14 else v

def normalize_rows(matrix):
    """ Normalize each row of the input matrix. """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 1e-14, norms, 1)
    return matrix / norms

