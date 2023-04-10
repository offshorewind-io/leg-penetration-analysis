import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def equivalent_diameter(A):

    return np.sqrt(A * 4 / np.pi)


def calculate_tip_area(spudcan, h):

    D_tip = np.interp(h, spudcan['Depth [m]'], spudcan['D [m]'])
    A_tip = np.pi * D_tip**2 / 4

    return A_tip


def bouyant_weight_cone(spudcan, grid, soil_profile):

    W_cone_grid = []
    for h in grid:

        if h >= 0:
            h_cone = -spudcan['Depth [m]'][0]
            D = equivalent_diameter(spudcan['Area [m2]'].max())
            i = soil_profile[(h >= soil_profile['Depth from [m]']) & (h < soil_profile['Depth to [m]'])].index[0]
            gamma = soil_profile.loc[i, 'Buoyant unit weight [kN/m3]']
        else:
            h_cone = h - spudcan['Depth [m]'][0]
            D = np.interp(h, spudcan['Depth [m]'], spudcan['D [m]'])
            gamma = soil_profile.loc[0, 'Buoyant unit weight [kN/m3]']
        V_c = np.pi * D** 2 / 4 * h_cone / 3
        W_cone = gamma * V_c
        W_cone_grid = W_cone_grid + [W_cone]

    return W_cone_grid


def soil_volume_weight(grid, soil_profile, A):

    d_z = soil_profile['Depth to [m]'] - soil_profile['Depth from [m]']
    gamma = soil_profile['Buoyant unit weight [kN/m3]']
    sigma_v_to = np.cumsum(d_z * gamma).tolist()
    W_soil_grid = []

    for h in grid:

        if h <= 0:
            W_soil = 0
        else:
            sigma_v = np.interp(h, [0] + soil_profile['Depth to [m]'].tolist(), [0] + sigma_v_to)
            W_soil = sigma_v * A

        W_soil_grid = W_soil_grid + [W_soil]

    return W_soil_grid


def backfill_volume_weight(spudcan, grid, soil_profile):

    A = spudcan['Area [m2]'].max()
    D = equivalent_diameter(A)
    W_backfill = 0
    backfill_started = False
    d_h = grid[1] - grid[0]
    W_backfill_grid = []

    for h in grid:
        if h > 0:
            i = soil_profile[(h >= soil_profile['Depth from [m]']) & (h < soil_profile['Depth to [m]'])].index[0]
            soil_type = soil_profile.loc[i, 'Soil type']
            gamma = soil_profile.loc[i, 'Buoyant unit weight [kN/m3]']
            s_u = soil_profile.loc[i, 'Peak undrained shear strength [kPa]']
            W_slice = gamma * A * d_h

            if soil_type == 'Clay':
                h_c = critical_cavity_depth(s_u, 0, gamma, D)
                if h > h_c:
                    backfill_started = True
            elif soil_type == 'Sand':
                backfill_started = True
        else:
            W_slice = 0

        W_backfill = W_backfill + W_slice * backfill_started
        W_backfill_grid = W_backfill_grid + [W_backfill]

    return W_backfill_grid


def critical_cavity_depth(s_um, rho, gamma, D):

    h_c_estimate = -1
    h_c_iteration = 0

    while np.abs(h_c_iteration - h_c_estimate) > 0.01:
        h_c_estimate = h_c_iteration
        s_uh = s_um + rho * h_c_estimate
        h_c_iteration = D * (s_uh / gamma / D)**0.55 - 1 / 4 * (s_uh / gamma / D)

    return h_c_iteration


def cone_weight_calc(h, s_um, rho, gamma, A, V_c):

    D = equivalent_diameter(A)
    h_c = critical_cavity_depth(s_um, rho, gamma, D)
    h_calc = np.minimum(np.maximum(0, h), h_c)
    W_cone = bouyant_weight_cone(gamma, V_c)

    return W_cone


def bearing_capacity_clay(s_u, A, N_c=6.14):

    Q_v0 = A * s_u * N_c

    return Q_v0


def equivalent_undrained_shear_strength(Q_v0, A, N_c=6.14):

    Q_v_norm = bearing_capacity_clay(1, A, N_c)
    s_u_eq = Q_v0 / Q_v_norm

    return s_u_eq


def N_c_Houlsby_Martin_2003(h, D, s_u, rho, beta, alpha=0.5):

    N_1 = 5.69 * (1 - 0.21 * np.cos(np.radians(beta) / 2)) * (1 + h / D)**0.34
    N_2 = 0.5 + 0.36 * (1 / np.tan(np.radians(beta) / 2))**1.5 - 0.4 * (h / D)**2
    N_c00 = N_1 + N_2 * D * rho / s_u
    N_c0a = N_c00 * (1 + (0.212 * alpha - 0.097 * alpha**2) * (1 - 0.53 * h / (D + h)))
    N_c0 = N_c0a + alpha / np.tan(np.radians(beta) / 2) * (1 + 1 / 6 / np.tan(np.radians(beta) / 2) * D * rho / s_u)

    return N_c0


def bearing_capacity_sand(h, phi, gamma, A):

    D = equivalent_diameter(A)
    phi_rad = np.radians(phi)
    zeta_h_gamma = 1  # Recommendation in Insafe
    zeta_s_q = 1 + np.tan(phi_rad)
    zeta_h_q = 1 + 2 * np.tan(phi_rad) * (1 - np.sin(phi_rad))**2 * np.arctan(h / D)
    N_q = np.exp(np.pi * np.tan(phi_rad)) * np.tan(np.pi / 4 + phi_rad / 2)**2

    # This implements the Eurocode 7 Formula - Cassidy and Houlsby (2002) present values for conical shaped footings
    N_gamma = 0.7 * (np.exp(np.pi * np.tan(phi_rad)) * np.tan(np.pi / 4 + phi_rad / 2)**2 - 1) * np.tan(phi_rad)

    Q_v0 = A * (
        1 / 2 * gamma * D * N_gamma * zeta_h_gamma +
        gamma * h * N_q * zeta_s_q * zeta_h_q
    )

    return Q_v0


def squeezing_capacity(h, h_base, A, s_u, Q_v0_base, N_c=6.14, n=3):

    D = equivalent_diameter(A)
    Q_v0 = bearing_capacity_clay(s_u, A, N_c)
    Q_v0_ratio = np.maximum(1, Q_v0_base / Q_v0)

    # This formula approximates the chart in Appendix A4.5.4
    d_h = h_base - h
    N_c_equivalent = N_c + np.tanh(2 * (Q_v0_ratio - 1)) * (D / d_h / n - 1)
    Q_v0_squeezing = np.minimum(A * N_c_equivalent * s_u, Q_v0_base)

    return Q_v0_squeezing


def punching_capacity(gamma, phi, h, h_base, A, Q_v_base):

    D = equivalent_diameter(A)
    phi_rad = np.radians(phi)

    s_u_eq = equivalent_undrained_shear_strength(Q_v_base, A)
    K_s = (2.5 * (s_u_eq / gamma / D)**0.6) / np.tan(phi_rad)
    Q_v0 = Q_v_base + gamma / 2 * (h_base**2 - h**2) * np.pi * D * K_s * np.tan(phi_rad)

    return Q_v0


def punching_capacity_clay(s_u, h, h_base, A, Q_v_base):

    D = equivalent_diameter(A)
    P = np.pi * D

    # Appendix A 4.5.3 in Insafe JIP report
    Q_v0 = Q_v_base + (h_base - h) * s_u * P * 3 / 4

    return Q_v0


def calculate_grid_capacity(grid, soil_profile, spudcan):

    A = spudcan['Area [m2]'].max()
    D = equivalent_diameter(A)
    spudcan['D [m]'] = [equivalent_diameter(A) for A in spudcan['Area [m2]']]

    Q_v0_uniform_grid = []
    Q_v0_mechanism_grid = []

    grid_reverse = grid[::-1]
    i_previous_iteration = len(soil_profile) - 1
    Q_v0_top_next_layer = 0

    for h_exact in grid_reverse:
        h = np.round(h_exact, 4)  # this is to avoid small numerical errors when making the grid
        if h < 0:
            i = -1
            if i != i_previous_iteration:
                Q_v0_top_next_layer_mechanism = Q_v0_mechanism + 0
                Q_v0_top_next_layer_uniform = Q_v0_uniform + 0

            A_tip = calculate_tip_area(spudcan, h)
            Q_v0_mechanism = Q_v0_top_next_layer_mechanism * A_tip / A
            Q_v0_uniform = Q_v0_top_next_layer_uniform * A_tip / A

        else:
            i = soil_profile[(h >= soil_profile['Depth from [m]']) & (h < soil_profile['Depth to [m]'])].index[0]

            soil_type = soil_profile.loc[i, 'Soil type']
            gamma = soil_profile.loc[i, 'Buoyant unit weight [kN/m3]']
            phi = soil_profile.loc[i, 'Peak angle of friction [deg]']
            s_u = soil_profile.loc[i, 'Peak undrained shear strength [kPa]']
            h_base = soil_profile.loc[i, 'Depth to [m]']
            if soil_type == 'Clay':
                N_c = N_c_Houlsby_Martin_2003(h, D, s_u, rho=0, beta=180)

            if i == len(soil_profile)-1:
                if soil_type == 'Sand':
                    Q_v0_uniform = bearing_capacity_sand(h, phi, gamma, A)
                elif soil_type == 'Clay':
                    Q_v0_uniform = bearing_capacity_clay(s_u, A, N_c)
                Q_v0_mechanism = Q_v0_uniform + 0

            else:
                if i != i_previous_iteration:
                    Q_v0_top_next_layer = Q_v0_mechanism + 0

                if soil_type == 'Sand':
                    Q_v0_punching = punching_capacity(gamma, phi, h, h_base, A, Q_v0_top_next_layer)
                    Q_v0_uniform = bearing_capacity_sand(h, phi, gamma, A)
                    Q_v0_mechanism = np.minimum(Q_v0_punching, Q_v0_uniform)

                elif soil_type == 'Clay':
                    Q_v0_squeezing = squeezing_capacity(h, h_base, A, s_u, Q_v0_top_next_layer, N_c)
                    Q_v0_punching = punching_capacity_clay(s_u, h, h_base, A, Q_v0_top_next_layer)
                    Q_v0_uniform = bearing_capacity_clay(s_u, A, N_c)
                    Q_v0_mechanism = np.minimum(np.maximum(Q_v0_squeezing, Q_v0_uniform), Q_v0_punching)

        i_previous_iteration = i + 0
        Q_v0_uniform_grid = Q_v0_uniform_grid + [Q_v0_uniform]
        Q_v0_mechanism_grid = Q_v0_mechanism_grid + [Q_v0_mechanism]

    results = pd.DataFrame(data={
        'h [m]': grid,
        'Q_v0 uniform [kN]': Q_v0_uniform_grid[::-1],
        'Q_v0 mechanism [kN]': Q_v0_mechanism_grid[::-1]
    })

    results['W_cone [kN]'] = bouyant_weight_cone(spudcan, grid, soil_profile)
    results['W_backfill [kN]'] = backfill_volume_weight(spudcan, grid, soil_profile)
    results['W_soil [kN]'] = soil_volume_weight(grid, soil_profile, A)
    results['Q_v uniform [kN]'] = results['Q_v0 uniform [kN]'] + results['W_soil [kN]'] + results['W_cone [kN]'] - results['W_backfill [kN]']
    results['Q_v mechanism [kN]'] = results['Q_v0 mechanism [kN]'] + results['W_soil [kN]'] + results['W_cone [kN]'] - results['W_backfill [kN]']

    return results


if __name__ == '__main__':

    soil_profile = pd.DataFrame(data={
        'Depth to [m]': [2, 4, 8, 10],
        'Soil type': ['Sand', 'Clay', 'Sand', 'Clay'],
        'Buoyant unit weight [kN/m3]': [9, 9, 9, 9],
        'Peak undrained shear strength [kPa]': [np.nan, 100, np.nan, 300],
        'Peak angle of friction [deg]': [35, np.nan, 40, np.nan]
    })

    spudcan = pd.DataFrame(data={
        'Depth [m]': [-0.8, 0, 1],
        'Area [m2]': [0, 124.5, 124.5]
    })

    soil_profile['Depth from [m]'] = [0] + soil_profile['Depth to [m]'].tolist()[:-1]
    soil_profile['Layer thickness [m]'] = soil_profile['Depth to [m]'] - soil_profile['Depth from [m]']

    colors = {'Sand': 'sandybrown', 'Clay': 'steelblue'}
    soil_profile['Color'] = [colors[soil_type] for soil_type in soil_profile['Soil type']]
    soil_profile['Color'] = [colors[soil_type] for soil_type in soil_profile['Soil type']]

    d_z = 0.1
    grid = np.arange(spudcan['Depth [m]'][0], soil_profile['Depth to [m]'].tolist()[-1], d_z).tolist()
    results = calculate_grid_capacity(grid, soil_profile, spudcan)

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=(
        "Leg penetration assessment", "Soil profile"))

    fig.add_trace(go.Scatter(x=results["Q_v0 uniform [kN]"], y=results["h [m]"], name='Uniform 0'), 1, 1)
    fig.add_trace(go.Scatter(x=results["Q_v0 mechanism [kN]"], y=results["h [m]"], name='Mechanism 0'), 1, 1)
    fig.add_trace(go.Scatter(x=results["W_cone [kN]"], y=results["h [m]"], name='Cone'), 1, 1)
    fig.add_trace(go.Scatter(x=results["W_backfill [kN]"], y=results["h [m]"], name='Backfill'), 1, 1)
    fig.add_trace(go.Scatter(x=results["W_soil [kN]"], y=results["h [m]"], name='Soil'), 1, 1)
    fig.add_trace(go.Scatter(x=results["Q_v uniform [kN]"], y=results["h [m]"], name='Uniform'), 1, 1)
    fig.add_trace(go.Scatter(x=results["Q_v mechanism [kN]"], y=results["h [m]"], name='Mechanism'), 1, 1)
    fig.add_trace(go.Bar(name='Soil profile', x=[" "] * len(soil_profile), y=soil_profile["Layer thickness [m]"], marker={'color': soil_profile["Color"]}), 1, 2)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(rangemode="tozero")
    fig['layout']['yaxis']['title'] = 'Depth (m)'
    fig['layout']['xaxis']['title'] = 'Capacity (kN)'
    fig.show()