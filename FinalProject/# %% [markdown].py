# %% [markdown]
# # Final Project - Zhengyang Kris Weng Submission 12/10/2024

# %% [markdown]
# ![image.png](attachment:image.png)
# 
# two bodies, six degrees of freedom, includes impacts, has external forces (for shaking the cup),
# and is planar

# %%
import sympy as sym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objects as go

# %%
def TransToRp(T):
    """Converts a homogeneous transformation matrix into a rotation matrix
    and position vector

    :param T: A homogeneous transformation matrix
    :return R: The corresponding rotation matrix,
    :return p: The corresponding position vector.

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        (np.array([[1, 0,  0],
                   [0, 0, -1],
                   [0, 1,  0]]),
         np.array([0, 0, 3]))
    """
    R = T[:3, :3]
    p = T[:3, 3]
    return R, p


def TransInv(T):
    """Inverts a homogeneous transformation matrix

    :param T: A homogeneous transformation matrix
    :return: The inverse of T
    Uses the structure of transformation matrices to avoid taking a matrix
    inverse, for efficiency.

    Example input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1,  0, 0,  0],
                  [0,  0, 1, -3],
                  [0, -1, 0,  0],
                  [0,  0, 0,  1]])
    """
    R, p = TransToRp(T)
    Rt = R.T
    return sym.Matrix.vstack(sym.Matrix.hstack(Rt, -Rt * p), sym.Matrix([[0, 0, 0, 1]]))
    
def se3ToVec(se3mat):
    """ Converts an se3 matrix into a spatial velocity vector

    :param se3mat: A 4x4 matrix in se3
    :return: The spatial velocity 6-vector corresponding to se3mat

    Example Input:
        se3mat = np.array([[ 0, -3,  2, 4],
                           [ 3,  0, -1, 5],
                           [-2,  1,  0, 6],
                           [ 0,  0,  0, 0]])
    Output:
        np.array([1, 2, 3, 4, 5, 6])
    """
    return sym.Matrix([se3mat[2, 1], se3mat[0, 2], se3mat[1, 0], se3mat[0, 3], se3mat[1, 3], se3mat[2, 3]])

# %%
# Declare variables
t, m, M, g, l, L_box = sym.symbols('t m M g l L') # L bugged lagrange again =/
x_box = sym.Function('x_box')(t)
y_box = sym.Function('y_box')(t)
theta_box = sym.Function('theta_box')(t)
x_jack = sym.Function('x_jack')(t)
y_jack = sym.Function('y_jack')(t)
theta_jack = sym.Function('theta_jack')(t)

q = sym.Matrix([x_box, y_box, theta_box, x_jack, y_jack, theta_jack])
qdot = q.diff(t)
qddot = qdot.diff(t)

# %%
# Transformations

# world to box
g_w_b = sym.Matrix([[sym.cos(theta_box), -sym.sin(theta_box), 0, x_box], [sym.sin(theta_box), sym.cos(theta_box), 0, y_box], [0, 0, 1, 0], [0, 0, 0, 1]])
g_b_b1 = sym.Matrix([[1, 0, 0, L_box], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
g_b_b2 = sym.Matrix([[1, 0, 0, 0], [0, 1, 0, -L_box], [0, 0, 1, 0], [0, 0, 0, 1]])
g_b_b3 = sym.Matrix([[1, 0, 0, -L_box], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
g_b_b4 = sym.Matrix([[1, 0, 0, 0], [0, 1, 0, L_box], [0, 0, 1, 0], [0, 0, 0, 1]])

# world to jack
g_w_j = sym.Matrix([[sym.cos(theta_jack), -sym.sin(theta_jack), 0, x_jack], [sym.sin(theta_jack), sym.cos(theta_jack), 0, y_jack], [0, 0, 1, 0], [0, 0, 0, 1]])
g_j_j1 = sym.Matrix([[1, 0, 0, l], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
g_j_j2 = sym.Matrix([[1, 0, 0, 0], [0, 1, 0, -l], [0, 0, 1, 0], [0, 0, 0, 1]])
g_j_j3 = sym.Matrix([[1, 0, 0, -l], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
g_j_j4 = sym.Matrix([[1, 0, 0, 0], [0, 1, 0, l], [0, 0, 1, 0], [0, 0, 0, 1]])

# world to box sides
g_w_b1 = g_w_b * g_b_b1
g_w_b2 = g_w_b * g_b_b2
g_w_b3 = g_w_b * g_b_b3
g_w_b4 = g_w_b * g_b_b4

# world to jack ends
g_w_j1 = g_w_j * g_j_j1
g_w_j2 = g_w_j * g_j_j2
g_w_j3 = g_w_j * g_j_j3
g_w_j4 = g_w_j * g_j_j4

# %%
# impact relations
g_w_b_list = [g_w_b1, g_w_b2, g_w_b3, g_w_b4]
g_w_j_list = [g_w_j1, g_w_j2, g_w_j3, g_w_j4]

g_b_j = [[TransInv(g_w_b) * g_w_j for g_w_j in g_w_j_list] for g_w_b in g_w_b_list]
# display(g_b_j)

# Velocity
v_box = se3ToVec(np.array(TransInv(g_w_b) * g_w_b.diff(t)))
v_jack = se3ToVec(np.array(TransInv(g_w_j) * g_w_j.diff(t)))

j_jack = 4*(m/4)*l**2
j_box = 4*(M/4)*L_box**2
# display(j_box)

I_box = sym.Matrix([[0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, j_box, 0, 0, 0],
                    [0, 0, 0, M, 0, 0],
                    [0, 0, 0, 0, M, 0],
                    [0, 0, 0, 0, 0, M]])
I_jack = sym.Matrix([[0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, j_jack, 0, 0, 0],
                    [0, 0, 0, m, 0, 0],
                    [0, 0, 0, 0, m, 0],
                    [0, 0, 0, 0, 0, m]])

# Kinetic energy
T_box = 0.5 * v_box.dot(I_box * v_box)
T_jack = 0.5 * v_jack.dot(I_jack * v_jack)

KE = T_box + T_jack

# Potential Energy
PE = m*g*y_jack + M*g*y_box

# Calculate the Lagrangian:
L = KE.simplify() - PE.simplify()
print("EL Equation:")
display(L)


# %%
# EL equations
dLdq = sym.Matrix([L]).jacobian(q).T
dLdqdot = sym.Matrix([L]).jacobian(qdot).T
dLdqdotdt = dLdqdot.diff(t)

# Let there be force
f_y_box = M*g
f_theta_box = 50*sym.sin(sym.pi*t)
F = sym.Matrix([0, f_y_box, f_theta_box, 0, 0, 0])

EL_lhs = dLdqdotdt - dLdq
EL_rhs = F
EL_lhs.simplify()
EL_rhs.simplify()
EL_eqn = sym.Eq(EL_lhs, EL_rhs)
display(EL_eqn)

EL_subbed = EL_eqn.subs({M: 10, m: 1, l: 0.5, L_box: 5, g: 9.81})

# Solve
soln = sym.solve(EL_subbed, qddot, dict=True)
for sol in soln:
    print('\n\033[1mSymbolic Solution: ')
    for v in qddot:
        display(sym.Eq(v, sol[v].simplify()))

# %%
# Lambdify
x_box_ddot_func = sym.lambdify([q[0], q[1], q[2], q[3], q[4], q[5], qdot[0], qdot[1], qdot[2], qdot[3], qdot[4], qdot[5], t], soln[0][qddot[0]])
y_box_ddot_func = sym.lambdify([q[0], q[1], q[2], q[3], q[4], q[5], qdot[0], qdot[1], qdot[2], qdot[3], qdot[4], qdot[5], t], soln[0][qddot[1]])
theta_box_ddot_func = sym.lambdify([q[0], q[1], q[2], q[3], q[4], q[5], qdot[0], qdot[1], qdot[2], qdot[3], qdot[4], qdot[5], t], soln[0][qddot[2]])
x_jack_ddot_func = sym.lambdify([q[0], q[1], q[2], q[3], q[4], q[5], qdot[0], qdot[1], qdot[2], qdot[3], qdot[4], qdot[5], t], soln[0][qddot[3]])
y_jack_ddot_func = sym.lambdify([q[0], q[1], q[2], q[3], q[4], q[5], qdot[0], qdot[1], qdot[2], qdot[3], qdot[4], qdot[5], t], soln[0][qddot[4]])
theta_jack_ddot_func = sym.lambdify([q[0], q[1], q[2], q[3], q[4], q[5], qdot[0], qdot[1], qdot[2], qdot[3], qdot[4], qdot[5], t], soln[0][qddot[5]])

def dynamics(s, t):
    sdot = np.array([
        s[6],
        s[7],
        s[8],
        s[9],
        s[10],
        s[11],
        x_box_ddot_func(*s, t),
        y_box_ddot_func(*s, t),
        theta_box_ddot_func(*s, t),
        x_jack_ddot_func(*s, t),
        y_jack_ddot_func(*s, t),
        theta_jack_ddot_func(*s, t)
    ])
    return sdot

# %%
# Define acceleration matrix:
qddot_Matrix = sym.Matrix([qdot[0], soln[0][qddot[0]],
                       qdot[1], soln[0][qddot[1]],
                       qdot[2], soln[0][qddot[2]],
                       qdot[3], soln[0][qddot[3]],
                       qdot[4], soln[0][qddot[4]],
                       qdot[5], soln[0][qddot[5]]])

# Define dummy symbols:
x_b_l, y_b_l, theta_b_l, x_j_l, y_j_l, theta_j_l, x_b_ldot, y_b_ldot, theta_b_ldot, x_j_ldot, y_j_ldot, theta_j_ldot = sym.symbols('x_box_l, y_box_l, theta_box_l, x_jack_l, y_jack_l, theta_jack_l, x_box_ldot, y_box_ldot, theta_box_ldot, x_jack_ldot, y_jack_ldot, theta_jack_ldot')

static_dict = {q[0]:x_b_l, q[1]:y_b_l, q[2]:theta_b_l,
              q[3]:x_j_l, q[4]:y_j_l, q[5]:theta_j_l,
              qdot[0]:x_b_ldot, qdot[1]:y_b_ldot, qdot[2]:theta_b_ldot,
              qdot[3]:x_j_ldot, qdot[4]:y_j_ldot, qdot[5]:theta_j_ldot}
qddot_d = qddot_Matrix.subs(static_dict)
qddot_lambdify = sym.lambdify([x_b_l, x_b_ldot ,y_b_l, y_b_ldot, theta_b_l, theta_b_ldot,
                           x_j_l, x_j_ldot ,y_j_l, y_j_ldot, theta_j_l, theta_j_ldot, t], qddot_d)

# %%
r_jack_hat = sym.Matrix([x_jack, y_jack, theta_jack, 1])

# Define impact constraints
phi_b_j = []
indices = [3, 7, 3, 7]  # Indices for each wall

for i in range(4):
    phi_b_j.append([(g_b_j[i][j][indices[i % 2]]).subs(static_dict) for j in range(4)])

# Accessing the constraints for each wall
phi_b1_j = phi_b_j[0]
phi_b2_j = phi_b_j[1]
phi_b3_j = phi_b_j[2]
phi_b4_j = phi_b_j[3]

# Define impact constraint
phi_subbed = sym.Matrix([phi for sublist in phi_b_j for phi in sublist])
phi_subbed.simplify()
display(phi_subbed)

# %%
# Compute the Hamiltonian:
H = (dLdqdot.T * qdot)[0] - L

# Compute expressions:
H_subbed = H.subs(static_dict)
dLdqdot_subbed = dLdqdot.subs(static_dict)
dphidq_subbed = phi_subbed.jacobian([x_b_l, y_b_l, theta_b_l, x_j_l, y_j_l, theta_j_l])

# Define dummy symbols for tau+:
lamb = sym.symbols(r'/lambda')
x_b_dot_Plus, y_b_dot_Plus, theta_b_dot_Plus, x_j_dot_Plus, y_j_dot_Plus, theta_j_dot_Plus = sym.symbols(r'x_box_dot_+, y_box_dot_+, theta_box_dot_+, x_jack_dot_+, y_jack_dot_+, theta_jack_dot_+')

impact_dict = {x_b_ldot:x_b_dot_Plus, y_b_ldot:y_b_dot_Plus, theta_b_ldot:theta_b_dot_Plus,
               x_j_ldot:x_j_dot_Plus, y_j_ldot:y_j_dot_Plus, theta_j_ldot:theta_j_dot_Plus}

# Evaluate expressions at tau+:
dLdqdot_subbed_post = dLdqdot_subbed.subs(impact_dict)
dphidq_subbed_post = dphidq_subbed.subs(impact_dict)
H_subbed_post = H_subbed.subs(impact_dict)

# %%
impact_eqns_list = []

# Define equations
lhs = sym.Matrix([dLdqdot_subbed_post[0] - dLdqdot_subbed[0],
              dLdqdot_subbed_post[1] - dLdqdot_subbed[1],
              dLdqdot_subbed_post[2] - dLdqdot_subbed[2],
              dLdqdot_subbed_post[3] - dLdqdot_subbed[3],
              dLdqdot_subbed_post[4] - dLdqdot_subbed[4],
              dLdqdot_subbed_post[5] - dLdqdot_subbed[5],
              H_subbed_post - H_subbed])

for i in range(phi_subbed.shape[0]):
    rhs = sym.Matrix([lamb*dphidq_subbed[i,0],
                  lamb*dphidq_subbed[i,1],
                  lamb*dphidq_subbed[i,2],
                  lamb*dphidq_subbed[i,3],
                  lamb*dphidq_subbed[i,4],
                  lamb*dphidq_subbed[i,5],
                  0])
    impact_eqns_list.append(sym.Eq(lhs, rhs))

# %%
post_list = [x_b_dot_Plus, y_b_dot_Plus, theta_b_dot_Plus,
            x_j_dot_Plus, y_j_dot_Plus, theta_j_dot_Plus]

def impact_update(s, impact_eqns, dum_list):
    """ This function updates the system after impact.
    It returns the updated s array after impact.
    
    Parameters:
    s: current state
    impact_eqns: impact equations to solve
    dum_list: list of dummy variables for post-impact velocities
    
    Returns:
    numpy array: updated state after impact, or original state if no valid solution found
    """
    subs_dict = {x_b_l:s[0], y_b_l:s[1], theta_b_l:s[2],
                 x_j_l:s[3], y_j_l:s[4], theta_j_l:s[5],
                 x_b_ldot:s[6], y_b_ldot:s[7], theta_b_ldot:s[8],
                 x_j_ldot:s[9], y_j_ldot:s[10], theta_j_ldot:s[11]}
    
    try:
        new_impact_eqns = impact_eqns.subs(subs_dict)
        impact_solns = sym.solve(new_impact_eqns, [x_b_dot_Plus, y_b_dot_Plus, theta_b_dot_Plus,
                                               x_j_dot_Plus, y_j_dot_Plus, theta_j_dot_Plus,
                                               lamb], dict=True)
        
        if not impact_solns:  # If no solutions found
            print("No solutions found for impact equations")
            return s  # Return original state
            
        if len(impact_solns) == 1:
            print("Single solution found - using it")
            sol = impact_solns[0]
        else:
            # Find the first solution with non-zero lambda
            valid_sol = None
            for sol in impact_solns:
                lamb_sol = sol[lamb]
                if abs(lamb_sol) >= 1e-06:
                    valid_sol = sol
                    break
            
            if valid_sol is None:
                print("No valid solutions with non-zero lambda found")
                return s  # Return original state
            sol = valid_sol

        # Create updated state array
        return np.array([
            s[0],  # q will be the same after impact
            s[1],
            s[2],
            s[3],
            s[4],
            s[5],
            float(sym.N(sol[dum_list[0]], 15)),  # q_dot will change after impact
            float(sym.N(sol[dum_list[1]], 15)),
            float(sym.N(sol[dum_list[2]], 15)),
            float(sym.N(sol[dum_list[3]], 15)),
            float(sym.N(sol[dum_list[4]], 15)),
            float(sym.N(sol[dum_list[5]], 15)),
        ])
        
    except Exception as e:
        print(f"Error in impact_update: {str(e)}")
        return s  # Return original state if anything goes wrong

# %%
phi_func = sym.lambdify([x_b_l, y_b_l, theta_b_l,
                     x_j_l, y_j_l, theta_j_l,
                     x_b_ldot, y_b_ldot, theta_b_ldot,
                     x_j_ldot, y_j_ldot, theta_j_ldot],
                    phi_subbed)

def impact_condition(s, phi_func, threshold = 1e-1):
    """ This function checks the systems for impact.
    In the case of an impact (abs(phi_val) < threshold),
    the function returns True and the row number of the
    phi matrix in which the impact accured.
    """
    phi_val = phi_func(*s)
    for i in range(phi_val.shape[0]):
        if phi_val[i] < threshold:
            return (True, i)
    return (False, None)


# %%
def integrate(f, xt, dt, time):
    """
    This function takes in an initial condition x(t) and a timestep dt,
    as well as a dynamical system f(x) that outputs a vector of the
    same dimension as x(t). It outputs a vector x(t+dt) at the future
    time step.
    
    Parameters
    ============
    dyn: Python function
        derivate of the system at a given step x(t), 
        it can considered as \dot{x}(t) = func(x(t))
    x0: NumPy array
        current step x(t)
    dt: 
        step size for integration
    time:
        step time
        
    Return
    ============
    new_x: 
        value of x(t+dt) integrated from x(t)
    """
    k1 = dt * f(xt, time)
    k2 = dt * f(xt+k1/2., time)
    k3 = dt * f(xt+k2/2., time)
    k4 = dt * f(xt+k3, time)
    new_xt = xt + (1/6.) * (k1+2.0*k2+2.0*k3+k4)

    return new_xt


def simulate_impact(f, x0, tspan, dt, integrate):
    """
    This function takes in an initial condition x0, a timestep dt,
    a time span tspan consisting of a list [min_time, max_time],
    as well as a dynamical system f(x) that outputs a vector of the
    same dimension as x0. It outputs a full trajectory simulated
    over the time span of dimensions (xvec_size, time_vec_size).
    
    Parameters
    ============
    f: Python function
        derivate of the system at a given step x(t), 
        it can considered as \dot{x}(t) = func(x(t))
    x0: NumPy array
        initial conditions
    tspan: Python list
        tspan = [min_time, max_time], it defines the start and end
        time of simulation
    dt:
        time step for numerical integration
    integrate: Python function
        numerical integration method used in this simulation

    Return
    ============
    x_traj:
        simulated trajectory of x(t) from t=0 to tf
    """
    N = int((max(tspan)-min(tspan))/dt)
    x = np.copy(x0)
    tvec = np.linspace(min(tspan),max(tspan), N)
    xtraj = np.zeros((len(x0), N))
    time = 0
    for i in range(N):
        time = time + dt
        (impact, impact_num) = impact_condition(x, phi_func, 1e-1)
        if impact is True:
            x = impact_update(x, impact_eqns_list[impact_num], post_list)
            xtraj[:, i]=integrate(f, x, dt, time)
        else:
            xtraj[:, i]=integrate(f, x, dt, time)
        x = np.copy(xtraj[:,i]) 
    return xtraj

# %%
# Simulate the motion:
tspan = [0, 10]
dt = 0.01
# s = [x_box, y_box, theta_box, x_jack, y_jack, theta_jack, x_box_dot, y_box_dot, theta_box_dot, x_jack_dot, y_jack_dot, theta_jack_dot]
s0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, -1.57, 0, 0, 0])

N = int((max(tspan) - min(tspan))/dt)
tvec = np.linspace(min(tspan), max(tspan), N)
traj = simulate_impact(dynamics, s0, tspan, dt, integrate)

plt.figure()
plt.plot(tvec, traj[0], label='x_box')
plt.plot(tvec, traj[1], label='y_box')
plt.plot(tvec, traj[2], label='theta_box')
plt.title('Box Motion Simulation')
plt.xlabel('t')
plt.ylabel('state')
plt.show()

plt.figure()
plt.plot(tvec, traj[3], label='x_jack')
plt.plot(tvec, traj[4], label='y_jack')
plt.plot(tvec, traj[5], label='theta_jack')
plt.title('Jack Motion Simulation')
plt.xlabel('t')
plt.ylabel('state')
plt.show()

plt.figure()
plt.plot(tvec, traj[6], label='x_box_dot')
plt.plot(tvec, traj[7], label='y_box_dot')
plt.plot(tvec, traj[8], label='theta_box_dot')
plt.title('Box Velocity Simulation')
plt.xlabel('t')
plt.ylabel('state')
plt.show()

plt.figure()
plt.plot(tvec, traj[9], label='x_jack_dot')
plt.plot(tvec, traj[10], label='y_jack_dot')
plt.plot(tvec, traj[11], label='theta_jack_dot')
plt.title('Jack Velocity Simulation')
plt.xlabel('t')
plt.ylabel('state')
plt.show()


