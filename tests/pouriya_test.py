import numpy
from pointing_utils.optimal_control.SOFCstepper import SOFCStepper
from pointing_utils.optimal_control import lqg_ih


timestep = 4e-2
TF = 5
ntrials = 10

I = 0.25
b = 0.2
ta = 0.03
te = 0.04

a1 = -1*(b/(ta*te*I))
a2 = -1*(1/(ta*te) + (1/ta + 1/te)*b/I)
a3 = -1*(b/I + 1/ta + 1/te)
b1 = 1/(ta*te*I)


A = numpy.array([   [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, a1, a2, a3]    ])

B = numpy.array([[ 0, 0, 0, b1]]).reshape((-1,1))


C = numpy.array([   [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]
                        ])



D = numpy.array([   [0.01, 0, 0],
                    [0, 0.01, 0],
                    [0, 0, 0.05]
                    ])

F = numpy.diag([0.01, 0.01, 0.01, 0.01])
Y = 0.08*B
G = 0.03*numpy.diag([1,0.1,0.01,0.001])


Q = numpy.diag([1, 0.01, 0, 0])
R = numpy.array([[1e-4]])
U = numpy.diag([1, 0.1, 0.01, 0])


D = D*0.35
G = G*0.35


###############                         K and L config                  ###########################
#########################               for K and L I used lgq_ih to use the function compute kalman matrix
#########################                phillis1985family.plot_trajectories  or SOFC stepper ?
X0 = [-0.5,0,0,0]
init_value =[X0,X0]
Ac, Bc, Cc = A, B, C

K,L= lqg_ih.compute_kalman_matrices(Ac, Bc, Cc, D, F, G, Q, R, U, Y)

'''Mov=phillis1985family.plot_trajectories(
timestep,
    TF,
    A,
    B,
    C,
    D,
    Ac,
    Bc,
    Cc,
    F,
    G,
    K,
    L,
    Y,
    noise="on",
    ntrials=ntrials,
    init_value=init_value,
    
)'''

Mov = SOFCStepper.simulate(
    timestep,
    TF,
    A,
    B,
    C,
    D,
    Ac,
    Bc,
    Cc,
    F,
    G,
    K,
    L,
    Y,
    noise="on",
    ntrials=ntrials,
    init_value=init_value,

)

result = SOFCStepper.identify_A_B(
    Mov[:,:, 0, 0],
    timestep,
    TF,
    C,
    D,
    Ac,
    Bc,
    Cc,
    F,
    G,
    K,
    L,
    Y,
    noise="off",
    ntrials=ntrials,
    init_value=init_value,
)