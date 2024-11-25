import numpy as np
from matplotlib import pyplot as plt

print("-------Subtask 2-------")
t = np.array([-3,2,11], dtype=np.float64)

print("t = {}, t x t x t = {}".format(t, np.cross(t, np.cross(t,t))))

a = np.array([-2,6,1], dtype=np.float64)

b = np.cross(t, a)

c = np.cross(a, t)

print("a = {}, b = {}, c = {}".format(a, b, c))
print("t*b = {}, a*b = {}".format(np.inner(t,b), np.inner(a,b)))

print("-------Subtask 3-------")

def t_hat(t):
    assert t.shape == (3,)

    return np.array([[0,-t[2], t[1]],[t[2], 0, -t[0]],[-t[1], t[0], 0]])


that = t_hat(t)

print("t_hat = \n{},\nt_hat*t = \n{},\nt_hat*a = \n{}".format(that, that@t.T, that@a))

print("-------Subtask 4-------")

v = np.random.rand(3)
print("v = {}. v x t == t_hat@v, {}".format(v, np.equal(np.cross(t, v), that@v)))

print("-------Subtask 5-------")

print("t_hat.T == -t_hat: \n{}".format(that.T == -1*that))

that2 = that@that

that3 = that2@that

print("t_hat^2.T = \n{}, \nt_hat^3.T = \n{}.".format(that2.T, that3.T))


print("-------Task 3----------")

def f(x):
    return np.exp(-x**2/2)

def f_p(x):
    return -x*np.exp(-x**2/2)

def f_pp(x):
    return (x**2-1)*np.exp(-x**2/2)

granularity=0.1
x = np.arange(start=-7,stop=7, step=granularity)
print("x = {}".format(x))
plt.plot(x, np.array([f(xi) for xi in x]), label="f(x)")

plt.plot(x, np.array([f_p(xi) for xi in x]), label="f\'(x)")

plt.plot(x, np.array([f_pp(xi) for xi in x]), label="f\'\'(x)")

plt.legend()
plt.show()

print("-------Subtask 10------")

def h_exp(x,t):
    return np.exp(-(x**2)/2*t)

def h(x,t):
    return (1/(np.sqrt(2*np.pi*t)))*h_exp(x,t)

def h_x(x,t):
    return (-x/t)*(1/(np.sqrt(2*np.pi*t)))*h_exp(x,t)

def h_t(x,t):
    return (x**2 - t)/(2*np.sqrt(2*np.pi)*t**(5/2))*h_exp(x,t)

def h_xx(x,t):
    return 2*h_t(x,t)

def plot(t):
    granularity = 0.1
    x = np.arange(start=-15,stop=15, step=granularity)
    plt.title("Plots for t={}".format(t))
    plt.plot(x, np.array([h(xi,t=t) for xi in x]), label="f(x,t)")
    plt.plot(x, np.array([h_x(xi, t=t) for xi in x]), label="f_x(x,t)")
    plt.plot(x, np.array([h_xx(xi, t=t) for xi in x]), label="f_t(x,t)")
    plt.plot(x, np.array([h_t(xi, t=t) for xi in x]), label="f_xx(x,t)")
    plt.legend()
    plt.show()

plot(1)
plot(2)
plot(4)
