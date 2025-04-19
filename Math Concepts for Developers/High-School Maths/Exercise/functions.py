import sympy
import math

# Ex 3
def solve_quadratic_equation(a, b, c):
    """
    Returns the real solutions of the quadratic equation ax^2 + bx + c = 0
    """
    if a == 0:
        if b != 0:
            return [-c / b]
        else:
            raise ValueError("Division by zero detected!")
    else:
        discriminant = b**2 - 4*a*c
        divisor = 2 * a
        
        if discriminant < 0:
            return []
        elif discriminant == 0:
            return [-b / divisor, -b / divisor] #Bonus 2 (2 solution, same number)
        else:
            square_root = math.sqrt(discriminant)
            return [(-b - square_root) / divisor, (-b + square_root) / divisor]
        
# Ex 4

# Without functions
def generate_fibonacci(n):
    return [1 if x == 1 or x == 2 else generate_fibonacci(x-1)[-1] +
        generate_fibonacci(x-2)[-1] for x in range(1, n+1)]

fibonacci_list = generate_fibonacci(20)
squared = [x**2 for x in fibonacci_list]
odd_squared = [x for x in squared if x % 2 != 0]
sum_odd_squared = sum(odd_squared)

from functools import reduce

# With functions
def generate_fibonacci2(n):
    return reduce(lambda seq, _: seq + [seq[-1] + seq[-2]], range(2, n), [1, 1])

fibonacci_list2 = generate_fibonacci2(20)
squared2 = list(map(lambda x: x**2, fibonacci_list2))
odd_squared2 = list(filter(lambda x: x % 2 != 0, squared))
sum_odd_squared2 = sum(odd_squared2)

# Without variables
def generate_fibonnaci_sum(n):
    return reduce(
    lambda acc, x: acc + x, 
    filter(
        lambda x: x % 2 != 0, 
        map(
            lambda x: x**2, 
            reduce(lambda seq, _: seq + [seq[-1] + seq[-2]], range(2, n), [1, 1])
        )
    )
)

# Ex 5

import math

x = 4
y = 5
b = 10  

lhs = math.log(x * y, b)

rhs = math.log(x, b) + math.log(y, b)

# print(f"Left side: {lhs}")
# print(f"Right side: {rhs}")
# print(f"Are both sides equal? {math.isclose(lhs, rhs)}")

x = 20
y = 5
b = 10

lhs = math.log(x / y, b)

rhs = math.log(x, b) - math.log(y, b)

# print(f"Left side: {lhs}")
# print(f"Right side: {rhs}")
# print(f"Are both sides equal? {math.isclose(lhs, rhs)}")

x = 3
n = 4
b = 10

lhs = math.log(x**n, b)

rhs = n * math.log(x, b)

# print(f"Left side: {lhs}")
# print(f"Right side: {rhs}")
# print(f"Are both sides equal? {math.isclose(lhs, rhs)}")

x = 10
b = 2

lhs = math.log(x, b)

rhs = math.log(x) / math.log(b)

# print(f"Left side: {lhs}")
# print(f"Right side: {rhs}")
# print(f"Are both sides equal? {math.isclose(lhs, rhs)}")

b = 10  
log_1 = math.log(1, b)

# print(f"log_{b}(1) = {log_1}")

# Ex 9
import numpy as np
import matplotlib.pyplot as plt

def plot_math_function(f, min_x, max_x, num_points):
    x = np.linspace(min_x, max_x, num_points)
    vectorized_function = np.vectorize(f)
    y = vectorized_function(x)

    ax = plt.gca()
    ax.spines["bottom"].set_position("zero")
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_visible(False)

    plt.plot(x,y)
    plt.show()

# Ex 10
def plot_math_functions(functions, min_x, max_x, num_points):
    x = np.linspace(min_x, max_x, num_points)
    vectorized_functions = [np.vectorize(f) for f in functions]
    ys = [v(x) for v in vectorized_functions]

    ax = plt.gca()
    ax.spines["bottom"].set_position("zero")
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_visible(False)

    [plt.plot(x,y) for y in ys]
    plt.show()