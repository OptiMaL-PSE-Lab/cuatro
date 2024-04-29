import numpy as np
import math
import random

def mat_func(X1,X2,func):
  output = np.zeros(shape = X1.shape)
  I, J = X1.shape
  for i in range(I):
    for j in range(J):
      output[i,j] = func([X1[i,j],X2[i,j]])
  return output

def black_box(x):
  return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2

def quad_approx(a,x):
    return a[0] + a[1]*x[0] + a[2]*x[1] + 0.5*a[3]*x[0]**2 + a[4]*x[0]*x[1] + 0.5*a[5]*x[1]**2
     
def least_squares(a, surrogate, samples, samples_eval):
  result = 0
  for i in range(len(samples)):
    result += (samples_eval[i] - surrogate(a, samples[i]))**2
  return result

def distance_constraint(point, center, max_distance):
  return max_distance**2 - ((point[0]-center[0])**2 + (point[1]-center[1])**2) # If positive, ensures the point is within max_distance of center

def convexity_constraint(a):
  trc = a[3]+a[5]
  det = a[3]*a[5]-a[4]**2

  # Because math.sqrt is always positive, the following represents the most
  # negative eigenvalue. If this is positive, then the Hessian is positive
  # definite
  return trc - math.sqrt(trc**2 - 4*det)

# Returns distance between vectors a and b
def distance(a,b, output_square = False):
  output = 0
  for i in range(len(a)):
    output += (a[i]-b[i])^2
  if output_square:
    return output
  else:
    return math.sqrt(output)

# Rotates a point around a pivot counter clockwise by a given angle
def rotate_around(point,pivot,angle):
    arm = point - pivot
    rotate_mat = np.array([[math.cos(angle) , -math.sin(angle)],
                           [math.sin(angle) ,  math.cos(angle)]])
    return rotate_mat@arm + pivot

# If x is inside bounds, returns the distance to the closest edge
# If x is outside bounds, returns the negative of the distance to the closest edge
def inside_bounds(x, bounds):
    most_negative_deviation  = math.inf
    for i in range(len(bounds)):
        from_min = x[i] - min(bounds[i])
        from_max = max(bounds[i]) - x[i]
        most_negative_deviation = min(most_negative_deviation, from_min, from_max)
    return most_negative_deviation

# Vice versa
def outside_bounds(x, bounds):
    return -inside_bounds(x,bounds)

# Returns the given bounds centered around a point
# def bounds_centered_on(center, bounds = None, diameter = None):
#     centered_bounds = np.empty((len(center),2))
#     if bounds:
#         for i in range(len(center)):
#             bound_range = max(bounds[i]) - min(bounds[i])
#             centered_bounds[i] = np.array([-bound_range/2, bound_range/2]) + center[i]
#         return centered_bounds
#     elif diameter:
#         for i in range(len(center)):
#             centered_bounds[i] = np.array([-diameter/2, diameter/2]) + center[i]
#         return centered_bounds
#     else:
#         raise ValueError("Must specify either bounds or radius")

# Returns center of a given set of bounds
def center_of_bounds(bounds):
    return np.array([(max(b)+min(b))/2 for b in bounds])
    

# Generates n_s samples using LHS on a n-dimensional hypercube
def latin_hypercube(n_s,n):
  S = np.zeros((n_s,n))
  for i in range(n):
    S[:,i] = np.array([[shuffle(n_s)]])
  return S

# Generates a list of 0 to N-1 in a random order
# The naming follows an analogy of a deck of cards
def shuffle(N):
  deck = list(range(N))
  shuffled = []
  for j in range(N):
    card_number = np.random.randint(0,len(deck)) # generates a random integer from 0 to the current length of the deck (exclusive)
    shuffled.append(deck[card_number]) # adds the card to the shuffled deck
    deck.remove(deck[card_number]) # removes the card from the deck
  return shuffled

# def random_radial_sample(center, radius):
#     arg = 2*math.pi*random.random()
#     mod = math.sqrt(radius**2*random.random())
#     return np.array([mod*math.cos(arg), mod*math.sin(arg)]) + center

def random_bounded_sample(bounds):
    s = np.empty(len(bounds))
    for i in range(len(bounds)):
        range_n = max(bounds[i]) - min(bounds[i])
        s[i] = random.random()*range_n + min(bounds[i])
    return s

# Returns x and y coordinates for a circle. To be used in plt.plot()
def circle(center, radius, resolution = 20):
    angle = np.linspace(0, 2*math.pi, resolution)
    x = radius * np.cos(angle) + center[0]
    y = radius * np.sin(angle) + center[1]
    return (x,y)

# Returns x and y coordinates for a rectangle given bounds. To be used in plt.plot()
def rect(bounds):
    x = [bounds[0,0],bounds[0,0],bounds[0,1],bounds[0,1],bounds[0,0]]
    y = [bounds[1,0],bounds[1,1],bounds[1,1],bounds[1,0],bounds[1,0]]
    return (x,y)

def path(*points):
    path = np.vstack([p for p in points]).transpose()
    return path

















