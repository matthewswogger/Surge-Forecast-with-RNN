import numpy as np
import matplotlib.pyplot as plt

def populate_horizontal_line_between_surge_points(line):
    min_x = min(line)[0]
    max_x = max(line)[0]
    space_between_points = 5
    new_point = min_x
    points_list = [min_x]
    while new_point < max_x:
        new_point += space_between_points
        points_list.append(new_point)
    return points_list

def create_x_y_pairs(line, points_list):
    gradient = []
    rip_apart_line = [point for point in line]
    for point in points_list:
        count = 0
        for origin_point in rip_apart_line:
            y = origin_point[1]
            if point == origin_point[0] and count == 0:
                gradient.append(origin_point)
                rip_apart_line.pop(0)
                count = 1
            elif count == 0:
                gradient.append([point, y, 1.0])
                count = 1
    return gradient

def create_gradient(x_y_list, line):
    for i in xrange(len(line)-1):
        start_surge = line[i][2]
        surge_jump = (line[i][2] - line[i+1][2])/26.0
        if i == 0:
            start = 1
            stop = 26
        elif i == 1:
            start = 27
            stop = 52
        elif i == 2:
            start = 53
            stop = 78
        elif i == 3:
            start = 79
            stop = 104
        elif i == 4:
            start = 105
            stop = 130
        elif i == 5:
            start = 131
            stop = 156
        elif i == 6:
            start = 157
            stop = 182
        elif i == 7:
            start = 183
            stop = 208
        else:
            start = 209
            stop = 234
        for j in xrange(start,stop):
            if surge_jump > 0:
                start_surge -= surge_jump
                x_y_list[j][2] = round(start_surge,4)
            elif surge_jump < 0:
                start_surge -= surge_jump
                x_y_list[j][2] = round(start_surge,4)
            else:
                x_y_list[j][2] = start_surge
    return x_y_list

def populate_horizontal_gradient_line_between_surge_points(line):
    points_list = populate_horizontal_line_between_surge_points(line)
    x_y_list = create_x_y_pairs(line, points_list)
    gradient = create_gradient(x_y_list, line)
    return gradient

def horizontal_lines(all_gradients):
    alphas = np.linspace(0,1,150)
    surges = np.linspace(0.9,1.6,150)
    for grad_surge in all_gradients:
        for i, surge in enumerate(surges):
            if grad_surge[2] < surge:
                grad_surge[2] = alphas[i]
                break
            if grad_surge[2] >= 1.6:
                grad_surge[2] = 1.0
                break
    return all_gradients
####################################################
def another_function(gradient_0,gradient_1,gradient_2,gradient_3,gradient_4):
    short_gradients = [gradient_0,gradient_1,gradient_2,gradient_3,gradient_4]
    first = zip(gradient_0,gradient_1)
    second = zip(gradient_1,gradient_2)
    third = zip(gradient_2,gradient_3)
    fourth = zip(gradient_3,gradient_4)

    alphas = np.linspace(0,1,150)
    surges = np.linspace(0.9,1.6,150)
    for gradient in short_gradients:
        for grad_surge in gradient:
            for i, surge in enumerate(surges):
                if grad_surge[2] < surge:
                    grad_surge[2] = alphas[i]
                    break
                if grad_surge[2] >= 1.6:
                    grad_surge[2] = 1.0
                    break
    return first, second, third, fourth

def almost_done(thing):
    for segment in thing:
        start = segment[0][1]
        end = segment[1][1]

        start_surge = segment[0][2]
        end_surge = segment[1][2]
        surge_jump = (start_surge - end_surge)/20

        y_list = []
        for y in xrange(start,end,6):
            y_list.append([segment[0][0],y,0.0])
        y_list = y_list[1:]

        for spot in y_list:
            if surge_jump > 0:
                start_surge -= surge_jump
                spot[2] = round(start_surge,4)
            elif surge_jump < 0:
                start_surge -= surge_jump
                spot[2] = round(start_surge,4)
            else:
                spot[2] = start_surge
            plt.scatter(spot[0], spot[1], c='#99173C', s=3, alpha=spot[2], edgecolors= 'none')



#################################################################################



line_0 = [[440,175,1.0],[570,175,1.0],[700,175,1.0],[830,175,1.0],[960,175,1.0],[1090,175,1.0],[1220,175,1.0],[1350,175,1.0],[1480,175,1.0]]

line_1 = [[440,295,1.0],[570,295,1.0],[700,295,1.0],[830,295,1.5],[960,295,1.4],[1090,295,1.2],[1220,295,1.1],[1350,295,1.0],[1480,295,1.0]]

line_2 = [[440,415,1.0],[570,415,1.7],[700,415,1.6],[830,415,1.4],[960,415,1.3],[1090,415,1.0],[1220,415,1.2],[1350,415,1.0],[1480,415,1.0]]

line_3 = [[440,535,1.0],[570,535,1.0],[700,535,1.0],[830,535,1.0],[960,535,1.0],[1090,535,1.2],[1220,535,1.1],[1350,535,1.0],[1480,535,1.0]]

line_4 = [[440,655,1.0],[570,655,1.0],[700,655,1.0],[830,655,1.0],[960,655,1.0],[1090,655,1.0],[1220,655,1.0],[1350,655,1.0],[1480,655,1.0]]

#       forecast,real
# point_16: 1.6 : 1.9
# point_17: 1.7 : 1.6
# point_14: 1.3 : 1.2
# point_15: 1.4 : 1.4
# point_12: 1.2 : 1.0
# point_13: 1.0 : 1.0
# point_25: 1.1 : 1.0
# point_26: 1.2 : 1.0
# point_27: 1.3 : 1.2
# point_24: 1.0 : 1.0
# point_29: 1.5 : 1.6
# point_28: 1.4 : 1.4
# point_0 : 1.1 : 1.0
# point_1 : 1.2 : 1.0
# point_2 : 1.4 : 1.3
# point_3 : 1.5 : 1.4

gradient_0 = populate_horizontal_gradient_line_between_surge_points(line_0)
gradient_1 = populate_horizontal_gradient_line_between_surge_points(line_1)
gradient_2 = populate_horizontal_gradient_line_between_surge_points(line_2)
gradient_3 = populate_horizontal_gradient_line_between_surge_points(line_3)
gradient_4 = populate_horizontal_gradient_line_between_surge_points(line_4)

all_gradients = gradient_0 + gradient_1 + gradient_2 + gradient_3 + gradient_4
all_gradients = horizontal_lines(all_gradients)

im = plt.imread('map.png')
implot = plt.imshow(im)

line_5 = [[440,175,1.0],[570,175,1.0],[700,175,1.0],[830,175,1.0],[960,175,1.0],[1090,175,1.0],[1220,175,1.0],[1350,175,1.0],[1480,175,1.0]]

line_6 = [[440,295,1.0],[570,295,1.0],[700,295,1.0],[830,295,1.5],[960,295,1.4],[1090,295,1.2],[1220,295,1.1],[1350,295,1.0],[1480,295,1.0]]

line_7 = [[440,415,1.0],[570,415,1.7],[700,415,1.6],[830,415,1.4],[960,415,1.3],[1090,415,1.0],[1220,415,1.2],[1350,415,1.0],[1480,415,1.0]]

line_8 = [[440,535,1.0],[570,535,1.0],[700,535,1.0],[830,535,1.0],[960,535,1.0],[1090,535,1.2],[1220,535,1.1],[1350,535,1.0],[1480,535,1.0]]

line_9 = [[440,655,1.0],[570,655,1.0],[700,655,1.0],[830,655,1.0],[960,655,1.0],[1090,655,1.0],[1220,655,1.0],[1350,655,1.0],[1480,655,1.0]]


gradient_5 = populate_horizontal_gradient_line_between_surge_points(line_5)
gradient_6 = populate_horizontal_gradient_line_between_surge_points(line_6)
gradient_7 = populate_horizontal_gradient_line_between_surge_points(line_7)
gradient_8 = populate_horizontal_gradient_line_between_surge_points(line_8)
gradient_9 = populate_horizontal_gradient_line_between_surge_points(line_9)
first, second, third, fourth = another_function(gradient_5,gradient_6,gradient_7,gradient_8,gradient_9)

almost_done(first)
almost_done(second)
almost_done(third)
almost_done(fourth)

for point in all_gradients:
    plt.scatter(point[0], point[1], c='#99173C', s=3, alpha=point[2], edgecolors= 'none')
plt.savefig('forecast.png', bbox_inches='tight', dpi=600)
# plt.show()
