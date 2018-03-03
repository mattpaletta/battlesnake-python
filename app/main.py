from Queue import PriorityQueue

import bottle
import os
import numpy as np
from math import sqrt
from numba import jit
from scipy.ndimage import gaussian_filter


def euclidean_distance(start, end):
    sx, sy = (start["x"], start["y"])
    ex, ey = (end["x"], end["y"])
    return sqrt(abs(ex - sx)**2 + abs(ey - sy)**2)

def manhattan_distance(start, end):
    sx, sy = (start["x"], start["y"])
    ex, ey = (end["x"], end["y"])
    return abs(ex - sx) + abs(ey - sy)

#@jit
def get_neighboars(board, square):
    x = square["x"]
    y = square["y"]

    neighbours = []
    neighbours.append({"x": x + 1, "y": y}) if x + 1 < len(board) else None
    neighbours.append({"x": x - 1, "y": y}) if x - 1 >= 0 else None
    neighbours.append({"x": x ,"y": y + 1}) if y + 1 < len(board) else None
    neighbours.append({"x": x, "y": y - 1}) if y - 1 >= 0 else None

    return neighbours

#@jit
def find_shortest_path(graph, start, end, path=[]):
    path = path + [start]
    if (start["x"], start["y"]) == (end["x"], end["y"]):
        return path
    shortest = None
    neighbors = sorted(get_neighboars(graph, start), key=lambda x: manhattan_distance(x, end))
    for node in neighbors:
        if len(filter(lambda p: p["x"] == node["x"] and p["y"] == node["y"] ,path)) == 0: #and len(path) < manhattan_distance(start, end):
            newpath = find_shortest_path(graph, node, end, path)
            if newpath:
                return newpath
                #if not shortest or len(newpath) < len(shortest):
                    #shortest = newpath
    return shortest

@bottle.route('/')
def static():
    return "the server is running"


@bottle.route('/static/<path:path>')
def static(path):
    return bottle.static_file(path, root='static/')


@bottle.post('/start')
def start():
    data = bottle.request.json
    game_id = data.get('game_id')
    board_width = data.get('width')
    board_height = data.get('height')

    head_url = '%s://%s/static/head.png' % (
        bottle.request.urlparts.scheme,
        bottle.request.urlparts.netloc
    )

    # TODO: Do things with data

    return {
        'color': '#00FF00',
        'taunt': '{} ({}x{})'.format(game_id, board_width, board_height),
        'head_url': head_url
    }

#@jit
def choose_next_move(data):

    FOOD_VALUE = 1.0
    SNAKE_VALUE = -1.0
    PLANNING = 3

    # Add to game board
    board = np.zeros(shape=(data["width"], data["height"]), dtype=np.float32)
    health = data["you"]["health"]
    you = data["you"]["body"]["data"]
    snakes = data["snakes"]["data"]
    food = data["food"]["data"]
    head = you[0]

    def add_item(food, val):
        if type(food) == list:
            for f in food:
                board[int(f["x"])][int(f["y"])] = val
        else:
            board[int(food["x"])][int(food["y"])] = val

    # Add food to board:
    map(lambda f: add_item(f, FOOD_VALUE), food)

    # Add snakes to board:
    map(lambda x: add_item(x, SNAKE_VALUE), map(lambda b: b["body"]["data"], snakes))
    map(lambda x: add_item(x, SNAKE_VALUE), you)

    print("Finding shortest path")
    food_dist = map(lambda f: find_shortest_path(graph=board, start=head, end=f), food)
    print("Got clostest food")
    clostest_food_dist = min(map(lambda x: len(x), food_dist))
    if (health - PLANNING) <= clostest_food_dist:
        # Move towards food...
        clostest_food = filter(lambda f: len(f) == clostest_food_dist, food_dist)[0]

        if you["x"] > clostest_food[0]:
            return 0
        elif you["x"] < clostest_food[0]:
            return 1
        elif you["y"] > clostest_food[1]:
            return 2
        elif you["y"] > clostest_food[1]:
            return 3

    # Find biggest reward direction:
    blurred = gaussian_filter(input=board, cval=SNAKE_VALUE, sigma=2)
    options = get_neighboars(board=blurred, square=head)
    best = max(map(lambda p: blurred[p["x"]][p["y"]], options))

    best_option = list(filter(lambda p: blurred[p["x"]][p["y"]] == best, options))[0]

    if head["x"] > best_option["x"]:
        return 0
    elif head["x"] < best_option["x"]:
        return 1
    elif head["y"] > best_option["y"]:
        return 2
    elif head["y"] > best_option["y"]:
        return 3
    else:
        return 0


@bottle.post('/move')
def move():
    data = bottle.request.json
    print("Got Move!")
    # TODO: Do things with data
    direction = ['up', 'down', 'left', 'right'][choose_next_move(data)]

    print direction
    return {
        'move': direction,
        'taunt': 'battlesnake-python!'
    }


# Expose WSGI app (so gunicorn can find it)
application = bottle.default_app()

if __name__ == '__main__':
    bottle.run(
        application,
        host=os.getenv('IP', '0.0.0.0'),
        port=os.getenv('PORT', '8080'),
        debug = True)
