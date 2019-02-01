import turtle
import numpy as np
import math

#########################################################################
#                                                                       #
# Utility class to render robot onscreen using Python turtle graphics.  #
#                                                                       #
#########################################################################

class Renderer:

    def __init__(self, n=1):
        turtle.radians()
        self.n = n
        self.count = -1

    def render_clear():
        turtle.clear()
        self.count = -1

    def render_reset(self, env, raytrace=False):
        self.count += 1
        self.render_backdrop(env.obstacles, env.bot, raytrace)

    def render_backdrop(self, obstacles, bot, raytrace):
        if self.count % self.n == 0:
            turtle.hideturtle()
            turtle.colormode(255)
            turtle.pencolor((np.random.randint(255),np.random.randint(255),np.random.randint(255)))
            if raytrace:
                self.render_raytrace(obstacles, bot)
            turtle.pensize(3+np.random.randint(5))
            for o in obstacles:
                self.render_obstacle(o)
            self.render_reset_turtle(bot)

    def render_reset_turtle(self, bot):
            turtle.penup()
            turtle.goto(bot.x, bot.y)
            turtle.setheading(bot.heading)
            turtle.turtlesize(3)
            turtle.showturtle()

    def render_obstacle(self, obstacle):
        turtle.hideturtle()
        turtle.penup()
        turtle.goto(obstacle.x+obstacle.radius,obstacle.y)
        turtle.setheading(math.pi/2)
        turtle.pendown()
        turtle.circle(obstacle.radius)

    def render_raytrace(self, obstacles, bot):
        turtle.hideturtle()
        turtle.penup()
        turtle.pensize(1)
        turtle.goto(bot.x, bot.y)
        turtle.pendown()
        for o in obstacles:
            a,b,m,n = bot.bearings_to_ob(o)
            d = bot.distance_to_ob(o)
            self.render_reset_turtle(bot)
            turtle.pendown()
            turtle.goto(m[0], m[1])
            self.render_reset_turtle(bot)
            turtle.pendown()
            turtle.goto(n[0], n[1])
        self.render_reset_turtle(bot)

    def render_step(self, bot, force=False):
        if self.count % self.n == 0 or force:
            turtle.setheading(bot.heading)
            turtle.pendown()
            turtle.goto(bot.x, bot.y)
