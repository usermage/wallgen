import numpy as np
import itertools
import random
from random import getrandbits, randint
from functools import reduce
import time
import PIL.Image, PIL.ImageDraw
import pandas as pd

_best_ugl = [
                            " ", #0
                            "╴", #1
                            "╷", #2
                            "┐", #3
                            "╶", #4
                            "─", #5
                            "┌", #6
                            "┬", #7
                            "╵", #8
                            "┘", #9
                            "│", #10
                            "┤", #11
                            "└", #12
                            "┴", #13
                            "├", #14
                            "┼", #15
                        ]
#Определяет валидность координат (попадают ли они внутрь массива лабиринта)
def valid_co(x, y, n, m):
    return (x>=0) and (x<n) and (y>=0) and (y<m)
#Отрисовка лабиринта (консольная версия)
def show_nice(a):
    for id, i in enumerate(a):
        for ud, u in enumerate(i):
            print(" ", end="")
            if u[0]:
                print("─"*3, end="")
            else:
                print(" "*3, end="")
        print(" ")
        for ud, u in enumerate(i):
            if u[3]:
                print("│", end="")
            else:
                print(" ", end="")
            print(" "*3, end="")
        print("│")
    for ud, u in enumerate(i):
        print(" ", end="")
        print("─"*3, end="")
    print(" ")
    
#Реализация алгоритма Краскала
def Kruskal(n, m):
    a = np.ones((n,m,4), dtype=np.int32)
    b = np.zeros((n,m), dtype=np.int32)
    def get_way(px,py):
        if px == 0:
            if py == 1:
                return 1
            else:
                return 3
        else:
            if px == 1:
                return 2
            else: return 0
    for i in range(n):
        for u in range(m):
            b[i,u] = i*m + u
    wall = np.array([(i,u,j) for i in range(n) for u in range(m) for j in (1,2) if ((i<n-1) or (j==1)) and ((u<m-1) or (j==2))], dtype=np.int32)
    np.random.shuffle(wall)
    for ix,iy,ij in wall:
        if ij == 1:
            if b[ix, iy] != b[ix, iy+1]:
                a[ix, iy, 1] = 0
                a[ix, iy+1, 3] = 0
                if b[ix, iy] < b[ix, iy+1]:
                    link = b[ix, iy]
                    stack = [(ix, iy+1)]
                else:
                    link = b[ix, iy+1]
                    stack = [(ix, iy)]
            else:
                continue
        else:
            if b[ix, iy] != b[ix+1, iy]:
                a[ix, iy, 2] = 0
                a[ix+1, iy, 0] = 0
                if b[ix, iy] < b[ix+1, iy]:
                    link = b[ix, iy]
                    stack = [(ix+1, iy)]
                else:
                    link = b[ix+1, iy]
                    stack = [(ix, iy)]
            else:
                continue
        while stack:
            x, y = stack.pop()
            if b[x, y] == link:
                continue
            else:
                b[x, y] = link
            for px, py in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                if valid_co(x+px,y+py,n,m) and (a[x,y,get_way(px,py)] == 0) and (b[x+px,y+py] != link):
                    stack.append((x+px,y+py))
    return a

#Подсчёт тупиков (клеток лабиринта с тремя стенками)
def count_no_way(a):
    nway = 0
    for i in a:
        for u in i:
            if sum(u) == 3:
                nway += 1
    return nway

#Модификация алгоритма B2
def B2_3(n, m):
    a = np.zeros((n+1, m+1, 4), dtype=np.int32)
    b = np.zeros((n+1, m+1), dtype=np.int32)
    for i in range(1, n):
        for u in range(1, m):
            b[i,u] = i*m + u
    p = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    def _k(k):
        return (k+2)%4
    for i in range(1, n):
        for u in range(1, m):
            good = [k for k in range(4) if b[i,u] != b[i+p[k][0],u+p[k][1]]]
            k = random.choice(good)
            ii, uu = i+p[k][0], u+p[k][1]
            a[i,u,k] = 1
            a[ii,uu,_k(k)] = 1
            if b[i,u]<b[ii,uu]:
                link = b[i,u]
                stack = [(ii,uu)]
            else:
                link = b[ii,uu]
                stack = [(i,u)]
            while stack:
                x, y = stack.pop()
                if b[x, y] == link:
                    continue
                else:
                    b[x, y] = link
                for k in range(4):
                    if a[x,y,k] and (b[x+p[k][0],y+p[k][1]]!=link):
                        stack.append((x+p[k][0], y+p[k][1]))
    return a

#Перевод из удобного для построения B2 вида лабиринта в нормальный вид
def wall_to_norm(a):
    d = np.ones((a.shape[0]-1,a.shape[1]-1,4), dtype=np.bool)
    for i in range(1, a.shape[0]):
        for u in range(1, a.shape[1]):
            if (u!=a.shape[1]-1) and (not a[i, u, 0]):
                d[i-1, u-1, 1] = False
                d[i-1, u, 3] = False
            if (i!=a.shape[0]-1) and (not a[i, u, 3]):
                d[i-1, u-1, 2] = False
                d[i, u-1, 0] = False
    return d

#Вспомогательная функция для нахождения длины пути решения
def _rr_way(a, x, y, xx, yy, z):
    if (x == a.shape[0]-1) and (y == a.shape[1]-1):
        return z
    if (not a[x,y,1]) and (y+1 != yy):
        mb = _rr_way(a, x, y+1, x, y, z+1)
        if mb is not None:
            return mb
    if (not a[x,y,2]) and (x+1 != xx):
        mb = _rr_way(a, x+1, y, x, y, z+1)
        if mb is not None:
            return mb
    if (not a[x,y,0]) and (x-1 != xx):
        mb = _rr_way(a, x-1, y, x, y, z+1)
        if mb is not None:
            return mb
    if (not a[x,y,3]) and (y-1 != yy):
        mb = _rr_way(a, x, y-1, x, y, z+1)
        if mb is not None:
            return mb
#Длина пути решения
def rr_way(a):
    return _rr_way(a, 0, 0, 0, 0, 0)

#Модификация алгоритма B2
def B2_4(n, m):
    a = np.zeros((n+1, m+1, 4), dtype=np.bool)
    b = np.zeros((n+1, m+1), dtype=np.int32)
    b[:,0] = 1
    b[:,-1] = 1
    b[0,:] = 1
    b[-1,:] = 1
    lost = (n-1)*(m-1)
    now = 1
    p = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    def _k(k):
        return (k+2)%4
    while lost:
        now += 1
        x, y = randint(1, n-1), randint(1, m-1)
        while (b[x, y]):
            x, y = randint(1, n-1), randint(1, m-1)
        while True:
            if b[x, y]!=now:
                b[x, y] = now
                lost -= 1
            good = [k for k in range(4) if b[x+p[k][0], y+p[k][1]] != now]
            if not good:
                x, y = randint(1, n-1), randint(1, m-1)
                while (b[x, y]) != now:
                    x, y = randint(1, n-1), randint(1, m-1)
                continue
            k = random.choice(good)
            xx, yy = x+p[k][0], y+p[k][1]
            a[x,y,k] = True
            a[xx,yy,_k(k)] = True
            if b[xx,yy] !=0:
                break
            x, y = xx, yy
    return a

#Модификация алгоритма B2
def B2_5(n, m):
    a = np.zeros((n+1, m+1, 4), dtype=np.bool)
    b = np.zeros((n+1, m+1), dtype=np.int32)
    b[:,0] = 1
    b[:,-1] = 1
    b[0,:] = 1
    b[-1,:] = 1
    lost = (n-1)*(m-1)
    now = 1
    p = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    def _k(k):
        return (k+2)%4
    while lost:
        now += 1
        x, y = randint(1, n-1), randint(1, m-1)
        while (b[x, y]):
            x, y = randint(1, n-1), randint(1, m-1)
        while True:
            if b[x, y]!=now:
                b[x, y] = now
                lost -= 1
            good = [k for k in range(4) if (b[x+p[k][0], y+p[k][1]] > 0) and (b[x+p[k][0], y+p[k][1]] != now)]
            if not good:
                good = [k for k in range(4) if b[x+p[k][0], y+p[k][1]] != now]
            if not good:
                x, y = randint(1, n-1), randint(1, m-1)
                while (b[x, y]) != now:
                    x, y = randint(1, n-1), randint(1, m-1)
                continue
            k = random.choice(good)
            xx, yy = x+p[k][0], y+p[k][1]
            a[x,y,k] = True
            a[xx,yy,_k(k)] = True
            if b[xx,yy] !=0:
                break
            x, y = xx, yy
    return a

#Модификация алгоритма B2
def B2_6(n, m):
    a = np.zeros((n+1, m+1, 4), dtype=np.bool)
    b = np.zeros((n+1, m+1), dtype=np.int32)
    b[:,0] = 1
    b[:,-1] = 1
    b[0,:] = 1
    b[-1,:] = 1
    lost = (n-1)*(m-1)
    now = 1
    p = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    def _k(k):
        return (k+2)%4
    while lost:
        now += 1
        x, y = randint(1, n-1), randint(1, m-1)
        while (b[x, y]):
            x, y = randint(1, n-1), randint(1, m-1)
        while True:
            if b[x, y]!=now:
                b[x, y] = now
                lost -= 1
            good = [k for k in range(4) if b[x+p[k][0], y+p[k][1]] != now]
            if not good:
                x, y = randint(1, n-1), randint(1, m-1)
                while (b[x, y]) != now:
                    x, y = randint(1, n-1), randint(1, m-1)
                continue
            for i in range(len(good)):
                if good[i]%2==((x)+(y))%2:
                    for u in range(1):
                        good.append(good[i])
            k = random.choice(good)
            xx, yy = x+p[k][0], y+p[k][1]
            a[x,y,k] = True
            a[xx,yy,_k(k)] = True
            if b[xx,yy] !=0:
                break
            x, y = xx, yy
    return a