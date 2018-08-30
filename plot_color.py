# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 10:23:39 2018

@author: Sirius
# specifying color from python RGB values
"""

import matplotlib.pyplot as plt
import matplotlib.colors as colors
colors.Normalize([48,186,238])
colors.to_rgb([48,186,238])

# 将0-255转化为16进制
c1 = '#%02x%02x%02x' % (48,186,238)
c2 = '#%02x%02x%02x' % (253,114,71)
c3 = '#%02x%02x%02x' % (41,141,198)
c4 = '#%02x%02x%02x' % (231,115,85)

plt.plot(range(10),c=c1)
plt.plot(range(10,20),c=c2)
plt.plot(range(20,30),c=c3)
plt.plot(range(30,40),c=c4)
