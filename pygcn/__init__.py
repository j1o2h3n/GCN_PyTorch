'''
作用是在文件夹中包含一个__init__.py，Python就会把
文件夹当作一个package，里面的py文件就能够在外面被
import了。总而言之就是为了方便被后续其他程序调用，
而对当前本工程无用。
'''

# __future__包是把下一个新版本的特性导入到当前版本，导入python未来支持的语言特征
from __future__ import print_function	# 将print从语言语法中移除，让你可以使用函数的形式
from __future__ import division	# python2导入精确除法，例如1/3=0，导入后1/3=0.33

from .layers import *
from .models import *
from .utils import *

'''
引用语句from .XXX import XXXX导入带上点（.），表示
从__init__.py的目录下导入模块。否则其他文件不知道
去哪导入，会出错。
'''
