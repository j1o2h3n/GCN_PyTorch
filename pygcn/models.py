# 此文件定义GCN模型
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution	# 简单的GCN层

	'''
	nfeat，底层节点的参数，feature的个数
	nhid，隐层节点个数
	nclass，最终的分类数
	dropout参数
	GCN由两个GraphConvolution层构成
	输出为输出层做log_softmax变换的结果
	'''


class GCN(nn.Module):	# nn.Module类的单继承
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
		# super()函数是用于调用父类(超类)的方法
		# super().__init__()表示子类既能重写__init__()方法又能调用父类的方法
		# https://www.runoob.com/w3cnote/python-extends-init.html
		

        self.gc1 = GraphConvolution(nfeat, nhid)
		# self.gc1代表GraphConvolution()，gc1输入尺寸nfeat，输出尺寸nhid
        self.gc2 = GraphConvolution(nhid, nclass)
		# self.gc2代表GraphConvolution()，gc2输入尺寸nhid，输出尺寸ncalss
        self.dropout = dropout
		# dropout参数

    def forward(self, x, adj):	# 前向传播
        x = F.relu(self.gc1(x, adj))
		# x是输入特征，adj是邻接矩阵
		# self.gc1(x, adj)是执行GraphConvolution中forward
		# 得到（随机初始化系数*输入特征*邻接矩阵+偏置）
		# 经过relu函数
        x = F.dropout(x, self.dropout, training=self.training)
		# 输入x，dropout参数是self.dropout
		# training=self.training表示将模型整体的training状态参数传入dropout函数，没有此参数无法进行dropout
		
        x = self.gc2(x, adj)
		# gc2层
        return F.log_softmax(x, dim=1)	
		# 输出为输出层做log_softmax变换的结果，dim表示log_softmax将计算的维度
		
