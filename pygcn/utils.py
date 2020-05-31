# 此文件是定义了一些需要的工具函数
import numpy as np	# 导入numpy包，且用np名称等价
import scipy.sparse as sp	# scipy.sparse稀疏矩阵包，且用sp名称等价
import torch


def encode_onehot(labels):	# 将标签转换为one-hot编码形式
    classes = set(labels)	# set()函数就是提取输入的组成元素，且进行不重复无序的排列输出
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in 
					enumerate(classes)}
	# 这一句主要功能就是进行转化成dict字典数据类型，且键为元素，值为one-hot编码
	# enumerate()将可遍历对象组合成一个含数据下标和数据的索引序列
	# for i,c in XXX 将XXX序列进行循环遍历赋给(i,c)，这里是i得数据下标，c得数据
	# len()返回元素的个数，np.identity()函数创建对角矩阵，返回主对角线元素为1，其余元素为0
	# 矩阵[i,:]是仅保留第一维度的下标i的元素和第二维度所有元素，直白来看就是提取了矩阵的第i行
	# {}生成了字典，c:xxx 是字典的形式，c作为键，xxx作为值，在for in循环下进行组成字典
	# c:xxx在for in前面这种结构我还是没查到所以然，只是python跑出来看到了结果明白了怎么运行
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
							 dtype=np.int32)
	# array()是numpy是数组格式，dtype是数组元素的数据类型，list()用于将元组转换为列表
	# map(function, iterable)是对指定序列iterable中的每一个元素调用function函数，
	# 根据提供的函数对指定序列做映射，返回包含每次function函数返回值的新列表
    # 这句话的意思就是将输入一一对应one-hot编码进行输出
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):	# 加载数据
    """加载引文网络数据集（目前仅限cora）"""
    print('Loading {} dataset...'.format(dataset))
	# format()后面的内容cora，填入前面大括号{}中替代，显示输出Loading cora dataset...

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
	# 论文样本的独自信息的数组
	# np.genfromtxt()函数用于从.csv文件或.tsv文件中生成数组
    # np.genfromtxt(fname, dtype)
    # frame：文件名	../data/cora/cora.content		dtype：数据类型	str字符串								
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
	# 提取样本的特征，并将其转换为csr矩阵（压缩稀疏行矩阵），用行索引、列索引和值表示矩阵
	# [:, 1:-1]是指行全部选中、列选取第二列至倒数第二列，float32类型
	# 这句功能就是去除论文样本的编号和类别，留下每篇论文的词向量，并将稀疏矩阵编码压缩
    labels = encode_onehot(idx_features_labels[:, -1])
	# 提取论文样本的类别标签，并将其转换为one-hot编码形式

    # 构建图
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # 提取论文样本的编号id数组
    idx_map = {j: i for i, j in enumerate(idx)}
	# 由样本id到样本索引的映射字典
	# enumerate()将可遍历对象组合成一个含数据下标和数据的索引序列
	# {}生成了字典，论文编号id作为索引的键，顺序数据下标值i作为键值:0,1,2,...
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
	# 论文样本之间的引用关系的数组
	# np.genfromtxt()函数用于从.csv文件或.tsv文件中生成数组
	# np.genfromtxt(fname, dtype)
    # frame：文件名	../data/cora/cora.cites		dtype：数据类型	int32	
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
	# 将论文样本之间的引用关系用样本字典索引之间的关系表示，
	# 说白了就是将论文引用关系数组中的数据(论文id）替换成对应字典的索引键值
	# list()用于将元组转换为列表。flatten()是将关系数组降为一维，默认按一行一行排列
	# map()是对降维后的一维关系数组序列中的每一个元素调用idx_map.get进行字典索引，
	# 即将一维的论文引用关系数组中论文id转化为对应的键值数据
	# .shape是读取数组的维度，.reshape()是将一维数组复原成原来维数形式
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
	# 构建图的邻接矩阵，用坐标形式的稀疏矩阵表示，非对称邻接矩阵
	# edges.shape[0]表示引用关系数组的维度数（行数），np.ones全1的n维数组
	# edges[:, 0]被引用论文的索引数组做行号row，edges[:, 1]引用论文的索引数组做列号col
	# labels.shape[0]总论文样本的数量，做方阵维数
	# 前面说白了就是引用论文的索引做列，被引用论文的索引做行，然后在这个矩阵面填充1，其余填充0			

    # 建立对称邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
	# 将非对称邻接矩阵转变为对称邻接矩阵（有向图转无向图）
	# A.multiply(B)是A与B的Hadamard乘积，A>B是指按位将A大于B的位置进行置1其余置0（仅就这里而言可以这么理解，我没找到具体解释）
	# adj=adj+((adj转置)⊙(adj.T > adj))-((adj)⊙(adj.T > adj))
	# 基本上就是将非对称矩阵中的非0元素进行对应对称位置的填补，得到对称邻接矩阵
	
    features = normalize(features)
	# features是样本特征的压缩稀疏矩阵，行规范化稀疏矩阵，具体函数后面有定义
    adj = normalize(adj + sp.eye(adj.shape[0]))
	# 对称邻接矩阵+单位矩阵，并进行归一化
	# 这里即是A+I，添加了自连接的邻接矩阵
	# adj=D^-1(A+I)

	# 分割为train，val，test三个集，最终数据加载为torch的格式并且分成三个数据集
    idx_train = range(140)		# 0~139，训练集索引列表
    idx_val = range(200, 500)	# 200~499，验证集索引列表
    idx_test = range(500, 1500)	# 500~1499，测试集索引列表
	# range()创建整数列表

    features = torch.FloatTensor(np.array(features.todense()))	# 将特征矩阵转化为张量形式
	# .todense()与.csr_matrix()对应，将压缩的稀疏矩阵进行还原
    labels = torch.LongTensor(np.where(labels)[1])
	# np.where(condition)，输出满足条件condition(非0)的元素的坐标，np.where()[1]则表示返回列的索引、下标值
	# 说白了就是将每个标签one-hot向量中非0元素位置输出成标签
	# one-hot向量label转常规label：0,1,2,3,……
    adj = sparse_mx_to_torch_sparse_tensor(adj)
	# 将scipy稀疏矩阵转换为torch稀疏张量，具体函数下面有定义

    idx_train = torch.LongTensor(idx_train)	# 训练集索引列表
    idx_val = torch.LongTensor(idx_val)		# 验证集索引列表
    idx_test = torch.LongTensor(idx_test)	# 测试集索引列表
	# 转化为张量
	
    return adj, features, labels, idx_train, idx_val, idx_test
	# 返回（样本关系的对称邻接矩阵的稀疏张量，样本特征张量，样本标签，
	#		训练集索引列表，验证集索引列表，测试集索引列表）


def normalize(mx):	# 这里是计算D^-1A，而不是计算论文中的D^-1/2AD^-1/2	
    """行规范化稀疏矩阵"""
    # 这个函数思路就是在邻接矩阵基础上转化出度矩阵，并求D^-1A随机游走归一化拉普拉斯算子
	# 函数实现的规范化方法是将输入左乘一个D^-1算子，就是将矩阵每行进行归一化
	
	rowsum = np.array(mx.sum(1))
	# .sum(1)计算输入矩阵的第1维度求和的结果，这里是将二维矩阵的每一行元素求和
    r_inv = np.power(rowsum, -1).flatten()
	# rowsum数组元素求-1次方，flatten()返回一个折叠成一维的数组（默认按行的方向降维）
	# 求倒数
	
    r_inv[np.isinf(r_inv)] = 0.
	# isinf()测试元素是否为正无穷或负无穷,若是则返回真，否则是假，最后返回一个与输入形状相同的布尔数组
	# 如果某一行全为0，则倒数r_inv算出来会等于无穷大，将这些行的r_inv置为0
	# 这句就是将数组中无穷大的元素置0处理
    r_mat_inv = sp.diags(r_inv)	# 稀疏对角矩阵
	# 构建对角元素为r_inv的对角矩阵
	# sp.diags()函数根据给定的对象创建对角矩阵，对角线上的元素为给定对象中的元素
    mx = r_mat_inv.dot(mx)	# 点积
	# 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘
	# 所谓矩阵点积就是两个矩阵正常相乘而已
	
    return mx	# D^-1A


def accuracy(output, labels):	# 准确率，此函数可参考学习借鉴复用
    preds = output.max(1)[1].type_as(labels)
	# max(1)返回每一行最大值组成的一维数组和索引,output.max(1)[1]表示最大值所在的索引indice
	# type_as()将张量转化为labels类型
    correct = preds.eq(labels).double()
	# eq是判断preds与labels是否相等，相等的话对应元素置1，不等置0
    correct = correct.sum()
	# 对其求和，即求出相等(置1)的个数
    return correct / len(labels)	# 计算准确率


	# Scipy中的sparse matrix转换为PyTorch中的sparse matrix，此函数可参考学习借鉴复用
	# 构建稀疏张量，一般需要Coo索引、值以及形状大小等信息
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将scipy稀疏矩阵转换为torch稀疏张量。"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
	# tocoo()是将此矩阵转换为Coo格式，astype()转换数组的数据类型
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
	# vstack()将两个数组按垂直方向堆叠成一个新数组
	# torch.from_numpy()是numpy中的ndarray转化成pytorch中的tensor
	# Coo的索引
    values = torch.from_numpy(sparse_mx.data)
	# Coo的值
    shape = torch.Size(sparse_mx.shape)
	# Coo的形状大小
    return torch.sparse.FloatTensor(indices, values, shape)	# sparse.FloatTensor()构造构造稀疏张量
	
