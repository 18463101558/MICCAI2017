import numpy as np
import tensorflow as tf

def tensorflow_categorical(count, seed):
    assert count > 0
    arr = [1.] + [.0 for _ in range(count-1)]
    return tf.random_shuffle(arr, seed)

# Returns a random array [x0, x1, ...xn] where one is 1 and the others
# are 0. Ex: [0, 0, 1, 0].
def rand_one_in_array(count, seed=None):
    if seed is None:
        seed = np.random.randint(1, 10e6)
    return tensorflow_categorical(count=count, seed=seed)

class JoinLayerGen:
    '''
    JoinLayerGen将为全局droppath路径和全局droppout路径初始化种子。
    这些种子将用于创建随机张量，子层将使用它们来确定它们是否必须使用全局droppout以及它采用的路径。
    '''
    def __init__(self, width, global_p=0.5):#3 0.5 false
        self.global_p = global_p#0.5
        self.width = width#3 也就是跟c相同的家伙 代表了柱的数量
        self.switch_seed = np.random.randint(1, 10e6)#设置选择dropout种子
        self.path_seed = np.random.randint(1, 10e6)
        self.is_global = self._build_global_switch()#随机设置使用最长路径
        self.path_array = self._build_global_path_arr()#设置一个path选择arry，但是保证至少有一个是正常工作的

    def _build_global_path_arr(self):
        # 决定选择哪一条
        return rand_one_in_array(seed=self.path_seed, count=self.width)#跟column数量是相同的

    def _build_global_switch(self):
        return self.switch_seed % 2==1 #近似地随机产生一个以50%概率选择全局路径的变量

    def get_join_layer(self, drop_p):
        global_switch = self.is_global
        global_path = self.path_array
        #return JoinLayer(drop_p=drop_p, is_global=global_switch, global_path=global_path)
        #drop率，是否选择全局冻结，全局mask位置，是否强制使用最长路径

def fractal_net( drop_path, global_p=0.5, dropout=None):
    b=1#block数量
    c=3#column数量
    def f(z):
        output = z
        # 初始化JoinLayerGen，用于派生共享相同全局丢弃路径的JoinLayers 这一步主要是设置随机数种子
        join_gen = JoinLayerGen(width=c, global_p=global_p)#3 0.5 false
        for i in range(b):#b是堆叠块的个数
            print(i)
            dropout_i = dropout[i] if dropout else None
            output = fractal_block(join_gen=join_gen,
                                   c=c,
                                   drop_p=drop_path,#0.15
                                   dropout=dropout_i)(output)#drop率
            #output = MaxPooling2D(pool_size=(2,2), strides=(2,2))(output)#每个block之间缩小分辨率
        return output
    return f
def fractal_block(join_gen, c,drop_p, dropout=None):
    def f(z):
        columns = [[z] for _ in range(c)]#c这里等于3，所以这里将z复制了三份，成为一个新数组，代表一共有三条路径
        for row in range(2**(c-1)):#网络深度从0到2**(c-1) - 1 先遍历深度
            t_row = []
            for col in range(c):
                prop = 2**(col)#该深度对应的层是否有卷积的必要
                # Add blocks
                if (row+1) % prop == 0:#因为index是从0开始的，所以得加1，并且比如对于col=2，也就是最后一个，那么他只有一次卷积的机会
                    t_col = columns[col]#因为始终只有三个柱，这里是将对应柱的上一个取出来，用于下一次卷积
                    t_col.append(fractal_conv(filter=filter,
                                              nb_col=nb_col,
                                              nb_row=nb_row,
                                              dropout=dropout)(t_col[-1]))#执行对应的卷积
                    t_row.append(col)#t_row在当前深度所有需要合并层的index吗

            # Merge (if needed)
            if len(t_row) > 1:#注意不需要合并的在if这里就过滤掉了
                merging = [columns[x][-1] for x in t_row]#这一个-1有点意思，是因为这里 [[z] for _ in range(c)]，z外面有两个中括号，所以取出来的也就是前面一层的结果
                merged  = join_gen.get_join_layer(drop_p=drop_p)(merging)
                for i in t_row:
                    columns[i].append(merged)
        return columns[0][-1]
    return f