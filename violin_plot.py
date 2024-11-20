import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#设置绘图风格
plt.style.use('ggplot')
#处理中文乱码
# plt.rcParams['font.sans-serif'] = ['SimHei']
#坐标轴负号的处理
plt.rcParams['axes.unicode_minus']=False
# 读取数据
tips = pd.read_csv('./pca_pro/nerel_pca_pro.csv')
# 绘制分组小提琴图
sns.violinplot(x = "real label", # 指定x轴的数据
               y = "real label probabilit", # 指定y轴的数据
               hue = "predict true or not", # 指定分组变量
            #    hue_order = ["predict correctly","predict incorrectly"],
               data = tips, # 指定绘图的数据集
               order = ['NATIONALITY', 'CITY', 'EVENT', 'PERCENT', 'PRODUCT', 'PENALTY', 'NUMBER', 'PROFESSION', 'STATE_OR_PROVINCE', 'AGE', 'LOCATION', 'LANGUAGE', 'IDEOLOGY', 'ORGANIZATION', 'TIME', 'COUNTRY', 'DISTRICT', 'DATE', 'DISEASE', 'PERSON', 'CRIME', 'AWARD', 'MONEY', 'RELIGION', 'FACILITY', 'ORDINAL', 'LAW', 'FAMILY', 'WORK_OF_ART'], # 指定x轴刻度标签的顺序
               scale = 'count', # 以预测正确和错误的个数调节小提琴图左右的宽度
               split = True, # 将小提琴图从中间割裂开，形成不同的密度曲线；
               palette = 'RdBu' # 指定预测正确与预测错误对应的颜色
              )
# plt.xticks(rotation=45)
# 添加图形标题
plt.title('NEREL')
# plt.xlabel("",loc='right',labelpad=1)
# plt.ylabel("Probability",loc='top',labelpad=1)
plt.annotate("Probability",xy=(-1,1.49))
plt.annotate("Label category",xy=(26,-0.463))
plt.xticks([0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28],['$NATIONALITY$', '$CITY$', '$EVENT$','$PERCENT$','$PRODUCT$','$PENALTY$','$NUMBER$','$PEOFESSION$','$S\_O\_P$','$AGE$','$LOCATION$','$LANGUAGE$','$IDEOLOGY$','$ORGANIZATION$','$TIME$','$COUNTRY$','$DISTRICT$','$DATE$','$DISEASE$','$PERSON$','$CRIME$','$AWARD$','$MONEY$','$RELIGION$','$FACILITY$','$ORDINAL$','$LAW$','$FAMILY$','$W\_O\_A$'],fontsize=5,rotation=45)
# 设置图例
plt.legend(loc = 'upper right', ncol = 2)

#控制横纵坐标的值域
plt.axis([-1,29,-0.5,1.5])
# 显示图形
plt.show()
# plt.savefig("CADEC.jpg")
