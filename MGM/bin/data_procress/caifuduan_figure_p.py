# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from collections import Counter
from pandas import DataFrame
from pandas import Series
import echarts as ec

'''加载数据'''
def load_data(filename):
    return pd.read_csv(filename, error_bad_lines=False)

'''将series类型的数据转换为list'''
def df_to_list(df):
    df = df.tolist()
    res = []
    for each in df:
        vals = [k for k in str(each).split("|") if not pd.isnull(
            k) and k != 'nan' and k != 'None' and k != "-999" and k != "" and k != "投资" and k != "理财" and k != "资产配置"]
        res += list(set(vals))
    return res


'''填充开户时间和生日：注意，这里面有些是缺失数据，所以用-999填充'''
def fill_opentime_birth_date(x):
    if pd.isnull(x):
        return -999
    else:
        return 2017 - int(str(x)[:4])

'''年龄的划分，这里有可能需要将年龄进一步的合并，特征是30-40岁之间的'''
def process_age(x):
    if x <= 30:
        return "<=30"
    elif x <= 40:
        return "<=40"
    elif x <= 50:
        return "<=50"
    elif x <= 60:
        return "<=60"
    else:
        return ">60"

'''处理客户级别的，将客户分成不同的级别'''
def process_customer_level(x):
    if (x <= 2 and x >= 1):
        return "<=2"
    elif x == 3:
        return "3"
    elif x >= 3:
        return "3+"

'''开户时间的处理'''
def process_open_time(x):
    if x >= 5:
        return "5+"
    else:
        return str(x)

'''处理浮点数的问题，保留两位小数，主要是因为round函数存在问题'''
def remove_2_xiaoshu(x):
    return '%.2f' % x

'''处理省市的问题'''
def process_province(x):
    jiangsu_str = "江苏省"
    zhejiang_str = "浙江省"
    beijing_str = "北京市"
    shanghai_str = "上海市"
    shandong_str = "山东省"
    guangdong_str = "广东省"
    if (x != jiangsu_str) and (x != zhejiang_str) and (x != beijing_str) and (x != shanghai_str) and (
                x != shandong_str):
        return 'other'
    else:
        return x

'''加载相应的数据，提取相应的数据特征'''
def load_data():
    data_fir = "./data/sf/zh/"

    '''step1:取出全量数据和mgm数据，并提取其中相应的数据特征'''
    mgm_data = pd.read_csv(data_fir + "final_version_data.csv")
    mgm_data = mgm_data[
        ["ecif_id", "recommend_num", "sex", "birth_date", "ftrade", "fhighest_education", "customer_level",
         "province_name", "audit_status", "card_type", "customer_source", "fcustomer_status", "finvest_status",
         "open_time"]]
    all_data = pd.read_csv(data_fir + "MGM_feature_nv.csv")
    # print len(all_data[all_data['fhighest_education'].notnull()])*1.0/len(all_data)
    all_data = all_data[
        ["ecif_id", "recommend_num", "sex", "birth_date", "ftrade", "fhighest_education", "customer_level",
         "province_name", "audit_status", "card_type", "customer_source", "fcustomer_status", "finvest_status",
         "open_time"]]

    '''step2.1:处理全量数据和mgm数据中的open_time和age问题'''
    mgm_data['open_time'] = mgm_data['open_time'].apply(lambda x: 2017 - int(str(x)[:4]))
    mgm_data['age'] = mgm_data['birth_date'].apply(lambda x: 2017 - int(str(x)[:4]))
    mgm_data.drop('birth_date', axis=1, inplace=True)

    all_data['open_time'] = all_data['open_time'].apply(lambda x: fill_opentime_birth_date(x))
    all_data['age'] = all_data['birth_date'].apply(lambda x: fill_opentime_birth_date(x))
    all_data.drop('birth_date', axis=1, inplace=True)

    '''step2.2:处理全量数据中数据为空的问题，并填充数据'''
    for column in all_data.columns.tolist():
        all_data.loc[(all_data[column].isnull()), column] = -999

    '''step2.3:处理全量数据和mgm中ftrade以及fhighest_education==-10这部分，其实他们也是空的'''
    mgm_data.loc[(mgm_data['ftrade'] == -10), 'ftrade'] = -999
    mgm_data.loc[(mgm_data['fhighest_education'] == -10), 'fhighest_education'] = -999

    all_data.loc[(all_data['ftrade'] == -10), 'ftrade'] = -999
    all_data.loc[(all_data['fhighest_education'] == -10), 'fhighest_education'] = -999

    # print len(all_data[all_data['fhighest_education']==-999])*1.0/len(all_data)

    '''step2.4:处理全量数据和mgm数据中的customer_level和open_time,province_name以及age离散化的问题'''
    mgm_data['customer_level'] = mgm_data['customer_level'].apply(lambda x: process_customer_level(x))
    mgm_data['open_time'] = mgm_data['open_time'].apply(lambda x: process_open_time(x))
    mgm_data['province_name'] = mgm_data['province_name'].apply(lambda x: process_province(x))
    mgm_data['age'] = mgm_data['age'].apply(lambda x: process_age(x))

    all_data['customer_level'] = all_data['customer_level'].apply(lambda x: process_customer_level(x))
    all_data['open_time'] = all_data['open_time'].apply(lambda x: process_open_time(x))
    all_data['province_name'] = all_data['province_name'].apply(lambda x: process_province(x))
    all_data['age'] = all_data['age'].apply(lambda x: process_age(x))

    return mgm_data, all_data


'''将既有数字又有类别型的数据全部转换为类别型，并去除数字型中相应的浮点数'''
def convert_to_category(x):
    if "." in x:
        val = x[:x.index('.')]
        # val='%f' % x##"'" + str(val) + "'"
        return str(val)
    else:
        return x

'''画出财富端的图表'''
def plot_caifuduan_figure():
    import numpy as np
    '''load data'''
    mgm_data, all_data = load_data()
    '''data description'''
    # print mgm_data.head()
    # print all_data.head()
    '''plot figure'''
    diff_list = ["sex", "age", "ftrade", "fhighest_education", "customer_level", "province_name", "audit_status",
                 "card_type", "customer_source", "fcustomer_status", "finvest_status", "open_time"]
    '''define data range'''
    mgm_data_4_gt = mgm_data[mgm_data['recommend_num'] >= 4]
    mgm_data_4_lt = mgm_data[mgm_data['recommend_num'] < 4]
    all_data = all_data

    nototation_list = ['推荐人数>=4', '推荐人数<4', '整体']
    data_list = [mgm_data_4_gt, mgm_data_4_lt, all_data]

    '''数据表的映射关系'''
    sex_dict={'0':'男','1':'女'}
    card_type_dict={'0':'身份证','1':'学生证','2':'工作证','3':'军官证','4':'其他','6':'台湾通行证','7':'护照'}
    audit_status_dict={'1':'未提交','2':'待审核','3':'审核拒绝','4':'待归档','5':'归档打回','6':'已归档','7':'预审核中','8':'审核拒绝'}
    fhighest_education_dict={'0':'中学','1':'本科 / 大专','2':'研究生及以上','3':'小学及以下'}

    '''ensemble data'''
    for column in diff_list:
        fig_data_label_list=[]
        for data in data_list:
            dd = data[column].value_counts()*1.0 / len(data) * 100
            dd = dd.reset_index()
            dd[column] = dd[column].apply(lambda x: remove_2_xiaoshu(x))
            dd['index'] = dd['index'].astype(np.str)
            dd['index'] = dd['index'].apply(lambda x: convert_to_category(x))
            dd.sort(columns=column,inplace=True)
            label_dd = dd['index'].tolist()
            data_dd = dd[column].tolist()

            data_label_dict={}
            for i in range(len(label_dd)):
                data_label_dict[label_dd[i]]=data_dd[i]
            fig_data_label_list.append(data_label_dict)

            print column,data_dd[i],data_label_dict
        '''sorted data by data label,carefully'''
        keys=set()
        for data_label_dict in fig_data_label_list:
            keys=keys.union(set(data_label_dict.keys()))

        label_list=list(keys)
        label_list.sort()

        fig_data_list=[]
        for data_label_dict in fig_data_label_list:
            cur_data_list=[0]*len(label_list)
            for k,v in data_label_dict.items():
                if k in label_list:
                    ind=label_list.index(k)
                    cur_data_list[ind]=v
            fig_data_list.append(cur_data_list)

        '''mapping the data label'''
        if (column == 'sex') or (column == 'card_type') or (column == 'audit_status') or (
            column == 'fhighest_education'):
            tmp_label_list = []
            obj=eval(column+"_dict")
            for k in label_list:
                if k in obj.keys():
                    tmp_label_list.append(obj[k])
                else:
                    tmp_label_list.append(k)
            # print label_list
            # print tmp_label_list
            label_list = tmp_label_list
        else:
            pass

        '''plot figure'''
        chart = ec.Echart(True, theme='macarons')

        chart.use(ec.Title('mgm用户推荐人数为（>=4,<4）,整体对比图', column, x='center'))
        chart.use(ec.Tooltip(trigger='axis'))
        chart.use(ec.Legend(data=nototation_list, position=['center', 'bottom']))
        chart.use(ec.Toolbox(show='true', feature=ec.Feature()))
        chart.use(ec.Axis(param='x', type='category', data=label_list))
        chart.use(ec.Axis(param='y', type='value'))
        itemStyle = ec.ItemStyle(normal={'label': {'show': 'true', 'position':'top','formatter':'{c}'}})
        for i in range(len(fig_data_list)):
            chart.use(ec.Bar(name=nototation_list[i], data=fig_data_list[i],itemStyle=itemStyle))# barWidth=25,markPoint=ec.MarkPoint(), markLine=ec.MarkLine(),
        chart.plot()

def main():
    plot_caifuduan_figure()

if __name__ == '__main__':
    main()
