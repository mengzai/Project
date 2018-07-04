# encoding=utf8
from echarts import *
import pandas as pd

data=[1                    ,
1                    ,
1                    ,
0.9999583535058539   ,
0.9999583535058539   ,
0.9999583535058539   ,
0.9999583535058539   ,
0.9999583535058539   ,
0.9999531476940857   ,
0.9999479418823174   ,
0.9999453389764333   ,
0.9999245157293603   ,
0.999112409093512    ,
0.9983966099753765   ,
0.9971680383980676   ,
0.9971654354921835   ,
0.997155023868647    ,
0.9971342006215739   ,
0.9971263919039215   ,
0.9970795395980072   ,
0.9961737283503304   ,
0.9961294789503001   ,
0.9960487888678922   ,
0.9933105318777884   ,
0.9926910402773657   ,
0.9926884373714815   ,
0.9713888585216536   ,
0.9708760860624801   ,
0.9699910980618763   ,
0.9688249962257864   ,
0.9660971508592192   ,
0.9659279619767508   ,
0.9659279619767508   ,
0.9659279619767508   ,
0.9657145236942523   ,
0.9654334098587664   ,
0.9619377072563811   ,
0.9554070163931012   ,
0.9551232996517312   ,
0.9536214229565887   ,
0.8930075536328758   ,
0.8923958707501054   ,
0.8922214760558688   ,
0.8917008948790429   ,
0.8916306164201715   ,
0.8832388478497395   ,
0.8830228066613567   ,
0.8701488341584545   ,
0.8589355156096266   ,
0.855031156783433    ,
0.8497550665563035   ,
0.8403013123851468   ,
0.8402987094792627   ,
0.8241268552211689   ,
0.8001801210871817   ,
0.7829124434518697   ,
0.7503032385355011   ,
0.7503032385355011   ,
0.7503032385355011   ,
0.7503032385355011   ,
0.7503032385355011   ,
0.7503032385355011   ,
0.7503032385355011   ,
0.7503032385355011   ,
0.7503032385355011   ,
0.7503032385355011   ,
0.7503032385355011   ,
0.7502043281119042   ,
0.7037190319272436   ,
0.7037190319272436   ,
0.690917940789097    ,
0.690917940789097    ,
0.6892416693997179   ,
0.678512491345338    ,
0.6308975340069654   ,
0.6308975340069654   ,
0.6308975340069654   ,
0.6308975340069654   ,
0.6308975340069654   ,
0.6308975340069654   ,
0.6308975340069654   ,
0.6308975340069654   ,
0.6308975340069654   ,
0.6308975340069654   ,
0.6308975340069654   ,
0.610941054593348    ,
0.5763458324873889   ,
0.47137063818046465  ,
0.4656468481412649   ,
0.44497456960951204  ,
0.4444357680914974   ,
0.3869193567698979   ,
0.3832232304144347   ,
0.22641897414273296  ,
0.2108379795203365   ,
0.16545891833643078  ,
0.15385516390498352  ,
0.1270348216749179   ,
0.08591151161156314  ,
0.08478965917550353  ,
0.028788139078467202 ,
0.02457663735794641  ,
0.02373850166325686  ,
0.01135647837245501  ,
0.0042635598382033705,
3.383777649367754E-5 ,
0
]

index=[
"phone                      ",
"id_number                  ",
"label                      ",
"is_abnormal                ",
"product_type               ",
"apply_max_amount           ",
"loan_type                  ",
"system_source_name         ",
"city_id                    ",
"id_type                    ",
"long_repayment_term        ",
"cus_age_id                 ",
"mobile_type                ",
"cus_age                    ",
"live_province              ",
"live_city                  ",
"education                  ",
"marriage                   ",
"org_type                   ",
"register_province          ",
"live_postcard              ",
"year_income                ",
"entry_date                 ",
"register_city              ",
"org_province               ",
"org_city                   ",
"max_diploma                ",
"contact_num                ",
"house                      ",
"a_annual_income            ",
"resident_province          ",
"credit_grade               ",
"grade_version              ",
"credit_level               ",
"org_provice                ",
"domicile_province          ",
"resident_city              ",
"credit_high_limit          ",
"domicile_city              ",
"recruitment_date           ",
"company_position           ",
"live_times                 ",
"live_case                  ",
"risk_industry              ",
"industry1                  ",
"month_pay                  ",
"house_condition            ",
"register_flag              ",
"loan_purpose               ",
"job_position               ",
"support_persons            ",
"has_car                    ",
"in_city_years              ",
"gender                     ",
"provide_for_count          ",
"loan_count                 ",
"wjq_count                  ",
"jq_count                   ",
"total_count                ",
"ewjq_count                 ",
"ejq_count                  ",
"etotal_count               ",
"applying_count             ",
"apply_passed_count         ",
"apply_reject_count         ",
"apply_total_count          ",
"query_times                ",
"apply_times                ",
"caution                    ",
"score                      ",
"fee_months                 ",
"month_payment              ",
"accept_moth_repay          ",
"month_other_income         ",
"card_cnt                   ",
"n1                         ",
"n2                         ",
"n3                         ",
"n4                         ",
"n5                         ",
"latest_month_income        ",
"end_balance                ",
"month_income               ",
"cash_flow_type             ",
"consumption_habits         ",
"register_postcard          ",
"house_address              ",
"channel_source             ",
"relation                   ",
"credit_cards               ",
"identity_address_province  ",
"identity_address_city      ",
"pledge_num                 ",
"house_amount               ",
"has_children               ",
"id_validity_date           ",
"fam_stable                 ",
"verifying_result           ",
"child_nums                 ",
"monthly_outlay             ",
"customer_ident_type        ",
"corp_regist_capital        ",
"borrower_type              ",
"share_proportion           ",
"money_source               ",
"house_type                 "
]

def reduce_num(x):
    if x<0.1:
        y = "0%-10%"
    elif x<0.2:
        y = "10%-20%"
    elif x<0.3:
        y = "20%-30%"
    elif x<0.4:
        y = "30%-40%"
    elif x<0.5:
        y = "40%-50%"
    elif x<0.6:
        y = "50%-60%"
    elif x<0.7:
        y = "60%-70%"
    elif x<0.8:
        y = "70%-80%"
    elif x<0.9:
        y = "80%-90%"
    else:
        y = "90%+"
    return  y

df_1=pd.read_csv('data',delimiter='\t')
print df_1
data = df_1.loc[:,"coverage_rate"].apply(reduce_num)
print data.value_counts()

indexx=[
"90%+     ",
"60%-70%  ",
"80%-90%  ",
"70%-80%  ",
"0%-10%   ",
"40%-50%  ",
"10%-20%  ",
"20%-30%  ",
"30%-40%  ",
"50%-60%  ",
]

data1=["40",
"16",
"15",
"15",
" 9",
" 4",
" 3",
" 2",
" 2",
" 1",

]
v_data = [float(ii) for ii in data1]
v_index = [str(ii) for ii in indexx]
title_name ="覆盖率分布图"

##ploat
max_num = max(max(v_data),max(v_data))*1.1
chart = Echart(True,theme='macarons')
chart.use(Title(title_name,'特征字段覆盖率分布','center'))
chart.use(Tooltip(trigger='axis'))
chart.use(Toolbox(show='true',feature=Feature()))
chart.use(Axis(param='x',type='category',data=v_index))
chart.use(Axis(param='y',type='value',max=round(max_num,1)))
chart.use(Bar(name='人数',data=v_data,markPoint=MarkPoint(),markLine=MarkLine(),barWidth=50))
chart.plot(-1)