#coding=utf-8
import time
import sys
from operator import itemgetter

#衰减曲线:max(-(x-N)*(x+N)*1.0/(N*N),0)*int(Mlabel)。
#input:x至今为止实际期数  Mlabel:逾期期数
# output:max(-(x-N)*(x+N)*1.0/(N*N),0)*int(Mlabel)
def decay_curve(x,Mlabel):
	if Mlabel==1:  N=6
	elif Mlabel==2:N=12
	else:          N=24
	return max(-(x-N)*(x+N)*1.0/(N*N),0)*int(Mlabel)

#input:有顺序的逾期情况
#out_put:衰减之后的M6_M1的字典值
def Digital_bit_comparison_double(overdue_list,time_decay,index_dict):
	M1_M6_decay={}
	Current_index=0
	for Mlabel in overdue_list:
		Current_period = index_dict[time_decay[Current_index][2]]  # 至当前时间 的期数
		if int(Mlabel)<7 and int(Mlabel)>0: #至于对M1_M6 才会衰减
			if M1_M6_decay.has_key(Mlabel):#输入当前期数进行衰减合并
				M1_M6_decay[Mlabel]+=decay_curve(int(Current_period),int(Mlabel))
			else:M1_M6_decay[Mlabel]=decay_curve(int(Current_period),int(Mlabel))

		Current_index+=1

	for i  in range(1,7):#加入没有在Mlabel  M1_M6 里面的label设置为0。0
		if M1_M6_decay.has_key(i):
			pass
		else:M1_M6_decay[i]=0.0
	M1_M6_decay_list=[(M1_M6_decay[k]*1.0) for k in sorted(M1_M6_decay.keys(),reverse=True)]  #以便将数据降续排序[M6,M5,M4,M3,M2,M1]
	# M1_M6_decay=sorted(M1_M6_decay.items(), lambda x, y: cmp(x[0], y[0]),reverse=True)  #降序列排序
	return M1_M6_decay_list

#对单个list解析  得到逾期期数list
def find_max_overdue_term(info_list):
	overdue_list=[]
	info_list=info_list.split(",")
	for lines in info_list:
		term, overdue_term, overdue_sart, overdue_end, over_due_amount = lines.split(';')
		overdue_list.append(int(overdue_term))
	return overdue_list

#处理单个合同
def single_contract(ori_line):
	early_target,issue_amount, M7_cur_loan_balance_all, Return_period = int(ori_line[11]),float(ori_line[7]), float(str(ori_line[8])), 0
	if ori_line[-1] in ["", "NULL", "null", "\N"]:  # 如果逾期计算为NULL,  则直接返回 ori_line
		return ori_line

	overdue_list = find_max_overdue_term(ori_line[-1])


	max_overdue_term = max(overdue_list)  #得到单笔合同的逾期情况Mi
	min_overdue_term = min(overdue_list)

	if max_overdue_term < 7:  # M7_cur_loan_balance_all  当前贷款余额设置为0.0
		M7_cur_loan_balance_all = 0.0

	if early_target == 1:  # 提前还清处理  如果提前还清 则置为M0  还款期数为最大
		max_overdue_term = 0
		Return_period = len(overdue_list)+min_overdue_term

	else:  # 没有提前还款则  overdue_list的长度为已还期数
		Return_period = overdue_list.count(0)  # 得到正常还款的期数
	return overdue_list,min_overdue_term,max_overdue_term, M7_cur_loan_balance_all,Return_period, issue_amount


#input:data
#output_ #max_overdue_term:最大逾期期数;M7_cur_loan_balance_all:M7剩余还款,
# M1_M6_decay:M1_M6的时间衰减list,Return_period:已还期数,issue_amount:放款金额
def pross(li):
	issue_amount=0.0
	M7_cur_loan_balance_all=0.0
	Return_period=0
	merge_overdue_list=[]
	M1_M6_decay=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

	time_decay = []
	num=0
	for line in range(0,len(li)):
		every_line = li[line]
		if every_line[-1] in ["", "NULL", "null", "\N"]:
			continue

		overdue_list_temp, min_overdue_term_temp, max_overdue_term_temp, M7_cur_loan_balance_all_temp, Return_period_temp, issue_amount_temp = single_contract(
			every_line)

		issue_amount += issue_amount_temp  # 合同总金额为几个分合同合同金额的加和
		M7_cur_loan_balance_all += M7_cur_loan_balance_all_temp  #
		Return_period += Return_period_temp

        #对M1_M6 进行衰减
		for term_every in every_line[-1].split(","):
			Make_to_time= term_every.split(";")[2].split(" ")[0]        #得到【每期还款开始时间】
			temp=time.mktime(time.strptime(Make_to_time, '%Y-%m-%d'))   #每期还款开始时间 作为后面排序可用
			time_decay.append([num, temp,Make_to_time])
			num+=1

		merge_overdue_list.extend(overdue_list_temp)  # 对逾期期数list进行合并,以备后期找到 M
		index_dict={}

		index_dict_num=0
		if line==len(li)-1:
			sorted_time_decay=time_decay
			for i in sorted(sorted_time_decay, key=lambda l: (l[1]), reverse=True):  #降续排列表示,时间最越大,表示距离最新一遍还款记录时间越近
				if index_dict.has_key(i[2]):  #
					pass
				else:
					index_dict[i[2]]=index_dict_num
					index_dict_num+=1
			M1_M6_decay = Digital_bit_comparison_double(merge_overdue_list, time_decay,index_dict)

	max_overdue_term=max(merge_overdue_list)

	if max_overdue_term<0:
		max_overdue_term=0

	ascending_order=[max_overdue_term, M7_cur_loan_balance_all]
	ascending_order.extend(M1_M6_decay)
	ascending_order.extend([-Return_period, -issue_amount])

	li.append(ascending_order)
	return li

#input:段内分数,key:leval
#output range_expansion:扩展之后的分数
def get_grade(percent,key):
	Scoring_range={0:[800,1000],1:[700,800],2:[600,700],3:[500,600],4:[400,500],5:[300,400],6:[200,300],7:[0,200]}
	range_expansion=Scoring_range[key][0]+percent*(Scoring_range[key][1]-Scoring_range[key][0])
	return range_expansion


def final_score(M_dict,M_dict_index_differ_count,index_sorted,result_arr):
	final_score_dict={}
	Duplicate_removal={}
	for key,val in M_dict.items():
		leval_num=M_dict_index_differ_count[key]
		num_spilt=0
		for every_pople in val:
			percent = num_spilt * 1.0 / leval_num
			if Duplicate_removal.has_key(tuple(every_pople[1:])):  #特征一样,分数相同
				final_score_dict[every_pople[0]]=Duplicate_removal[tuple(every_pople[1:])]
			else:
				Duplicate_removal[tuple(every_pople[1:])]=[num_spilt,percent,get_grade(percent,key)]
				final_score_dict[every_pople[0]]=[num_spilt,percent,get_grade(percent,key)]
		        	num_spilt += 1
	return final_score_dict


#[max_overdue_term, M7_cur_loan_balance_all，M6,M5,M4,M3,M2,M1,-Return_period(已还期数的负数，由于降续排序，已还期数越大应该分数越大)，-issue_amount(合同金额，同理)] 进行降续排序
#out_put   index_sorted排序之后的index:      M_dict:根据Mi进行字典的分类,在每个字典的value[[],[],[]]是有序的。
def sort_compare(result_arr):
	index_sorted=[i[0] for i in sorted(result_arr, key=lambda l: ( l[1],l[2], l[3],l[4], l[5],l[6], l[7],l[8], l[9],l[10]), reverse=True)]
	M_dict={}
	M_dict_index_differ_count={}
	diff_leval_num = []
	for line in index_sorted:
		M_dict.setdefault(result_arr[line][1], []).append(result_arr[line])  #对一个M设置一个字典,字典内部为有序的list,有助于段内排序及分数
		if tuple(result_arr[line][1:]) not in diff_leval_num:
			diff_leval_num.append(tuple(result_arr[line][1:]))
			if M_dict_index_differ_count.has_key(result_arr[line][1]):
				M_dict_index_differ_count[result_arr[line][1]]+=1
			else:
				M_dict_index_differ_count[result_arr[line][1]]=1
	return index_sorted,M_dict,M_dict_index_differ_count


def main():

	result_arr=[]
	data_befor = []
	De_duplication_id={}
	index_num = 0

	for line in sys.stdin:
		li = line.rstrip("\n").split("A")
		De_duplication_id.setdefault(li[2], []).append(li) #将合同按人为维度进行聚合：以身份证号码为key建立字典


	for key,val in De_duplication_id.items():

		result=pross(val)

		ascending_order=[index_num]  #在排序第一列加入index  使后期找到相对应其原信息
		ascending_order.extend(result[-1])
		data_befor.append(result[0][:3])
		result_arr.append(ascending_order)
		index_num+=1

	index_sorted,M_dict,M_dict_index_differ_count=sort_compare(result_arr)#list 排序


	final_score_dict=final_score(M_dict_index_differ_count,M_dict,index_sorted,result_arr) #score
	idx=0
	idx_list=[] #排名分数一样的排名相同
	for line in index_sorted:
                if str(final_score_dict[line][2]) not in idx_list:
			idx += 1
			idx_list.append(str(final_score_dict[line][2]))
                write_row=[str(data_befor[line][2])]
		write_row.extend([str(idx),str(final_score_dict[line][0]),str(final_score_dict[line][2])])
                
		if not isinstance(write_row, list):
                      continue
                sys.stdout.write("%s\n" %("\t".join(write_row)))

if __name__ == "__main__":
    main()

