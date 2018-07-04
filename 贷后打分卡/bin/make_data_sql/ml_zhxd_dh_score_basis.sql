use ml;
insert overwrite table `ml.zhxd_dh_score_basis`
select
t1.apply_id,--进件编号
t1.contract_no as contract_no,--合同号
t1.customer_identity_no as id_number,--证件号码
t1.contract_start_date,--合同开始日期
t1.contract_end_date,--合同结束日期
ii.instalment,--当前所属期数
t1.contract_amount,--合同金额
t1.issue_amount,--放款金额

t3.cur_loan_balance_all , --当前贷款余额(M7合同不为0)
t1.loan_term, --放款期数
t3.mn,--逾期期数
case when t1.contract_status in('14','15') then 1 else 0  end early_target, --14:提前结清   15:放款退回结清
bb.remain_capital + bb.remain_interest + bb.remain_amerce + bb.remain_forfeit + bb.remain_other_fee as remain_total,--逾期金额
t3.is_write_off, --是否核销合同
if (re.contract_no is null,0,1) as is_first_overdue, --首期逾期标识
re.first_term_remain_total,--首期逾期金额
pp.each_term_repayment_start_date ,--每期应还日期
pp.each_term_business_date ,--每期实际还款日期
pp.each_term_overdue_amt --每期逾期金额
from

--贷后数据
(
select * from
   (
    select *,
    row_number() over(partition by apply_id order by contract_start_date desc) as rownum
    from dw.fact_tcsv_contract
   ) t
   where t.rownum=1
) t1

--计算当期期数
join (
select contract_no, max(instalment) as instalment, max(repayment_start_date) as repayment_start_date
from dw.fact_tcsv_repayment_plan
where repayment_start_date <=date_sub(from_unixtime(unix_timestamp(),'yyyy-MM-dd'),1)
group by contract_no
) ii on t1.contract_no=ii.contract_no


--逾期金额
left join (
select contract_no, max(overdue_instalment) as overdue_instalment,
max(overdue_days) as overdue_days,
sum(case when settle_flag = 1 then should_capital else coalesce(remain_capital, 0) end) as remain_capital,
sum(case when settle_flag = 1 then should_interest else coalesce(remain_interest, 0) end) as remain_interest,
sum(case when settle_flag = 1 then should_amerce else coalesce(remain_amerce, 0) end) as remain_amerce,
sum(case when settle_flag = 1 then should_forfeit else coalesce(remain_forfeit, 0) end) as remain_forfeit,
sum(case when settle_flag = 1 then should_other_fee else coalesce(remain_other_fee, 0) end) as remain_other_fee
from dw.fact_tcsv_repayment_plan
where repayment_start_date <=date_sub(from_unixtime(unix_timestamp(),'yyyy-MM-dd'),1)
and overdue_flag=2
group by contract_no
) bb on t1.contract_no=bb.contract_no

--显示每期期数（去掉提前还款的记录，但是保留提前还款的第一笔记录（有应还金额））
left join (
select contract_no,
sum(case when t.prepayment_date_current is not null
or t.should_capital+t.should_interest+t.should_instalment_fee+t.should_amerce+t.should_forfeit+t.should_other_fee=0  --提前结清
 then 1 else 0 end) as target,
concat_ws(',', collect_list(concat(instalment,':', coalesce(business_date,prepayment_date_current,''))) ) as each_term_business_date,
concat_ws(',', collect_list(concat(instalment,':', coalesce(repayment_start_date,''))) ) as each_term_repayment_start_date,
concat_ws(',', collect_list(concat(instalment,':', ( case when overdue_flag=2 then
case when settle_flag = 1
then should_capital + should_interest + should_amerce + should_forfeit + should_other_fee
else remain_capital + remain_interest + remain_amerce + remain_forfeit + remain_other_fee
end
else 0 end)
)
)
) each_term_overdue_amt
from dw.fact_tcsv_repayment_plan t
where t.repayment_start_date <=date_sub(from_unixtime(unix_timestamp(),'yyyy-MM-dd'),1)
group by t.contract_no
) pp on t1.contract_no=pp.contract_no

left join dm.tcsv_contract_loan_balance t3 on t1.contract_no=t3.contract_no and t3.partition_date=date_sub(from_unixtime(unix_timestamp(),'yyyy-MM-dd'),1)

--首期逾期标识
left join (select t.contract_no,
case when settle_flag = 1
then should_capital + should_interest + should_amerce + should_forfeit + should_other_fee
else remain_capital + remain_interest + remain_amerce + remain_forfeit + remain_other_fee
end as first_term_remain_total --首期逾期金额
from dw.fact_tcsv_repayment_plan t
where t.instalment=1
and t.overdue_flag=2
) re on t1.contract_no = re.contract_no

where t1.product_category_id in('1','3','7','10','13' )--城市信贷：精英贷、新薪贷、新薪宜楼贷、精英贷（银行合作）、新薪贷（银行合作）
and t1.contract_start_date<=date_sub(from_unixtime(unix_timestamp(),'yyyy-MM-dd'),1);

