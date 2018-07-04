################################  信贷+线下宜人贷
######全部的callhisid
create table dianxiao_label_xinhexin_tmp_all_v2
as
select *
from

(
select* 
from
ods.newdxwh_callhistory
where calltime is not null and calltime>='2016-01-01'  --打电话时开始时间不为NULL，且在16年1月1号之后
) b

join
(
  select 
  c1.phone1 as phone, 
  c2.SENDAPPLY_TIME as jinjiantime,   --提交申请时间
  c3.inserttime,  --插入日期
  c3.customerid as customerid2
  from
  ods.icp_T_BEE_CUSTOMER c1
  join
  ods.icp_T_BEE_TRANSPORT c2
  on c1.CUSTOMER_ID=c2.CUSTOMER_ID
  join
  ods.newdxwh_customer c3
  on c1.phone1=c3.customerphone1   --通过手机号 关联
  where c2.IS_VALID=1 and c2.SENDAPPLY_TIME is not null and c3.inserttime>='2016-01-01' and c3.customerphone1 is not null and c1.phone1 is not null and (unix_timestamp(c2.SENDAPPLY_TIME)>unix_timestamp(c3.inserttime))
) c
--IS_VALID 申请是否有效（0：否  1：是）不允许空值
--SENDAPPLY_TIME  提交申请时间
--逻辑  申请有效，并且已经提交进件，在电销中插入时间>='2016-01-01',保证电话都不为null，进件时间>电销中的时间 进件时间与打电话时间<=60天  （前面-后面）
on b.customerid = c.customerid2
where (unix_timestamp(c.jinjiantime)-unix_timestamp(b.calltime)>0) and (datediff(to_date(c.jinjiantime),to_date(b.calltime))<=60);


######同一个 customerid 每天只取最后一条callhisid  当customerid相同且拨打时间相同时候 to_date变为天啦，保证一天一个客户取最后一条数据
create table dianxiao_label_xindai_tmp_all_v2
as
 select *
 from
 (
  select *,
  row_number() over(partition by e.customerid, to_date(e.calltime) order by e.calltime desc) as rownumber
  from dianxiao_label_xinhexin_tmp_all_v2 e
  ) z
where z.rownumber = 1;



################################  车贷
######全部的callhisid
create table dianxiao_label_chedai_tmp_all_v2
as 
select *
from 
( 
select * 
from 
ods.newdxwh_callhistory 
where calltime is not null and calltime>='2016-01-01' 
) b 
join 
(  
  select jinjian.mobile as phone,jinjian.intotime as jinjiantime,c3.inserttime, c3.customerid as customerid2 
  from 
  ods.autoloan_al_apply jinjian
  join 
  ods.newdxwh_customer c3 
  on jinjian.mobile=c3.customerphone1 
  where jinjian.intotime is not null and c3.inserttime>='2016-01-01' and c3.customerphone1 is not null and (unix_timestamp(jinjian.intotime)>unix_timestamp(c3.inserttime))  
) c 
on b.customerid = c.customerid2 
where (unix_timestamp(c.jinjiantime)-unix_timestamp(b.calltime)>0) and (datediff(to_date(c.jinjiantime),to_date(b.calltime))<=60);  
 
######同一个 customerid 每天只取最后一条callhisid
create table dianxiao_label_chedai_tmp_distinct_all_v2
as
 select *
 from
 (
  select *,
  row_number() over(partition by e.customerid, to_date(e.calltime) order by e.calltime desc) as rownumber
  from dianxiao_label_chedai_tmp_all_v2 e
  ) z
where z.rownumber = 1;