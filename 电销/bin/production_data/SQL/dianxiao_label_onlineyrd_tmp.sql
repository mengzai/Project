create table dianxiao_label_onlineyrd_tmp
as
SELECT b.customer_guid 
from
ods.newdxwh_yrd_SUM_BOR_INFO_WHGDDX a
join 
ods.newdxwh_yx_yirendaiapply b
on upper(a.CDS_NUM)= upper(reflect('org.apache.commons.codec.digest.DigestUtils', 'md5Hex',b.papersnumber));


##############################  剔除线上宜人贷之后的负样本
######同一个 customerid 每天只取最后一条callhisid
create table dianxiao_bad_tmp_all
as
 select *
 from
 (
  select *,
  row_number() over(partition by e.customerid, to_date(e.calltime) order by e.calltime desc) as rownumber
  from
  (
  select * 
    from 
    ods.newdxwh_callhistory   a
    where a.calltime is not null and a.calltime>='2016-01-01' 
    and a.customerid is not null and  a.customerid not in (select b.customer_guid as id from dianxiao_label_onlineyrd_tmp b)
   ) e
  ) z
where z.rownumber = 1;