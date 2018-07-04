create table dianxiao_7_v2
as
select all.*, old.gap,
case when old.gap is null then 0
else 1 end as is_old
from
(
select * from dianxiao_good_feature_7_v2
union all
select * from
(
    select label,customerid,callhisid,call_times,connect_times,has_callin,has_staff_hangup,avg_waittime,min_waittime,max_waittime,avg_onlinetime,min_onlinetime,max_onlinetime,all_call_waittime,all_call_onlinetime,comments,datasource,location,callresult,calltime,mobile_type,phone
    from
    (
    select *,row_number() over(order by call_times asc) as rownumber
    from
    test.dianxiao_bad_feature_7_v2
    ) tmp
    where round(tmp.rownumber%4, 1)=0
    ) tmp
) all
left join
dianxiao_old_tag old
on all.callhisid=old.callhisid;