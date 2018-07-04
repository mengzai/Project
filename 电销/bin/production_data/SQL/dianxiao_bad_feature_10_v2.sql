create table dianxiao_bad_feature_10_v2
as
select 0 as label, feature2.customerid, feature1.*, feature2.comments, feature2.datasource, feature2.location, feature2.callresult, feature2.calltime,
case when find_in_set(substr(cus.customerphone1,1,3),'134,135,136,137,138,139,150,151,152,157,158,159,182,183,184,187,188')>0 then 1 --yidong
     when find_in_set(substr(cus.customerphone1,1,3),'130,131,132,155,156,185,186,145')>0 then 2 --liantong
     when find_in_set(substr(cus.customerphone1,1,3),'133,153,180,181,189,173,177')>0 then 3 --dianxi
     else 4 end mobile_type, cus.customerphone1 as phone
from
(
select final.callhisid, count(*) as call_times, sum(coalesce(final.is_connect,0)) as connect_times,
case when sum(final.is_callin)>=1 then 1 when sum(final.is_callin) is null then null else 0 end as has_callin,
case when sum(final.is_staff_hangup)>=1 then 1 when sum(final.is_staff_hangup) is null then null else 0 end as has_staff_hangup,
if(sum(final.is_connect)>=1, avg(final.waittime), NULL) as avg_waittime,
if(sum(final.is_connect)>=1, min(coalesce(final.waittime, 999999999)), NULL) as min_waittime,
if(sum(final.is_connect)>=1, max(coalesce(final.waittime, 0)), NULL) as max_waittime,
if(sum(final.is_connect)>=1, avg(final.onlinetime), NULL) as avg_onlinetime,
if(sum(final.is_connect)>=1, min(coalesce(final.onlinetime, 999999999)), NULL) as min_onlinetime,
if(sum(final.is_connect)>=1, max(coalesce(final.onlinetime, 0)), NULL) as max_onlinetime,
concat_ws(',', collect_list(concat(coalesce(calltime2,''),'@',coalesce(waittime,'')))) as all_call_waittime,
concat_ws(',', collect_list(concat(coalesce(calltime2,''),'@', coalesce(onlinetime,'')))) as all_call_onlinetime
from
(
select tmp1.customerid, tmp1.callhisid, tmp1.calltime as calltime,tmp2.calltime as calltime2, tmp2.waittime, tmp2.onlinetime, tmp2.is_connect,
tmp2.is_staff_hangup, tmp2.is_callin
from
(
select *
from
dianxiao_bad_tmp_all
where calltime>='2016-10-01' and calltime<='2016-10-31'
) tmp1
join
(
select customerid, callhisid, calltime,
if(connecttime is not null and calltime is not null, (unix_timestamp(connecttime)-unix_timestamp(calltime)), NULL) as waittime,
if(offtime is not null and connecttime is not null, (unix_timestamp(offtime)-unix_timestamp(connecttime)), NULL)  as onlinetime,
case when connecttime is not null then 1 else 0 end as is_connect,
case when hanguptype=2 then 1 when hanguptype is null then null else 0 end as is_staff_hangup,
case when calldirection=1 then 1  when calldirection is null then null else 0 end as is_callin
from
ods.newdxwh_callhistory
where  calltime<='2016-10-31'
) tmp2
on tmp1.customerid=tmp2.customerid
where tmp1.calltime>=tmp2.calltime
and tmp1.callhisid not in (select b.callhisid as id from dianxiao_label_xinhexin_tmp_all_v2 b
                           union all
                           select c.callhisid as id from  dianxiao_label_chedai_tmp_all_v2 c
                           union all
                           select d.callhisid as id from dianxiao_label_dropout_v2 d)
) final
group by final.callhisid
) feature1
join
ods.newdxwh_callhistory feature2
on feature1.callhisid=feature2.callhisid
left join
ods.newdxwh_customer cus
on cus.customerid=feature2.customerid
where cus.customerphone1 is not null;