create table qk_dianxiao_phone_count_7
as
select callhisid, sum(one_month_innum) as one_month_innum,sum(two_month_innum) as two_month_innum ,sum(three_month_innum) as three_month_innum from
(
     select c.phone,d.create_time,t.calltime,t.callhisid,
     case when (t.calltime>d.create_time) and (datediff(to_date(t.calltime),to_date(d.create_time))<=30) then 1 else 0 end as one_month_innum,
     case when (t.calltime>d.create_time) and (datediff(to_date(t.calltime),to_date(d.create_time))<=60) then 1 else 0 end as two_month_innum,
     case when (t.calltime>d.create_time) and (datediff(to_date(t.calltime),to_date(d.create_time))<=90) then 1 else 0 end as three_month_innum
     from ods.qk_tb_clues d
     left join  ods.qk_tb_customers c
     on c.id=d.customer_id
     join  dianxiao_7_v2 t
    on c.phone=t.phone
) final
group by final.callhisid;
  
create table qk_dianxiao_phone_count_10
as
select callhisid, sum(one_month_innum) as one_month_innum,sum(two_month_innum) as two_month_innum ,sum(three_month_innum) as three_month_innum from
(
     select c.phone,d.create_time,t.calltime,t.callhisid,
     case when (t.calltime>d.create_time) and (datediff(to_date(t.calltime),to_date(d.create_time))<=30) then 1 else 0 end as one_month_innum,
     case when (t.calltime>d.create_time) and (datediff(to_date(t.calltime),to_date(d.create_time))<=60) then 1 else 0 end as two_month_innum,
     case when (t.calltime>d.create_time) and (datediff(to_date(t.calltime),to_date(d.create_time))<=90) then 1 else 0 end as three_month_innum
     from ods.qk_tb_clues d
     left join  ods.qk_tb_customers c
     on c.id=d.customer_id
     join  dianxiao_10_v2 t
    on c.phone=t.phone
) final
group by final.callhisid;