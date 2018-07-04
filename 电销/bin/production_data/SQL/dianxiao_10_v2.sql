create table dianxiao_10_v2
as
select all.*, old.gap,
case when old.gap is null then 0
else 1 end as is_old
from
(
select * from dianxiao_good_feature_10_v2
union all
select * from dianxiao_bad_feature_10_v2
) all
left join
dianxiao_old_tag old
on all.callhisid=old.callhisid;