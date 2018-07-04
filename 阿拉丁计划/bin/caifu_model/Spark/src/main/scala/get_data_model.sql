create table test.alading_train_4_10
as
select 
CASE WHEN ecif_label.label is not null  THEN 1 ELSE 0 END as label,
all_ecif_id.ecif_id,tr.*,ac.*,co.*,ky.*,ord.*
from 
(select distinct ecif_id ,datadate from
(
select trade.ecif_id_trade as ecif_id,trade.datadate from alading_trade_features trade
union all
select activity.ecif_id_activity as ecif_id ,activity.datadate from alading_activity_features activity
union all
select communicate.ecif_id_communicate as ecif_id,communicate.datadate from alading_communicate_features communicate
union all
select kyc.ecif_id_kyc as ecif_id,kyc.datadate from alading_kyc_features kyc
union all
select order2.ecif_id_order as ecif_id,order2.datadate from alading_order_features order2
) distinct_id
) all_ecif_id
--加载特征
left join    alading_trade_features tr
on tr.ecif_id_trade=all_ecif_id.ecif_id
and tr.datadate=all_ecif_id.datadate
left join  alading_activity_features ac
on ac.ecif_id_activity=all_ecif_id.ecif_id
and ac.datadate=all_ecif_id.datadate
left join alading_communicate_features co
on co.ecif_id_communicate=all_ecif_id.ecif_id
and co.datadate=all_ecif_id.datadate
left join alading_kyc_features ky
on ky.ecif_id_kyc=all_ecif_id.ecif_id
and  ky.datadate=all_ecif_id.datadate
left join alading_order_features ord
on ord.ecif_id_order=all_ecif_id.ecif_id 
and ord.datadate=all_ecif_id.datadate
left join alading_good_label ecif_label
on ecif_label.ecif_id=all_ecif_id.ecif_id 
and  ecif_label.datadate=all_ecif_id.datadate limit 10;



alter table alading_trade_features change ecif_id ecif_id_trade double;
alter table alading_activity_features change ecif_id ecif_id_activity double;
alter table alading_communicate_features change ecif_id ecif_id_communicate double;
alter table alading_kyc_features change ecif_id ecif_id_kyc double;
alter table alading_order_features change ecif_id ecif_id_order double;