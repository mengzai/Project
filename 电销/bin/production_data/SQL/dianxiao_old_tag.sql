##############################找到老客户
create table dianxiao_old_tag
as
select z1.customerid, z1.callhisid, z1.calltime, z1.jinjiantime, z1.phone, case when find_in_set(substr(z1.phone,1,3),'134,135,136,137,138,139,150,151,152,157,158,159,182,183,184,187,188')>0 then 1 --yidong
     when find_in_set(substr(z1.phone,1,3),'130,131,132,155,156,185,186,145')>0 then 2 --liantong
     when find_in_set(substr(z1.phone,1,3),'133,153,180,181,189,173,177')>0 then 3 --dianxi
     else 4 end mobile_type,  z1.gap
     --保证电话为联通电信移动这三种电话信息
 from
 (
  select e1.*,
  row_number() over(partition by e1.callhisid order by to_date(e1.jinjiantime) desc) as rownumber  --老客户中选择进件时间最早的
  from
	  (
		    select b1.customerid, b1.callhisid, b1.calltime, d1.jinjiantime, d1.phone, datediff(to_date(b1.calltime),to_date(d1.jinjiantime)) as gap --打电话时间-进件时间为gap
		    from   
	    	(
		    select *
		    from
		    ods.newdxwh_callhistory
		    where calltime is not null
		    ) b1   
	    join   
		    ( 
		    --综合信贷和车贷的手机号 客户id union all  并选择与电销中有重复的客户
		    select c1.phone, c1.jinjiantime, c3.customerid as customerid2
		    from   
			    (
			    	select tmp1.phone, tmp1.jinjiantime
			    	from
				    (
				    	select c1.phone1 as phone, c2.SENDAPPLY_TIME as jinjiantime from
				    	ods.icp_T_BEE_CUSTOMER c1
				   	 	join
				    	ods.icp_T_BEE_TRANSPORT c2
				    	on c1.CUSTOMER_ID=c2.CUSTOMER_ID
				    	where c2.IS_VALID=1 and c2.SENDAPPLY_TIME is not null and c1.phone1 is not null
				    ) tmp1
			    union all
					select tmp2.phone, tmp2.jinjiantime
					from
					    (
					    select jinjian.mobile as phone,jinjian.intotime as jinjiantime 
					    from   
					    ods.autoloan_al_apply jinjian 
					    where jinjian.intotime is not null and jinjian.mobile is not null
					    ) tmp2
			    ) c1
		    join
		    ods.newdxwh_customer c3
		    on c1.phone=c3.customerphone1
		    where  c3.inserttime>'2016-01-01' and c3.customerphone1 is not null  
		    ) d1
	    on b1.customerid = d1.customerid2
	    where (unix_timestamp(d1.jinjiantime)<unix_timestamp(b1.calltime))--进件时间在打电话之前表示这个合同已经之前有合同啦，表示老客户
 	) e1
 ) z1
where z1.rownumber = 1;