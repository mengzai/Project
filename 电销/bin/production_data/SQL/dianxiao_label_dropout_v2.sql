create table dianxiao_label_dropout_v2
as
	    select b1.customerid, b1.callhisid, b1.calltime, d1.jinjiantime
	    from   
	    (
	    select *
	    from
	    ods.newdxwh_callhistory
	    where calltime is not null and calltime>='2016-01-01'
	    ) b1   
    join   
    	(  
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
	    where  c3.inserttime>='2016-01-01' and c3.customerphone1 is not null  
    ) d1
    on b1.customerid = d1.customerid2
    where (datediff(to_date(d1.jinjiantime),to_date(b1.calltime))>60);