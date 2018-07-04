add file ./line_on.py;

drop table if exists `test.zhxd_dh_final_score`;
create table if not exists `test.zhxd_dh_final_score` (
 `id_number` string,
 `idx` string,
 `group_idx` string,
 `score` string,
 `dtime` string
);



SET mapreduce.reduce.memory.mb=8192;
set mapreduce.map.memory.mb=4096;
insert overwrite table `test.zhxd_dh_final_score`
select
  transform(g.*)
  using 'python line_on.py'
  as(   id_number,
        idx,
        group_idx,
        score,
        dtime
        )
  from
  (     select
        t.apply_id,
        t.contract_no,
        t.id_number,
        t.contract_start_date,
        t.contract_end_date,
        t.instalment,
        t.contract_amount,
        t.issue_amount,
        t.cur_loan_balance_all,
        t.loan_term,
        t.mn,
        t.early_target,
        t.remain_total,
        t.is_write_off,
        t.is_first_overdue,
        t.first_term_remain_total,
        t.each_term_repayment_start_date,
        t.each_term_business_date,
        t.each_term_overdue_amt,
        t.leval
        from test.zhxd_dh_score_overdue t
  )g;
