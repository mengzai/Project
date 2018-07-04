add file /opt/program/work/liliwang/dh_score/line_on/overdue.py;
insert overwrite table `ml.zhxd_dh_score_overdue`
select
  transform(g.*)
  using 'python overdue.py'
  as(   apply_id,
        contract_no,
        id_number,
        contract_start_date,
        contract_end_date,
        instalment,
        contract_amount,
        issue_amount,
        cur_loan_balance_all,
        loan_term,
        mn,
        early_target,
        remain_total,
        is_write_off,
        is_first_overdue,
        first_term_remain_total,
        each_term_repayment_start_date,
        each_term_business_date,
        each_term_overdue_amt,
        leval)
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
        t.each_term_overdue_amt
        from ml.zhxd_dh_score_basis t
  )g;
