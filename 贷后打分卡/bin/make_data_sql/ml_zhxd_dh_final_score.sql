add file /opt/program/work/liliwang/dh_score/line_on/dh_score_time.py;

set mapreduce.map.memory.mb=20480;
set mapreduce.reduce.memory.mb=20480;
set mapred.task.timeout=6000000;

insert overwrite table `ml.zhxd_dh_final_score`
select
  transform(g.*)
  using 'python dh_score_time.py' --添加合同时间的截止时间，如果后期上线时间应该为当天时间的前一天
  as(   id_number,
        idx,
        group_idx,
        score,
        dtime
        )
  from
  (     select *
        from ml.zhxd_dh_score_overdue t
        order by t.id_number) g;
