#!/bin/sh
DATE1=$(date "+%Y-%m-%d %H:%M:%S")
echo "etl_date is $DATE1"
DATE2=$(date -d last-day +%Y-%m-%d)
echo "date_date is $DATE2"

DATE3=$(date -d last-saturday +%Y-%m-%d)
echo "last saturday is $DATE3"
if [ $DATE2 = $DATE3 ]; then   --≈–∂œlast-day=last-sunday£¨º¥ΩÒÃÏ «–«∆⁄“ª
sqoop eval  --connect "jdbc:mysql://10.130.98.99:3308/cjgwaptest?useUnicode=true&characterEncoding=utf8"   --username "tk_cjgwaptest_rw"   --password "qEVQh7a7T9wUMKpt967n"  -e "delete FROM cs_score"
sqoop export -D mapred.child.java.opts="-Djava.security.egd=file:/dev/../dev/urandom"   --connect "jdbc:mysql://10.130.98.99:3308/cjgwaptest?useUnicode=true&characterEncoding=utf8"   --username "tk_cjgwaptest_rw"   --password "qEVQh7a7T9wUMKpt967n"   --input-null-string '\\N'  --input-null-non-string '\\N'   --table cs_score  --columns "id,idx,group_idx,score,dtime"  --export-dir /user/hive/warehouse/ml/zhxd_dh_final_score  --input-fields-terminated-by '\001';
echo "load success"
else
echo "load none"
fi
