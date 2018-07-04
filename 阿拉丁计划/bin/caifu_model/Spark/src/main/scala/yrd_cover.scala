package cn.creditease.invest.test
/**
  * Created by Administrator on 2017/8/17.
  */

import org.apache.spark.rdd.RDD
import org.apache.spark.{Accumulator, SparkConf, SparkContext}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types.{DoubleType, LongType, StringType, StructField, StructType}

object yrd_cover {
  def fieldDescribe(sc: SparkContext,hiveCtx: HiveContext,tableName:String) = {
    val df = hiveCtx.sql(s"select * from $tableName")
    val dfSize: Double = df.count().toDouble
    //assert(dfSize>0)
    val columnName: Array[String] = df.columns
    df.show()
    val rawDescrbe: SchemaRDD = df.describe(columnName.toSeq:_*)

    // save DataFrame
    val fieldName: Array[String] = rawDescrbe.columns.drop(1)
    val dataTranspose: Array[Row] = fieldName
      .map(x =>(Array(tableName,x) ++ rawDescrbe.select(x).map(x => x.toString.replaceAll("\\[|\\]","")).collect()))
      .map(x=>Row(x:_*))

    // Trans to DataFrame
//    val dfName: Array[String] =  Array("table_name","field_name","count","mean","stddev","min","max")
//    val data: RDD[Row] = sc.parallelize(dataTranspose)
//    //DataFrame (field: Array[String],data: RDD[Row])
//    val schema = StructType(dfName.map(fieldName => StructField(fieldName, StringType, true)))
//    val dfDescrOut = hiveCtx.createDataFrame(data, schema).withColumn("coverage_rate",col("count")/dfSize)
//    dfDescrOut.show()
//
//    // insert into hive
//    dfDescrOut.registerTempTable("tmp_table")
//    hiveCtx.sql(s"use test")
//    hiveCtx.sql(s"insert into table testtable_state_describe select * from tmp_table")
  }


  def file_name()={
    val all_file_name=
        List(
          "yrd_abatement_advance_apply                        ",
          "yrd_actual_cashflow                                ",
          "yrd_actual_repayment                               ",
          "yrd_actual_repayment_de_al_n                       ",
          "yrd_actual_repayment_detail                        ",
          "yrd_actual_repayment_detail_al                     ",
          "yrd_actual_repayment_detail_allot                  ",
          "yrd_actual_return                                  ",
          "yrd_actual_return_allot                            ",
          "yrd_admin_user                                     ",
          "yrd_app_credit_investigation                       ",
          "yrd_app_supplement_contact                         ",
          "yrd_app_supplement_file                            ",
          "yrd_app_supplement_info                            ",
          "yrd_app_t02_rmc_all_res_trust                      ",
          "yrd_application_bank_water                         ",
          "yrd_application_borrower_info                      ",
          "yrd_application_contact_info                       ",
          "yrd_application_credit_ufo                         ",
          "yrd_application_denied_info                        ",
          "yrd_application_denied_rules                       ",
          "yrd_application_dic                                ",
          "yrd_application_image_file                         ",
          "yrd_application_info                               ",
          "yrd_application_info_other                         ",
          "yrd_application_log                                ",
          "yrd_application_step                               ",
          "yrd_application_transport                          ",
          "yrd_apply_credit_investigation                     ",
          "yrd_apply_more_info                                ",
          "yrd_auction_balance                                ",
          "yrd_audit_application                              ",
          "yrd_audit_contact_info                             ",
          "yrd_audit_log                                      ",
          "yrd_b_apply_info                                   ",
          "yrd_b_basic_info                                   ",
          "yrd_b_basic_info_his                               ",
          "yrd_bl_worker_detail                               ",
          "yrd_borrow_rate                                    ",
          "yrd_businesscode                                   ",
          "yrd_c_address_info                                 ",
          "yrd_c_address_info_his                             ",
          "yrd_c_contact_info                                 ",
          "yrd_c_contact_info_creditaudit                     ",
          "yrd_c_contact_info_his                             ",
          "yrd_c_user_balance_payment                         ",
          "yrd_c_user_balance_payment_his                     ",
          "yrd_c_user_regist_info                             ",
          "yrd_contract_clause                                ",
          "yrd_cps_relate_user_info                           ",
          "yrd_data_citys                                     ",
          "yrd_data_dic_items                                 ",
          "yrd_deduct_apply_bill                              ",
          "yrd_discount_code_t                                ",
          "yrd_draw_info                                      ",
          "yrd_draw_info_his                                  ",
          "yrd_draw_user_bank                                 ",
          "yrd_dw_t02_application_borrower_info               ",
          "yrd_enterprise_info                                ",
          "yrd_finace_observe_income_info                     ",
          "yrd_finace_product                                 ",
          "yrd_finaces_manager                                ",
          "yrd_finance_exit_process                           ",
          "yrd_finance_renew_process                          ",
          "yrd_flow_account                                   ",
          "yrd_flow_business                                  ",
          "yrd_flow_cash                                      ",
          "yrd_flow_dic_cash_type                             ",
          "yrd_lender_auction_brief                           ",
          "yrd_lender_info                                    ",
          "yrd_loan                                           ",
          "yrd_loan_per_period                                ",
          "yrd_loan_per_period_n                              ",
          "yrd_loan_per_period_trade                          ",
          "yrd_loan_per_period_trade_n                        ",
          "yrd_loop_loan_info                                 ",
          "yrd_organization                                   ",
          "yrd_p_borrower_product                             ",
          "yrd_p_product                                      ",
          "yrd_pay_bill_main                                  ",
          "yrd_pay_order                                      ",
          "yrd_pay_source                                     ",
          "yrd_product_application_chl                        ",
          "yrd_product_audit_channel                          ",
          "yrd_product_info                                   ",
          "yrd_product_info_modify                            ",
          "yrd_product_rate                                   ",
          "yrd_r_uid_ecifid                                   ",
          "yrd_regist_channel                                 ",
          "yrd_repayment_plan                                 ",
          "yrd_repayment_plan_change                          ",
          "yrd_repayment_statistics                           ",
          "yrd_return_overdue                                 ",
          "yrd_rmc_all_result_m3                              ",
          "yrd_rmc_all_result_m3_month                        ",
          "yrd_s_bank                                         ",
          "yrd_s_city                                         ",
          "yrd_s_deined                                       ",
          "yrd_selluser                                       ",
          "yrd_smp_application_borrower_info                  ",
          "yrd_smp_application_info                           ",
          "yrd_smp_audit_application                          ",
          "yrd_student_info                                   ",
          "yrd_sum_bor_info_shxcdx                            ",
          "yrd_sum_bor_info_whgddx                            ",
          "yrd_tla_transfer_loan_apply                        ",
          "yrd_total_m6                                       ",
          "yrd_tpw_recharge                                   ",
          "yrd_transport_channel                              ",
          "yrd_trust_abatement_penalty                        ",
          "yrd_trust_contract_clause                          ",
          "yrd_trust_contract_detail                          ",
          "yrd_trust_contract_download                        ",
          "yrd_trust_loan_info                                ",
          "yrd_trust_loan_info_log                            ",
          "yrd_trust_prepayment_refund                        ",
          "yrd_trust_public_repayment_rec                     ",
          "yrd_trust_repayment_plan                           ",
          "yrd_trust_repayment_record                         ",
          "yrd_trust_repayment_record_detail                  ",
          "yrd_trust_repayment_statistics                     ",
          "yrd_trust_sys_deduct_info                          ",
          "yrd_trust_sys_deduct_info_log                      ",
          "yrd_trust_sys_payment_info                         ",
          "yrd_trust_sys_payment_info_log                     ",
          "yrd_uceas                                          ",
          "yrd_umbrella_borrower                              ",
          "yrd_umbrella_data_citys                            ",
          "yrd_umbrella_seller                                ",
          "yrd_umbrella_sellorg                               ",
          "yrd_user_account_new                               ",
          "yrd_user_bank                                      ",
          "yrd_user_denied_loan                               ",
          "yrd_userinfoaudit                                  ",
          "yrd_userinfoauditdesc                              ",
          "yrd_v_yrd_application_info                         ",
          "yrd_v_yrd_c_address_info                           ",
          "yrd_v_yrd_contract_clause                          ",
          "yrd_whitelist_off_line                             ",
          "yrd_yrd_all_sum                                    ",
          "yrd_yrd_data_citys                                 ",
          "yrd_yrd_loan_orders                                ",
          "yrd_yrd_organization                               ",
          "yrd_yrd_os                                         ",
          "yrd_yrd_selluser                                   ",
          "yrdgengmei_customer_info_reportenter_daily         ",
          "yrdgengmei_loan_apply_info_dailyreport_daily       ",
          "yrdgengmei_loan_apply_info_supervise_daily         ",
          "yrdgengmei_loan_badload_supervise_monthly          ",
          "yrdgengmei_loan_detail_supervise_monthly           ",
          "yrdgengmei_loan_refund_detail_monthly              ",
          "yrdgengmei_loan_refund_detail_supervise_monthly    ",
          "yrdgengmei_loan_refund_reportenter_daily           ",
          "yrdgengmei_revenue_monthly                         ",
          "yrdgengmei_surplus_info_dailyreport_daily          ",
          "yrdgengmei_surplus_info_dailyreport_monthly        ")
    val file_name=List(
      "bs_tsm_user"
    )
    file_name
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("fieldDescribe")
    val sc = new SparkContext(conf)
    val hiveCtx = new HiveContext(sc)
    sc.setLogLevel("ERROR")

    val tableList = file_name()
    hiveCtx.sql(s"use test")

    val tableDescribe = fieldDescribe(sc,hiveCtx,_:String)
    for (tableName <- tableList){
      tableDescribe("ods."+tableName)
    }
  }
}