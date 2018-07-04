#!/usr/bin/env python

FORMAT = "%Y-%m-%d"
ENABLE_DEBUG = False

import sys
import datetime
from optparse import OptionParser

def squash_stat_info(stat_dict, enable_debug=False):
    keys = stat_dict.keys()
    keys.sort()
    flattern_result_list = []
    for key in keys:
        times = len(stat_dict[key])
        total_amount = sum(map(lambda x : round(float(x[-1]), 2), stat_dict[key]))
        result_item = ["m%d" %key, times, total_amount]
        if enable_debug:
            debug_items = []
            for items in stat_dict[key]:
                debug_items.append("plan:%s, actual:%s, overdue_days:%d, amount:%.2f" %(items))
            result_item.append(debug_items)
        flattern_result_list.append(tuple(result_item))
    return flattern_result_list


def _update_info_to_stat_dict_inline_(stat_dict, overdue_level, planed_date, actual_date, overdue_days, overdue_amt):
    if not stat_dict.has_key(overdue_level):
        stat_dict[overdue_level] = []
    stat_dict[overdue_level].append((planed_date, actual_date, overdue_days, overdue_amt))


def update_info_to_stat(stat_dict, cutoff_date, repayment_start_date_planed, business_dates_actual, overdue_amount, index):
    if business_dates_actual is "":
        business_dates_actual = cutoff_date
    [date_repayment_start_date_planed, date_business_dates_actual] = map(lambda x : datetime.datetime.strptime(x, FORMAT), [repayment_start_date_planed, business_dates_actual])
    days_delta = (date_business_dates_actual - date_repayment_start_date_planed).days
    level = (days_delta - 1) / 30 + 1
    ## hack m0
    if (level == 1) and (days_delta < 1):
        level = 0
    level = level if level <= 7 else 7
    return ';'.join(map(lambda x:str(x),[index,level,date_repayment_start_date_planed,date_business_dates_actual,overdue_amount]))
    _update_info_to_stat_dict_inline_(stat_dict, level, repayment_start_date_planed, business_dates_actual, days_delta, overdue_amount)

def process(li, cutoff_date, enable_debug):
    [each_repayment_start_dates, each_term_business_dates, each_term_overdue_amts] = infos =  li[-3:]
    assert(len(li) >= 3)
    if (each_repayment_start_dates == each_term_business_dates == each_term_overdue_amts == ""):
        li.append("")
        return li
    #print infos
    [each_repayment_start_dates, each_term_business_dates, each_term_overdue_amts] = \
                                                                        infos = \
            [
              dict(i) for i in \
                   map(lambda x : [] if x == "" or x == "\N" \
		                     else map(lambda y : tuple(
				                               map(lambda z : int(z) if z.isdigit() else z,
                                                                   y.split(":")
                                                                  )
				                              ),
                                 x.split(",")),
                   infos)
            ]
    #print infos

    stat_dict = dict()
    info_list = []
    for (index, repayment_start_date_planed) in each_repayment_start_dates.items():
        business_dates_actual = each_term_business_dates.get(index, "")
        overdue_amount = round(float(each_term_overdue_amts.get(index, "0.0")), 2)
        info_list.append(update_info_to_stat(stat_dict, cutoff_date, repayment_start_date_planed, business_dates_actual, overdue_amount, index))
    #print info_list 
    """
    result = squash_stat_info(stat_dict, enable_debug)
    ret = []
    for item in result:
        if enable_debug:
            ret.append("%s:%s:%s:%s" %(item))
        else:
            ret.append("%s:%s:%s" %item)
    li.append(",".join(ret))
    """
    li.append(','.join(info_list))
    return li


def main():
    '''
    try:
        [enable_debug] = sys.argv[1:]
        enable_debug = True if enable_debug.lower() in ["debug", "true"] else False
    except:
        enable_debug = ENABLE_DEBUG
    '''
    parser = OptionParser()
    parser.add_option("-d", "--debug", action="store_true",
                      dest="enable_debug",
                      default=ENABLE_DEBUG,
                      help="enable debug or not")
    parser.add_option("-c", "--cutoff", action="store",
                      dest="cutoff_date",
                      default=datetime.datetime.now(),
                      help="set cutoff date")
    (options, args) = parser.parse_args()
    enable_debug, cutoff_date_str = options.enable_debug, options.cutoff_date
    
    oneday = datetime.timedelta(days=1)
    cutoff_date_str = cutoff_date_str - oneday
    cutoff_date = cutoff_date_str.strftime(FORMAT)

    for line in sys.stdin:
        li = line.rstrip("\n").split("\t")
        result = process(li, cutoff_date, enable_debug)
        if not isinstance(result, list):
            continue
        sys.stdout.write("%s\n" %("\t".join(result)))

if __name__ == "__main__":
    main()

