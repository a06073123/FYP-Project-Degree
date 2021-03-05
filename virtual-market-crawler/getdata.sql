SELECT DATE(Time), sum(SumCount), avg(MinPrice) FROM bdo.trade_record
WHERE bdo.trade_record.MainKey BETWEEN 7001 AND 7005
group by DATE(Time);