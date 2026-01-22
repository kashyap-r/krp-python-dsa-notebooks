-- Running Balance Problem 
-- Given a transaction (account_id, amount, txn_dt) 
-- Comute running balance per account 

Select account_id, 
	txn_dt, 
	sum(amount) over (partition by account_id order by txn_dt) as running_balance
from transactions 

-- Deduplication with window functions

Select * 
from (
Select *, 
	row_number() over (partition by id order by updated_at DESC) as rn 
from raw_table ) t 
where rn = 1 

-- Normalization challenge 
-- Given a denormalized table of user addresses and transactions, break into normmalized tables and write queries joining them efifciently


-- You are given a transactions table containing payments and refunds 
-- for each user, calculate their daily net transaction amount 
-- Net = sum (payment) - sum(refund)  (for every customer)
-- Output user_id, txn_date, amount

Select
	user_id, 
	txn_date,
	amount, 
	SUM (case 
			when txn_type = 'PAYMENT' THEN amount 
			WHEN txn_type = 'REFUND' THEN -amount)
		end) as net_amount
from transactions 
group by user_id, date(txn_ts)
order by user_id, txn_date

-- if it is a very large table then partition by date 
Select
	user_id, 
	txn_date,
	amount, 
	SUM (case 
			when txn_type = 'PAYMENT' THEN amount 
			WHEN txn_type = 'REFUND' THEN -amount)
		end) OVER (PARTITION by user_id, txn_date) as net_amount
from transactions 
group by user_id, date(txn_ts)
order by user_id, txn_date

	from transaction 

--- If you have to create an index for faster access do it on user_id, txn_date

--- Problem: Identify suspicious users whose daily transaction amount is 3x higher than their 7day moving average 
-- Hints 
	-- Aggregate daily first 
	-- Window functions 
	-- Moving average 


-- daily total
with daily_totals AS (
select user_id, 
		txn_date, 
		sum (case when txn_type = 'PAYMENT' then amount
		else -amount)
		end ) as daily_amount
from transactions
group by user_id, txn_date 
),
-- 7day moving average
moving_average as (
Select user_id,
	txn_date, 
	daily_amount, 
	avg(daily_amount) over (partition by user_id order by txn_date rows between 7 preceeding and 1 preceeding ) as 7d_average
	from daily_totals
)
Select * from moving_average 
where daily_amount > (3 * 7d_average)

/*
Follow up 
- What happens it fewer than 7 days 
	It will aggregate whatever data is available 
- would you exclude weekends 
	WHERE EXTRACT(DOW FROM txn_date) NOT IN (0,6)
	
- how would you productionize this 
	compute this and write it into a daily fact table 
	as define the threshold for fradulent transaction  


- Batch vs streaming
	
*/


/* 
Problem: For each user, find the longest consecutive streak of successful transactions 
Hints: Needs ordering 
Needs grouping fo consecutive rows 

*/

With ranked as (
Select user_id, 
txn_date, 
status, 
row_number() over (partition by user_id order by txn_date) as rn1, 
row_number() over (partition by user_id, status order by txn_Date) as rn2 
from transactions 
), 
groups as (
Select user_idm 
status, 
rn1 - rn2 as grp 
from ranked 
where status = 'SUCCESS'
)
Select user_id
max(count(*)) as longest_success_streak
from groups
group by user_id, grp 


--- Alternate 
WITH ordered_tx AS (
    SELECT
        customer_id,
        payment_id,
        payment_date,
        CASE WHEN amount > 0 THEN 1 ELSE 0 END AS success_flag,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY payment_date) AS rn
    FROM payment
),
streaks AS (
    SELECT
        customer_id,
        payment_id,
        payment_date,
        success_flag,
        rn,
        rn - SUM(success_flag) OVER (PARTITION BY customer_id ORDER BY rn) AS grp
    FROM ordered_tx
    WHERE success_flag = 1
),
grouped AS (
    SELECT
        customer_id,
        grp,
        COUNT(*) AS streak_length
    FROM streaks
    GROUP BY customer_id, grp
)
SELECT
    customer_id,
    MAX(streak_length) AS longest_success_streak
FROM grouped
GROUP BY customer_id
ORDER BY longest_success_streak DESC;

/*
Follow ups 
- What is timestamps are the same 
	Add a secondary ordering column like payment_id in the "ordered_tx"
		ROW_NUMBER() OVER ( PARTITION BY customer_id   ORDER BY payment_date, payment_id ) AS rn
- what of data arrives late
	Always compute streaks based on payment_date not dat ingestion date 
	In production environment, run this as a nighly job when payment transactions are not happening in a geography 
	and write a reconciliation procedure to run when the data is complete. 
	
- how would you index this?
	composite index on customer_id, payment_id, payment_date
	or consider a partial index something like on a condition where payment > 0 
*/


/*
Problem: Deduplication with business rules 
Table: Payments 
Question: You receive duplicate payment records, keep only the latest version per payment_id 

Hint: have to use a windowing function, and filter on rank (to get the latest version)

*/

select payment_id, user_id, amount
from (
Select *, 
	row_number() over (partition by payment_id order by updated_ts desc) as rn
from payments 
) t
where rn = 1 

/*
Question: For each transaction, calculate percentage contribution to the user's monthly total

1 Calculate monthly total 
2. patition by user + month 

*/

Select 
	user_id, 
	txn_date, 
	amount, 
	amount * 100.0 / sum(amount) over (partition by user_id, date_trunc('month', txn_Date)) as pct_contribution 
from transactions


-- alternate 

WITH monthly_totals AS (
    SELECT
        customer_id,
        DATE_TRUNC('month', payment_date) AS month,
        SUM(amount) AS monthly_total
    FROM payment
    GROUP BY customer_id, DATE_TRUNC('month', payment_date)
)
SELECT
    p.payment_id,
    p.customer_id,
    p.payment_date,
    p.amount,
    mt.monthly_total,
    ROUND( (p.amount * 100.0 / mt.monthly_total), 2 ) AS pct_contribution
FROM payment p
JOIN monthly_totals mt
  ON p.customer_id = mt.customer_id
 AND DATE_TRUNC('month', p.payment_date) = mt.month
ORDER BY p.customer_id, p.payment_date;



/*
what if total is zero ?
 USe CASE when amount = 0 Then null 
 else < calculate >
 end case
 
how would you round ? where would this logic live ETL or BI 
	In ETL 
		calculate and store in a fact table or a materialized view 
	In BI 
		round it off before showing it off 
*/ 

/*

*/
