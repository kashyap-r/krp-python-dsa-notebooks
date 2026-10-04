
-- Top 3 customers by monthly spend
-- Detect customers with sudden spending spikes

SELECT
  customer_id,
  payment_date,
  amount,
  amount - LAG(amount) OVER (PARTITION BY customer_id ORDER BY payment_date) AS delta
FROM payment
WHERE amount > 50;

-- Find the distribution of customer spending 
-- find the max and min of amount 
-- divide equally into 10 buckets 
-- assign amount to each bucket based on the threshold 

SELECT max(amount), min(amount)  
FROM payment

--------------------------------------------------
-- Joins & Aggregations (Foundation)
-- Tables: customer, payment, rental
/*
Inner vs Left joins
Anti-joins
GROUP BY correctness
*/
-- Customers with no payments 

Select c.customer_id 
from customer c
LEFT JOIN payment p on c.customer_id = p.customer_id 
where p.customer_id is NULL 

-- Monthly revenue 
select date_trunc('month', payment_date) as month, 
sum(amount) 
from payment 
group by month

/*
Window Functions 
Functions:
	ROW_NUMBER
	RANK
	DENSE_RANK
	LAG, LEAD
	SUM() OVER()
*/

-- Top 3 customers per month 
/*
CTEs & Subqueries
Correlated vs non-correlated
CTE vs inline subquery (performance discussion)
*/

-- Customers above average spend
WITH avg_spend AS (
  SELECT AVG(amount) avg_amt FROM payment
)
SELECT customer_id, SUM(amount)
FROM payment, avg_spend
GROUP BY customer_id
HAVING SUM(amount) > avg_amt;

-- Fraud / Anomaly Style SQL
-- Sudden payment spikes
SELECT
  customer_id,
  payment_date,
  amount,
  amount - LAG(amount) OVER (
    PARTITION BY customer_id ORDER BY payment_date
  ) AS delta
FROM payment
WHERE amount > 50;

-- 
-- Multiple payments within short time window
SELECT customer_id, payment_date
FROM payment p1
WHERE EXISTS (
  SELECT 1
  FROM payment p2
  WHERE p1.customer_id = p2.customer_id
    AND p2.payment_date BETWEEN
        p1.payment_date - INTERVAL '10 minutes'
        AND p1.payment_date
);

/*
Practice explaining:

Why this query works
Edge cases
Performance concerns
*/

-- Query Optimization
EXPLAIN ANALYZE
SELECT * FROM payment WHERE customer_id = 341;

/*
Sequential scan vs index scan
Why indexes help / don’t help
When to avoid indexes
*/



--- Some Important Queries ---

-- Top3 customers per month by spend
/*
First group by monthly spend
then rank

*/

-- Query 1 

SELECT
  customer_id,
  DATE_TRUNC('month', payment_date) AS month,
  SUM(amount) AS total_spend,
  RANK() OVER (
    PARTITION BY DATE_TRUNC('month', payment_date)
    ORDER BY SUM(amount) DESC
  ) AS rnk
FROM payment 
GROUP BY customer_id, month

 
-- this does not include ties 
Select 
	rnk,
	to_char(month, 'Mon-YYYY') as month,
	customer_id, 
	total_spend
FROM (
	Select date_trunc('month', payment_date) as month, 
		customer_id,
		sum(amount) as total_spend,
		row_number() over (
			partition by date_trunc('month', payment_date)
			order by sum(amount) desc
		) as rnk 
	from payment 
	group by month, customer_id
) ranked 
where rnk <= 3 
order by month, total_spend desc ;


-- Top-N per group (PayPal LOVES this) (Using CTE)
WITH monthly_spend AS (
  SELECT
    customer_id,
    DATE_TRUNC('month', payment_date) AS month,
    SUM(amount) AS total_amount
  FROM payment
  GROUP BY customer_id, month
)
SELECT customer_id, month, total_amount
FROM (
  SELECT *,
         RANK() OVER (PARTITION BY month ORDER BY total_amount DESC) rnk
  FROM monthly_spend
) t
WHERE rnk <= 3;

/*
Note: “I aggregate first, then rank. Window functions don’t collapse rows, which is why they’re ideal for top-N problems.”
Window functions preserve rows → perfect for ranking.
*/

/*
2️⃣ Deduplication (classic)(latest record per customer)

Question
Keep latest payment per customer
*/

SELECT *
FROM (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY customer_id
           ORDER BY payment_date DESC
         ) rn
  FROM payment
) t
WHERE rn = 1;

-- “ROW_NUMBER guarantees exactly one row, unlike RANK.”
/*
3️⃣ Anti-join (customers with no payments)
*/

SELECT c.customer_id
FROM customer c
LEFT JOIN payment p
  ON c.customer_id = p.customer_id
WHERE p.customer_id IS NULL;

-- “I prefer LEFT JOIN + IS NULL over NOT IN because of NULL safety.”

/*
4️⃣ Rolling window (7-day revenue)
*/

SELECT
  payment_date::date,
  SUM(amount) OVER (
    ORDER BY payment_date::date
    RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW
  ) AS rolling_7d_revenue
FROM payment;

-- Monthly revenue
SELECT DATE_TRUNC('month', payment_date) month,
       SUM(amount)
FROM payment
GROUP BY month;



/*
5️⃣ LAG / LEAD (behavior change)
*/
SELECT customer_id
FROM payment
GROUP BY customer_id
HAVING MAX(payment_date) < CURRENT_DATE - INTERVAL '30 days';

-- Payment-to-payment delta (LAG)
SELECT customer_id, payment_date,
       amount - LAG(amount) OVER (
         PARTITION BY customer_id ORDER BY payment_date
       ) delta
FROM payment;


/*
ADVANCED / DIFFERENTIATORS
6️⃣ Customers inactive for 30+ days
*/

SELECT customer_id
FROM payment
GROUP BY customer_id
HAVING MAX(payment_date) < CURRENT_DATE - INTERVAL '30 days';

/*
7️⃣ Percentile ranking (Senior signal) - Percentile ranking of customers
*/

SELECT
  customer_id,
  SUM(amount) AS total_spend,
  PERCENT_RANK() OVER (ORDER BY SUM(amount)) AS pct_rank
FROM payment
GROUP BY customer_id;

/*
8️⃣ Find gaps in activity (date gaps)
*/

SELECT
  customer_id,
  payment_date,
  payment_date - LAG(payment_date) OVER (
    PARTITION BY customer_id ORDER BY payment_date
  ) AS gap
FROM payment;

-- Customers inactive for 30 days
Select customer_id 
from payment 
group by customer_id 
having max(payment_date) < current_date - INTERVAL '30 days'

-- Customers with increasing spend
SELECT customer_id
FROM (
  SELECT customer_id,
         amount - LAG(amount) OVER (
           PARTITION BY customer_id ORDER BY payment_date
         ) delta
  FROM payment
) t
GROUP BY customer_id
HAVING MIN(delta) > 0;



/*
9️⃣ Correlated subquery → rewrite (they WILL ask this)
Bad (correlated):
*/

SELECT *
FROM payment p
WHERE amount >
  (SELECT AVG(amount)
   FROM payment
   WHERE customer_id = p.customer_id);

-- Good (set-based):
WITH avg_amt AS (
  SELECT customer_id, AVG(amount) avg_amount
  FROM payment
  GROUP BY customer_id
)
SELECT p.*
FROM payment p
JOIN avg_amt a
  ON p.customer_id = a.customer_id
WHERE p.amount > a.avg_amount;

-- Note: “This avoids repeated scans and is easier to optimize.”




/*
🔟 Fraud-style pattern (PayPal-ish)
Multiple payments in short window:
*/

SELECT p1.customer_id, p1.payment_date
FROM payment p1
JOIN payment p2
  ON p1.customer_id = p2.customer_id
 AND p2.payment_date BETWEEN
     p1.payment_date - INTERVAL '10 minutes'
     AND p1.payment_date
 AND p1.payment_id <> p2.payment_id;

 
-- 1️⃣1️⃣ Running total per customer 
SELECT customer_id, payment_date,
       SUM(amount) OVER (
         PARTITION BY customer_id ORDER BY payment_date
       ) running_total
FROM payment;

-- 1️⃣2️⃣ Payments above global average

SELECT *
FROM payment
WHERE amount > (SELECT AVG(amount) FROM payment);

-- 1️⃣3️⃣ Multiple payments in 10-minute window (fraud-like)
SELECT p1.customer_id, p1.payment_date
FROM payment p1
JOIN payment p2
 ON p1.customer_id = p2.customer_id
AND p2.payment_date BETWEEN
    p1.payment_date - INTERVAL '10 minutes'
    AND p1.payment_date
AND p1.payment_id <> p2.payment_id;


-- 1️⃣4️⃣ Gaps in customer activity
SELECT customer_id,
       payment_date - LAG(payment_date) OVER (
         PARTITION BY customer_id ORDER BY payment_date
       ) gap
FROM payment;

-- 1️⃣5️⃣ Customers with payments every month
SELECT customer_id
FROM payment
GROUP BY customer_id
HAVING COUNT(DISTINCT DATE_TRUNC('month', payment_date)) =
       (SELECT COUNT(DISTINCT DATE_TRUNC('month', payment_date)) FROM payment);

-- 1️⃣6️⃣ Most recent N payments per customer
SELECT *
FROM (
  SELECT *, ROW_NUMBER() OVER (
    PARTITION BY customer_id ORDER BY payment_date DESC
  ) rn
  FROM payment
) t
WHERE rn <= 3;

-- 1️⃣7️⃣ Revenue contribution %
WITH total AS (
  SELECT SUM(amount) total_amt FROM payment
)
SELECT customer_id,
       SUM(amount) / total_amt * 100 AS pct
FROM payment, total
GROUP BY customer_id, total_amt;

-- 1️⃣8️⃣ Customers whose spend doubled MoM
WITH m AS (
  SELECT customer_id,
         DATE_TRUNC('month', payment_date) mth,
         SUM(amount) amt
  FROM payment
  GROUP BY customer_id, mth
)
SELECT *
FROM (
  SELECT *,
         amt / LAG(amt) OVER (
           PARTITION BY customer_id ORDER BY mth
         ) ratio
  FROM m
) t
WHERE ratio >= 2;

-- 1️⃣9️⃣ Payments on weekends
SELECT *
FROM payment
WHERE EXTRACT(DOW FROM payment_date) IN (0,6);

-- 2️⃣0️⃣ Customers with exactly one payment
SELECT customer_id
FROM payment
GROUP BY customer_id
HAVING COUNT(*) = 1;

-- SECTION C — SENIOR / STAFF-LEVEL (21–30)
-- 2️⃣1️⃣ Explain why NOT IN is dangerous

SELECT *
FROM customer
WHERE customer_id NOT IN (SELECT customer_id FROM payment);
/*
❌ Breaks if subquery returns NULL
✅ Use LEFT JOIN instead

If payment.customer_id contains even ONE NULL, the entire condition becomes UNKNOWN, and returns zero rows.

Why?
	SQL uses 3-valued logic: TRUE / FALSE / UNKNOWN
	
	x NOT IN (1, 2, NULL) → UNKNOWN for all x

Senior-level answer

“NOT IN is unsafe in the presence of NULLs. I prefer LEFT JOIN + IS NULL or NOT EXISTS, which are NULL-safe.”
*/
-- Correct alternatives
-- Option 1: LEFT JOIN (preferred)
SELECT c.customer_id
FROM customer c
LEFT JOIN payment p
  ON c.customer_id = p.customer_id
WHERE p.customer_id IS NULL;

-- Option 2: NOT EXISTS
SELECT c.customer_id
FROM customer c
WHERE NOT EXISTS (
  SELECT 1
  FROM payment p
  WHERE p.customer_id = c.customer_id
);


-- 2️⃣2️⃣ Index impact analysis
/*
What they test

Do you understand how the database executes SQL, not just syntax
*/
EXPLAIN ANALYZE
SELECT * FROM payment WHERE customer_id = 10;

Before index:
	Sequential Scan & Scans entire table
Then:

CREATE INDEX idx_payment_customer ON payment(customer_id);

/*
Index Scan
Faster for selective queries

Senior-level explanation

“Indexes help when predicates are selective. For low-cardinality columns or heavy writes, indexes can degrade performance.”

*/

-- 2️⃣3️⃣ Cardinality reduction (performance mindset)
SELECT customer_id, SUM(amount)
FROM payment
WHERE payment_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY customer_id;

/*
Why this is good

	Filters before aggregation	
	Reduces rows early	
	Improves memory & CPU usage

	“I push filters as early as possible to reduce intermediate result size.”
*/

-- 2️⃣4️⃣ Avoid SELECT *

Say: Increases IO, brittle schemas, breaks index-only scans.

Problems
	Reads unnecessary columns → IO waste
	Breaks index-only scans
	Fragile when schema changes
	Harder to cache

Senior answer (verbatim)

“SELECT * increases IO, hides intent, and prevents index-only scans. I always project only required columns.”


-- 2️⃣5️⃣ CTE vs subquery discussion

Say:

“Before PG12, CTEs were optimization fences.”


When to use CTE

	Readability	
	Logical decomposition
	Reuse

When NOT to
	Performance-critical paths (older PG versions)

Senior phrasing

“I use CTEs for clarity, but I’m mindful of optimization fences in older Postgres versions.”


-- 2️⃣6️⃣ DISTINCT vs GROUP BY
SELECT DISTINCT customer_id FROM payment;

vs

SELECT customer_id FROM payment GROUP BY customer_id;

Difference
	DISTINCT → de-dup rows	
	GROUP BY → aggregation semantics

Rule of thumb
	No aggregates → DISTINCT	
	Aggregates → GROUP BY

Senior note
“GROUP BY is more flexible; DISTINCT is simpler but less expressive.”

-- 2️⃣7️⃣ Window function without ORDER BY
SUM(amount) OVER (PARTITION BY customer_id)

Say: Same value per partition.

Meaning
	Same value repeated for each row in partition
	Not cumulative

Use-case
	Compare row value vs partition total

Say this
	“Without ORDER BY, the window spans the entire partition.”

-- 2️⃣8️⃣ Detect duplicate rows
SELECT customer_id, payment_date, COUNT(*)
FROM payment
GROUP BY customer_id, payment_date
HAVING COUNT(*) > 1;

What they test
	Data quality mindset
	Debugging skills

Senior framing
	“This is usually a signal of upstream idempotency or ingestion issues.”

-- 2️⃣9️⃣ NULL-safe aggregation
SELECT SUM(COALESCE(amount,0)) FROM payment;

Why needed
	Aggregates ignore NULLs	
	But arithmetic with NULL → NULL

Senior framing

“I normalize NULLs explicitly to avoid semantic ambiguity.”

-- 3️⃣0️⃣ Explain query, don’t just write it (This is the most important one.)

Say this sentence verbatim:

“I think in set-based operations, reduce data early, and use window functions when row-level context matters.”


-----------------------------------------------------------------------------------------------------------------------

