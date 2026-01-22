Select * from employees
-- emp_no, birth_date, first_name, last_name, gender, hire_date

select * from departments
-- dept_no, dept_name

select * from salaries
-- emp_no, salary

select * from dept_emp
-- emp_no, dept_no, from_date, to_date


Select emp_no, 
department, 
salary, 
avg(salary) over (partition by department) as dept_avg_salary 
from employees

SELECT de.dept_no,
       depts.dept_name,
       AVG(s.salary) AS avg_salary
FROM dept_emp de
INNER JOIN salaries s
    ON de.emp_no = s.emp_no
INNER JOIN departments depts
    ON de.dept_no = depts.dept_no
WHERE de.to_date > CURRENT_DATE
GROUP BY de.dept_no, depts.dept_name;

-- Is stupid usage -- because we have to explicityly restrict the output to single row per department using distinct
SELECT DISTINCT de.dept_no,
       depts.dept_name,
       AVG(s.salary) OVER (PARTITION BY de.dept_no) AS avg_salary
FROM dept_emp de
INNER JOIN salaries s
    ON de.emp_no = s.emp_no
INNER JOIN departments depts
    ON de.dept_no = depts.dept_no
WHERE de.to_date > CURRENT_DATE;
/*
🔎 How this works:
- AVG(s.salary) OVER (PARTITION BY de.dept_no) computes the average salary per department without collapsing rows.
- DISTINCT ensures you only get one row per department (otherwise you’d see repeated rows for each employee).
*/

/*
“For each employee, show their salary along with the average salary of their department. The result should list every employee, 
not just one row per department.”

For each employee in the company, display their salary along with the average salary of their department, the overall company average salary, 
and their rank within the department by salary. Finally, filter the results to show only the top 3 employees per department, including ties at 
3rd place if multiple employees share the same salary.

*/

Select dept_no,
       dept_name,
       emp_no,
       salary,
       dept_avg_salary,
       company_avg_salary,
       dept_salary_rank
FROM (
	SELECT 
		de.emp_no, 
		de.dept_no,
	    depts.dept_name,
	    s.salary,
		AVG(s.salary) OVER (PARTITION BY de.dept_no) AS dept_avg_salary,
		AVG(s.salary) OVER () as company_avg_salary,
		RANK() OVER (PARTITION BY de.dept_no ORDER BY s.salary DESC) as dept_salary_rank
	FROM dept_emp de
	INNER JOIN salaries s
	    ON de.emp_no = s.emp_no
	INNER JOIN departments depts
	    ON de.dept_no = depts.dept_no
	WHERE de.to_date > CURRENT_DATE
) ranked 
where dept_salary_rank <= 3
order by dept_no, dept_salary_rank;

---- Include ties 

SELECT dept_no,
       dept_name,
       emp_no,
       salary,
       dept_avg_salary,
       company_avg_salary,
       dept_salary_rank
FROM (
    SELECT de.dept_no,
           depts.dept_name,
           de.emp_no,
           s.salary,
           AVG(s.salary) OVER (PARTITION BY de.dept_no) AS dept_avg_salary,
           AVG(s.salary) OVER () AS company_avg_salary,
           DENSE_RANK() OVER (
               PARTITION BY de.dept_no
               ORDER BY s.salary DESC
           ) AS dept_salary_rank
    FROM dept_emp de
    INNER JOIN salaries s
        ON de.emp_no = s.emp_no
    INNER JOIN departments depts
        ON de.dept_no = depts.dept_no
    WHERE de.to_date > CURRENT_DATE
) ranked
WHERE dept_salary_rank <= 3
ORDER BY dept_no, dept_salary_rank;

