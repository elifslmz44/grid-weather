-- Analytical marts: the kind of questions an analyst would actually ask of the star schema.

-- mean demand by season
CREATE OR REPLACE VIEW mart_season AS
SELECT season,
       ROUND(AVG(nd_mean), 0)  AS mean_demand_mw,
       COUNT(*)                AS days
FROM v_daily
GROUP BY season
ORDER BY mean_demand_mw DESC;

-- weekday vs weekend
CREATE OR REPLACE VIEW mart_weekday AS
SELECT CASE WHEN is_weekend THEN 'weekend' ELSE 'weekday' END AS day_type,
       ROUND(AVG(nd_mean), 0) AS mean_demand_mw
FROM v_daily
GROUP BY 1
ORDER BY mean_demand_mw DESC;

-- demand response by 1 C temperature bin (the relationship the whole project exploits)
CREATE OR REPLACE VIEW mart_temp_response AS
SELECT CAST(ROUND(temp_mean_c) AS INTEGER) AS temp_c,
       ROUND(AVG(nd_mean), 0)              AS mean_demand_mw,
       COUNT(*)                            AS days
FROM v_daily
WHERE temp_mean_c IS NOT NULL
GROUP BY 1
HAVING COUNT(*) >= 5
ORDER BY temp_c;
