-- Star schema built purely in SQL from the staging views.
-- dim_date : one conformed date dimension every fact joins to.
-- fact_demand_daily / fact_weather_daily : additive daily measures.

CREATE OR REPLACE TABLE dim_date AS
SELECT
    d                                                   AS date_key,
    EXTRACT(year  FROM d)                               AS year,
    EXTRACT(month FROM d)                               AS month,
    EXTRACT(day   FROM d)                               AS day,
    EXTRACT(dow   FROM d)                               AS dow,          -- 0 = Sunday
    EXTRACT(doy   FROM d)                               AS day_of_year,
    (EXTRACT(dow FROM d) IN (0, 6))                     AS is_weekend,
    CASE
        WHEN EXTRACT(month FROM d) IN (12, 1, 2) THEN 'winter'
        WHEN EXTRACT(month FROM d) IN (3, 4, 5)  THEN 'spring'
        WHEN EXTRACT(month FROM d) IN (6, 7, 8)  THEN 'summer'
        ELSE 'autumn'
    END                                                 AS season
FROM (SELECT DISTINCT CAST(timestamp_utc AS DATE) AS d FROM stg_demand_halfhourly);

-- half-hourly -> daily demand, expressed as SQL (mirrors the pandas aggregation in clean.py)
CREATE OR REPLACE TABLE fact_demand_daily AS
SELECT
    CAST(timestamp_utc AS DATE)     AS date_key,
    AVG(nd)                         AS nd_mean,
    MAX(nd)                         AS nd_max,
    MIN(nd)                         AS nd_min,
    MAX(nd) - MIN(nd)               AS nd_range,
    AVG(nd) * 24                    AS nd_total_mwh,
    COUNT(*)                        AS n_periods
FROM stg_demand_halfhourly
GROUP BY 1;

-- hourly -> daily weather (population-weighted mean temperature)
CREATE OR REPLACE TABLE fact_weather_daily AS
SELECT
    CAST(timestamp_utc AS DATE)     AS date_key,
    AVG(gb_temp_pop_weighted)       AS temp_mean_c,
    MIN(gb_temp_pop_weighted)       AS temp_min_c,
    MAX(gb_temp_pop_weighted)       AS temp_max_c
FROM stg_weather_hourly
GROUP BY 1;

-- convenience view: the daily grain with its dimension attributes attached
CREATE OR REPLACE VIEW v_daily AS
SELECT f.date_key, d.year, d.month, d.season, d.is_weekend,
       f.nd_mean, f.nd_range, w.temp_mean_c
FROM fact_demand_daily f
JOIN dim_date d USING (date_key)
LEFT JOIN fact_weather_daily w USING (date_key);
