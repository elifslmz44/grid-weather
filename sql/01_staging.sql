-- Staging: load the cleaned pipeline outputs straight from CSV. DuckDB reads CSV natively,
-- so the "warehouse" needs no import step -- these views point at the tidy processed files.
-- (__DEMAND_CSV__ / __WEATHER_CSV__ are substituted with absolute paths by src/warehouse.py.)
CREATE OR REPLACE VIEW stg_demand_halfhourly AS
    SELECT * FROM read_csv_auto('__DEMAND_CSV__', header=true);

CREATE OR REPLACE VIEW stg_weather_hourly AS
    SELECT * FROM read_csv_auto('__WEATHER_CSV__', header=true);
