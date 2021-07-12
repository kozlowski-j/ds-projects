--1. Return all records from the TRIPS table with the names of the starting and end stations for trips starting on Alcatraz Ave at Shattuck Ave.

SELECT a.*
    ,b.name as "start_station_name"
    ,c.name as "end_station_name"
    FROM trips a
        JOIN station_info b
            ON a.start_station_id = b.station_id
        JOIN station_info c
            ON a.end_station_id = c.station_id
WHERE b.name LIKE 'Alcatraz Ave at Shattuck Ave';

--2. Return the number and average trip length (in seconds) by the region names of the starting station for 2018 trips.

SELECT c.name, count(*) as "count", AVG(( CAST( end_date AS DATE ) - CAST( start_date AS DATE ) ) * 86400) as "avg_duration_sec"
FROM trips a
    JOIN station_info b
        ON a.start_station_id = b.station_id
    JOIN regions c
        ON b.region_id = c.region_id
WHERE EXTRACT(YEAR FROM a.start_date) = 2018
GROUP BY c.name;


--3. Return a (unique) list of 'id' and abbreviated names of all stations, at which at least 5 trips ended - save the results to the table STATION_A

--DROP TABLE STATION_A;
CREATE TABLE STATION_A AS 
SELECT
    a.end_station_id
    ,b.short_name
    ,count(*) as cnt
FROM trips a
    JOIN station_info b
        ON b.station_id = a.end_station_id
GROUP BY a.end_station_id, b.short_name
HAVING count(*) >= 5
ORDER BY 3;

--4. Return a (unique) list of the 'id' of all end stations to which no trips have been recorded, from a station where there are currently no bikes available - store the results in table STATION_B.

CREATE TABLE STATION_B AS 
    SELECT DISTINCT(end_station_id)
    FROM trips
    WHERE start_station_id not in (SELECT station_id
                                    FROM station_status 
                                    WHERE num_bikes_available = 0);


--5. Combine the results from steps 3. and 4. returning only the station ids that appear in both lists.
SELECT *
FROM station_a a
    INNER JOIN station_b b
        ON a.end_station_id = b.end_station_id;


--6. For each region, indicate the longest trip that began in that region (the end station may be in a different region).
--Consider only trips that began in 2018. 
--(hint: estimate the length of the trip by calculating the length in a straight line, use the STATION_INFO.STATION_GEOM field and Pythagoras' theorem).

CREATE OR REPLACE FUNCTION distance (Lat1 IN NUMBER,
                                     Lon1 IN NUMBER,
                                     Lat2 IN NUMBER,
                                     Lon2 IN NUMBER) RETURN NUMBER IS
BEGIN
  RETURN SQRT( POWER(Lat1 - Lat2, 2) + POWER(Lon1 - Lon2, 2));
END;

CREATE OR REPLACE VIEW trips_distances AS
SELECT region_id, trip_id, distance(start_station_lat, start_station_lon,
                                    end_station_lat, end_station_lon) as distance
FROM (
    SELECT b.region_id, a.trip_id
                ,TO_NUMBER(trim(SUBSTR(SUBSTR(b.station_geom, 7), 0, REGEXP_INSTR(SUBSTR(b.station_geom, 7), ' '))), '999.9999999999999') as start_station_lat
                ,TO_NUMBER(trim(REPLACE(SUBSTR(b.station_geom, REGEXP_INSTR(b.station_geom, ' ')), ')', '')), '999.9999999999999') as start_station_lon
                ,TO_NUMBER(trim(SUBSTR(SUBSTR(c.station_geom, 7), 0, REGEXP_INSTR(SUBSTR(c.station_geom, 7), ' '))), '999.9999999999999') as end_station_lat
                ,TO_NUMBER(trim(REPLACE(SUBSTR(c.station_geom, REGEXP_INSTR(c.station_geom, ' ')), ')', '')), '999.9999999999999') as end_station_lon
                FROM trips a
                    JOIN station_info b
                        ON a.start_station_id = b.station_id
                    JOIN station_info c
                        ON a.end_station_id = c.station_id
                    WHERE EXTRACT(YEAR FROM a.start_date) = 2018
);

SELECT 
    b.name as region
    ,c.trip_id
    ,a.max_distance

FROM (
    SELECT 
        region_id
        ,MAX(distance) as max_distance
    FROM trips_distances 
    GROUP BY region_id
    ) a
        JOIN regions b
            ON a.region_id = b.region_id
        JOIN trips_distances c
            ON a.max_distance = c.distance;


--7. For each trip, return a list containing the start date and end date of the trip and its 'id', the name of the start and end stations (and their 'id'). In the same table, also return the start and end dates, along with the name of the end station previous trips that started at the same start station.

SELECT d.*, e.name as previous_trip_end_station_name
FROM (
    SELECT
        a.start_station_id
        ,a.trip_id
        ,a.start_date
        ,a.end_date
        ,b.name as start_station_name
        ,a.end_station_id
        ,c.name as end_station_name
        ,LAG(a.trip_id) OVER(
                        PARTITION BY a.start_station_id
                        ORDER BY a.start_date
                    ) previous_trip_id
        ,LAG(a.start_date) OVER(
                        PARTITION BY a.start_station_id
                        ORDER BY a.start_date
                    ) previous_trip_start_date
        ,LAG(a.end_date) OVER(
                        PARTITION BY a.start_station_id
                        ORDER BY a.start_date
                    ) previous_trip_end_date
        ,LAG(a.end_station_id) OVER(
                        PARTITION BY a.start_station_id
                        ORDER BY a.start_date
                    ) previous_trip_end_station_id
    FROM
        trips a
        LEFT JOIN station_info b
            ON a.start_station_id = b.station_id
        LEFT JOIN station_info c
            ON a.end_station_id = c.station_id) d
    LEFT JOIN station_info e
        ON d.previous_trip_end_station_id = e.station_id
ORDER BY 1, 3;

