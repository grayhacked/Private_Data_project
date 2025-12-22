WITH data_scope AS (
    SELECT
         displayed_search_id as search_id
         --,SUM(VIEW_COUNT) AS sid_view_year
         ,SUM(VIEW_COUNT) AS sid_view
    FROM SEARCH_SMT.SMT_SCH_AGG_SEARCH_LIST_TRACKING_KPI
    WHERE 0 = 0
      AND SNAPSHOT_DATE between '2024-10-01' and '2025-11-12'
      AND site_id IN (100, 206, 201, 202, 205)
      AND search_session_origin = 'INTERN'
      AND NOT COALESCE(IS_BOT, FALSE)
    GROUP BY 1
),
cum_views AS (
    SELECT
        search_id
        --,sid_view_year
        ,sid_view
        --,SUM(sid_view_year) OVER (ORDER BY sid_view_year DESC ROWS UNBOUNDED PRECEDING) AS cum_view_year
        ,SUM(sid_view) OVER (ORDER BY sid_view DESC ROWS UNBOUNDED PRECEDING) AS cum_view
    FROM data_scope
),
total AS (
    SELECT
        --SUM(sid_view_year) AS total_view_year
        SUM(sid_view) AS total_view
    FROM data_scope
),
tails AS (
    SELECT
        cv.search_id
        ,ds.sid_view
        ,cv.cum_view
        ,cv.cum_view / t.total_view * 100 AS tail,
    FROM cum_views cv
    JOIN data_scope ds ON ds.search_id = cv.search_id
    CROSS JOIN total t
),
tail_filter AS (
    SELECT *
    FROM tails
    WHERE 0 = 0
        AND tail <= 50
)
,categ as(
    SELECT DISTINCT t.search_id,
        tail,
        PRODUCT_STEERING_FAMILY_LEVEL1_NAME as CAT1,
        PRODUCT_STEERING_FAMILY_LEVEL2_NAME as CAT2,
        PRODUCT_STEERING_FAMILY_LEVEL3_NAME as CAT3,
        SUM(PRODUCT_ORDER_TURNOVER) AS va_categ,
        SUM(CLICK_COUNT) AS click
    FROM DATA_PRD.SEARCH_SMT.SMT_SCH_AGG_SEARCH_LIST_OFFER_PERFORMANCE perf
    inner join tail_filter t on t.search_id = perf.search_id
    WHERE 0=0
        AND perf.snapshot_date BETWEEN '2024-10-01' and '2025-11-12'
        AND PAGE_NUMBER = 1
    GROUP BY 1,2,3,4,5
    QUALIFY ROW_NUMBER() OVER (PARTITION BY t.search_id ORDER BY va_categ DESC, click DESC) = 1
)
SELECT
    to_timestamp(perf.snapshot_date) as date
    ,displayed_search_id AS search_id
    ,tail
    ,CAT1
    ,CAT2
    ,CAT3
    ,SUM(perf.view_count) AS vues_LR
    ,SUM(perf.click_count) + SUM(perf.direct_from_lr_add_to_basket_count) AS clics_LR
    ,SUM(perf.turnover) AS VA_LR
    ,SUM(perf.order_count) AS commandes_LR
FROM DATA_PRD.SEARCH_SMT.SMT_SCH_AGG_SEARCH_LIST_TRACKING_KPI perf
INNER JOIN categ on categ.search_id = perf.displayed_search_id
WHERE 0 = 0
    AND perf.displayed_search_id <> ''
    and snapshot_date between '2024-10-01' and '2025-11-12'
GROUP BY all
ORDER BY 1,vues_LR desc 