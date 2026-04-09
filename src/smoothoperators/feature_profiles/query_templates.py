"""This module contains jinja sql templates"""

PARQUET_MIGRATION_TEMPLATE = '''
SET max_partitions_per_insert_block = 500;
CREATE TABLE IF NOT EXISTS {{ db_name }}.{{ table_name }}
ENGINE=MergeTree
ORDER BY event_time_unix
{% if partition_by %}
PARTITION BY {{ partition_by }}
{% else %}
PARTITION BY toYYYYMMDD(event_timestamp_UTC)
{% endif %}
SETTINGS index_granularity = 8192
AS (
    {% if deduplication_keys %}
    SELECT DISTINCT ON ({% for k in deduplication_keys[:-1] %}
                        {{ k }},
                        {% endfor %}
                        {{ deduplication_keys[-1] }})
    {% else %}
    SELECT
    {% endif %}

      {% for col in columns[:-1] %}
      {{ col.source_name }} AS {{ col.target_name }},
      {% endfor %}
      {{ columns[-1].source_name }} AS {{ columns[-1].target_name }}

    FROM file('{{ file_path }}', Parquet)
    WHERE isNotNull({{ event_time_unix }})
          {% if scorable_event %}
          AND {{ scorable_event }} > 0
          {% endif %}
          {% if start_time %}
          AND {{ event_time_unix }} >= {{ start_time }}
          {% endif %}
)
'''

SIN_TEMPLATE = '''
ALTER TABLE {{ database_name }}.{{ table_name }}
ADD COLUMN IF NOT EXISTS {{ target_name }} Float32
DEFAULT SIN(2 * PI() * ({{ time_column }}) / ({{ period }}));
'''

COS_TEMPLATE = '''
ALTER TABLE {{ database_name }}.{{ table_name }}
ADD COLUMN IF NOT EXISTS {{ target_name }} Float32
DEFAULT COS(2 * PI() * ({{ time_column }}) / ({{ period }}));
'''

# This template assumes that the aggregation over the windows will be
# computed in seconds. Please ensure that either a column in seconds
# is being used or that proper unit conversion is being done
# elsewhere.

WINDOW_AGGREGATION_TEMPLATE = '''
CREATE TABLE IF NOT EXISTS {{ db_name }}.{{ target_table_name }}
ENGINE=MergeTree
ORDER BY ({{ order_by | join(", ") }})
SETTINGS index_granularity = 8192
AS (
    WITH
        60 as minute,
	60 * minute as hour,
	24 * hour as day,
        7 * day as week,
        4 * week as month
    SELECT
        {%- for agg in aggregations %}
        {{ agg.expression }} OVER {{ agg.window }} AS {{ agg.col_name }},
        {%- endfor %}
        {%-for k in select_keys %}
        {{ k }}{{"," if not loop.last}}
        {%- endfor %}
    FROM {{ db_name }}.{{ source_table_name }}
    WINDOW
    {%- for w in window_expressions %}
      {{ w.name }} as ({{ w.expression }}){{"," if not loop.last}}
    {%- endfor %}
)
'''
WINDOW_AGGREGATION_TEMPLATE_MILISECONDS = '''
CREATE TABLE IF NOT EXISTS {{ db_name }}.{{ target_table_name }}
ENGINE=MergeTree
ORDER BY ({{ order_by | join(", ") }})
SETTINGS index_granularity = 8192
AS (
    WITH
        60000 as minute,
	60 * minute as hour,
	24 * hour as day,
        7 * day as week,
        4 * week as month
    SELECT
        {%- for agg in aggregations %}
        {{ agg.expression }} OVER {{ agg.window }} AS {{ agg.col_name }},
        {%- endfor %}
        {%-for k in select_keys %}
        {{ k }}{{"," if not loop.last}}
        {%- endfor %}
    FROM {{ db_name }}.{{ source_table_name }}
    WINDOW
    {%- for w in window_expressions %}
      {{ w.name }} as ({{ w.expression }}){{"," if not loop.last}}
    {%- endfor %}
)
'''

WINDOW_TEMPLATE = '''
{%- if partition_by %} PARTITION BY {{ partition_by }} {%- endif %}
{%- if order_by %} ORDER BY {{ order_by }} {%- endif %}
{%- if rows %} ROWS BETWEEN {{ rows.bottom }} PRECEDING AND {{ rows.top }} {{"PRECEDING" if rows.top is defined else "CURRENT ROW"}} {%- endif %}
{%- if range_ %} RANGE BETWEEN {{ range_.bottom }} PRECEDING AND {{ range_.top }} {{"PRECEDING" if range_.top  is defined else "CURRENT ROW"}} {%- endif %}
'''
