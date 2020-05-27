#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset

echo "Materializing table..."

echo '
with recent_updates as (
  select
    *
  from dbt_buffer.publish_updates where synced_at > timestamp_sub(current_timestamp(), interval 180 day)
     and profile_service = "instagram"

), extracted_hashtags as (
  select
    id
    , created_at
    , regexp_extract_all(text, "#[a-zA-Z0-9_.+-]+") as hashtags
  from recent_updates
  where array_length(regexp_extract_all(text, "#[a-zA-Z0-9_.+-]+")) > 0

), parsed_hashtags as (
  select
    distinct replace(lower(array_to_string(hashtags, " ")), ".", "") as hashtag_line
    , array_length(regexp_extract_all(replace(lower(array_to_string(hashtags, " ")), ".", ""), "#")) AS n
  from extracted_hashtags
)

select
  hashtag_line
from parsed_hashtags
where n > 3' | \
bq query --destination_table "buffer-data:temp_exploration.updates_hashtags" --nouse_legacy_sql --replace -n 0

echo "Exporting table to CSV..."
bq extract --noprint_header "buffer-data:temp_exploration.updates_hashtags"  "gs://buffer-temp/updates_hashtags.csv"

echo "Downloading table CSV"
gsutil cp "gs://buffer-temp/updates_hashtags.csv" data/
