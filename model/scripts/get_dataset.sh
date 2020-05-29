#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset

# echo "Materializing table..."
# echo 'select array_to_string(hashtags, " ") from buffer-data.dbt_buffer.predict_hashtags_posts' | \
# bq query --destination_table "buffer-data:temp_exploration.updates_hashtags" --nouse_legacy_sql --replace -n 0

# echo "Exporting table to CSV..."
# bq extract --noprint_header "buffer-data:temp_exploration.updates_hashtags"  "gs://buffer-temp/datasets/updates_hashtags/parts/*"

# echo "Merging files..."
# gsutil compose "gs://buffer-temp/datasets/updates_hashtags/parts/*" "gs://buffer-temp/datasets/updates_hashtags/updates_hashtags.csv"

echo "Downloading table CSV..."
gsutil cp "gs://buffer-temp/datasets/updates_hashtags/updates_hashtags.csv" data/

echo "Done."
