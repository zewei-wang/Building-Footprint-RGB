#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# This script downloads data used in this tutorial from S3 buckets.

# Exit when error occurs
set -e

echo "===== Syncing RGB+LiDAR merged data (~22GB) ... ====="
# Use 'sync' to only download missing or updated files
aws s3 sync s3://aws-satellite-lidar-tutorial/data/ ./data/ --no-sign-request

echo "===== Syncing pretrained model weights (617MB) ... ====="
# Use 'sync' to only download missing or updated files
aws s3 sync s3://aws-satellite-lidar-tutorial/models/ ./models/ --no-sign-request

echo "===== Syncing completes. ====="
