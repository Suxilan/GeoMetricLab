# Losses Module

This module contains training losses and loss composition helpers used by GeoMetricLab.

## Scope

- Metric-learning losses for retrieval training
- Wrapper utilities around third-party loss libraries
- Custom loss components that fit the GeoEncoder training pipeline

## Public focus

The maintained loss surface is centered on instance-level and supervised-scene retrieval objectives.

## Structure

- `instance_losses.py`: retrieval-oriented loss definitions and miners
- `custom/`: local custom loss building blocks