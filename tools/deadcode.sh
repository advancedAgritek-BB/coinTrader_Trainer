#!/usr/bin/env bash
vulture src --min-confidence 70 | tee VULTURE.txt
