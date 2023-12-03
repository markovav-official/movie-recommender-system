# Movie Recommender System Project

## Overview

This repository contains the implementation of a movie recommender system using Nearest Neighbors and LightGCN models on the MovieLens 100K dataset.

## Author

- Name: Andrei Markov
- Email: `a.markov@innopolis.university`
- Group Number: B21-AAI-01

## Repository Structure

- `data`: Contains datasets.
- `models`: Stores trained and serialized models.
- `notebooks`: Includes Jupyter notebooks for data exploration and model training.
- `references`: Data dictionaries, manuals, and other explanatory materials.
- `reports`: Contains reports.
- `benchmark`: Dataset used for evaluation and the evaluation script.

## Running the benchmark

- Create a virtual environment `python -m venv venv` in project root and activate it
- Install all libraries `pip install -r requirements.txt`
- Go to `benchmark` folder and run `python evaluate.py [all/nearest_neighbors/lightgcn]`
